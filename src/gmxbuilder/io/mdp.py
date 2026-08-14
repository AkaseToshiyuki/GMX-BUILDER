"""MDP file writer — generates standard multi-stage MD protocols.

Supports:
  - minim:       Energy minimization (steepest descent)
  - nvt_heat:    NVT heating with strong restraints
  - nvt_eq:      NVT equilibration (decaying restraints)
  - npt_eq:      NPT equilibration (decaying restraints, semi-isotropic for membranes)
  - prod:        Production run (no restraints)
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import re


class MDPWriter:
    """Write GROMACS .mdp parameter files with production-quality parameters."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        output_dir: str | Path,
        params: dict,
        eq_stages: list[dict] | None = None,
        prod_iters: list[dict] | None = None,
        minimization: dict | None = None,
    ) -> list[Path]:
        """Generate the full MDP suite and return the list of written paths.

        Parameters
        ----------
        output_dir : Path
        params : dict
            Authoritative system context (force-field family, membrane class,
            coupling-group count, and available restraint macros). Legacy
            callers may still provide version-1 global values; new clients use
            stage-owned settings returned by ``normalize_simulation_config``.
        eq_stages : list[dict] or None
            Per-stage equilibration configs. Each dict can override:
            bb, sc, lipid, dih, dt, nsteps, ensemble, tcoupl, tau_t,
            nstcomm, tau_p, ref_p, compress.
            Each supplied dt must declare ``dt_unit`` (``fs`` or ``ps``).
            Legacy callers may use the explicit boolean ``dt_fs`` flag.
            If None, uses built-in default schedule.
        prod_iters : list[dict] or None
            Production iteration configs. Each dict: nsteps, dt, nstxout,
            tcoupl, tau_t, tau_p, ref_p, compress, nstcomm.
            If None, uses a single default production run.
        minimization : dict or None
            Independent minimization settings, including restraints,
            non-bonded parameters, constraints, and advanced overrides.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        self.validate_protocol(params, eq_stages, prod_iters, minimization)
        schedule_source = _default_schedule(params) if eq_stages is None else eq_stages
        hydrated_schedule: list[dict] = []
        for source_index, stage in enumerate(schedule_source):
            ensemble = str(
                stage.get("ensemble", "nvt" if source_index < 2 else "npt")
            ).lower()
            hydrated = _stage_defaults(params, ensemble)
            hydrated.update(copy.deepcopy(stage))
            hydrated_schedule.append(hydrated)
        schedule = [
            stage for stage in hydrated_schedule if stage.get("enabled", True)
        ]
        # Convert the explicitly declared unit exactly once.  Never infer a
        # timestep unit from magnitude: 0.5 can legitimately mean either
        # 0.5 fs or 0.5 ps and the latter is unsafe for atomistic MD.
        for rst in schedule:
            _convert_timestep_to_ps(rst)

        # ---- Stage 0: Minimization ----
        minim_config = _default_minimization(params)
        for legacy_key, stage_key in {
            "em_integrator": "integrator",
            "em_nsteps": "nsteps",
            "em_ftol": "emtol",
            "em_step": "emstep",
            "em_nstlist": "nstlist",
            "em_constraints": "constraints",
            "em_overrides": "mdp_overrides",
        }.items():
            if legacy_key in params:
                minim_config[stage_key] = copy.deepcopy(params[legacy_key])
        if minimization is not None:
            minim_config.update(copy.deepcopy(minimization))
        minim = _build_minim(params, minim_config)
        minim = _apply_mdp_overrides(
            minim, minim_config.get("mdp_overrides", {})
        )
        paths.append(self._write(output_dir / "mini.mdp", minim))

        # ---- Stages 1-N: Equilibration ----
        enabled_schedule = [
            (index, stage) for index, stage in enumerate(hydrated_schedule)
            if stage.get("enabled", True)
        ]
        for active_index, (source_index, _source_stage) in enumerate(enabled_schedule):
            rst = schedule[active_index]
            stage_num = source_index + 1
            dt = rst.get("dt", params.get("dt", 0.002))
            nsteps = rst.get("nsteps", int(params.get("eq_nsteps", 250000)))
            is_first = active_index == 0
            ensemble = rst.get("ensemble", "nvt" if source_index < 2 else "npt")

            if ensemble in ("nvt",):
                content = _build_nvt(params, rst, dt, nsteps, is_first, stage_num)
            else:
                content = _build_npt(params, rst, dt, nsteps, stage_num, is_first)
            content = _apply_mdp_overrides(content, params.get("mdp_overrides", {}))
            content = _apply_mdp_overrides(content, rst.get("mdp_overrides", {}))
            paths.append(self._write(output_dir / f"equili_{stage_num}.mdp", content))

        # ---- Production ----
        production_source = (
            _default_production(params) if prod_iters is None else prod_iters
        )
        production: list[dict] = []
        for stage in production_source:
            if not stage.get("enabled", True):
                continue
            hydrated = _stage_defaults(params, "npt")
            hydrated.update(copy.deepcopy(stage))
            for _ in range(int(stage.get("repeat", 1))):
                production.append(copy.deepcopy(hydrated))
        for pr in production:
            pr.pop("repeat", None)
            _convert_timestep_to_ps(pr)
        for active_index, pr in enumerate(production):
            prod_params = dict(params)
            prod_params.update(pr)
            suffix = f"_{active_index + 1}" if len(production) > 1 else ""
            content = _build_prod(prod_params)
            content = _apply_mdp_overrides(content, params.get("mdp_overrides", {}))
            content = _apply_mdp_overrides(content, pr.get("mdp_overrides", {}))
            paths.append(self._write(output_dir / f"production{suffix}.mdp", content))

        return paths

    @staticmethod
    def validate_protocol(
        params: dict,
        eq_stages: list[dict] | None = None,
        prod_iters: list[dict] | None = None,
        minimization: dict | None = None,
    ) -> None:
        """Reject malformed or unsafe user-editable MDP protocol values."""
        if not isinstance(params, dict):
            raise ValueError("MDP parameters must be an object")
        unknown_global = sorted(set(params) - _GLOBAL_KEYS)
        if unknown_global:
            raise ValueError(
                "unknown global MDP setting(s): " + ", ".join(unknown_global)
            )
        _validate_stage(params, "global settings")
        if params.get("pcoupl_type", "auto") not in {"auto", "isotropic", "semisotropic"}:
            raise ValueError("global settings pcoupl_type must be auto, isotropic, or semisotropic")
        has_membrane = bool(params.get("has_membrane", True))
        _validate_comm_groups(
            params.get("comm_grps", "System"), has_membrane, "global settings"
        )
        for key in ("em_nsteps", "em_nstlist"):
            if key in params:
                value = _finite_number(params[key], f"global settings {key}")
                if isinstance(params[key], bool) or not value.is_integer() or value < 1:
                    raise ValueError(
                        f"global settings {key} must be a positive integer"
                    )
        if params.get("em_constraints", "h-bonds") not in {
            "none", "h-bonds", "all-bonds", "h-angles", "all-angles"
        }:
            raise ValueError("global settings em_constraints value is not supported")
        if params.get("em_integrator", "steep") not in {"steep", "cg"}:
            raise ValueError("global settings em_integrator must be steep or cg")
        if "gen_seed" in params:
            seed = _finite_number(params["gen_seed"], "global settings gen_seed")
            if isinstance(params["gen_seed"], bool) or not seed.is_integer() or seed < -1:
                raise ValueError("global settings gen_seed must be an integer of -1 or greater")
        if "n_tc_groups" in params:
            groups = _finite_number(
                params["n_tc_groups"], "global settings n_tc_groups"
            )
            if not groups.is_integer() or not 1 <= groups <= 3:
                raise ValueError("global settings n_tc_groups must be 1, 2, or 3")
        for flag in (
            "has_membrane", "protein_position_restraints",
            "lipid_position_restraints", "lipid_dihedral_restraints",
        ):
            if flag in params and not isinstance(params[flag], bool):
                raise ValueError(f"global settings {flag} must be true or false")
        for key in (
            "em_ftol", "em_step", "temperature", "rlist", "rvdw",
            "rcoulomb", "fourierspacing", "compressibility",
        ):
            if key in params and _finite_number(
                params[key], f"global settings {key}"
            ) <= 0:
                raise ValueError(f"global settings {key} must be positive")
        _validate_nonbond_geometry(params, "global settings")
        for label, stages in (("equilibration", eq_stages), ("production", prod_iters)):
            if stages is None:
                continue
            if not isinstance(stages, list) or len(stages) > 32:
                raise ValueError(f"{label} stages must be a list with at most 32 entries")
            if not stages:
                raise ValueError(f"at least one {label} stage must be enabled")
            for index, stage in enumerate(stages, 1):
                if not isinstance(stage, dict):
                    raise ValueError(f"{label} stage {index} must be an object")
                unknown_stage = sorted(set(stage) - _STAGE_KEYS)
                if unknown_stage:
                    raise ValueError(
                        f"unknown {label} stage {index} setting(s): "
                        + ", ".join(unknown_stage)
                    )
                if "enabled" in stage and not isinstance(stage["enabled"], bool):
                    raise ValueError(f"{label} stage {index} enabled must be true or false")
                if label == "equilibration" and "repeat" in stage:
                    raise ValueError(
                        f"{label} stage {index} repeat is only valid for production"
                    )
                if not stage.get("enabled", True):
                    continue
                _validate_stage(stage, f"{label} stage {index}")
                if _force_field_family(params) == "charmm" and str(
                    stage.get("dispcorr", "no")
                ).lower() not in {"no", "none"}:
                    raise ValueError(
                        f"{label} stage {index}: CHARMM36/CHARMM36m requires "
                        "DispCorr=no with its force-switch non-bonded protocol"
                    )
                _validate_comm_groups(
                    stage.get("comm_grps", params.get("comm_grps", "System")),
                    has_membrane,
                    f"{label} stage {index}",
                )
                _validate_override_comm_groups(
                    stage.get("mdp_overrides", {}),
                    has_membrane,
                    f"{label} stage {index} overrides",
                )
            if not any(stage.get("enabled", True) for stage in stages):
                raise ValueError(f"at least one {label} stage must be enabled")
        _validate_overrides(params.get("mdp_overrides", {}), "global MDP overrides")
        _validate_overrides(params.get("em_overrides", {}), "minimization MDP overrides")
        _validate_override_comm_groups(
            params.get("mdp_overrides", {}), has_membrane, "global MDP overrides"
        )
        _validate_minimization(minimization, params)

    @staticmethod
    def normalize_simulation_config(
        config: object | None,
        context: dict | None = None,
    ) -> dict[str, object]:
        """Return the version-2, stage-owned simulation configuration.

        Workflow controls and execution hardware are deliberately kept out of
        the MDP context.  Legacy version-1 task state is migrated by copying
        former global MDP values into each stage only when that stage did not
        already provide an explicit value.
        """
        raw = {} if config is None else config
        if not isinstance(raw, dict):
            raise ValueError("simulation parameters must be an object")
        unknown = sorted(set(raw) - _SIMULATION_CONFIG_KEYS)
        if unknown:
            raise ValueError(
                "unknown simulation setting(s): " + ", ".join(unknown)
            )
        schema_version = raw.get("schema_version", 1)
        if isinstance(schema_version, bool) or schema_version not in {1, 2}:
            raise ValueError("simulation schema_version must be 1 or 2")
        if schema_version == 2:
            misplaced = sorted(
                set(raw) & (_GLOBAL_KEYS | {"system_name", "mdp_overrides_text"})
            )
            if misplaced:
                raise ValueError(
                    "schema version 2 requires MDP values to belong to "
                    "minimization, an equilibration stage, or a production stage; "
                    "misplaced setting(s): " + ", ".join(misplaced)
                )
        browser_text = raw.get("mdp_overrides_text", "")
        if browser_text not in (None, ""):
            raise ValueError(
                "mdp_overrides_text is browser-only state; reload the task "
                "and enter overrides in the relevant simulation stage"
            )

        mdp_context = dict(context or {})
        for key in _CONTEXT_KEYS:
            if key in raw and key not in mdp_context:
                mdp_context[key] = raw[key]

        legacy_stage = {
            key: copy.deepcopy(raw[key])
            for key in _LEGACY_STAGE_KEYS
            if key in raw
        }
        legacy_overrides = copy.deepcopy(raw.get("mdp_overrides", {}))

        eq_source = raw.get("eq_stages")
        if eq_source is None:
            eq_source = _default_schedule(mdp_context)
        prod_source = raw.get("prod_iters")
        if prod_source is None:
            prod_source = _default_production(mdp_context)

        def hydrate(stages: object, label: str) -> list[dict]:
            if not isinstance(stages, list):
                raise ValueError(f"{label} stages must be a list")
            hydrated: list[dict] = []
            for index, stage in enumerate(stages, 1):
                if not isinstance(stage, dict):
                    raise ValueError(f"{label} stage {index} must be an object")
                ensemble = str(stage.get("ensemble", "npt")).lower()
                defaults = _stage_defaults(mdp_context, ensemble)
                defaults.update(copy.deepcopy(legacy_stage))
                stage_overrides = copy.deepcopy(stage.get("mdp_overrides", {}))
                defaults.update(copy.deepcopy(stage))
                defaults["mdp_overrides"] = {
                    **legacy_overrides,
                    **stage_overrides,
                }
                hydrated.append(defaults)
            return hydrated

        eq_stages = hydrate(eq_source, "equilibration")
        prod_iters = hydrate(prod_source, "production")

        minim = _default_minimization(mdp_context)
        legacy_min_map = {
            "em_integrator": "integrator",
            "em_nsteps": "nsteps",
            "em_ftol": "emtol",
            "em_step": "emstep",
            "em_nstlist": "nstlist",
            "em_constraints": "constraints",
        }
        for old_key, new_key in legacy_min_map.items():
            if old_key in raw:
                minim[new_key] = copy.deepcopy(raw[old_key])
        for key in _NONBOND_KEYS:
            if key in raw:
                minim[key] = copy.deepcopy(raw[key])
        if "em_overrides" in raw:
            minim["mdp_overrides"] = copy.deepcopy(raw["em_overrides"])
        explicit_minim = raw.get("minimization")
        if explicit_minim is not None:
            if not isinstance(explicit_minim, dict):
                raise ValueError("minimization settings must be an object")
            minim.update(copy.deepcopy(explicit_minim))

        MDPWriter.validate_protocol(
            mdp_context, eq_stages, prod_iters, minim
        )
        return {
            "schema_version": 2,
            "minimization": minim,
            "eq_stages": eq_stages,
            "prod_iters": prod_iters,
            "hardware": copy.deepcopy(raw.get("hardware", {})),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _write(path: Path, content: str) -> Path:
        path.write_text(content)
        return path


_MDP_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_INDEX_GROUP = re.compile(r"^[A-Za-z][A-Za-z0-9_+-]*$")
_BASE_COM_GROUPS = frozenset({"System", "SOLU", "SOLV"})
_MEMBRANE_COM_GROUPS = frozenset({"MEMB", "SOLU_MEMB"})
_STAGE_KEYS = frozenset({
    "enabled", "repeat", "bb", "sc", "lipid", "dih", "dt", "dt_unit",
    "dt_fs", "nsteps",
    "ensemble", "tcoupl", "tau_t", "temperature", "comm_mode", "comm_grps",
    "nstcomm", "constraints", "pcoupl", "pcoupl_type", "tau_p", "ref_p",
    "compress", "nstlist", "nstxout_compressed", "nstxout", "nstvout",
    "nstfout", "nstcalcenergy", "nstenergy", "nstlog", "mdp_overrides",
    "rlist", "vdw_modifier", "rvdw_switch", "rvdw", "rcoulomb",
    "fourierspacing", "dispcorr", "gen_seed",
})
_NONBOND_KEYS = frozenset({
    "rlist", "vdw_modifier", "rvdw_switch", "rvdw", "rcoulomb",
    "fourierspacing", "dispcorr", "nstlist",
})
_MINIMIZATION_KEYS = frozenset({
    "integrator", "nsteps", "emtol", "emstep", "nstlist", "constraints",
    "bb", "sc", "lipid", "dih", "mdp_overrides",
}) | _NONBOND_KEYS
_GLOBAL_KEYS = (_STAGE_KEYS - {"enabled", "repeat"}) | frozenset({
    "em_nsteps", "em_nstlist", "em_constraints", "em_ftol", "em_step",
    "em_integrator", "em_overrides", "eq_nsteps", "prod_nsteps",
    "production_nsteps", "force_field", "force_field_family", "has_membrane",
    "n_tc_groups", "protein_position_restraints", "lipid_position_restraints",
    "lipid_dihedral_restraints", "compressibility", "gen_seed",
})
_CONTEXT_KEYS = frozenset({
    "force_field", "force_field_family", "has_membrane", "n_tc_groups",
    "protein_position_restraints", "lipid_position_restraints",
    "lipid_dihedral_restraints",
})
_LEGACY_STAGE_KEYS = (_STAGE_KEYS - {
    "enabled", "repeat", "bb", "sc", "lipid", "dih", "dt", "dt_unit",
    "dt_fs", "nsteps", "ensemble", "mdp_overrides",
})
_SIMULATION_CONFIG_KEYS = frozenset({
    "schema_version", "minimization", "eq_stages", "prod_iters", "hardware",
    "system_name", "mdp_overrides_text",
}) | _GLOBAL_KEYS


def _finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _declared_timestep_unit(stage: dict, label: str) -> str:
    """Return the explicit timestep unit, accepting the legacy boolean flag."""
    has_unit = "dt_unit" in stage
    has_legacy = "dt_fs" in stage
    if has_unit and has_legacy:
        raise ValueError(
            f"{label} must use dt_unit or legacy dt_fs, not both"
        )
    if has_unit:
        unit = str(stage["dt_unit"]).strip().lower()
        if unit not in {"fs", "ps"}:
            raise ValueError(f"{label} dt_unit must be fs or ps")
        return unit
    if has_legacy:
        if not isinstance(stage["dt_fs"], bool):
            raise ValueError(f"{label} dt_fs must be true or false")
        return "fs" if stage["dt_fs"] else "ps"
    raise ValueError(
        f"{label} timestep requires an explicit dt_unit ('fs' or 'ps')"
    )


def _convert_timestep_to_ps(stage: dict) -> None:
    """Convert an explicitly-unit-labelled stage copy to GROMACS ps."""
    if "dt" not in stage:
        return
    unit = _declared_timestep_unit(stage, "stage")
    if unit == "fs":
        stage["dt"] = float(stage["dt"]) / 1000.0
    else:
        stage["dt"] = float(stage["dt"])
    stage.pop("dt_unit", None)
    stage.pop("dt_fs", None)


def _validate_nonbond_geometry(values: dict, label: str) -> None:
    if "rvdw_switch" not in values or values["rvdw_switch"] in (None, ""):
        return
    switch = _finite_number(values["rvdw_switch"], f"{label} rvdw_switch")
    cutoff = _finite_number(values.get("rvdw", 1.2), f"{label} rvdw")
    if switch < 0 or switch >= cutoff:
        raise ValueError(f"{label} rvdw_switch must be non-negative and below rvdw")


def _validate_comm_groups(value: object, has_membrane: bool, label: str) -> None:
    text = str(value).strip()
    groups = text.split()
    if not groups:
        raise ValueError(f"{label} comm_grps must select at least one index group")
    if any(not _INDEX_GROUP.fullmatch(group) for group in groups):
        raise ValueError(f"{label} comm_grps contains an invalid index group name")
    if len(groups) != len(set(groups)):
        raise ValueError(f"{label} comm_grps must not repeat an index group")
    available = set(_BASE_COM_GROUPS)
    if has_membrane:
        available.update(_MEMBRANE_COM_GROUPS)
    unknown = sorted(set(groups) - available)
    if unknown:
        raise ValueError(
            f"{label} comm_grps references unavailable index group(s): "
            + ", ".join(unknown)
        )
    if "System" in groups and len(groups) != 1:
        raise ValueError(f"{label} comm_grps cannot combine System with another group")
    if "SOLU_MEMB" in groups and ({"SOLU", "MEMB"} & set(groups)):
        raise ValueError(
            f"{label} comm_grps cannot overlap SOLU_MEMB with SOLU or MEMB"
        )


def _validate_override_comm_groups(
    overrides: dict | None, has_membrane: bool, label: str
) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if str(key).lower().replace("_", "-") == "comm-grps":
            _validate_comm_groups(value, has_membrane, label)


def _validate_overrides(overrides: dict, label: str) -> None:
    if overrides in (None, {}):
        return
    if not isinstance(overrides, dict) or len(overrides) > 64:
        raise ValueError(f"{label} must contain at most 64 key/value entries")
    for key, value in overrides.items():
        if not _MDP_KEY.fullmatch(str(key)):
            raise ValueError(f"Invalid MDP key {key!r} in {label}")
        text = str(value)
        if not text or len(text) > 256 or "\n" in text or "\r" in text or "\x00" in text:
            raise ValueError(f"Invalid value for MDP key {key!r} in {label}")


def _validate_stage(stage: dict, label: str) -> None:
    if "enabled" in stage and not isinstance(stage["enabled"], bool):
        raise ValueError(f"{label} enabled must be true or false")
    for key in ("nsteps", "nstcomm", "nstxout", "nstvout", "nstfout",
                "nstxout_compressed", "nstcalcenergy", "nstenergy", "nstlog"):
        if key in stage:
            value = stage[key]
            number = _finite_number(value, f"{label} {key}")
            if isinstance(value, bool) or not number.is_integer() or number < 0:
                raise ValueError(f"{label} {key} must be a non-negative integer")
    if "nsteps" in stage and _finite_number(
        stage["nsteps"], f"{label} nsteps"
    ) < 1:
        raise ValueError(f"{label} nsteps must be positive")
    if "repeat" in stage:
        value = stage["repeat"]
        repeat = _finite_number(value, f"{label} repeat")
        if isinstance(value, bool) or not repeat.is_integer() or not 1 <= repeat <= 100:
            raise ValueError(f"{label} repeat must be an integer from 1 to 100")
    if "gen_seed" in stage:
        value = stage["gen_seed"]
        seed = _finite_number(value, f"{label} gen_seed")
        if isinstance(value, bool) or not seed.is_integer() or seed < -1:
            raise ValueError(f"{label} gen_seed must be an integer of -1 or greater")
    if "dt" in stage:
        timestep = _finite_number(stage["dt"], f"{label} timestep")
        unit = _declared_timestep_unit(stage, label)
        timestep_fs = timestep if unit == "fs" else timestep * 1000.0
        if not 0 < timestep_fs <= 5.0:
            raise ValueError(f"{label} timestep must be in (0, 5] fs")
    elif "dt_unit" in stage or "dt_fs" in stage:
        raise ValueError(f"{label} declares a timestep unit without dt")
    if stage.get("comm_mode", "linear") not in {"linear", "angular", "none"}:
        raise ValueError(f"{label} comm_mode must be linear, angular, or none")
    if stage.get("ensemble", "npt") not in {"nvt", "npt"}:
        raise ValueError(f"{label} ensemble must be nvt or npt")
    if str(stage.get("tcoupl", "v-rescale")).lower() not in {
        "v-rescale", "nose-hoover", "berendsen", "no",
    }:
        raise ValueError(f"{label} tcoupl value is not supported")
    if str(stage.get("pcoupl", "C-rescale")).lower() not in {
        "c-rescale", "berendsen", "parrinello-rahman", "no",
    }:
        raise ValueError(f"{label} pcoupl value is not supported")
    if stage.get("constraints", "h-bonds") not in {
        "none", "h-bonds", "all-bonds", "h-angles", "all-angles"
    }:
        raise ValueError(f"{label} constraints value is not supported")
    if str(stage.get("vdw_modifier", "Potential-shift")).lower() not in {
        "potential-shift", "force-switch", "potential-switch", "none",
    }:
        raise ValueError(f"{label} vdw_modifier value is not supported")
    if str(stage.get("dispcorr", "no")).lower() not in {
        "no", "none", "ener", "enerpres",
    }:
        raise ValueError(f"{label} dispcorr value is not supported")
    for key in (
        "tau_t", "tau_p", "temperature", "rlist", "rvdw", "rcoulomb",
        "fourierspacing", "compress",
    ):
        if key in stage and _finite_number(stage[key], f"{label} {key}") <= 0:
            raise ValueError(f"{label} {key} must be positive")
    if "ref_p" in stage:
        _finite_number(stage["ref_p"], f"{label} ref_p")
    if stage.get("comm_mode", "linear") != "none" and "nstcomm" in stage and int(stage["nstcomm"]) < 1:
        raise ValueError(f"{label} nstcomm must be positive when COM removal is enabled")
    for key in ("bb", "sc", "lipid", "dih"):
        if key in stage and _finite_number(stage[key], f"{label} {key}") < 0:
            raise ValueError(f"{label} {key} restraint must be non-negative")
    _validate_nonbond_geometry(stage, label)
    _validate_overrides(stage.get("mdp_overrides", {}), f"{label} MDP overrides")


def _validate_minimization(minimization: dict | None, params: dict) -> None:
    if minimization is None:
        return
    if not isinstance(minimization, dict):
        raise ValueError("minimization settings must be an object")
    unknown = sorted(set(minimization) - _MINIMIZATION_KEYS)
    if unknown:
        raise ValueError(
            "unknown minimization setting(s): " + ", ".join(unknown)
        )
    if minimization.get("integrator", "steep") not in {"steep", "cg"}:
        raise ValueError("minimization integrator must be steep or cg")
    for key in ("nsteps", "nstlist"):
        value = minimization.get(key)
        if value is None:
            continue
        number = _finite_number(value, f"minimization {key}")
        if isinstance(value, bool) or not number.is_integer() or number < 1:
            raise ValueError(f"minimization {key} must be a positive integer")
    for key in ("emtol", "emstep", "rlist", "rvdw", "rcoulomb", "fourierspacing"):
        if key in minimization and _finite_number(
            minimization[key], f"minimization {key}"
        ) <= 0:
            raise ValueError(f"minimization {key} must be positive")
    if minimization.get("constraints", "h-bonds") not in {
        "none", "h-bonds", "all-bonds", "h-angles", "all-angles"
    }:
        raise ValueError("minimization constraints value is not supported")
    for key in ("bb", "sc", "lipid", "dih"):
        if key in minimization and _finite_number(
            minimization[key], f"minimization {key}"
        ) < 0:
            raise ValueError(f"minimization {key} restraint must be non-negative")
    _validate_nonbond_geometry(minimization, "minimization")
    _validate_overrides(
        minimization.get("mdp_overrides", {}), "minimization MDP overrides"
    )
    if _force_field_family(params) == "charmm" and str(
        minimization.get("dispcorr", "no")
    ).lower() not in {"no", "none"}:
        raise ValueError(
            "CHARMM36/CHARMM36m requires DispCorr=no with its force-switch "
            "non-bonded protocol"
        )


def _apply_mdp_overrides(content: str, overrides: dict | None) -> str:
    """Replace generated MDP keys without emitting duplicate directives."""
    if not overrides:
        return content
    _validate_overrides(overrides, "MDP overrides")
    normalized = {str(key).lower().replace("_", "-"): str(value) for key, value in overrides.items()}
    seen: set[str] = set()
    output = []
    assignment = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_-]*)(\s*=\s*)(.*)$")
    for line in content.splitlines():
        match = assignment.match(line)
        if match:
            key = match.group(2).lower().replace("_", "-")
            if key in normalized:
                line = f"{match.group(1)}{match.group(2)}{match.group(3)}{normalized[key]}"
                seen.add(key)
        output.append(line)
    for key, value in normalized.items():
        if key not in seen:
            output.append(f"{key:<24s}= {value}")
    return "\n".join(output).rstrip() + "\n"


# =============================================================================
# Default restraint schedule — decaying over 6 stages
# =============================================================================

def _force_field_family(params: dict) -> str:
    family = str(params.get("force_field_family", "")).strip().lower()
    if family:
        return family
    force_field = str(params.get("force_field", "")).strip().lower()
    if force_field.startswith("charmm"):
        return "charmm"
    if force_field.startswith("opls"):
        return "opls"
    return "amber"


def _nonbond_defaults(params: dict) -> dict[str, object]:
    if _force_field_family(params) == "charmm":
        return {
            "rlist": 1.2,
            "vdw_modifier": "Force-switch",
            "rvdw_switch": 1.0,
            "rvdw": 1.2,
            "rcoulomb": 1.2,
            "fourierspacing": 0.12,
            "dispcorr": "no",
        }
    return {
        "rlist": 1.0,
        "vdw_modifier": "Potential-shift",
        "rvdw_switch": None,
        "rvdw": 1.0,
        "rcoulomb": 1.0,
        "fourierspacing": 0.12,
        "dispcorr": "EnerPres",
    }


def _stage_defaults(params: dict, ensemble: str) -> dict[str, object]:
    has_membrane = bool(params.get("has_membrane", True))
    values: dict[str, object] = {
        "enabled": True,
        "bb": 4000 if has_membrane else 400,
        "sc": 2000 if has_membrane else 40,
        "lipid": 1000 if has_membrane else 0,
        "dih": 1000 if has_membrane else 0,
        "tcoupl": "v-rescale",
        "tau_t": 1.0,
        "temperature": params.get("temperature", 310.15),
        "comm_mode": "linear",
        "comm_grps": "SOLU_MEMB SOLV" if has_membrane else "SOLU SOLV",
        "nstcomm": 100,
        "constraints": "h-bonds",
        "nstlist": 20,
        "nstxout_compressed": 5000,
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstcalcenergy": 100,
        "nstenergy": 1000,
        "nstlog": 1000,
        "gen_seed": params.get("gen_seed", -1),
        "mdp_overrides": {},
        **_nonbond_defaults(params),
    }
    if ensemble == "npt":
        values.update({
            "pcoupl": "C-rescale",
            "pcoupl_type": "semisotropic" if has_membrane else "isotropic",
            "tau_p": 5.0,
            "ref_p": 1.0,
            "compress": "4.5e-5",
        })
    return values


def _default_minimization(params: dict) -> dict[str, object]:
    has_membrane = bool(params.get("has_membrane", True))
    values: dict[str, object] = {
        "integrator": "steep",
        "nsteps": 50000 if has_membrane else 5000,
        "emtol": 1000.0,
        "emstep": 0.01,
        "nstlist": 10,
        "constraints": "h-bonds",
        "bb": 4000 if has_membrane else 400,
        "sc": 2000 if has_membrane else 40,
        "lipid": 1000 if has_membrane else 0,
        "dih": 1000 if has_membrane else 0,
        "mdp_overrides": {},
        **_nonbond_defaults(params),
    }
    values["nstlist"] = 10
    return values

def _default_schedule(params: dict) -> list[dict]:
    """Return a system-specific equilibration protocol.

    Membranes use a six-stage restraint release with NVT followed by
    semi-isotropic NPT.  Solvated non-membrane systems use one restrained
    NVT stage before isotropic NPT production.

    NOTE: dt values are in PS (GROMACS convention), not fs.
    """
    if not params.get("has_membrane", True):
        return [{
            "bb": 400, "sc": 40, "lipid": 0, "dih": 0,
            "dt": 0.001, "dt_unit": "ps", "nsteps": 125000, "ensemble": "nvt",
            "comm_grps": "SOLU SOLV",
            "nstxout_compressed": 5000,
        }]
    return [
        {"bb":4000, "sc":2000, "lipid":1000, "dih":1000, "dt":0.001, "dt_unit":"ps", "nsteps":125000, "ensemble":"nvt", "comm_grps":"SOLU_MEMB SOLV"},
        {"bb":2000, "sc":1000, "lipid":400,  "dih":400,  "dt":0.001, "dt_unit":"ps", "nsteps":125000, "ensemble":"nvt", "comm_grps":"SOLU_MEMB SOLV"},
        {"bb":1000, "sc":500,  "lipid":400,  "dih":200,  "dt":0.001, "dt_unit":"ps", "nsteps":125000, "ensemble":"npt", "comm_grps":"SOLU_MEMB SOLV"},
        {"bb":500,  "sc":200,  "lipid":200,  "dih":200,  "dt":0.002, "dt_unit":"ps", "nsteps":250000, "ensemble":"npt", "comm_grps":"SOLU_MEMB SOLV"},
        {"bb":200,  "sc":50,   "lipid":40,   "dih":100,  "dt":0.002, "dt_unit":"ps", "nsteps":250000, "ensemble":"npt", "comm_grps":"SOLU_MEMB SOLV"},
        {"bb":50,   "sc":0,    "lipid":0,    "dih":0,    "dt":0.002, "dt_unit":"ps", "nsteps":250000, "ensemble":"npt", "comm_grps":"SOLU_MEMB SOLV"},
    ]


def _default_production(params: dict) -> list[dict]:
    """Return restart-friendly production chunks for the system class."""
    if params.get("has_membrane", True):
        return [{
            "nsteps": params.get("prod_nsteps", 5_000_000),
            "dt": 0.002,
            "dt_unit": "ps",
            "repeat": 5,
            "nstxout_compressed": 10_000,
            "comm_grps": "SOLU_MEMB SOLV",
        }]
    return [{
        "nsteps": params.get("prod_nsteps", 500_000),
        "dt": 0.002,
        "dt_unit": "ps",
        "repeat": 10,
        "nstxout_compressed": 50_000,
        "comm_grps": "SOLU SOLV",
    }]


# =============================================================================
# Common MDP sections
# =============================================================================

def _nonbond_params(params: dict, nstlist: int = 20, stage: dict | None = None) -> str:
    """Return force-field-specific non-bonded defaults.

    CHARMM36 uses the published 1.0--1.2 nm force-switch protocol and no
    analytical dispersion correction.  Amber/GAFF/Lipid21 and OPLS use a
    1.0 nm shifted cutoff with long-range dispersion correction.  Explicit
    stage values remain available for expert protocols, except for the known
    incompatible CHARMM + dispersion-correction combination.
    """
    stage = stage or {}
    family = _force_field_family(params)
    defaults = _nonbond_defaults(params)

    def selected(key: str):
        return stage.get(key, params.get(key, defaults[key]))

    dispcorr = str(selected("dispcorr"))
    if family == "charmm" and dispcorr.lower() not in {"no", "none"}:
        raise ValueError(
            "CHARMM36/CHARMM36m requires DispCorr=no with its force-switch "
            "non-bonded protocol"
        )
    switch_line = ""
    rvdw_switch = selected("rvdw_switch")
    if rvdw_switch is not None and str(rvdw_switch).strip():
        switch_line = f"\nrvdw_switch             = {rvdw_switch}"
    return f""";
cutoff-scheme           = Verlet
nstlist                 = {stage.get('nstlist', nstlist)}
rlist                   = {selected('rlist')}
vdwtype                 = Cut-off
vdw-modifier            = {selected('vdw_modifier')}{switch_line}
rvdw                    = {selected('rvdw')}
coulombtype             = PME
rcoulomb                = {selected('rcoulomb')}
fourierspacing          = {stage.get('fourierspacing', params.get('fourierspacing', 0.12))}
DispCorr                = {dispcorr}"""


def _temp_coupling(params: dict, stage: dict | None = None) -> str:
    """Temperature coupling with separate groups for protein, membrane, solvent."""
    stage = stage or {}
    n_groups = int(params.get("n_tc_groups", 3))
    has_membrane = params.get("has_membrane", True)
    if not has_membrane and n_groups >= 3:
        n_groups = 2  # Solvator: only SOLU + SOLV
    if n_groups == 3:
        grps = "SOLU MEMB SOLV"
    elif n_groups == 2:
        grps = "SOLU_MEMB SOLV" if has_membrane else "SOLU SOLV"
    else:
        grps = "System"
    ref_t = stage.get("temperature", params.get("temperature", 310.15))
    ref_t_vals = " ".join(str(ref_t) for _ in range(n_groups))
    return f""";
	tcoupl                  = {stage.get('tcoupl', params.get('tcoupl', 'v-rescale'))}
	tc-grps                 = {grps}
	tau-t                   = {' '.join([str(stage.get('tau_t', 1.0))] * n_groups)}
	ref-t                   = {ref_t_vals}"""

def _pressure_coupling(params: dict, stage: dict | None = None) -> str:
    """Semi-isotropic pressure coupling for membrane systems."""
    stage = stage or {}
    pcoupl_type = stage.get("pcoupl_type", params.get("pcoupl_type", "auto"))
    if pcoupl_type == "auto":
        pcoupl_type = "semisotropic" if params.get("has_membrane", True) else "isotropic"
    compress = stage.get("compress", params.get("compressibility", "4.5e-5"))
    if pcoupl_type == "semisotropic":
        comp_str = f"{compress}  {compress}"
    else:
        comp_str = compress
    ref_p = stage.get("ref_p", params.get("ref_p", 1.0))
    rendered_type = {
        "isotropic": "Isotropic",
        "semisotropic": "Semiisotropic",
        "anisotropic": "Anisotropic",
        "surface-tension": "Surface-Tension",
    }.get(str(pcoupl_type).lower(), str(pcoupl_type))
    return f""";
pcoupl                  = {stage.get('pcoupl', params.get('pcoupl', 'C-rescale'))}
pcoupltype              = {rendered_type}
tau-p                   = {stage.get('tau_p', params.get('tau_p', 5.0))}
compressibility         = {comp_str}
ref-p                   = {str(ref_p) + '  ' + str(ref_p) if pcoupl_type == 'semisotropic' else str(ref_p)}
refcoord-scaling        = com"""


def _define_macros(rst: dict, params: dict) -> str:
    """Position restraint and dihedral restraint define macros."""
    has_membrane = params.get("has_membrane", True)
    protein_restraints = params.get("protein_position_restraints", True)
    lipid_restraints = params.get("lipid_position_restraints", has_membrane)
    lipid_dihedrals = params.get("lipid_dihedral_restraints", has_membrane)
    macros: list[str] = []
    if protein_restraints or lipid_restraints:
        macros.append("-DPOSRES")
    if protein_restraints:
        macros.extend([
            f"-DPOSRES_FC_BB={rst['bb']:.1f}",
            f"-DPOSRES_FC_SC={rst['sc']:.1f}",
        ])
    if lipid_restraints:
        macros.append(f"-DPOSRES_FC_LIPID={rst['lipid']:.1f}")
    if lipid_dihedrals:
        macros.extend(["-DDIHRES", f"-DDIHRES_FC={rst['dih']:.1f}"])
    if not macros:
        return "; no position-restraint macros are required for this system"
    return "define                  = " + " ".join(macros)


def _output_control(params: dict, stage: dict | None = None) -> str:
    stage = stage or {}
    return f"""nstxout-compressed      = {stage.get('nstxout_compressed', params.get('nstxout_compressed', 5000))}
nstxout                 = {stage.get('nstxout', params.get('nstxout', 0))}
nstvout                 = {stage.get('nstvout', params.get('nstvout', 0))}
nstfout                 = {stage.get('nstfout', params.get('nstfout', 0))}
nstcalcenergy           = {stage.get('nstcalcenergy', params.get('nstcalcenergy', 100))}
nstenergy               = {stage.get('nstenergy', params.get('nstenergy', 1000))}
nstlog                  = {stage.get('nstlog', params.get('nstlog', 1000))}"""


def _com_motion(params: dict | None = None, stage: dict | None = None) -> str:
    if params is None:
        params = {}
    stage = stage or {}
    grps = stage.get("comm_grps", params.get("comm_grps", "System"))
    return f""";
nstcomm                 = {stage.get('nstcomm', params.get('nstcomm', 100))}
comm-mode               = {stage.get('comm_mode', params.get('comm_mode', 'linear'))}
comm-grps               = {grps}"""


# =============================================================================
# Stage builders
# =============================================================================

def _build_minim(params: dict, minimization: dict) -> str:
    return f"""; Energy Minimization — generated by GMXBUILDER
{_define_macros(minimization, params)}
integrator              = {minimization.get('integrator', 'steep')}
emtol                   = {minimization.get('emtol', 1000.0)}
emstep                  = {minimization.get('emstep', 0.01)}
nsteps                  = {minimization.get('nsteps', 50000 if params.get('has_membrane', True) else 5000)}
{_nonbond_params(params, nstlist=int(minimization.get('nstlist', 10)), stage=minimization)}
;
constraints             = {minimization.get('constraints', 'h-bonds')}
constraint-algorithm    = Lincs
;
tcoupl                  = no
pcoupl                  = no
continuation            = no
"""


def _build_nvt(params: dict, rst: dict, dt: float, nsteps: int, is_first: bool, stage_num: int) -> str:
    lines = [f"; NVT Equilibration Stage {stage_num} — generated by GMXBUILDER"]
    lines.append(_define_macros(rst, params))
    lines.append("integrator              = md")
    lines.append(f"dt                      = {dt:.6f}")
    lines.append(f"nsteps                  = {nsteps}")
    lines.append(_output_control(params, rst))
    lines.append(_nonbond_params(params, stage=rst))
    lines.append(_temp_coupling(params, rst))
    lines.append(";")
    lines.append(f"constraints             = {rst.get('constraints', params.get('constraints', 'h-bonds'))}")
    lines.append("constraint-algorithm    = Lincs")
    lines.append(_com_motion(params, rst))

    if is_first:
        seed = rst.get("gen_seed", params.get("gen_seed", -1))
        temp = rst.get("temperature", params.get("temperature", 310.15))
        lines.append(";")
        lines.append("gen-vel                 = yes")
        lines.append(f"gen-temp                = {temp}")
        lines.append(f"gen-seed                = {seed}")
        lines.append("continuation            = no")
    else:
        lines.append(";")
        lines.append("gen-vel                 = no")
        lines.append("continuation            = yes")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def _build_npt(
    params: dict,
    rst: dict,
    dt: float,
    nsteps: int,
    stage_num: int,
    is_first: bool = False,
) -> str:
    lines = [f"; NPT Equilibration Stage {stage_num} — generated by GMXBUILDER"]
    lines.append(_define_macros(rst, params))
    lines.append("integrator              = md")
    lines.append(f"dt                      = {dt:.6f}")
    lines.append(f"nsteps                  = {nsteps}")
    lines.append(_output_control(params, rst))
    lines.append(_nonbond_params(params, stage=rst))
    lines.append(_temp_coupling(params, rst))
    lines.append(_pressure_coupling(params, rst))
    lines.append(";")
    lines.append(f"constraints             = {rst.get('constraints', params.get('constraints', 'h-bonds'))}")
    lines.append("constraint-algorithm    = Lincs")
    if is_first:
        seed = rst.get("gen_seed", params.get("gen_seed", -1))
        temp = rst.get("temperature", params.get("temperature", 310.15))
        lines.append("gen-vel                 = yes")
        lines.append(f"gen-temp                = {temp}")
        lines.append(f"gen-seed                = {seed}")
        lines.append("continuation            = no")
    else:
        lines.append("continuation            = yes")
    lines.append(_com_motion(params, rst))
    lines.append("")
    return "\n".join(lines)


def _build_prod(params: dict) -> str:
    default_nsteps = 5_000_000 if params.get("has_membrane", True) else 500_000
    lines = ["; Production Run — generated by GMXBUILDER"]
    lines.append("integrator              = md")
    lines.append(f"dt                      = {params.get('dt', 0.002):.6f}")
    lines.append(
        f"nsteps                  = "
        f"{params.get('nsteps', params.get('prod_nsteps', default_nsteps))}"
    )
    lines.append(_output_control(params, params))
    lines.append(_nonbond_params(params, stage=params))
    lines.append(_temp_coupling(params))
    lines.append(_pressure_coupling(params))
    lines.append(";")
    lines.append(f"constraints             = {params.get('constraints', 'h-bonds')}")
    lines.append("constraint-algorithm    = Lincs")
    lines.append("continuation            = yes")
    lines.append(_com_motion(params, params))
    lines.append("")
    return "\n".join(lines)
