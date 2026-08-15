"""Martini 3 MDP, index and run-script generation owned by the CG module."""

from __future__ import annotations

import math
import re
from pathlib import Path

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError


def normalize_protocol(raw: dict | None, *, has_membrane: bool) -> dict:
    raw = dict(raw or {})
    allowed = {
        "temperature",
        "pressure",
        "production_ns",
        "output_interval_ps",
        "minimization_steps",
        "minimization_tolerance",
        "minimization_step_nm",
        "eq1_duration_ns",
        "eq1_timestep_fs",
        "eq1_temperature",
        "eq1_tau_t",
        "eq2_duration_ns",
        "eq2_timestep_fs",
        "eq2_temperature",
        "eq2_tau_t",
        "eq2_pressure",
        "eq2_tau_p",
        "production_timestep_fs",
        "production_temperature",
        "production_tau_t",
        "production_pressure",
        "production_tau_p",
        "energy_interval_ps",
        "log_interval_ps",
        "comm_mode",
        "comm_interval",
        "has_membrane",
        "equilibration_1",
        "equilibration_2",
        "use_gpu",
        "gpu_ids",
        "threads",
        "mpi_ranks",
        "system_name",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ModuleConfigError("Unknown Martini simulation setting(s): " + ", ".join(unknown))

    def number(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ModuleConfigError(f"{name} must be between {low:g} and {high:g}")
        return value

    # temperature/pressure remain accepted for task/checkpoint compatibility,
    # but every dynamics stage owns its actual coupling values.
    legacy_temperature = number("temperature", 310.0, 250.0, 370.0)
    legacy_pressure = number("pressure", 1.0, 0.1, 100.0)
    production_ns = number("production_ns", 1000.0, 1.0, 100000.0)
    output_ps = number("output_interval_ps", 100.0, 1.0, production_ns * 1000.0)
    energy_ps = number("energy_interval_ps", 20.0, 0.02, production_ns * 1000.0)
    log_ps = number("log_interval_ps", 20.0, 0.02, production_ns * 1000.0)
    threads_number = number("threads", 8, 1, 1024)
    mpi_number = number("mpi_ranks", 1, 1, threads_number)
    if not threads_number.is_integer() or not mpi_number.is_integer():
        raise ModuleConfigError("threads and mpi_ranks must be integers")
    threads = int(threads_number)
    mpi_ranks = int(mpi_number)
    if threads % mpi_ranks:
        raise ModuleConfigError("threads must be divisible by mpi_ranks")
    for boolean_name in ("equilibration_1", "equilibration_2", "use_gpu"):
        if boolean_name in raw and not isinstance(raw[boolean_name], bool):
            raise ModuleConfigError(f"{boolean_name} must be true or false")
    use_gpu = raw.get("use_gpu", True)
    gpu_ids = str(raw.get("gpu_ids", "0")).strip()
    if use_gpu and not re.fullmatch(r"\d+(?:,\d+)*", gpu_ids):
        raise ModuleConfigError("gpu_ids must be a comma-separated list such as 0 or 0,1")
    if use_gpu:
        selected = gpu_ids.split(",")
        if len(set(selected)) != len(selected):
            raise ModuleConfigError("gpu_ids must not contain duplicates")
        if len(selected) > mpi_ranks:
            raise ModuleConfigError("selected GPU count cannot exceed mpi_ranks")
    minimization_steps_number = number("minimization_steps", 20000, 100, 1_000_000)
    comm_interval_number = number("comm_interval", 100, 1, 1_000_000)
    if not minimization_steps_number.is_integer() or not comm_interval_number.is_integer():
        raise ModuleConfigError("minimization_steps and comm_interval must be integers")
    comm_mode = str(raw.get("comm_mode", "Linear")).strip().lower()
    if comm_mode not in {"linear", "none"}:
        raise ModuleConfigError("comm_mode must be Linear or None")
    if "has_membrane" in raw and not isinstance(raw["has_membrane"], bool):
        raise ModuleConfigError("has_membrane must be true or false")
    return {
        "minimization_steps": int(minimization_steps_number),
        "minimization_tolerance": number("minimization_tolerance", 200.0, 1.0, 10000.0),
        "minimization_step_nm": number("minimization_step_nm", 0.005, 0.0001, 0.1),
        "eq1_duration_ns": number("eq1_duration_ns", 1.0, 0.001, 1000.0),
        "eq1_timestep_fs": number("eq1_timestep_fs", 10.0, 1.0, 20.0),
        "eq1_temperature": number("eq1_temperature", legacy_temperature, 250.0, 370.0),
        "eq1_tau_t": number("eq1_tau_t", 1.0, 0.1, 20.0),
        "eq2_duration_ns": number("eq2_duration_ns", 10.0, 0.001, 10000.0),
        "eq2_timestep_fs": number("eq2_timestep_fs", 20.0, 1.0, 20.0),
        "eq2_temperature": number("eq2_temperature", legacy_temperature, 250.0, 370.0),
        "eq2_tau_t": number("eq2_tau_t", 1.0, 0.1, 20.0),
        "eq2_pressure": number("eq2_pressure", legacy_pressure, 0.1, 100.0),
        "eq2_tau_p": number("eq2_tau_p", 5.0, 0.1, 50.0),
        "production_timestep_fs": number("production_timestep_fs", 20.0, 1.0, 20.0),
        "production_temperature": number(
            "production_temperature", legacy_temperature, 250.0, 370.0
        ),
        "production_tau_t": number("production_tau_t", 1.0, 0.1, 20.0),
        "production_pressure": number("production_pressure", legacy_pressure, 0.1, 100.0),
        "production_tau_p": number("production_tau_p", 5.0, 0.1, 50.0),
        "production_ns": production_ns,
        "output_interval_ps": output_ps,
        "energy_interval_ps": energy_ps,
        "log_interval_ps": log_ps,
        "comm_mode": "Linear" if comm_mode == "linear" else "None",
        "comm_interval": int(comm_interval_number),
        "equilibration_1": bool(raw.get("equilibration_1", True)),
        "equilibration_2": bool(raw.get("equilibration_2", True)),
        "use_gpu": use_gpu,
        "gpu_ids": gpu_ids,
        "threads": threads,
        "mpi_ranks": mpi_ranks,
        "has_membrane": bool(has_membrane),
        "system_name": re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(raw.get("system_name", "martini3_system"))
        )[:64]
        or "martini3_system",
    }


def _common(config: dict, dt_ps: float) -> list[str]:
    stride = max(1, int(round(config["output_interval_ps"] / dt_ps)))
    energy_stride = max(1, int(round(config["energy_interval_ps"] / dt_ps)))
    log_stride = max(1, int(round(config["log_interval_ps"] / dt_ps)))
    return [
        "cutoff-scheme = Verlet",
        "nstlist = 20",
        "verlet-buffer-tolerance = -1",
        "rlist = 1.35",
        "coulombtype = Reaction-Field",
        "rcoulomb = 1.1",
        "epsilon-r = 15",
        "epsilon-rf = 0",
        "vdwtype = Cut-off",
        "vdw-modifier = Potential-shift-verlet",
        "rvdw = 1.1",
        "DispCorr = no",
        "constraints = none",
        "pbc = xyz",
        f"nstenergy = {energy_stride}",
        f"nstlog = {log_stride}",
        "nstxout-compressed = " + str(stride),
        "compressed-x-precision = 100",
        f"comm-mode = {config['comm_mode']}",
        f"nstcomm = {config['comm_interval']}",
        "comm-grps = System",
    ]


def write_mdp_files(directory: Path, config: dict) -> list[tuple[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    pressure_type = "semiisotropic" if config["has_membrane"] else "isotropic"
    compress = "3e-4 3e-4" if config["has_membrane"] else "3e-4"
    stages: list[tuple[str, str]] = []
    mini = [
        "integrator = steep",
        f"nsteps = {config['minimization_steps']}",
        f"emtol = {config['minimization_tolerance']:g}",
        f"emstep = {config['minimization_step_nm']:g}",
        "nstlist = 20",
        "cutoff-scheme = Verlet",
        "verlet-buffer-tolerance = -1",
        "rlist = 1.35",
        "coulombtype = Reaction-Field",
        "rcoulomb = 1.1",
        "epsilon-r = 15",
        "epsilon-rf = 0",
        "vdwtype = Cut-off",
        "vdw-modifier = Potential-shift-verlet",
        "rvdw = 1.1",
        "DispCorr = no",
        "constraints = none",
        "lincs-iter = 4",
        "lincs-order = 8",
        # Steepest descent can rotate a constrained CG backbone substantially
        # while retaining sub-per-mille constraint error.  Keep a hard 60°
        # warning gate while avoiding false 30° warnings during relaxation.
        "lincs-warnangle = 60",
        "pbc = xyz",
    ]
    (directory / "mini.mdp").write_text("\n".join(mini) + "\n", encoding="utf-8")
    stages.append(("mini", "mini.mdp"))

    def dynamics(
        name: str,
        ns: float,
        dt_fs: float,
        temperature: float,
        tau_t: float,
        *,
        npt: bool,
        pressure: float = 1.0,
        tau_p: float = 5.0,
        gen_vel: bool,
    ) -> None:
        dt = dt_fs / 1000.0
        lines = [
            "integrator = md",
            f"dt = {dt:g}",
            f"nsteps = {int(round(ns * 1000 / dt))}",
            "continuation = no" if gen_vel else "continuation = yes",
            "tcoupl = v-rescale",
            "tc-grps = System",
            f"tau-t = {tau_t:g}",
            f"ref-t = {temperature:g}",
        ]
        if gen_vel:
            lines += ["gen-vel = yes", f"gen-temp = {temperature:g}", "gen-seed = -1"]
        else:
            lines += ["gen-vel = no"]
        if npt:
            ref_p = f"{pressure:g} {pressure:g}" if config["has_membrane"] else f"{pressure:g}"
            lines += [
                "pcoupl = C-rescale",
                f"pcoupltype = {pressure_type}",
                f"tau-p = {tau_p:g}",
                f"ref-p = {ref_p}",
                f"compressibility = {compress}",
            ]
        else:
            lines += ["pcoupl = no"]
        lines += _common(config, dt)
        filename = f"{name}.mdp"
        (directory / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
        stages.append((name, filename))

    if config["equilibration_1"]:
        dynamics(
            "equilibration_1",
            config["eq1_duration_ns"],
            config["eq1_timestep_fs"],
            config["eq1_temperature"],
            config["eq1_tau_t"],
            npt=False,
            gen_vel=True,
        )
    if config["equilibration_2"]:
        dynamics(
            "equilibration_2",
            config["eq2_duration_ns"],
            config["eq2_timestep_fs"],
            config["eq2_temperature"],
            config["eq2_tau_t"],
            npt=True,
            pressure=config["eq2_pressure"],
            tau_p=config["eq2_tau_p"],
            gen_vel=not config["equilibration_1"],
        )
    dynamics(
        "production",
        config["production_ns"],
        config["production_timestep_fs"],
        config["production_temperature"],
        config["production_tau_t"],
        npt=True,
        pressure=config["production_pressure"],
        tau_p=config["production_tau_p"],
        gen_vel=not (config["equilibration_1"] or config["equilibration_2"]),
    )
    return stages


def write_index(system, path: Path) -> None:
    groups: dict[str, list[int]] = {"System": list(range(1, system.num_atoms + 1))}
    names = {
        ComponentKind.PROTEIN: "Protein",
        ComponentKind.MEMBRANE: "Membrane",
        ComponentKind.SOLVENT: "Solvent",
        ComponentKind.IONS: "Ions",
    }
    for component in system.components:
        name = names.get(component.kind)
        if name:
            groups.setdefault(name, []).extend(int(i) + 1 for i in component.atom_indices)
    groups["Solute"] = sorted(groups.get("Protein", []) + groups.get("Membrane", []))
    groups["Solvent_Ions"] = sorted(groups.get("Solvent", []) + groups.get("Ions", []))
    with path.open("w", encoding="utf-8") as handle:
        for name, indices in groups.items():
            if not indices:
                continue
            handle.write(f"[ {name} ]\n")
            for start in range(0, len(indices), 15):
                handle.write(" ".join(str(value) for value in indices[start : start + 15]) + "\n")


def write_run_script(path: Path, stages: list[tuple[str, str]], config: dict) -> None:
    gpu = ""
    if config["use_gpu"]:
        # Martini's validated protocol uses Reaction-Field electrostatics, so
        # requesting PME offload is both meaningless and rejected by GROMACS.
        gpu = f' -nb gpu -gpu_id "{config["gpu_ids"].replace(",", "")}"'
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'GMX="${GMX:-gmx}"',
        f"NTMPI=${{NTMPI:-{config['mpi_ranks']}}}",
        f"NTOMP=${{NTOMP:-{config['threads'] // config['mpi_ranks']}}}",
        'command -v "$GMX" >/dev/null || { echo "GROMACS executable not found: $GMX" >&2; exit 127; }',
        'coord="input.gro"',
        'checkpoint=""',
    ]
    for stage, filename in stages:
        lines += [
            f'echo "[GMXBUILDER] Preparing {stage}"',
            'if [[ -n "$checkpoint" ]]; then',
            f'  "$GMX" grompp -f "mdp/{filename}" -c "$coord" -t "$checkpoint" -p topol.top -n index.ndx -o "{stage}.tpr" -maxwarn 0',
            "else",
            f'  "$GMX" grompp -f "mdp/{filename}" -c "$coord" -p topol.top -n index.ndx -o "{stage}.tpr" -maxwarn 0',
            "fi",
            f'"$GMX" mdrun -deffnm "{stage}" -ntmpi "$NTMPI" -ntomp "$NTOMP"{gpu}',
            f'coord="{stage}.gro"',
            f'checkpoint="{stage}.cpt"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
