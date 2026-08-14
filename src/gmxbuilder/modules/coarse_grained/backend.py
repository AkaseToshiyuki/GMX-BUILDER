"""Task-private COBY orchestration for the independent Martini 3 workflow."""

from __future__ import annotations

import contextlib
import io
import math
import shutil
from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROReader
from gmxbuilder.modules.coarse_grained.assets import materialize_assets
from gmxbuilder.modules.coarse_grained.common import (
    coby_lipid_tokens,
    normalize_composition,
    strict_bool,
    task_root,
    task_step_dir,
    write_topology_texts,
)


def _finite_number(value: object, label: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModuleConfigError(f"{label} must be numeric") from exc
    if not low <= number <= high:
        raise ModuleConfigError(f"{label} must be between {low:g} and {high:g}")
    return number


def normalize_environment(config: dict, metadata: dict, coordinates=None) -> dict:
    environment = str(metadata.get("cg_environment", config.get("environment", "bilayer"))).lower()
    include_protein = metadata.get("cg_include_protein", True) is True
    rotations = {
        axis: _finite_number(config.get(f"rotate_{axis}", 0.0), f"Rotation {axis.upper()}", -180.0, 180.0)
        for axis in "xyz"
    }
    z_offset = _finite_number(config.get("z_offset", 0.0), "Protein Z offset", -8.0, 8.0)
    seed_value = config.get("seed", metadata.get("seed", 42))
    if isinstance(seed_value, bool) or not isinstance(seed_value, (int, float)):
        raise ModuleConfigError("Random seed must be an integer")
    if not float(seed_value).is_integer():
        raise ModuleConfigError("Random seed must be an integer")
    seed = int(seed_value)
    if seed < 0 or seed > 2_147_483_647:
        raise ModuleConfigError("Random seed must be between 0 and 2147483647")
    normalized = {
        "environment": environment,
        "include_protein": include_protein,
        "rotate_x": rotations["x"],
        "rotate_y": rotations["y"],
        "rotate_z": rotations["z"],
        "z_offset": z_offset,
        "seed": seed,
    }
    protein_extent = np.zeros(3, dtype=float)
    if include_protein and coordinates is not None:
        protein_coordinates = np.asarray(coordinates, dtype=float)
        if (
            protein_coordinates.ndim != 2
            or protein_coordinates.shape[1] != 3
            or not np.isfinite(protein_coordinates).all()
        ):
            raise ModuleConfigError("Mapped protein coordinates are invalid")
        if len(protein_coordinates):
            centered = protein_coordinates - protein_coordinates.mean(axis=0)
            angles = [math.radians(rotations[axis]) for axis in "xyz"]
            cx, cy, cz = (math.cos(value) for value in angles)
            sx, sy, sz = (math.sin(value) for value in angles)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            protein_extent = np.ptp(centered @ (rx @ ry @ rz).T, axis=0)
    normalized["protein_extent_nm"] = protein_extent.tolist()
    if environment == "bilayer":
        upper = normalize_composition(
            config.get("upper_leaflet", [{"name": "POPC", "ratio": 1}]), label="Upper"
        )
        asymmetric = strict_bool(config, "asymmetric", False)
        if asymmetric and "lower_leaflet" not in config:
            raise ModuleConfigError(
                "Asymmetric bilayer mode requires an explicit lower-leaflet composition"
            )
        lower = normalize_composition(
            config.get("lower_leaflet", [
                {"name": item["name"], "ratio": item["ratio"]} for item in upper
            ]),
            label="Lower",
        )
        if not asymmetric and lower != upper:
            raise ModuleConfigError(
                "Lower-leaflet composition differs from the upper leaflet; "
                "enable asymmetric bilayer mode to use it"
            )
        raw_count = config.get("n_lipids_per_leaflet", 150)
        if isinstance(raw_count, bool):
            raise ModuleConfigError("Lipids per leaflet must be an integer")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError("Lipids per leaflet must be an integer") from exc
        if count != raw_count or not 64 <= count <= 5000:
            raise ModuleConfigError("Lipids per leaflet must be an integer from 64 to 5000")
        from gmxbuilder.modules.coarse_grained.assets import load_manifest

        lipid_manifest = load_manifest()["lipids"]
        upper_apl = sum(item["ratio"] * lipid_manifest[item["name"]]["apl_nm2"] for item in upper)
        lower_apl = sum(item["ratio"] * lipid_manifest[item["name"]]["apl_nm2"] for item in lower)
        weighted_apl = 0.5 * (upper_apl + lower_apl)
        protein_xy_area = float(max(protein_extent[0], protein_extent[1]) ** 2)
        # The box is server-derived.  Keep the explicit leaflet packing area,
        # but always add the same three-nanometre total protein clearance that
        # the PBC safety gate requires.  Previously Z used only 1 nm and then
        # immediately failed its own 3 nm validator.
        required_xy = float(max(protein_extent[0], protein_extent[1])) + 3.0
        required_z = float(protein_extent[2]) + 2.0 * abs(z_offset) + 3.0
        box_xy = max(math.sqrt(count * weighted_apl + protein_xy_area), required_xy, 5.0)
        dry_box_z = max(6.0, required_z)
        normalized.update({
            "upper_leaflet": upper,
            "lower_leaflet": lower,
            "asymmetric": asymmetric,
            "n_lipids_per_leaflet": count,
            "weighted_apl_nm2": weighted_apl,
            "box_xy": box_xy,
            "box_z": dry_box_z,
            "dry_box_z": dry_box_z,
        })
    elif environment == "solution":
        padding = 1.5
        box_xy = max(float(max(protein_extent[0], protein_extent[1])) + 2.0 * padding, 5.0)
        box_z = max(float(protein_extent[2]) + 2.0 * abs(z_offset) + 2.0 * padding, 6.0)
        normalized.update({
            "box_xy": box_xy,
            "box_z": box_z,
            "dry_box_z": box_z,
        })
    else:
        raise ModuleConfigError("environment must be solution or bilayer")
    return normalized


def validate_protein_box(system, environment: dict, *, clearance_nm: float = 3.0) -> None:
    """Expand a server-derived PBC box before COBY can wrap the protein.

    COBY rotates centered coordinates in X/Y/Z order and then wraps them into
    the requested rectangular cell.  Three nanometres of total clearance also
    covers its internal bead-radius placement margin.  A wrapped protein can
    look fragmented and is not a safe starting structure, so the automatic
    dimensions are reconciled with the actual transformed coordinates before
    COBY.  Users never need to supply the physical box size.
    """
    if not environment.get("include_protein", True) or system.num_atoms == 0:
        return
    coordinates = np.asarray(system.structure.coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ModuleConfigError("Mapped protein coordinates are invalid")
    centered = coordinates - coordinates.mean(axis=0)
    angles = [math.radians(float(environment[f"rotate_{axis}"])) for axis in "xyz"]
    cx, cy, cz = (math.cos(value) for value in angles)
    sx, sy, sz = (math.sin(value) for value in angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotated = centered @ (rx @ ry @ rz).T
    extent = np.ptp(rotated, axis=0)
    required_xy = float(max(extent[0], extent[1]) + clearance_nm)
    required_z = float(extent[2] + 2.0 * abs(float(environment["z_offset"])) + clearance_nm)
    adjustments = {}
    if float(environment["box_xy"]) + 1e-9 < required_xy:
        adjustments["box_xy"] = [float(environment["box_xy"]), required_xy]
        environment["box_xy"] = required_xy
    if float(environment["box_z"]) + 1e-9 < required_z:
        adjustments["box_z"] = [float(environment["box_z"]), required_z]
        environment["box_z"] = required_z
    if adjustments:
        environment["automatic_box_adjustments"] = adjustments


def normalize_solvation(config: dict, metadata: dict) -> dict:
    environment = str(metadata.get("cg_environment", "bilayer"))
    include = strict_bool(config, "include_solvent", True)
    if environment == "solution" and not include:
        raise ModuleConfigError("Solution-phase Martini systems require solvent")
    salt = _finite_number(config.get("salt_molarity", 0.15), "Salt concentration", 0.0, 1.0)
    default_padding = 2.0 if environment == "bilayer" else 1.5
    padding = _finite_number(config.get("padding_nm", default_padding), "Solvent padding", 1.0, 8.0)
    return {
        "include_solvent": include,
        "salt_molarity": salt,
        "padding_nm": padding,
        "water_model": "W",
    }


def _materialize_inputs(system, config: dict, work: Path) -> tuple[list[str], str | None]:
    toppar = work / "toppar"
    materialize_assets(toppar)
    asset_names = list(__import__(
        "gmxbuilder.modules.coarse_grained.assets", fromlist=["load_manifest"]
    ).load_manifest()["files"])
    core_name = "martini_v3.0.0.itp"
    bonded_name = "martini_v3.0.0_ffbonded_v2.itp"
    asset_names = [core_name, bonded_name] + sorted(
        name for name in asset_names if name not in {core_name, bonded_name}
    )
    bundle = toppar / "martini3_bundle.top"
    bundle.write_text("".join(f'#include "{name}"\n' for name in asset_names), encoding="utf-8")
    itp_input = [f"file:{bundle}"]

    protein_arg = None
    if system.metadata.get("cg_include_protein", True):
        texts = dict(system.metadata.get("cg_topology_texts") or {})
        protein_texts = {
            name: text for name, text in texts.items()
            if name.endswith(".itp") and name != "martini_v3.0.0.itp"
        }
        if not protein_texts:
            raise ModuleConfigError("Mapped protein topology is missing")
        write_topology_texts(protein_texts, toppar)
        for name in sorted(protein_texts):
            itp_input.append(f"include:{toppar / name}")
        protein_source = (task_root(config) / str(system.metadata.get("cg_protein_pdb", ""))).resolve()
        task_dir = task_root(config)
        if task_dir not in protein_source.parents or not protein_source.is_file() or protein_source.is_symlink():
            raise ModuleConfigError("Task-owned mapped protein coordinates are missing")
        protein_path = work / "cg_protein.pdb"
        shutil.copy2(protein_source, protein_path)
        mapping = system.metadata.get("cg_mapping") or {}
        molecule_types = list(mapping.get("molecule_types") or [])
        if not molecule_types:
            raise ModuleConfigError("Mapped protein molecule types are missing")
        env = system.metadata.get("cg_environment_config") or {}
        parts = [
            f"file:{protein_path}",
            "moleculetypes:" + ":".join(molecule_types),
            # Use a full regular-bead radius rather than COBY's half-radius
            # default when carving solvent around a mapped protein.
            "buffer:0.264",
        ]
        for axis in "xyz":
            value = float(env.get(f"rotate_{axis}", 0.0))
            if value:
                parts.append(f"r{axis}:{value:g}")
        if float(env.get("z_offset", 0.0)):
            parts.append(f"center:0:0:{float(env['z_offset']):g}")
        protein_arg = " ".join(parts)
    return itp_input, protein_arg


def _integer_leaflet_composition(entries: list[dict], total: int) -> dict[str, int]:
    """Allocate an exact leaflet count with deterministic largest remainders."""
    exact = [(str(item["name"]), float(item["ratio"]) * total) for item in entries]
    counts = {name: int(math.floor(value)) for name, value in exact}
    remainder = total - sum(counts.values())
    ranked = sorted(exact, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for name, _value in ranked[:remainder]:
        counts[name] += 1
    return counts


def _membrane_command(environment: dict, corrections: dict | None = None) -> str:
    """Build one COBY bilayer command, including optional exact-count offsets."""
    corrections = corrections or {}
    parts = ["params:LTF", "leaflet:upper", *coby_lipid_tokens(environment["upper_leaflet"])]
    for name, value in sorted(dict(corrections.get("upper") or {}).items()):
        if int(value):
            parts.append(
                f"lipid_extra:name:{name}:extra_type:absolute:extra_val:{int(value)}"
            )
    parts.extend(["leaflet:lower", *coby_lipid_tokens(environment["lower_leaflet"])])
    for name, value in sorted(dict(corrections.get("lower") or {}).items()):
        if int(value):
            parts.append(
                f"lipid_extra:name:{name}:extra_type:absolute:extra_val:{int(value)}"
            )
    parts.extend(["leaflet:membrane", "protein_buffer:0.264", "kick:0.02"])
    return " ".join(parts)


def _built_leaflet_counts(builder) -> dict[str, dict[str, int]]:
    """Read final per-leaflet counts from the pinned COBY builder state."""
    membranes = getattr(builder, "MEMBRANES", None)
    if not isinstance(membranes, dict) or len(membranes) != 1:
        raise ModuleConfigError("COBY did not expose one completed bilayer")
    membrane = next(iter(membranes.values()))
    leaflets = membrane.get("leaflets") if isinstance(membrane, dict) else None
    if not isinstance(leaflets, dict):
        raise ModuleConfigError("COBY did not expose completed leaflet counts")
    result: dict[str, dict[str, int]] = {}
    for public_name, coby_name in (("upper", "upper_leaf"), ("lower", "lower_leaf")):
        leaflet = leaflets.get(coby_name)
        counts = leaflet.get("leaf_lipid_count_dict") if isinstance(leaflet, dict) else None
        if not isinstance(counts, dict):
            raise ModuleConfigError(f"COBY did not expose the {public_name}-leaflet count")
        result[public_name] = {str(name): int(value) for name, value in counts.items()}
    return result


def _leaflet_count_corrections(environment: dict, actual: dict) -> dict[str, dict[str, int]]:
    total = int(environment["n_lipids_per_leaflet"])
    desired = {
        "upper": _integer_leaflet_composition(environment["upper_leaflet"], total),
        "lower": _integer_leaflet_composition(environment["lower_leaflet"], total),
    }
    corrections: dict[str, dict[str, int]] = {"upper": {}, "lower": {}}
    for leaflet in ("upper", "lower"):
        if set(actual[leaflet]) - set(desired[leaflet]):
            unexpected = ", ".join(sorted(set(actual[leaflet]) - set(desired[leaflet])))
            raise ModuleConfigError(
                f"COBY generated unexpected {leaflet}-leaflet lipid type(s): {unexpected}"
            )
        corrections[leaflet] = {
            name: desired[leaflet][name] - int(actual[leaflet].get(name, 0))
            for name in desired[leaflet]
        }
    return corrections


def _has_corrections(corrections: dict[str, dict[str, int]]) -> bool:
    return any(value for leaflet in corrections.values() for value in leaflet.values())


def build_with_coby(system, config: dict, *, solvate: bool, final_salt: bool) -> tuple[Path, Path, str]:
    """Build one deterministic stage and return GRO, topology and COBY log."""
    try:
        from COBY import COBY
    except ImportError as exc:
        raise ModuleConfigError("Pinned COBY 1.0.14 is not installed") from exc

    work = task_step_dir(config) / "coby"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    itp_input, protein_arg = _materialize_inputs(system, config, work)
    env = dict(system.metadata.get("cg_environment_config") or {})
    solv = dict(system.metadata.get("cg_solvation_config") or {})
    kwargs: dict = {
        "randseed": int(env.get("seed", 42)),
        "box": [float(env.get("box_xy", 12.0)), float(env.get("box_xy", 12.0)), float(env.get("box_z", 14.0))],
        "box_type": "rectangular",
        "itp_input": itp_input,
        "out_sys": str(work / "system"),
        "out_top": str(work / "topol.top"),
        "out_log": str(work / "coby.log"),
        "system_name": "GMXBUILDER Martini 3 system",
        "verbose": 0,
    }
    if protein_arg:
        kwargs["protein"] = protein_arg
    existing_corrections = dict(env.get("lipid_count_corrections") or {})
    if env.get("environment") == "bilayer":
        kwargs["membrane"] = _membrane_command(env, existing_corrections)
    if solvate and solv.get("include_solvent", True):
        salt = float(solv.get("salt_molarity", 0.15)) if final_salt else 0.0
        kwargs["solvation"] = f"solv:W pos:NA neg:CL salt_molarity:{salt:g}"

    output_capture = io.StringIO()

    def _run_coby():
        try:
            with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                return COBY(**kwargs)
        except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            raise ModuleConfigError(f"COBY system construction failed: {detail}") from exc

    builder = _run_coby()
    if env.get("environment") == "bilayer":
        actual = _built_leaflet_counts(builder)
        residual = _leaflet_count_corrections(env, actual)
        if _has_corrections(residual):
            combined = {
                leaflet: {
                    name: int(dict(existing_corrections.get(leaflet) or {}).get(name, 0)) + delta
                    for name, delta in residual[leaflet].items()
                }
                for leaflet in ("upper", "lower")
            }
            for name in ("system.gro", "topol.top", "coby.log"):
                (work / name).unlink(missing_ok=True)
            kwargs["membrane"] = _membrane_command(env, combined)
            builder = _run_coby()
            actual = _built_leaflet_counts(builder)
            final_residual = _leaflet_count_corrections(env, actual)
            if _has_corrections(final_residual):
                details = "; ".join(
                    f"{leaflet}={sum(actual[leaflet].values())}"
                    for leaflet in ("upper", "lower")
                )
                raise ModuleConfigError(
                    "COBY could not satisfy the requested exact leaflet counts after "
                    f"deterministic correction ({details})"
                )
            env["lipid_count_corrections"] = combined
            system.metadata["cg_environment_config"] = env
    gro = work / "system.gro"
    top = work / "topol.top"
    if not gro.is_file() or not top.is_file():
        raise ModuleConfigError("COBY completed without coordinate and topology outputs")
    built_structure = GROReader().read(gro)
    try:
        fractional = built_structure.coordinates @ np.linalg.inv(
            built_structure.box_vectors
        )
    except np.linalg.LinAlgError as exc:
        raise ModuleConfigError("COBY produced a singular periodic box") from exc
    outside = np.any((fractional < -1e-5) | (fractional >= 1.0 + 1e-5), axis=1)
    if np.any(outside):
        raise ModuleConfigError(
            f"COBY produced {int(np.count_nonzero(outside))} bead(s) outside the "
            "primary periodic cell. Increase the box dimensions or reduce the "
            "protein offset; wrapped output was rejected."
        )
    # COBY must receive absolute inputs because the Web process cannot safely
    # change its process-wide cwd.  Convert emitted includes back to package-
    # relative paths before persisting; exports must be portable and must not
    # disclose server filesystem paths.
    topology_text = top.read_text(encoding="utf-8")
    topology_text = topology_text.replace(str(work) + "/", "")
    top.write_text(topology_text, encoding="utf-8")
    log = (work / "coby.log").read_text(encoding="utf-8", errors="replace") if (work / "coby.log").is_file() else output_capture.getvalue()
    return gro, top, log
