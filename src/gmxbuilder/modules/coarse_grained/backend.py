"""Task-private COBY orchestration for the independent Martini 3 workflow."""

from __future__ import annotations

import contextlib
import io
import math
import shutil
from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
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


def normalize_environment(config: dict, metadata: dict) -> dict:
    environment = str(metadata.get("cg_environment", config.get("environment", "bilayer"))).lower()
    include_protein = metadata.get("cg_include_protein", True) is True
    box_xy = _finite_number(config.get("box_xy", 12.0), "Box XY", 5.0, 40.0)
    box_z = _finite_number(config.get("box_z", 14.0 if environment == "bilayer" else 12.0), "Box Z", 6.0, 50.0)
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
        "box_xy": box_xy,
        "box_z": box_z,
        "rotate_x": rotations["x"],
        "rotate_y": rotations["y"],
        "rotate_z": rotations["z"],
        "z_offset": z_offset,
        "seed": seed,
    }
    if environment == "bilayer":
        upper = normalize_composition(
            config.get("upper_leaflet", [{"name": "POPC", "ratio": 1}]), label="Upper"
        )
        asymmetric = strict_bool(config, "asymmetric", False)
        lower = normalize_composition(
            config.get("lower_leaflet", upper if asymmetric else [
                {"name": item["name"], "ratio": item["ratio"]} for item in upper
            ]),
            label="Lower",
        )
        normalized.update({"upper_leaflet": upper, "lower_leaflet": lower, "asymmetric": asymmetric})
    return normalized


def validate_protein_box(system, environment: dict, *, clearance_nm: float = 3.0) -> None:
    """Reject a PBC box that would make COBY wrap a rotated mapped protein.

    COBY rotates centered coordinates in X/Y/Z order and then wraps them into
    the requested rectangular cell.  Three nanometres of total clearance also
    covers its internal bead-radius placement margin.  A wrapped protein can
    look fragmented and is not a safe starting structure, so validation must
    happen before COBY.
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
    failures = []
    if float(environment["box_xy"]) + 1e-9 < required_xy:
        failures.append(
            f"Box X/Y {environment['box_xy']:.2f} nm is smaller than the rotated protein "
            f"extent ({max(extent[0], extent[1]):.2f} nm); use at least {required_xy:.2f} nm"
        )
    if float(environment["box_z"]) + 1e-9 < required_z:
        failures.append(
            f"Box Z {environment['box_z']:.2f} nm is smaller than the positioned protein "
            f"extent ({extent[2]:.2f} nm); use at least {required_z:.2f} nm"
        )
    if failures:
        raise ModuleConfigError(
            "; ".join(failures) + ". Increase the box to prevent periodic wrapping of the protein"
        )


def normalize_solvation(config: dict, metadata: dict) -> dict:
    environment = str(metadata.get("cg_environment", "bilayer"))
    include = strict_bool(config, "include_solvent", True)
    if environment == "solution" and not include:
        raise ModuleConfigError("Solution-phase Martini systems require solvent")
    salt = _finite_number(config.get("salt_molarity", 0.15), "Salt concentration", 0.0, 1.0)
    return {"include_solvent": include, "salt_molarity": salt, "water_model": "W"}


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
    if env.get("environment") == "bilayer":
        upper = " ".join(coby_lipid_tokens(env["upper_leaflet"]))
        lower = " ".join(coby_lipid_tokens(env["lower_leaflet"]))
        kwargs["membrane"] = " ".join([
            "params:LTF", f"leaflet:upper {upper}", f"leaflet:lower {lower}",
            "leaflet:membrane protein_buffer:0.264 kick:0.02",
        ])
    if solvate and solv.get("include_solvent", True):
        salt = float(solv.get("salt_molarity", 0.15)) if final_salt else 0.0
        kwargs["solvation"] = f"solv:W pos:NA neg:CL salt_molarity:{salt:g}"

    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            COBY(**kwargs)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise ModuleConfigError(f"COBY system construction failed: {detail}") from exc
    gro = work / "system.gro"
    top = work / "topol.top"
    if not gro.is_file() or not top.is_file():
        raise ModuleConfigError("COBY completed without coordinate and topology outputs")
    # COBY must receive absolute inputs because the Web process cannot safely
    # change its process-wide cwd.  Convert emitted includes back to package-
    # relative paths before persisting; exports must be portable and must not
    # disclose server filesystem paths.
    topology_text = top.read_text(encoding="utf-8")
    topology_text = topology_text.replace(str(work) + "/", "")
    top.write_text(topology_text, encoding="utf-8")
    log = (work / "coby.log").read_text(encoding="utf-8", errors="replace") if (work / "coby.log").is_file() else output_capture.getvalue()
    if protein_arg and "Protein beads are outside pbc" in log:
        raise ModuleConfigError(
            "COBY detected mapped protein beads outside the periodic box. "
            "Increase the box dimensions or reduce the protein offset; wrapped protein coordinates were rejected"
        )
    return gro, top, log
