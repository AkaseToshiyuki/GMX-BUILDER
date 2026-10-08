"""Synchronous atomistic orientation previews used by web endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.io.pdb import PDBParser, PDBWriter
from gmxbuilder.modules.membrane.orient import compute_orientation, orient_protein
from gmxbuilder.modules.membrane.orient_module import OrientModule


def legacy_orientation_payload(
    pdb_path: str,
    algorithm: str,
    half_thickness: float | None,
) -> dict:
    """Compute the legacy orientation response off the event-loop thread."""
    parser = PDBParser()
    structure = parser.parse(pdb_path)
    z_offset, _, tilt_rad = compute_orientation(
        structure,
        algorithm=algorithm,
        half_thickness=half_thickness,
    )
    oriented = parser.parse(pdb_path)
    orient_protein(
        oriented,
        method=algorithm,
        half_thickness=half_thickness,
    )
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-orient-legacy-") as temporary_dir:
        oriented_path = Path(temporary_dir) / "oriented.pdb"
        PDBWriter.write(oriented, oriented_path)
        oriented_pdb = oriented_path.read_text(encoding="utf-8")
    return {
        "algorithm": algorithm,
        "z_offset": round(z_offset, 2),
        "tilt_degrees": round(np.degrees(tilt_rad), 1),
        "oriented_pdb": oriented_pdb,
    }


def generate_orientation_preview(task_dir: Path, config: dict) -> dict:
    """Run the real atomistic orientation module without persisting output."""
    structure_checkpoint = task_dir / "steps" / "structure"
    if not (structure_checkpoint / "system.npz").exists():
        raise FileNotFoundError(
            "Structure checkpoint is missing; run Check Structure Processing first."
        )

    system = System.load_checkpoint(structure_checkpoint)
    preview_config = dict(config)
    preview_config.setdefault("seed", system.metadata.get("seed", 42))
    module = OrientModule()
    module.validate_config(preview_config)
    result = module.execute(system, preview_config)
    if not result.success:
        raise RuntimeError("Orientation module reported failure")

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-orient-preview-") as temporary_dir:
        preview_path = Path(temporary_dir) / "viewer.pdb"
        result.system.write_viewer_pdb(preview_path)
        oriented_pdb = preview_path.read_text(encoding="utf-8")

    return {
        "status": "ok",
        "method": result.system.metadata.get(
            "_orientation_method",
            preview_config.get("method", "ppm"),
        ),
        "orientation": result.system.metadata.get("_orient_params", {}),
        "orientation_quality": result.system.metadata.get("_orientation_quality", {}),
        "oriented_pdb": oriented_pdb,
        "log": result.log,
    }
