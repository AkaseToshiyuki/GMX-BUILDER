"""WYSIWYG contract tests between Orient and Membrane checkpoints."""

from pathlib import Path

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from gmxbuilder.modules.membrane.orient_module import (
    OrientModule,
    assess_membrane_orientation,
)
from gmxbuilder.modules.membrane.orient import (
    _analyze_tm_helix_bundle,
    _find_best_ppm_orientation,
    _membrane_transfer_score,
    _scan_ppm_z_and_tilt,
)


def _synthetic_helical_bundle(tilt_degrees: float = 24.0) -> Structure:
    """Return four parallel idealised hydrophobic alpha helices."""
    angle = np.radians(tilt_degrees)
    axis = np.array([np.sin(angle), 0.0, np.cos(angle)])
    radial_u = np.array([0.0, 1.0, 0.0])
    radial_v = np.cross(axis, radial_u)
    coordinates = []
    atom_names = []
    resnames = []
    resids = []
    chain_ids = []
    offsets = [(-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)]
    for chain_index, (offset_u, offset_v) in enumerate(offsets):
        chain = chr(ord("A") + chain_index)
        for residue_index in range(24):
            phase = np.radians(100.0 * residue_index)
            axial = (residue_index - 11.5) * 0.15 * axis
            radial = 0.23 * (np.cos(phase) * radial_u + np.sin(phase) * radial_v)
            offset = offset_u * radial_u + offset_v * radial_v
            coordinates.append(axial + radial + offset)
            atom_names.append("CA")
            resnames.append("LEU")
            resids.append(residue_index + 1)
            chain_ids.append(chain)
    return Structure(
        coordinates=np.asarray(coordinates),
        box_vectors=np.eye(3) * 10.0,
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=chain_ids,
        elements=["C"] * len(coordinates),
    )


def test_orientation_viewer_uses_backend_coordinates_and_fixed_membrane():
    """Auto and manual previews must display backend Step 4 coordinates."""
    app_js = (
        Path(__file__).parents[1] / "src" / "gmxbuilder" / "web" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    redraw = app_js.split("function _redrawOrientViewer()", 1)[1].split(
        "window._redrawOrientViewer", 1
    )[0]
    assert "drawMembranePlane(v, 0.0, halfThick, 0.0, 0.0);" in redraw
    assert "planeZ" not in redraw
    assert "'/api/orient-preview/' + state.taskId" in app_js
    assert "await window._loadOrientationCheckpointPreview()" in app_js


def test_orientation_resume_uses_incremental_step_config():
    app_js = (
        Path(__file__).parents[1] / "src" / "gmxbuilder" / "web" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    restore = app_js.split("function _restoreOrientationConfig", 1)[1].split(
        "function invalidateOrientationCheck", 1
    )[0]
    assert "var savedConfig = taskState && taskState.step_orient_config" in restore
    assert "var savedResult = taskState && taskState.orient" in restore
    assert "_orientPhi = phi" in restore
    assert "_setOrientationModeUI();" in restore


def test_membrane_builder_preserves_checked_orientation(monkeypatch):
    structure = Structure(
        coordinates=np.array(
            [
                [-0.3, 0.0, -1.2],
                [0.2, 0.1, -0.7],
                [-0.1, -0.2, -0.2],
                [0.3, 0.0, 0.3],
                [-0.2, 0.2, 0.8],
                [0.1, -0.1, 1.3],
            ]
        ),
        box_vectors=np.eye(3) * 8.0,
        atom_names=["CA"] * 6,
        resnames=["LEU", "ILE", "VAL", "PHE", "LEU", "ILE"],
        resids=list(range(1, 7)),
        chain_ids=["A"] * 6,
        elements=["C"] * 6,
    )
    system = System(
        structure=structure,
        components=[Component("PROTEIN", ComponentKind.PROTEIN, np.arange(6))],
        metadata={"seed": 42},
    )

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.orient._find_best_ppm_orientation",
        lambda _structure, half_thickness=None: (
            np.array([0.0, 0.0, 1.0]),
            0.0,
            np.array([1.0, 0.0, 0.0]),
            0.0,
            0.0,
            0.0,
        ),
    )
    oriented = (
        OrientModule()
        .run(
            system,
            {
                "method": "manual",
                "z_offset": 0.4,
                "tilt": 20.0,
                "phi": 35.0,
                "half_thickness": 1.7,
            },
        )
        .system
    )
    assert oriented.metadata["_orientation_half_thickness_nm"] == 1.7
    checked_coordinates = oriented.structure.coordinates.copy()
    checked_params = dict(oriented.metadata["_orient_params"])

    result = MembraneBuilder().run(
        oriented,
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 64},
    )

    protein = result.system.component_by_name("PROTEIN")
    built_coordinates = result.system.coordinates[protein.atom_indices]
    translations = built_coordinates - checked_coordinates
    assert result.success
    assert np.allclose(translations, translations[0], atol=1e-10)
    assert result.system.metadata["_orient_params"] == checked_params
    assert not any("reversed" in entry or "untilted" in entry for entry in result.log)


def test_orientation_quality_warns_about_globular_overembedding():
    coordinates = []
    atom_names = []
    resnames = []
    resids = []
    for index in range(20):
        coordinates.append([0.1 * (index % 4), 0.1 * (index % 5), 0.05 * (index - 10)])
        atom_names.append("CA")
        resnames.append("LEU" if index % 2 == 0 else "ALA")
        resids.append(index + 1)
    structure = Structure(
        coordinates=np.asarray(coordinates),
        box_vectors=np.eye(3) * 8.0,
        atom_names=atom_names,
        resnames=resnames,
        resids=resids,
        chain_ids=["A"] * 20,
        elements=["C"] * 20,
    )
    system = System(
        structure=structure,
        components=[Component("PROTEIN", ComponentKind.PROTEIN, np.arange(20))],
    )

    quality = assess_membrane_orientation(system)

    assert quality["status"] == "warning"
    assert quality["core_fraction"] == 1.0
    assert any("over-embedding" in warning for warning in quality["warnings"])


def test_ppm_scan_returns_the_translation_that_was_scored():
    coords = np.array(
        [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -1.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.0],
        ]
    )
    energies = np.array([-1.0, -1.0, -1.0, 1.0, 1.0])

    z_offset, _tilt_axis, _tilt_angle, score = _scan_ppm_z_and_tilt(
        coords,
        energies,
        half_thickness=1.4,
        max_tilt=0.0,
        n_scans=51,
        tilt_improvement_threshold=0.95,
    )

    z_span = max(abs(coords[:, 2].min()), abs(coords[:, 2].max()), 2.1)
    scan_values = np.linspace(-z_span, z_span, 51)
    scan_scores = [_membrane_transfer_score(coords, energies, z, 1.4) for z in scan_values]
    assert z_offset > 0.0
    assert np.isclose(z_offset, scan_values[int(np.argmin(scan_scores))])
    assert np.isclose(score, min(scan_scores))


def test_ppm_uses_transmembrane_helix_consensus_for_bundle_axis():
    structure = _synthetic_helical_bundle(tilt_degrees=24.0)
    expected_axis = np.array([np.sin(np.radians(24.0)), 0.0, np.cos(np.radians(24.0))])

    analysis = _analyze_tm_helix_bundle(structure)
    ppm_axis, *_ = _find_best_ppm_orientation(structure)

    assert analysis["window_count"] >= 12
    assert analysis["confidence"] >= 0.72
    assert abs(float(np.dot(analysis["axis"], expected_axis))) > np.cos(np.radians(3.0))
    assert abs(float(np.dot(ppm_axis, expected_axis))) > np.cos(np.radians(3.0))


def test_orientation_quality_detects_tilted_bundle_and_buried_non_tm_region():
    structure = _synthetic_helical_bundle(tilt_degrees=28.0)
    extra_coordinates = np.array([[-0.5 + 0.2 * index, 2.2, 0.0] for index in range(6)])
    structure.coordinates = np.vstack([structure.coordinates, extra_coordinates])
    structure.atom_names.extend(["CA"] * 6)
    structure.resnames.extend(["GLU"] * 6)
    structure.resids.extend(list(range(1, 7)))
    structure.chain_ids.extend(["X"] * 6)
    structure.segids.extend([""] * 6)
    structure.elements.extend(["C"] * 6)
    structure.occupancies.extend([1.0] * 6)
    structure.tempfactors.extend([0.0] * 6)
    system = System(
        structure=structure,
        components=[
            Component(
                "PROTEIN",
                ComponentKind.PROTEIN,
                np.arange(structure.num_atoms),
            )
        ],
    )

    quality = assess_membrane_orientation(system)

    assert quality["tm_bundle_tilt_degrees"] > 20.0
    assert quality["non_tm_core_residue_count"] == 6
    assert any("transmembrane-helix bundle" in warning for warning in quality["warnings"])
    assert any("non-transmembrane" in warning for warning in quality["warnings"])
