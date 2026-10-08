"""Independent coordinate checks for the construction-only dry membrane policy."""

import numpy as np
import pytest

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.modules.export.exporter import ExportModule
from gmxbuilder.modules.ions.add_ions import IonBuilder
from gmxbuilder.modules.solvation.membrane_exclusion import assert_membrane_water_free
from gmxbuilder.modules.solvation.solvate import SolvationBuilder
from tests.test_solvation import _asymmetric_membrane_system


@pytest.mark.parametrize("model", ["tip3p", "tip4p", "spc", "spce"])
@pytest.mark.parametrize("prebuilt", [False, True])
def test_every_water_site_stays_outside_lipid_envelope_after_gro_roundtrip(
    tmp_path, model, prebuilt
):
    source = _asymmetric_membrane_system()
    source.metadata["water_model"] = model
    system = (
        SolvationBuilder()
        .run(
            source,
            {
                "water_model": model,
                "remove_overlap": False,
                "use_prebuilt_water": prebuilt,
                "box_padding": 2.0,
            },
        )
        .system
    )
    path = tmp_path / "solvated.gro"
    GROWriter.write(system.structure, path)
    structure = GROReader().read(path)
    lipid_indices = system.component_by_kind(ComponentKind.MEMBRANE)[0].atom_indices
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    lower, upper = (
        structure.coordinates[lipid_indices, 2].min(),
        structure.coordinates[lipid_indices, 2].max(),
    )
    z = structure.coordinates[water.atom_indices, 2]
    assert len(z) > 0
    assert not np.any((z >= lower) & (z <= upper))
    assert len(z) == water.metadata["n_molecules"] * (4 if model == "tip4p" else 3)
    assert np.any(z < lower) and np.any(z > upper)


def test_hydrogen_virtual_site_pore_edge_and_periodic_waters_are_removed_whole(monkeypatch):
    # Final lipid surfaces are z=2,6. Waters are at arbitrary XY positions,
    # including a pore and the periodic edge; O-only filtering would miss two.
    molecules = np.array(
        [
            [[2, 2, 4], [2, 2, 4.1], [2, 2, 3.9], [2, 2, 4]],
            [[4, 0, 1.95], [4, 0, 2.02], [4, 0, 1.90], [4, 0, 1.95]],
            [[1, 1, 6.05], [1, 1, 6.10], [1, 1, 6.06], [1, 1, 5.99]],
            [[1, 1, 12], [1, 1, 12.1], [1, 1, 11.9], [1, 1, 12]],
            [[1, 1, 1], [1, 1, 1.1], [1, 1, 0.9], [1, 1, 1]],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        SolvationBuilder, "_fill_from_prebuilt", lambda *args: (molecules.reshape(-1, 3).copy(), 5)
    )
    source = _asymmetric_membrane_system()
    source.metadata["water_model"] = "tip4p"
    system = (
        SolvationBuilder()
        .run(source, {"water_model": "tip4p", "box_padding": 2, "remove_overlap": False})
        .system
    )
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    assert water.metadata["n_molecules"] == 1
    np.testing.assert_allclose(system.coordinates[water.atom_indices], molecules[-1])


def test_water_that_enters_margin_after_gro_rounding_is_removed(monkeypatch, tmp_path):
    # The upper lipid surface is 6.000 nm. 6.0014 is outside the raw
    # 0.0011-nm margin, but GRO writes it as 6.001 nm, inside that margin.
    molecules = np.array(
        [
            [[1, 1, 6.0014]] * 3,
            [[1, 1, 6.0040]] * 3,
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        SolvationBuilder, "_fill_from_prebuilt", lambda *args: (molecules.reshape(-1, 3), 2)
    )
    system = (
        SolvationBuilder()
        .run(
            _asymmetric_membrane_system(),
            {"water_model": "tip3p", "box_padding": 2, "remove_overlap": False},
        )
        .system
    )
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    assert water.metadata["n_molecules"] == 1
    path = tmp_path / "solvated.gro"
    GROWriter.write(system.structure, path)
    saved = system.copy()
    saved.structure = GROReader().read(path)
    assert_membrane_water_free(saved)


def test_stale_checkpoint_fails_before_ion_mutation_or_export_cleanup(tmp_path):
    system = SolvationBuilder().run(_asymmetric_membrane_system(), {"box_padding": 2}).system
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    system.coordinates[water.atom_indices[1], 2] = 4
    sentinel = tmp_path / "input.gro"
    sentinel.write_text("keep")
    for module, config in [(IonBuilder(), {}), (ExportModule(), {"output_dir": str(tmp_path)})]:
        with pytest.raises(ModuleConfigError, match="Rerun solvation"):
            module.run(system, config)
    assert sentinel.read_text() == "keep"


def test_saved_membrane_reference_handles_wrapped_atoms_and_periodic_water():
    system = SolvationBuilder().run(_asymmetric_membrane_system(), {"box_padding": 2}).system
    lipid = system.component_by_kind(ComponentKind.MEMBRANE)[0]
    water = system.component_by_kind(ComponentKind.SOLVENT)[0]
    system.coordinates[lipid.atom_indices[0], 2] += 8
    assert_membrane_water_free(system)
    system.coordinates[water.atom_indices[0], 2] = -4
    with pytest.raises(ModuleConfigError, match="solvent sites"):
        assert_membrane_water_free(system)
