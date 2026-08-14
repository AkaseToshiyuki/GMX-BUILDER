"""Regression tests for environment-sensitive protonation input handling."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gmxbuilder.io.pdb import format_pdb_atom_name
from gmxbuilder.modules.modifications.protonation import (
    assign_all_protonations,
    assign_protonation,
    get_charge_adjustment,
    predict_pka_from_pdb,
)


def test_pdb_atom_names_follow_element_sensitive_alignment():
    assert format_pdb_atom_name("CA", "C") == " CA "
    assert format_pdb_atom_name("OD1", "O") == " OD1"
    assert format_pdb_atom_name("NA", "Na") == "NA  "
    assert format_pdb_atom_name("1HG", "H") == "1HG "


def test_propka_input_normalizes_legacy_left_aligned_atom_names(tmp_path, monkeypatch):
    pdb = tmp_path / "legacy.pdb"
    pdb.write_text(
        "ATOM      1 OD1  ASP A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )

    def fake_run(command, *, cwd, **_kwargs):
        normalized = Path(command[-1]).read_text()
        assert normalized.splitlines()[0][12:16] == " OD1"
        Path(cwd, "legacy.pka").write_text(
            "SUMMARY OF THIS PREDICTION\n"
            "ASP 1 A 4.20 3.80\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    predictions = predict_pka_from_pdb(pdb)

    assert predictions == [{
        "residue_name": "ASP",
        "chain": "A",
        "resid": 1,
        "model_pKa": 3.8,
        "predicted_pKa": 4.2,
        "shift": 0.4,
    }]


@pytest.mark.parametrize(
    "residue,pH,assigned,charge",
    [
        ("ASP", 3.8, "ASH", 0),
        ("ASP", 4.0, "ASP", -1),
        ("LYS", 10.4, "LYS", 1),
        ("LYS", 10.6, "LYN", 0),
        ("TYR", 10.0, "TYR", 0),
        ("TYR", 10.2, "TYM", -1),
        ("HIS", 5.9, "HSP", 1),
        ("HIS", 6.1, "HSE", 0),
    ],
)
def test_model_pka_assignment_branches(residue, pH, assigned, charge):
    result = assign_protonation(residue, pH)

    assert result["assigned_name"] == assigned
    assert result["charge"] == charge


def test_exact_model_pka_is_marked_as_ambiguous_microstate():
    result = assign_protonation("HIS", 6.0, his_tautomer="HSD")

    assert result["assigned_name"] == "HSD"
    assert result["ambiguous_at_pka"] is True


def test_all_assignments_preserve_sequence_indices():
    assignments = assign_all_protonations(["ALA", "ASP", "LYS"], pH=7.0)

    assert [assignment["index"] for assignment in assignments] == [0, 1, 2]
    assert [assignment["assigned_name"] for assignment in assignments] == [
        "ALA", "ASP", "LYS",
    ]


def test_charge_adjustment_uses_ph7_reference_state():
    result = get_charge_adjustment(["ASP", "LYS", "TYR"], pH=12.0)

    assert result["reference_pH"] == 7.0
    assert result["original_charge"] == 0  # ASP(-1) + LYS(+1) + TYR(0)
    assert result["new_charge"] == -2      # ASP(-1) + LYN(0) + TYM(-1)
    assert result["delta"] == -2


def test_propka_nonzero_exit_rejects_partial_output(tmp_path, monkeypatch):
    pdb = tmp_path / "protein.pdb"
    pdb.write_text("END\n")

    def fake_run(command, *, cwd, **_kwargs):
        Path(cwd, "protein.pka").write_text(
            "SUMMARY OF THIS PREDICTION\nASP 1 A 4.20 3.80\n"
        )
        return SimpleNamespace(returncode=2, stdout="", stderr="failed")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="PROPKA calculation failed"):
        predict_pka_from_pdb(pdb)
