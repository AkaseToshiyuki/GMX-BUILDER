"""Regression tests for environment-sensitive protonation input handling."""

from pathlib import Path
from types import SimpleNamespace

from gmxbuilder.io.pdb import format_pdb_atom_name
from gmxbuilder.modules.modifications.protonation import predict_pka_from_pdb


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
