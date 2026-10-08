"""Regression boundaries for coefficients, local overrides and stale trajectories."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as Obj

import pytest

from gmxbuilder.modules.membrane import parameter_provenance as provenance


def converter():
    path = Path(__file__).parents[1] / "tools/import_amber_lipid21.py"
    spec = importlib.util.spec_from_file_location("lipid21_importer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(scnb=6, scee=1.2):
    # Amber units: A=4*epsilon*sigma**12; B=4*epsilon*sigma**6.
    first, last = Obj(idx=0, nb_idx=1, charge=0.3), Obj(idx=3, nb_idx=1, charge=-0.4)
    term = Obj(
        atom1=first, atom4=last, improper=False, ignore_end=False, type=Obj(scnb=scnb, scee=scee)
    )
    return Obj(
        dihedrals=[term],
        ptr=lambda _: 1,
        parm_data={
            "NONBONDED_PARM_INDEX": [1],
            "LENNARD_JONES_ACOEF": [4 * 0.2 * 3**12],
            "LENNARD_JONES_BCOEF": [4 * 0.2 * 3**6],
        },
    )


def test_pair_conversion_retains_independent_lj_and_electrostatic_scales():
    for scnb, scee in ((2, 1.2), (6, 1.2), (1, 1), (4, 2)):
        row = converter()._explicit_pairs(source(scnb, scee))[0].split()
        assert row[:3] == ["1", "4", "2"]
        assert list(map(float, row[3:])) == pytest.approx(
            [1 / scee, 0.3, -0.4, 0.3, 0.2 * 4.184 / scnb], rel=1e-12
        )


def test_duplicate_proper_terms_do_not_duplicate_pairs_and_conflicts_fail():
    model = source()
    model.dihedrals *= 2
    assert len(converter()._explicit_pairs(model)) == 1
    model.dihedrals.append(source(scnb=2).dihedrals[0])
    with pytest.raises(ValueError, match="Conflicting"):
        converter()._explicit_pairs(model)


@pytest.mark.parametrize("scnb,scee", [(0, 1.2), (6, 0), (-1, 1), (6, float("nan"))])
def test_invalid_source_scaling_is_rejected(scnb, scee):
    with pytest.raises(ValueError, match="Invalid"):
        converter()._explicit_pairs(source(scnb, scee))


def test_improper_and_suppressed_endpoints_do_not_create_pairs():
    model = source()
    model.dihedrals[0].ignore_end = True
    assert converter()._explicit_pairs(model) == []
    model.dihedrals[0].ignore_end = False
    model.dihedrals[0].improper = True
    assert converter()._explicit_pairs(model) == []


def test_parameter_change_rejects_ready_flags_and_continuation(tmp_path, parameter_sources):
    metadata = {"lipid_name": "POPC", "force_field": "charmm36m", "lipid_ff": "charmm36m"}
    assert not provenance.parameters_current(metadata)
    metadata["parameter_fingerprint"] = provenance.fingerprint_from_metadata(metadata)
    work = tmp_path / "work"
    work.mkdir()
    (work / "topol.top").write_text("; resolved topology")
    provenance.record_work_parameters(work, metadata["parameter_fingerprint"])
    provenance.validate_work_parameters(work, metadata)
    (parameter_sources / "forcefield.itp").write_text("; changed coefficients")
    assert not provenance.parameters_current(metadata)
    with pytest.raises(ValueError, match="differ"):
        provenance.validate_work_parameters(work, metadata)


def test_topology_mutation_and_missing_provenance_reject_continuation(tmp_path, parameter_sources):
    metadata = {"lipid_name": "POPC", "force_field": "charmm36m", "lipid_ff": "charmm36m"}
    metadata["parameter_fingerprint"] = provenance.fingerprint_from_metadata(metadata)
    (tmp_path / "topol.top").write_text("original")
    provenance.record_work_parameters(tmp_path, metadata["parameter_fingerprint"])
    (tmp_path / "topol.top").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        provenance.validate_work_parameters(tmp_path, metadata)
    record = json.loads((tmp_path / "parameter-provenance.json").read_text())
    assert record["files"]["topol.top"] != provenance.file_hash(tmp_path / "topol.top")


@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
@pytest.mark.parametrize("lipid", ["PPCPL", "PPEPL"])
def test_plasmalogen_override_leaves_ester_chain_native(ff, lipid):
    from gmxbuilder.modules.forcefield.charmm_lipid_local import local_bonded_parameters
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    _, record = lipid_rtp_template(lipid, ff)
    vinyl = local_bonded_parameters(record, ("C33", "C32", "C31", "H1X"), 9, ff)
    assert vinyl == (("9", "180.000", "25.104000", "2"),)
    assert local_bonded_parameters(record, ("C211", "C210", "C29", "H91"), 9, ff) == ()
    _, ordinary = lipid_rtp_template("POPC", ff)
    assert local_bonded_parameters(ordinary, ("C211", "C210", "C29", "H91"), 9, ff) == ()


def test_unvalidated_mixed_model_is_blocked_but_its_validation_can_run(monkeypatch):
    from gmxbuilder.modules.forcefield.lipid_policy import (
        amber_mixed_validation_reason,
        rebuilding_library_entry,
    )
    from gmxbuilder.modules.membrane import equilibrated_library

    monkeypatch.setattr(
        equilibrated_library, "get_equilibrated_lipid_library", lambda: Obj(inspect=lambda *_: None)
    )
    assert "physical validation" in amber_mixed_validation_reason(["POPC", "CER16"])
    with rebuilding_library_entry(retry_failed_validation=True):
        assert amber_mixed_validation_reason(["POPC", "CER16"]) == ""


@pytest.mark.parametrize("name", ["GM1", "PAPI", "POP3", "SOP3"])
def test_recovered_parameters_admit_only_prepared_rebuilds(
    monkeypatch, name, unpopulated_default_lipid_library
):
    from gmxbuilder.modules.forcefield import gaff_backend
    from gmxbuilder.modules.forcefield.lipid_policy import (
        gaff_lipid_capability,
        rebuilding_library_entry,
    )

    monkeypatch.setattr(gaff_backend, "gaff_charge_method", lambda: "bcc")
    monkeypatch.setattr(gaff_backend, "cached_gaff_template", lambda *a, **kw: None)
    with rebuilding_library_entry(retry_failed_validation=True):
        allowed, reason = gaff_lipid_capability(name)
        assert not allowed
        assert "current identity-valid AM1-BCC" in reason
        assert gaff_lipid_capability("CAMP")[0]

    monkeypatch.setattr(gaff_backend, "cached_gaff_template", lambda *a, **kw: Obj())
    with rebuilding_library_entry(retry_failed_validation=True):
        assert gaff_lipid_capability(name) == (True, "")
    allowed, reason = gaff_lipid_capability(name)
    assert not allowed
    assert "awaits V4 physical validation" in reason
    with rebuilding_library_entry():
        assert not gaff_lipid_capability(name)[0]
    monkeypatch.setattr(gaff_backend, "gaff_charge_method", lambda: "gas")
    with rebuilding_library_entry(retry_failed_validation=True):
        assert not gaff_lipid_capability(name)[0]


def test_queue_default_excludes_overlapping_gaff_models():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from build_v4_library import capability_entries

    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability, lipid21_lipids

    defaults = capability_entries()
    comparison = capability_entries(include_gaff_overlap=True)
    overlap = {name for name in lipid21_lipids() if lipid21_capability(name)[0]}
    assert not any(e["family"] == "amber-gaff2" and e["lipid"] in overlap for e in defaults)
    assert len(comparison) - len(defaults) == len(overlap)
    assert all(e["lipid_ff"] == "amber-mixed" for e in defaults if e["family"] == "amber-gaff2")


def test_admission_only_policy_compatibility_is_exact_and_file_scoped(tmp_path):
    import hashlib

    relative = "modules/forcefield/lipid_policy.py"
    source = Path(__file__).parents[1] / "src/gmxbuilder" / relative
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == provenance._PIP_TEMPLATE_POLICY_HASH
    expected = "fc65f6baa9957e7010c9ffb19a95fc2349c289721807caf37877537f9aa068d9"
    assert (
        provenance._implementation_hash(source, relative, unchanged_policy=False)
        == provenance._CHECKDSH_POLICY_HASH
    )
    assert provenance._implementation_hash(source, relative) == expected
    assert provenance._implementation_hash(source, "io/top.py") == digest
    changed = tmp_path / "lipid_policy.py"
    changed.write_bytes(source.read_bytes() + b"\n# unreviewed future change\n")
    assert provenance._implementation_hash(changed, relative) == provenance.file_hash(changed)
    assert provenance._implementation_hash(changed, relative) != expected


def test_timestamp_only_gaff_compatibility_preserves_exact_parameter_identity(tmp_path):
    import hashlib

    from gmxbuilder.modules.membrane import parameter_provenance as provenance

    relative = "modules/forcefield/gaff_backend.py"
    source = Path(provenance.__file__).resolve().parents[2] / relative
    expected = "6a4ee9aa3953a370d5fe24482d331823a1b9d5effa0eb03cd2f2dc9937653ab5"
    # Reconstruct the immutable pre-fix source: only the equivalent UTC spelling
    # may differ. Parameter construction must not change under this alias.
    historical = (
        source.read_text()
        .replace("from datetime import datetime, timezone", "from datetime import UTC, datetime")
        .replace("datetime.now(timezone.utc)", "datetime.now(UTC)")
    )
    assert hashlib.sha256(historical.encode()).hexdigest() == expected
    assert provenance._implementation_hash(source, relative) == expected
    assert provenance._implementation_hash(source, "io/top.py") == provenance.file_hash(source)
    changed = tmp_path / "gaff_backend.py"
    changed.write_text(source.read_text() + "\n# A later change is not covered.\n")
    assert provenance._implementation_hash(changed, relative) == provenance.file_hash(changed)
    assert provenance._implementation_hash(changed, relative) != expected
