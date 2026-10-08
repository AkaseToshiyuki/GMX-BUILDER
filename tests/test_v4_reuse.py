"""V4 reuse namespace, provenance and publication boundaries; no dynamics."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.modules.membrane import v4_reuse
from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
from tests.test_v4_construction import pair  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import build_v4_library as queue  # noqa: E402
import v4_reuse_publish as publisher  # noqa: E402
from assemble_v4_library import assemble_entry  # noqa: E402


def test_allowlist_contains_exactly_the_reviewed_58_and_excludes_the_other_20():
    reviewed = set(
        (
            "BSM CER16 CER18 CER24 DAPC DAPE DAPG DEPC DGDG DLIPA DLIPC DLIPE DLIPG "
            "DLIPS DLPC DMPC DMPE DMPG DOPA DOPC DOPE DOPG DOPGD DOPS DPEPE DPPA DPPC "
            "DPPE DPPG DPPGD DPPS DSM DSPC LPC16 LPC18 LPE16 LYSPG MGDG NSM PAPC PMPC "
            "POPA POPC POPE POPG POPI POPS PSM PUPC SAPC SMPC SOPC SOPE SOPG SOPI SSM "
            "TMCL TOCL"
        ).split()
    )
    assert set(v4_reuse.audit_registry()["entries"]) == reviewed
    assert len(reviewed) == 58
    excluded = (
        "CAMP CHOL ERG SITO STIG POP2 POP3 SOP2 SOP3 GM1 PPCPL PPEPL PAPE PAPG "
        "PAPI PIPI SAPE SAPI SAPS SOPS"
    ).split()
    for name in excluded:
        assert v4_reuse.source_entry({"family": "charmm36-lipid", "lipid": name}) is None


def test_installed_file_cache_invalidates_after_parameter_change(tmp_path):
    path = tmp_path / "parameter.itp"
    path.write_text("original")
    files = {path.name: v4_reuse.file_digest(path)}
    assert v4_reuse.installed_matches(tmp_path, files)
    path.write_text("modified")
    assert not v4_reuse.installed_matches(tmp_path, files)


def test_exact_exporter_compatibility_is_opt_in_and_does_not_survive_edits(tmp_path):
    path = tmp_path / "exporter.py"
    path.write_text("audited prior implementation")
    prior = v4_reuse.file_digest(path)
    path.write_text("independently checked compatible implementation")
    current = v4_reuse.file_digest(path)
    files = {path.name: prior}
    compatibility = ((path.name, current, prior),)
    assert not v4_reuse.installed_matches(tmp_path, files)
    assert v4_reuse.installed_matches(tmp_path, files, compatibility=compatibility)
    path.write_text("unreviewed future implementation")
    assert not v4_reuse.installed_matches(tmp_path, files, compatibility=compatibility)


def test_exporter_compatibility_is_bound_to_audit_and_excludes_changed_parameters():
    registry = v4_reuse.audit_registry()
    compatibility = v4_reuse._exporter_compatibility(registry, "POPC")
    assert compatibility
    # Checking that a compatibility map exists is insufficient: every exporter
    # byte in the original audit must actually match through that exact map.
    assert v4_reuse.installed_matches(
        Path(v4_reuse.__file__).parents[2], registry["exporter_files"], compatibility=compatibility
    )
    for name in ("PPCPL", "PPEPL", "LYSPG", "POP2", "PAPI", "SAPI", "SOP2"):
        assert not v4_reuse._exporter_compatibility(registry, name)
    assert not v4_reuse._exporter_compatibility({**registry, "review_date": "changed"}, "POPC")


@pytest.mark.parametrize(
    "relative",
    [
        "io/source_document.py",
        "io/cell.py",
        "io/residue_identity.py",
        "modules/input/validation.py",
        "modules/input/connections.py",
        "core/exceptions.py",
    ],
)
def test_exporter_compatibility_rejects_modified_support_source(monkeypatch, relative):
    registry = v4_reuse.audit_registry()
    original = v4_reuse.installed_matches
    checked = []

    def modified_helper(root, files, **kwargs):
        if relative in files:
            checked.append(files)
            return False
        return original(root, files, **kwargs)

    monkeypatch.setattr(v4_reuse, "installed_matches", modified_helper)
    assert not v4_reuse._exporter_compatibility(registry, "POPC")
    assert len(checked) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_K", 330),
        ("composition", {"POPC": 80, "CER16": 20}),
        ("host_smiles", "CO"),
        ("smiles", "CCC"),
    ],
)
def test_condition_changes_cannot_acquire_an_audit_match(field, value):
    protocol = resolve_protocol("CER16", "charmm36-lipid")
    expected = v4_reuse.digest({k: v for k, v in protocol.items() if k not in {"family", "sha256"}})
    assert v4_reuse.matching_conditions(protocol, expected)
    protocol[field] = value
    protocol["sha256"] = v4_reuse.digest({k: v for k, v in protocol.items() if k != "sha256"})
    assert not v4_reuse.matching_conditions(protocol, expected)


def test_known_local_analysis_change_preserves_physical_reuse_audit():
    from gmxbuilder.modules.membrane.v4_platform import construction_policy as old_policy

    protocol = resolve_protocol("POPC", "charmm36-lipid")
    previous = {k: v for k, v in protocol.items() if k not in {"family", "sha256"}}
    previous["construction_policy"] = old_policy()
    expected = v4_reuse.digest(previous)
    assert v4_reuse.matching_conditions(protocol, expected)
    protocol["temperature_K"] += 1
    protocol["sha256"] = v4_reuse.digest({k: v for k, v in protocol.items() if k != "sha256"})
    assert not v4_reuse.matching_conditions(protocol, expected)


@pytest.fixture
def reusable(tmp_path, pair, monkeypatch):  # noqa: F811
    records, source_protocol = pair
    target_protocol = resolve_protocol("POPC", "charmm36-lipid")
    target = {
        "family": "charmm36-lipid",
        "lipid": "POPC",
        "force_field": "charmm36",
        "lipid_ff": "charmm36",
        "v4_protocol": target_protocol,
    }
    source = {
        **target,
        "family": "charmm36m-lipid",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
        "v4_protocol": source_protocol,
    }
    gmx = tmp_path / "gmx"
    gmx.write_text("synthetic compiler")
    itp = b"synthetic graph; physics covered by the separate production audit"
    checksum = __import__("hashlib").sha256(itp).hexdigest()
    evidence = {
        "source_protocol": source_protocol,
        "target_protocol": target_protocol,
        "audit_sha256": "a" * 64,
        "itp_sha256": {"POPC.itp": checksum},
        "gmx_sha256": v4_reuse.file_digest(gmx),
    }
    monkeypatch.setattr(v4_reuse, "reuse_evidence", lambda entry: evidence)
    monkeypatch.setattr(publisher, "reuse_evidence", lambda entry: evidence)
    from gmxbuilder.runtime import hardware

    monkeypatch.setattr(hardware, "find_gromacs_executable", lambda: str(gmx))
    from gmxbuilder.modules.membrane import v4_tpr

    def verify(gmx, tpr, protocol, ff, stage):
        assert ".reuse-" in str(tpr)
        return {
            "tpr_sha256": v4_reuse.file_digest(tpr),
            "effective": {
                "integrator": "md",
                "cutoff-scheme": "Verlet",
                "pbc": "xyz",
                "coulombtype": "PME",
                "tcoupl": "V-rescale",
                "vdw-modifier": "Force-switch",
                "DispCorr": "No",
                "pcoupl": "C-rescale",
                "dt": 0.002,
                "nstxout-compressed": 5000,
                "rvdw": 1.2,
                "rcoulomb": 1.2,
                "rvdw-switch": 1.0,
                "ref-t": [protocol["temperature_K"]],
            },
        }

    monkeypatch.setattr(v4_tpr, "verify_tpr", verify)
    roots = []
    for r, metadata in enumerate(records, 1):
        root = tmp_path / f"replica-{r}"
        roots.append(root)
        directory = root / source["family"] / source["lipid"]
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        for p in metadata["conformer_provenance"]:
            np.savez(
                directory / p["file"],
                coords=np.zeros((len(metadata["atom_names"]), 3)),
                atom_names=metadata["atom_names"],
            )
        work = tmp_path / "work" / source["family"] / source["lipid"] / f"replica-{r}"
        work.mkdir(parents=True)
        (work / "npt.tpr").write_bytes(b"synthetic TPR")
        (work / "POPC.itp").write_bytes(itp)
        (work / "topol.top").write_text('#include "POPC.itp"\n')
        from gmxbuilder.modules.membrane.parameter_provenance import record_work_parameters

        record_work_parameters(work, metadata["parameter_fingerprint"])
        (work / "v4-protocol.json").write_text(json.dumps(source_protocol))
    assert (
        assemble_entry(tmp_path, roots, source["family"], source["lipid"], tmp_path / "library")[
            "conformers"
        ]
        == 40
    )
    return tmp_path, target, source


def test_v4_publisher_preserves_source_and_reader_checks_copied_bytes(reusable):
    root, target, source = reusable
    originals = {str(p): v4_reuse.file_digest(p) for p in root.rglob("*") if p.is_file()}
    assert publisher.publish(root, target, source, 2)
    assert all(v4_reuse.file_digest(p) == h for p, h in originals.items())
    reader = EquilibratedLipidLibrary(roots=[root / "library"])
    entry = reader.inspect("POPC", "charmm36", "charmm36")
    assert entry is not None
    assert entry.metadata["v4_protocol"]["family"] == "charmm36-lipid"
    assert (
        entry.metadata["reused_from"]["source_metadata"]["v4_protocol"]["family"]
        == "charmm36m-lipid"
    )
    assert entry.metadata["n_conformations"] == 40
    assert len(entry.metadata["replica_records"]) == 2
    assert entry.metadata["reused_from"]["independent_trajectory_added"] is False
    entry.conformer_files[0].write_bytes(b"corrupted")
    assert reader.inspect("POPC", "charmm36", "charmm36") is None


def test_queue_rejects_finite_but_modified_reused_coordinates(reusable):
    root, target, source = reusable
    assert publisher.publish(root, target, source, 2)
    assert queue.entry_done(root, target, 2)
    path = next((root / "replica-1" / target["family"] / target["lipid"]).glob("conf_*.npz"))
    with np.load(path) as stored:
        contents = dict(stored)
    contents["coords"] = contents["coords"] + 0.1
    np.savez(path, **contents)
    assert not queue.entry_done(root, target, 2)


def test_v4_copy_hash_contract_does_not_reinterpret_legacy_reuse(tmp_path):
    assert v4_reuse.reused_coordinates_valid(
        tmp_path, {"reused_from": {"family": "charmm36m-lipid"}}
    )


def test_partial_publication_failure_rolls_back_only_its_own_targets(reusable, monkeypatch):
    root, target, source = reusable
    destination = root / "replica-2" / target["family"] / target["lipid"]
    rename = Path.rename

    def fail(self, other):
        if Path(other) == destination:
            raise OSError("injected publication failure")
        return rename(self, other)

    monkeypatch.setattr(Path, "rename", fail)
    assert not publisher.publish(root, target, source, 2)
    assert not (root / "replica-1" / target["family"] / target["lipid"]).exists()
    assert not destination.exists()
    assert queue.entry_done(root, source, 2)


def test_existing_target_work_is_never_replaced(reusable):
    root, target, source = reusable
    work = root / "work" / target["family"] / target["lipid"]
    work.mkdir(parents=True)
    (work / "sentinel").write_text("retained work")
    assert not publisher.publish(root, target, source, 2)
    assert (work / "sentinel").read_text() == "retained work"


@pytest.mark.parametrize(
    "defect", ["failed-source", "protocol", "hash", "coordinates-origin", "independent", "tpr"]
)
def test_malformed_reuse_provenance_fails_closed(reusable, defect):
    root, target, source = reusable
    assert publisher.publish(root, target, source, 2)
    path = root / "replica-1" / target["family"] / target["lipid"] / "metadata.json"
    metadata = json.loads(path.read_text())
    proof = metadata["reused_from"]
    if defect == "failed-source":
        proof["source_metadata"]["quality"]["passed"] = False
    elif defect == "protocol":
        metadata["v4_protocol"]["temperature_K"] += 1
    elif defect == "hash":
        proof["source_metadata_sha256"] = "0" * 64
    elif defect == "coordinates-origin":
        metadata["conformer_provenance"][0]["molecule_index"] += 1
    elif defect == "independent":
        proof["independent_trajectory_added"] = True
    else:
        proof["simulation"]["tpr_sha256"] = "invalid"
    with pytest.raises(ValueError):
        v4_reuse.validate_reused_metadata(metadata)
