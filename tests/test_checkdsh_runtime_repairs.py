"""Failure-path checks for installer and scientific reporting repairs."""

import importlib.util
import io
import json
import stat
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_token_migration_preserves_literal_secret_and_rotation_is_explicit(tmp_path):
    installer = helper("installer_state")
    original = "fixture-token-$(never-execute)"
    (tmp_path / "run-local.sh").write_text(f"export GMXBUILDER_ADMIN_TOKEN='{original}'\n")
    assert installer.admin_token(tmp_path) == original
    assert installer.admin_token(tmp_path) == original
    assert stat.S_IMODE((tmp_path / "admin-token").stat().st_mode) == 0o600
    assert installer.admin_token(tmp_path, rotate=True) != original


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "fifo"])
def test_gromacs_extraction_rejects_unsafe_members(tmp_path, kind):
    installer = helper("install_gromacs")
    archive = tmp_path / "source.tar.gz"
    member = tarfile.TarInfo("../outside" if kind == "traversal" else "member")
    member.type = {
        "traversal": tarfile.REGTYPE,
        "symlink": tarfile.SYMTYPE,
        "hardlink": tarfile.LNKTYPE,
        "fifo": tarfile.FIFOTYPE,
    }[kind]
    member.linkname = "../outside"
    with tarfile.open(archive, "w:gz") as stream:
        stream.addfile(member, io.BytesIO(b""))
    with pytest.raises(RuntimeError):
        installer._safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_failed_verified_download_preserves_cached_file(tmp_path, monkeypatch):
    installer = helper("install_gromacs")
    target = tmp_path / "source"
    target.write_bytes(b"previous")
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="checksum"):
        installer._fetch_verified("https://example.invalid/source", target, "0" * 64)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.download-*")) == []


def test_single_slot_can_use_its_complete_cpu_budget():
    from gmxbuilder.runtime.hardware import task_thread_allocation

    assert task_thread_allocation(1, cpu_cores=16, max_parallel=1) == 16
    assert task_thread_allocation(3, cpu_cores=17, max_parallel=3) * 3 <= 17


def test_gaff_lock_covers_unique_exact_artifacts():
    lock = json.loads((ROOT / "scripts/gaff-linux-64.lock.json").read_text())
    packages = lock["packages"]
    assert len(packages) == len({p["name"] for p in packages})
    assert len(packages) > 100
    for package in packages:
        assert package["url"].startswith("https://conda.anaconda.org/conda-forge/")
        assert len(bytes.fromhex(package["sha256"])) == 32
    assert {p["name"]: p["version"] for p in packages}["ambertools"] == "24.8"


def test_citations_separate_99sb_ildn_and_include_actual_ol24_propka():
    from gmxbuilder.runtime.citations import atomistic_citations

    def ids(metadata):
        return {r["id"] for r in atomistic_citations(metadata)["references"]}

    assert "amber99sb-ildn" not in ids({"force_field": "amber99sb"})
    assert {"amber99sb", "amber99sb-ildn"} <= ids({"force_field": "amber99sb-ildn"})
    assert {"ff14sb", "ol24", "propka"} <= ids(
        {"force_field": "amber14sb_ol24", "protonation_sources": ["PROPKA"]}
    )
    assert "propka" not in ids({"force_field": "amber14sb_ol24"})


def test_single_charge_witness_is_not_independent_corroboration():
    from gmxbuilder.modules.forcefield.charmm_charges import ChargeAssignment

    assert not ChargeAssignment(0, 0.0, 0.0, 4, 1).well_determined


def test_host_area_comparison_requires_recorded_denominator(tmp_path):
    compare = helper("compare_lipid_library")
    with pytest.raises(ValueError, match="denominator"):
        compare._leaflet_count(tmp_path)
    (tmp_path / "v4-protocol.json").write_text(json.dumps({"lipids_per_leaflet": 200}))
    assert compare._leaflet_count(tmp_path) == 200


def test_replica_publication_failure_restores_all_originals(tmp_path, monkeypatch):
    import os

    from gmxbuilder.modules.membrane.v4_reanalysis import publish_replicas

    pairs = []
    for index in range(2):
        source, target = tmp_path / f"new{index}", tmp_path / f"old{index}"
        source.mkdir()
        target.mkdir()
        (source / "metadata.json").write_text("new")
        (target / "metadata.json").write_text("original")
        pairs.append((source, target))
    original_replace = os.replace

    def fail_second(source, target):
        if Path(source) == pairs[1][0]:
            raise OSError("injected second-replica failure")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second-replica"):
        publish_replicas(pairs, tmp_path / "publication.json")
    assert [(target / "metadata.json").read_text() for _, target in pairs] == ["original"] * 2


def test_initial_leaflet_identity_survives_migration_and_rejects_reordering():
    import numpy as np

    from gmxbuilder.core.structure import Structure
    from gmxbuilder.modules.membrane.leaflet_identity import initial_leaflets

    initial = Structure(
        coordinates=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]),
        box_vectors=np.eye(3) * 4,
        atom_names=["C", "C"],
        resnames=["LIP", "LIP"],
        resids=[1, 2],
    )
    current = initial.copy()
    current.coordinates[:, 2] = [3.0, 1.0]
    assert initial_leaflets(initial, current, [[0], [1]]) == [([0], False), ([1], True)]
    current.resids = [2, 1]
    with pytest.raises(ValueError, match="order differs"):
        initial_leaflets(initial, current, [[0], [1]])


def test_plasmalogen_charges_match_west2020_original_stream():
    from gmxbuilder.modules.forcefield.lipid_policy import _make_charmm_plasmalogen
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
    from tests.prerequisites import forcefield_parameters_available

    if not forcefield_parameters_available("charmm36m"):
        pytest.skip("CHARMM36m parameters are installed separately")
    # Independent original PLA18 stream, SHA256 67d3f809...7550cb0f.
    expected = {
        "C3": 0.08,
        "HX": 0.08,
        "HY": 0.08,
        "O31": -0.36,
        "C31": 0.0,
        "H1X": 0.08,
        "C32": -0.20,
        "H2X": 0.08,
        "C23": 0.0,
        "H3R": 0.08,
        "H3S": 0.08,
        "C33": -0.18,
        "H3X": 0.09,
        "H3Y": 0.09,
    }
    for head in ("PE", "PC"):
        template = _make_charmm_plasmalogen(load_force_field_rtp("charmm36m"), head)
        charges = {row[0]: row[2] for row in template["atoms"]}
        assert {name: charges[name] for name in expected} == expected
        assert sum(charges.values()) == pytest.approx(0.0, abs=1e-6)


def test_installer_lock_blocks_new_submissions_and_keeps_health_readable(tmp_path, monkeypatch):
    import fcntl

    from starlette.responses import JSONResponse
    from starlette.testclient import TestClient

    from gmxbuilder.web.installation_guard import InstallationGuardMiddleware

    path = tmp_path / "install.lock"
    monkeypatch.setenv("GMXBUILDER_INSTALL_LOCK", str(path))

    async def app(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    client = TestClient(InstallationGuardMiddleware(app))
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert client.post("/api/tasks").status_code == 503
        assert client.get("/health").status_code == 200
    assert client.post("/api/tasks").status_code == 200


def test_installer_lock_remains_usable_after_managed_write_confinement(tmp_path):
    import subprocess
    import sys

    allowed = tmp_path / "managed"
    allowed.mkdir()
    lock = tmp_path / "state" / "install.lock"
    code = """
import fcntl
import os
import sys
from pathlib import Path
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from gmxbuilder.web.installation_guard import InstallationGuardMiddleware, prepare_installation_lock
from gmxbuilder.web.write_sandbox import restrict_writes

os.environ['GMXBUILDER_INSTALL_LOCK'] = sys.argv[2]
prepare_installation_lock()
restrict_writes(Path(sys.argv[1]))
async def app(scope, receive, send):
    await JSONResponse({'ok': True})(scope, receive, send)
client = TestClient(InstallationGuardMiddleware(app))
assert client.post('/api/upload-pdb').status_code == 200
with Path(sys.argv[2]).open('rb') as installer:
    fcntl.flock(installer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    response = client.post('/api/upload-pdb')
    assert response.status_code == 503
    assert 'Installation is in progress' in response.json()['error']
os.environ['GMXBUILDER_INSTALL_LOCK'] = str(Path(sys.argv[2]).with_name('missing.lock'))
response = client.post('/api/upload-pdb')
assert response.status_code == 503
assert response.json()['error'] == 'Submissions are temporarily unavailable; please retry'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(allowed), str(lock)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_nucleic_comparison_rejects_missing_terms_even_when_intersection_matches():
    validator = helper("verify_nucleic_forcefield")
    terms = {term: 0.0 for term in (*validator.EXACT_TERMS, *validator.TOLERATED, "Improper Dih.")}
    assert validator.missing_energy_terms(terms, terms) == []
    assert validator.missing_energy_terms({}, {})
    missing = {k: v for k, v in terms.items() if k != "Bond"}
    assert "merged: missing Bond" in validator.missing_energy_terms(terms, missing)


def test_gromacs_version_query_never_starts_installation():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/install_gromacs.py"), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == "2026.3"
