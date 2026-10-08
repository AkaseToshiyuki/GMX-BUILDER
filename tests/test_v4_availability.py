"""V4 manifest transitions and builder admission; no dynamics."""

import json

import numpy as np
import pytest

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.membrane import v4_availability as availability
from gmxbuilder.modules.membrane.builder import MembraneBuilder
from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from tests.test_v4_construction import pair  # noqa: F401
from tests.test_v4_reuse import reusable  # noqa: F401


def row(payload, source="charmm36m"):
    return next(
        e for e in payload["entries"] if e["lipid_name"] == "POPC" and e["lipid_ff"] == source
    )


def test_publication_and_revocation_refresh_the_same_list_without_restart(reusable):  # noqa: F811
    root, _, _ = reusable
    library = EquilibratedLipidLibrary([root / "library"])
    metadata = root / "library/charmm36m-lipid/POPC/metadata.json"
    original = metadata.read_text()
    metadata.unlink()
    first = availability.refresh_availability_list(library=library)
    assert not row(first)["ready"]
    metadata.write_text(original)
    second = availability.refresh_availability_list(library=library)
    assert row(second)["ready"]
    assert not row(second, "charmm36")["ready"]
    assert json.loads((root / "library/list.json").read_text()) == second
    # A forged list cannot unlock a missing or invalid entry.
    metadata.write_text("{invalid")
    (root / "library/list.json").write_text(json.dumps(second))
    third = availability.refresh_availability_list(library=library)
    assert not row(third)["ready"]
    with pytest.raises(ModuleConfigError, match="V4 acceptance"):
        availability.require_v4_lipids(["POPC"], "charmm36m", "charmm36m", library=library)


def test_single_replica_and_old_schema_do_not_unlock_a_pair(reusable):  # noqa: F811
    root, _, _ = reusable
    library = EquilibratedLipidLibrary([root / "library"])
    path = root / "library/charmm36m-lipid/POPC/metadata.json"
    original = json.loads(path.read_text())
    for mutation in (
        {"schema_version": 3},
        {"v4_protocol": None},
        {"replica_records": []},
        {"parameter_fingerprint": "stale"},
    ):
        path.write_text(json.dumps({**original, **mutation}))
        assert availability.accepted_entry(library, "POPC", "charmm36m", "charmm36m") is None


def test_default_and_legacy_default_override_both_use_v4(tmp_path, monkeypatch):
    from gmxbuilder.runtime import prebuilt_assets

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("GMXBUILDER_PREBUILT_AUTO_INSTALL", "0")
    old = tmp_path / ".cache/gmxbuilder/lipid_equilibrated"
    expected = tmp_path / ".cache/gmxbuilder/lipid_equilibrated_v4/library"
    for configured in (None, str(old)):
        if configured:
            monkeypatch.setenv("GMXBUILDER_LIPID_LIBRARY", configured)
        else:
            monkeypatch.delenv("GMXBUILDER_LIPID_LIBRARY", raising=False)
        assert EquilibratedLipidLibrary().roots == [expected]
        assert prebuilt_assets._configured_roots()[0] == expected


@pytest.mark.parametrize(
    "composition",
    [
        {"lipid_type": "POPC"},
        {
            "lipid_composition": {
                "upper": [{"name": "POPC", "ratio": 100}],
                "lower": [{"name": "POPC", "ratio": 50}, {"name": "POPE", "ratio": 50}],
            }
        },
    ],
)
def test_builder_rejects_unaccepted_pure_and_mixed_membranes_before_geometry(
    tmp_path, monkeypatch, composition
):
    from gmxbuilder.modules.membrane import equilibrated_library

    monkeypatch.setattr(equilibrated_library, "_library", EquilibratedLipidLibrary([tmp_path]))
    system = System(
        structure=Structure(coordinates=np.empty((0, 3)), box_vectors=np.eye(3) * 10),
        metadata={
            "force_field": "charmm36m",
            "lipid_ff": "charmm36m",
        },
    )
    with pytest.raises(ModuleConfigError, match="V4 acceptance"):
        MembraneBuilder().run(system, composition)


def test_builtin_guard_does_not_replace_task_custom_lipid_validation(tmp_path):
    availability.require_v4_lipids(
        ["CUSTOM"], "amber14sb", "gaff2", library=EquilibratedLipidLibrary([tmp_path])
    )


def test_list_endpoint_reads_verified_snapshot_without_writing_library(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from gmxbuilder.web.server import app
    from gmxbuilder.web.server_parts import option_catalog

    library = EquilibratedLipidLibrary([tmp_path])
    service = option_catalog.OptionCatalog(library, dependencies=lambda: 0)
    monkeypatch.setattr(option_catalog, "build_ui_options", lambda **kwargs: {"lipids": []})
    service.refresh()
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(option_catalog, "get_catalog", lambda: service)
    with TestClient(app) as client:
        response = client.get("/api/lipid-library-list")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == "ready"
    assert not (tmp_path / "list.json").exists()
    assert not any(entry["ready"] for entry in response.json()["entries"])


def test_non_object_list_is_repaired_instead_of_disabling_other_builders(tmp_path):
    (tmp_path / "list.json").write_text("null")
    result = availability.refresh_availability_list(library=EquilibratedLipidLibrary([tmp_path]))
    assert result["library_version"] == 4
    assert json.loads((tmp_path / "list.json").read_text()) == result


def test_read_only_library_still_reports_current_publication_and_revocation(
    reusable,  # noqa: F811
    monkeypatch,
):
    root, _, _ = reusable
    library = EquilibratedLipidLibrary([root / "library"])
    metadata = root / "library/charmm36m-lipid/POPC/metadata.json"
    original = metadata.read_text()
    metadata.unlink()
    old = availability.refresh_availability_list(library=library)

    def denied(**kwargs):
        raise PermissionError("Library writes are outside the Web storage boundary")

    monkeypatch.setattr(availability.tempfile, "mkstemp", denied)
    metadata.write_text(original)
    current = availability.refresh_availability_list(library=library)
    assert row(current)["ready"]
    assert json.loads((root / "library/list.json").read_text()) == old
    metadata.write_text("{invalid")
    assert not row(availability.refresh_availability_list(library=library))["ready"]


@pytest.mark.parametrize("command", ["build", "queue"])
def test_retired_v3_commands_fail_before_starting_a_job(command):
    from click.testing import CliRunner

    from gmxbuilder.app import main

    result = CliRunner().invoke(main, ["lipid-library", command])
    assert result.exit_code != 0
    assert "retired" in result.output
    assert "scripts/build_v4_library.py" in result.output


def test_accepted_v4_entry_clears_historical_quarantine_only_after_acceptance(
    reusable,  # noqa: F811
    monkeypatch,
):
    from gmxbuilder.modules.forcefield import lipid_policy
    from gmxbuilder.modules.membrane import equilibrated_library

    root, _, _ = reusable
    library = EquilibratedLipidLibrary([root / "library"])
    monkeypatch.setattr(equilibrated_library, "_library", library)
    monkeypatch.setitem(lipid_policy._CHARMM_LIBRARY_UNAVAILABLE, ("charmm36m", "POPC"), "old hold")
    assert lipid_policy.charmm_lipid_capability("POPC", "charmm36m")[0]
    (root / "library/charmm36m-lipid/POPC/metadata.json").unlink()
    assert not lipid_policy.charmm_lipid_capability("POPC", "charmm36m")[0]


def test_saved_checkpoint_cannot_export_a_now_unaccepted_lipid(tmp_path, monkeypatch):
    from gmxbuilder.core.component import Component
    from gmxbuilder.core.enums import ComponentKind
    from gmxbuilder.modules.membrane import equilibrated_library
    from gmxbuilder.pipeline.step_executor import StepRunner

    monkeypatch.setattr(
        equilibrated_library, "_library", EquilibratedLipidLibrary([tmp_path / "empty"])
    )
    runner = StepRunner(tmp_path / "task", pipeline_type="pure-membrane")
    source = System(
        structure=Structure(
            coordinates=np.zeros((1, 3)),
            box_vectors=np.eye(3),
            atom_names=["P"],
            resnames=["POPC"],
            resids=[1],
            elements=["P"],
        ),
        metadata={"force_field": "charmm36m", "lipid_ff": "charmm36m"},
    )
    source.add_component(
        Component(name="MEMBRANE", kind=ComponentKind.MEMBRANE, atom_indices=np.array([0]))
    )
    source.save_checkpoint(runner.step_dir("membrane"))
    result = runner.finalize_from_checkpoint("membrane")
    assert result["status"] == "error" and result["step"] == "membrane"
    assert "V4 acceptance" in result["error"]


def test_no_builtin_lipids_do_not_require_installing_global_assets(monkeypatch):
    def unavailable():
        raise AssertionError("unrelated global assets must not be installed")

    monkeypatch.setattr(availability, "get_equilibrated_lipid_library", unavailable)
    availability.require_v4_lipids([], "amber14sb", "lipid21")
    availability.require_v4_lipids(["CUSTOM"], "amber14sb", "gaff2")
