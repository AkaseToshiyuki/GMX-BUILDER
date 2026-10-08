"""The manifest: what it records, and what it must never record.

Its point is that a reviewer can rebuild the system from the package alone.
That makes two properties load-bearing and easy to break silently: it has to
be a configuration the builder actually accepts, and it must not carry paths
from the machine that produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.pipeline.config import PipelineConfig
from gmxbuilder.pipeline.provenance import (
    adopt_task_context,
    build_manifest,
    file_inventory,
    readable_summary,
    record_step,
)


def _system() -> System:
    return System(
        structure=Structure(coordinates=np.zeros((1, 3)), box_vectors=np.eye(3) * 5.0),
        metadata={"seed": 11},
    )


def _metrics(atoms: int = 1) -> dict:
    return {"num_atoms": atoms, "box_dimensions_nm": [5.0, 5.0, 5.0], "components": []}


def _manifest(system, **kwargs) -> dict:
    return build_manifest(
        system, system_name=kwargs.get("name", "probe"), seed=11, files=kwargs.get("files", [])
    )


# --------------------------------------------------------------------------
# Replay


def test_the_manifest_is_a_configuration_the_builder_accepts(tmp_path):
    """Not a report *about* the build -- the build, in the schema that runs it."""
    system = _system()
    record_step(
        system, "forcefield", {"name": "charmm36m", "water_model": "tip3p"}, _metrics(), 2.0
    )
    record_step(
        system, "membrane", {"lipid_type": "POPC", "n_lipids_per_leaflet": 128}, _metrics(), 9.0
    )

    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(_manifest(system)))

    replayed = PipelineConfig.from_yaml(path)
    assert replayed.seed == 11
    assert replayed.system_name == "probe"
    assert replayed.module_config("forcefield") == {"name": "charmm36m", "water_model": "tip3p"}
    assert replayed.module_config("membrane") == {
        "lipid_type": "POPC",
        "n_lipids_per_leaflet": 128,
    }


def test_the_provenance_block_does_not_disturb_the_replay():
    """Both halves live in one file, so one cannot drift from the other."""
    system = _system()
    record_step(system, "input", {}, _metrics(), 1.0)
    manifest = _manifest(system)

    assert "provenance" in manifest
    config = PipelineConfig(**manifest)
    assert config.modules == {"input": {}}


def test_a_rerun_step_is_recorded_as_the_settings_that_produced_the_checkpoint():
    """The user can re-run a step; the record must show the last run, not the first."""
    system = _system()
    record_step(system, "membrane", {"n_lipids_per_leaflet": 64}, _metrics(), 3.0)
    record_step(system, "solvation", {"box_padding": 2.0}, _metrics(), 1.0)
    record_step(system, "membrane", {"n_lipids_per_leaflet": 200}, _metrics(), 8.0)

    manifest = _manifest(system)
    assert manifest["modules"]["membrane"] == {"n_lipids_per_leaflet": 200}
    # And the pipeline still reads in order, not with membrane pushed to the end.
    assert [entry["step"] for entry in manifest["provenance"]["steps"]] == [
        "membrane",
        "solvation",
    ]


# --------------------------------------------------------------------------
# What must never leave the machine


@pytest.mark.parametrize("key", ["_task_dir", "_step_dir", "output_dir", "pdb", "task_id"])
def test_no_host_path_reaches_the_manifest(key, tmp_path):
    """The package is downloaded. A server path in it is a leak, not a detail."""
    system = _system()
    record_step(system, "input", {key: "/home/someone/tasks/abc123", "pH": 7.4}, _metrics(), 1.0)

    manifest = _manifest(system)
    rendered = __import__("json").dumps(manifest)
    assert "/home/someone" not in rendered
    assert "abc123" not in rendered
    # The scientific request survives; only the location is dropped.
    assert manifest["modules"]["input"]["pH"] == 7.4


def test_the_input_structure_is_identified_by_digest_not_shipped(tmp_path):
    """The uploaded file is the user's own and is not redistributed."""
    upload = tmp_path / "6xyz.pdb"
    upload.write_text("HEADER    A REAL UPLOAD\nEND\n")

    system = _system()
    record_step(system, "input", {"pdb": str(upload)}, _metrics(), 1.0)
    manifest = _manifest(system)

    identity = manifest["provenance"]["input_structure"]
    assert identity["filename"] == "6xyz.pdb"
    assert identity["bytes"] == upload.stat().st_size
    assert len(identity["sha256"]) == 64
    assert str(tmp_path) not in __import__("json").dumps(manifest)
    # A replay has to name a file, and this is the only name it can name.
    assert manifest["modules"]["input"]["pdb"] == "./6xyz.pdb"


def test_a_missing_upload_leaves_the_record_smaller_rather_than_wrong():
    system = _system()
    record_step(system, "input", {"pdb": "/gone/never-existed.pdb"}, _metrics(), 1.0)
    assert "input_structure" not in _manifest(system)["provenance"]


# --------------------------------------------------------------------------
# Task context and the inventory


def test_the_upload_time_comes_from_the_task_the_build_ran_in(tmp_path):
    (tmp_path / "state.json").write_text(
        '{"created_at": "2026-09-04T10:00:00+00:00", "original_filename": "7abc.pdb",'
        ' "task_type": {"id": "membrane-bilayer"}}'
    )
    system = _system()
    adopt_task_context(system, tmp_path)
    record_step(system, "input", {}, _metrics(), 1.0)

    task = _manifest(system)["provenance"]["task"]
    assert task["created_at"] == "2026-09-04T10:00:00+00:00"
    assert task["original_filename"] == "7abc.pdb"
    assert task["task_type_id"] == "membrane-bilayer"


def test_a_command_line_build_has_no_task_and_says_nothing_about_one(tmp_path):
    """`adopt_task_context` must not raise where there is no task to adopt."""
    system = _system()
    adopt_task_context(system, tmp_path / "no-such-task")
    adopt_task_context(system, None)
    record_step(system, "input", {}, _metrics(), 1.0)
    assert "task" not in _manifest(system)["provenance"]


def test_the_build_duration_is_the_sum_of_the_steps():
    system = _system()
    record_step(system, "input", {}, _metrics(), 1.25)
    record_step(system, "membrane", {}, _metrics(), 10.5)
    assert _manifest(system)["provenance"]["build_duration_s"] == pytest.approx(11.75)


def test_the_inventory_reports_every_file_with_its_size(tmp_path):
    (tmp_path / "structure").mkdir()
    (tmp_path / "structure" / "input.gro").write_text("x" * 40)
    (tmp_path / "README.txt").write_text("y" * 10)
    (tmp_path / "package.zip").write_bytes(b"z" * 100)

    entries = file_inventory(tmp_path)
    by_path = {entry["path"]: entry["bytes"] for entry in entries}

    assert by_path == {"README.txt": 10, "structure/input.gro": 40}, (
        "the archive cannot list its own size, and nothing else may be omitted"
    )


def test_the_prose_summary_reports_what_was_asked_and_what_came_out():
    system = _system()
    record_step(
        system,
        "membrane",
        {"lipid_type": "POPC", "n_lipids_per_leaflet": 128},
        {"num_atoms": 176048, "box_dimensions_nm": [8.4, 8.4, 11.0], "components": []},
        10.6,
    )
    summary = readable_summary(system)

    assert "n_lipids_per_leaflet = 128" in summary
    assert "num_atoms = 176048" in summary
    assert "box_dimensions_nm = 8.4, 8.4, 11" in summary
    assert "10.6" in summary


def test_a_system_built_outside_the_step_pipeline_says_so():
    assert "not built through the step pipeline" in readable_summary(_system())
