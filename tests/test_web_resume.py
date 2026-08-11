"""Regression tests for resuming persisted web tasks."""

import asyncio
from datetime import datetime, timezone

from gmxbuilder.web import server
from gmxbuilder.web.task_manager import TaskManager


def test_resume_reparses_pdb_for_structure_step(tmp_path, monkeypatch, small_pdb_file):
    """Resume must restore the residue sequence consumed by Step 3."""
    monkeypatch.setattr(server.task_manager, "root", tmp_path)
    task = server.task_manager.create_task(filename="input.pdb")
    server.task_manager.save_uploaded_pdb(
        task["task_id"], "input.pdb", small_pdb_file.read_bytes()
    )
    server.task_manager.update_state(
        task["task_id"],
        {"pdb_info": {"filename": "input.pdb"}},
    )

    resumed = asyncio.run(server.api_task_resume(task["task_id"]))

    assert resumed["pdb_content"]
    assert resumed["pdb_info_full"]["num_atoms"] == 5
    assert resumed["sequences"]
    assert resumed["sequences"][0]["residues"]


def test_task_ttl_is_configurable_for_create_and_access(tmp_path, monkeypatch):
    monkeypatch.setenv("GMXBUILDER_TASK_TTL_HOURS", "168")
    manager = TaskManager(root=tmp_path)
    created = manager.create_task(filename="persistent.pdb")
    created_at = datetime.fromisoformat(created["created_at"])
    expires_at = datetime.fromisoformat(created["expires_at"])
    assert (expires_at - created_at).total_seconds() == 168 * 3600

    monkeypatch.setattr(server.task_manager, "root", tmp_path)
    accessed = asyncio.run(server.api_task_status(created["task_id"]))
    refreshed = datetime.fromisoformat(accessed["expires_at"])
    remaining_hours = (refreshed - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 167.9 < remaining_hours <= 168
