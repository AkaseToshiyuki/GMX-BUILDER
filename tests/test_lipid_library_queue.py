from pathlib import Path

from gmxbuilder.modules.membrane import library_queue


def test_queue_uses_separate_release_jobs_and_two_resource_lanes(tmp_path, monkeypatch):
    jobs = {
        "amber14sb": [{"force_field": "amber14sb", "lipid_name": "POPC", "ready": False}],
        "charmm36m": [{"force_field": "charmm36m", "lipid_name": "POPC", "ready": False}],
        "charmm36": [{"force_field": "charmm36", "lipid_name": "POPC", "ready": False}],
    }

    class Library:
        def coverage(self, fields):
            return jobs[fields[0]]

    calls = []
    monkeypatch.setattr(library_queue, "EquilibratedLipidLibrary", Library)
    monkeypatch.setattr(
        library_queue, "configure_runtime_resources", lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        library_queue, "configured_cpu_ids", lambda: tuple(range(8)),
    )
    monkeypatch.setattr(
        library_queue, "configured_gpu_devices", lambda: ("0", "1"),
    )
    monkeypatch.setattr(library_queue, "configured_task_threads", lambda: 4)
    monkeypatch.setattr(
        library_queue,
        "_run_job",
        lambda job, **kwargs: (
            calls.append((job, kwargs)) or (job, True, str(tmp_path / "job.log"))
        ),
    )

    results = library_queue.run_library_queue(log_dir=tmp_path)

    assert [item[0]["force_field"] for item in results] == [
        "amber14sb", "charmm36m", "charmm36",
    ]
    assert {call[1]["cpus"] for call in calls} == {
        (0, 1, 2, 3), (4, 5, 6, 7),
    }
    assert {call[1]["gpu"] for call in calls} == {"0", "1"}


def test_queue_refuses_nonproduction_npt(tmp_path):
    try:
        library_queue.run_library_queue(npt_ps=100.0, log_dir=Path(tmp_path))
    except ValueError as exc:
        assert "at least 500 ps" in str(exc)
    else:
        raise AssertionError("short non-production queue was accepted")
