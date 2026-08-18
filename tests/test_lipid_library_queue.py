import json
from pathlib import Path

from click.testing import CliRunner

from gmxbuilder.app import main
from gmxbuilder.modules.membrane import library_queue


def test_queue_uses_every_configured_gpu_with_disjoint_cpu_lanes(tmp_path, monkeypatch):
    jobs = {
        "amber14sb": [
            {
                "force_field": "amber14sb",
                "lipid_name": "POPC",
                "ready": False,
                "lipid_ff": "gaff2",
                "parameter_family": "amber-gaff2",
            }
        ],
        "charmm36m": [
            {
                "force_field": "charmm36m",
                "lipid_name": "POPC",
                "ready": False,
                "lipid_ff": "charmm36m",
                "parameter_family": "charmm36m-lipid",
            }
        ],
        "charmm36": [
            {
                "force_field": "charmm36",
                "lipid_name": "POPC",
                "ready": False,
                "lipid_ff": "charmm36",
                "parameter_family": "charmm36-lipid",
            }
        ],
    }

    class Library:
        def coverage(self, fields):
            return jobs[fields[0]]

        def inspect_failure(self, *_args, **_kwargs):
            return None

    calls = []
    monkeypatch.setattr(library_queue, "EquilibratedLipidLibrary", Library)
    monkeypatch.setattr(
        library_queue,
        "configure_runtime_resources",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        library_queue,
        "configured_cpu_ids",
        lambda: tuple(range(12)),
    )
    monkeypatch.setattr(
        library_queue,
        "configured_gpu_devices",
        lambda: ("0", "1", "2"),
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
        "amber14sb",
        "charmm36m",
        "charmm36",
    ]
    assert {call[1]["cpus"] for call in calls} == {
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    }
    assert {call[1]["gpu"] for call in calls} == {"0", "1", "2"}


def test_queue_refuses_nonproduction_npt(tmp_path):
    try:
        library_queue.run_library_queue(npt_ps=100.0, log_dir=Path(tmp_path))
    except ValueError as exc:
        assert "at least 500 ps" in str(exc)
    else:
        raise AssertionError("short non-production queue was accepted")


def test_queue_records_worker_failure_and_continues(tmp_path, monkeypatch):
    jobs = [
        {
            "force_field": "charmm36m",
            "lipid_name": name,
            "lipid_ff": "charmm36m",
            "parameter_family": "charmm36m-lipid",
            "ready": False,
        }
        for name in ("POPC", "POPE")
    ]

    class Library:
        def coverage(self, _fields):
            return jobs

        def inspect_failure(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(library_queue, "EquilibratedLipidLibrary", Library)
    monkeypatch.setattr(
        library_queue,
        "configure_runtime_resources",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(library_queue, "configured_cpu_ids", lambda: tuple(range(4)))
    monkeypatch.setattr(library_queue, "configured_gpu_devices", lambda: ())
    monkeypatch.setattr(library_queue, "configured_task_threads", lambda: 4)

    def worker(job, **_kwargs):
        if job["lipid_name"] == "POPC":
            raise RuntimeError("synthetic failure")
        return job, True, str(tmp_path / "pope.log")

    monkeypatch.setattr(library_queue, "_run_job", worker)

    results = library_queue.run_library_queue(force_fields=("charmm36m",), log_dir=tmp_path)

    assert [(result[0]["lipid_name"], result[1]) for result in results] == [
        ("POPC", False),
        ("POPE", True),
    ]
    assert "synthetic failure" in Path(results[0][2]).read_text()


def test_queue_batches_never_compete_for_same_force_field_lipid_lock():
    jobs = [
        {
            "force_field": "amber14sb",
            "lipid_name": "POPS",
            "lipid_ff": "lipid21",
            "parameter_family": "amber-lipid21",
        },
        {
            "force_field": "amber14sb",
            "lipid_name": "POPS",
            "lipid_ff": "gaff2",
            "parameter_family": "amber-gaff2",
        },
        {
            "force_field": "amber14sb",
            "lipid_name": "PSM",
            "lipid_ff": "lipid21",
            "parameter_family": "amber-lipid21",
        },
        {
            "force_field": "charmm36m",
            "lipid_name": "POPS",
            "lipid_ff": "charmm36m",
            "parameter_family": "charmm36m-lipid",
        },
    ]

    batches = library_queue._unique_lock_batches(jobs, 3)

    assert sorted(len(batch) for batch in batches) == [1, 3]
    for batch in batches:
        keys = [(job["force_field"], job["lipid_name"]) for job in batch]
        assert len(keys) == len(set(keys))


def test_worker_invokes_one_exact_backend_and_uses_distinct_log_name(
    tmp_path,
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(library_queue.shutil, "which", lambda _name: "/bin/gmxbuilder")

    def fake_watchdog(command, **_kwargs):
        captured["command"] = command
        return 0

    monkeypatch.setattr(library_queue, "_run_with_watchdog", fake_watchdog)
    job = {
        "force_field": "amber14sb",
        "lipid_name": "POPS",
        "lipid_ff": "gaff2",
        "parameter_family": "amber-gaff2",
    }

    _job, success, log_path = library_queue._run_job(
        job,
        gpu="2",
        cpus=(0, 1, 2, 3),
        npt_ps=1000.0,
        log_dir=tmp_path,
    )

    assert success
    assert captured["command"] == [
        "/bin/gmxbuilder",
        "lipid-library",
        "build",
        "--force-field",
        "amber14sb",
        "--lipid",
        "POPS",
        "--lipid-ff",
        "gaff2",
        "--npt-ps",
        "1000.0",
    ]
    assert Path(log_path).name == "amber14sb-POPS-amber-gaff2-gpu2.log"


def test_status_json_distinguishes_terminal_unavailable_from_pending(monkeypatch):
    jobs = [
        {
            "force_field": "amber14sb",
            "lipid_name": "POPC",
            "ready": True,
            "unavailable": False,
            "parameter_family": "amber-lipid21",
        },
        {
            "force_field": "amber14sb",
            "lipid_name": "POPS",
            "ready": False,
            "unavailable": True,
            "parameter_family": "amber-gaff2",
        },
    ]

    class Library:
        def coverage(self, _force_fields):
            return jobs

    monkeypatch.setattr(
        "gmxbuilder.modules.membrane.equilibrated_library.EquilibratedLipidLibrary",
        Library,
    )
    result = CliRunner().invoke(
        main,
        ["lipid-library", "status", "--force-field", "amber14sb", "--json-output"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "schema_version": 1,
        "force_fields": ["amber14sb"],
        "total": 2,
        "ready": 1,
        "unavailable": 1,
        "pending": 0,
        "complete": True,
    }
