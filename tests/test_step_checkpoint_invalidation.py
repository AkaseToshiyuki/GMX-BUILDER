from gmxbuilder.pipeline.step_executor import StepRunner


def test_rerun_invalidates_only_downstream_checkpoints(tmp_path):
    runner = StepRunner(tmp_path, pipeline_type="membrane-bilayer")
    for step in ("input", "forcefield", "structure", "orient", "membrane"):
        directory = runner.step_dir(step)
        directory.mkdir(parents=True)
        (directory / "sentinel.txt").write_text(step, encoding="utf-8")

    invalidated = runner.invalidate_downstream("structure")

    assert invalidated == ["orient", "membrane"]
    assert runner.step_dir("input").exists()
    assert runner.step_dir("forcefield").exists()
    assert runner.step_dir("structure").exists()
    assert not runner.step_dir("orient").exists()
    assert not runner.step_dir("membrane").exists()
