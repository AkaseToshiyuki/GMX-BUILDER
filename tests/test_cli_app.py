from pathlib import Path

from gmxbuilder.app import _prepare_cli_build_config
from gmxbuilder.pipeline.config import PipelineConfig


def test_cli_build_binds_top_level_output_name_and_seed(tmp_path):
    configured_output = tmp_path / "configured"
    override_output = tmp_path / "override"
    config = PipelineConfig(
        system_name="documented_system",
        output_dir=configured_output,
        seed=31415,
        modules={
            "input": {"pdb": "input.pdb"},
            "orient": {"method": "ppm"},
            "membrane": {"lipid_type": "POPC"},
            "export": {"write_mdp": True},
        },
    )

    prepared = _prepare_cli_build_config(config, str(override_output))

    assert prepared.output_dir == Path(override_output)
    assert prepared.modules["export"]["output_dir"] == str(override_output)
    assert prepared.modules["export"]["system_name"] == "documented_system"
    assert prepared.modules["input"]["seed"] == 31415
    assert prepared.modules["orient"]["seed"] == 31415
    assert prepared.modules["membrane"]["seed"] == 31415
    assert config.output_dir == configured_output
    assert config.modules["export"] == {"write_mdp": True}
