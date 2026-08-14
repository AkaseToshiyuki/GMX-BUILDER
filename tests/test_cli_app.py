from pathlib import Path

from click.testing import CliRunner

from gmxbuilder.app import _prepare_cli_build_config, main
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


def test_public_server_rejects_development_reload(monkeypatch):
    monkeypatch.setenv("GMXBUILDER_DEPLOYMENT_MODE", "public")
    monkeypatch.setenv("GMXBUILDER_AUTH_USER", "researcher")
    monkeypatch.setenv("GMXBUILDER_AUTH_PASSWORD", "correct-horse-battery-staple")
    monkeypatch.setenv("GMXBUILDER_TRUSTED_PROXIES", "127.0.0.1/32")
    monkeypatch.setenv("GMXBUILDER_CORS_ORIGINS", "https://gmxbuilder.example.org")
    result = CliRunner().invoke(main, ["serve", "--reload"])
    assert result.exit_code != 0
    assert "not permitted in public mode" in result.output
