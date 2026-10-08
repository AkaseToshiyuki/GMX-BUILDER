import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
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


def test_serve_defaults_and_explicit_wildcard_disable_capability_access_logs(monkeypatch):
    monkeypatch.delenv("GMXBUILDER_DEPLOYMENT_MODE", raising=False)
    monkeypatch.delenv("GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT", raising=False)
    monkeypatch.setenv("GMXBUILDER_MAX_BUILDS", "1")

    import uvicorn

    import gmxbuilder.runtime.hardware as hardware_module

    hardware = SimpleNamespace(
        configured_task_slots=1,
        configured_cpu_cores=1,
        configured_task_threads=1,
        detected_cpu_threads=1,
        gmx_installed=False,
        gmx_version="",
        gmx_path="",
        configured_gpu_count=0,
        configured_gpu_devices=(),
        warnings=(),
    )
    monkeypatch.setattr(
        hardware_module,
        "configure_runtime_resources",
        lambda **_kwargs: hardware,
    )

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda web_app, **kwargs: calls.append((web_app, kwargs)))
    server_stub = ModuleType("gmxbuilder.web.server")
    server_stub.app = object()
    monkeypatch.setitem(sys.modules, "gmxbuilder.web.server", server_stub)

    default = CliRunner().invoke(main, ["serve"])
    assert default.exit_code == 0, default.output
    assert calls[-1][1]["host"] == "127.0.0.1"
    assert calls[-1][1]["port"] == 7788
    assert calls[-1][1]["access_log"] is False

    # A bare wildcard bind no longer authorizes itself: it is the broadest
    # exposure and needs the same explicit opt-in as any other non-loopback
    # address, so the server must refuse to start.
    refused = CliRunner().invoke(main, ["serve", "--host", "0.0.0.0"])
    assert refused.exit_code != 0
    assert "unauthenticated non-loopback" in refused.output
    assert calls[-1][1]["host"] == "127.0.0.1"

    wildcard = CliRunner().invoke(main, ["serve", "--host", "0.0.0.0", "--allow-unsafe-deployment"])
    assert wildcard.exit_code == 0, wildcard.output
    assert calls[-1][1]["host"] == "0.0.0.0"
    assert calls[-1][1]["access_log"] is False
    assert "unsafe non-loopback deployment is enabled" in wildcard.output


@pytest.fixture
def listed_v4_availability(monkeypatch):
    """CLI rendering consumes an explicit readiness snapshot, never the host cache."""
    from gmxbuilder.modules.membrane import v4_availability

    entries = [
        {"lipid_name": "CHOL", "lipid_ff": source, "ready": True}
        for source in ("charmm36", "charmm36m", "lipid21")
    ] + [
        {"lipid_name": "SAPI", "lipid_ff": "charmm36", "ready": True},
        {"lipid_name": "SAPI", "lipid_ff": "charmm36m", "ready": False},
        {"lipid_name": "22RHC", "lipid_ff": "gaff2", "ready": True},
    ]
    monkeypatch.setattr(
        v4_availability, "refresh_availability_list", lambda **_kwargs: {"entries": entries}
    )


def test_list_lipids_states_which_force_fields_can_build_each(listed_v4_availability):
    """Registry membership is not buildability.

    A lipid can be described here and still be refused at force-field
    selection, so the listing says which force fields can actually build it
    rather than leaving that to be discovered several steps into a workflow.
    """
    from click.testing import CliRunner

    from gmxbuilder.app import main

    result = CliRunner().invoke(main, ["list-lipids"])
    assert result.exit_code == 0

    lines = {line.split()[0]: line for line in result.output.splitlines() if line.startswith("  ")}

    def labels(name: str) -> set[str]:
        rendered = lines[name].rsplit("[", 1)[1].rstrip("]")
        return {item.strip() for item in rendered.split(",")} - {"none"}

    # V4 readiness, including per-family failures, controls the displayed labels.
    assert {"C36", "C36m", "Lipid21"} <= labels("CHOL")
    # The supplied snapshot accepts SAPI only under CHARMM36.
    assert "C36" in labels("SAPI")
    assert "C36m" not in labels("SAPI")
    # Absent readiness must remain visible as unavailable in the unfiltered list.
    assert labels("20AHC") == set()
    assert "[none]" in lines["20AHC"]


def test_list_lipids_names_the_unbuildable_ones_on_stderr(listed_v4_availability):
    from click.testing import CliRunner

    from gmxbuilder.app import main

    result = CliRunner().invoke(main, ["list-lipids"])
    assert result.exit_code == 0
    assert "cannot be built by any bundled force field" in result.output


def test_list_lipids_can_filter_to_one_force_field(listed_v4_availability):
    from click.testing import CliRunner

    from gmxbuilder.app import main

    result = CliRunner().invoke(main, ["list-lipids", "--force-field", "lipid21"])
    assert result.exit_code == 0
    # Only entries explicitly ready for Lipid21 should survive this filter.
    assert "CHOL" in result.output
    # An oxysterol that Lipid21 does not cover must not appear under it.
    assert "22RHC" not in result.output


def test_list_lipids_rejects_an_unknown_force_field():
    from click.testing import CliRunner

    from gmxbuilder.app import main

    result = CliRunner().invoke(main, ["list-lipids", "--force-field", "nonsense"])
    assert result.exit_code != 0
    assert "Unknown force field" in result.output
