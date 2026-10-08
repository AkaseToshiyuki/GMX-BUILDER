"""The shipped example configuration must actually work.

It did not. It used `composition_upper` / `composition_lower`, `orient.algorithm`
and a top-level `simparams` block -- none of which any module accepts -- so a
user following it got a validation error rather than a build. Nothing referenced
the file, which is why it could drift for as long as it did.

These tests validate every module block against the real module that consumes
it, so a key renamed in the code fails here instead of in someone's terminal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.pipeline.step_executor import _get_module, get_pipeline_steps
from tests.prerequisites import requires_forcefield

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "build_example.yaml"


@pytest.fixture(scope="module")
def example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_the_example_parses_and_names_a_system(example):
    assert example["system_name"]
    assert example["output_dir"]
    assert isinstance(example["modules"], dict)


def test_every_module_block_uses_keys_that_module_accepts(example):
    """The failure this file shipped with: keys no module recognises."""
    failures = []
    for name, config in example["modules"].items():
        module = _get_module(name, "membrane-bilayer")
        probe = dict(config)
        # input.pdb points at a placeholder path; existence is checked
        # separately by the module and is not what this test is about.
        if name == "input":
            continue
        try:
            module.validate_config(probe)
        except ModuleConfigError as error:
            message = str(error)
            if "Unsupported" in message or "must be" in message and "option" in message:
                failures.append(f"{name}: {message}")
            elif "Unsupported" in message:
                failures.append(f"{name}: {message}")
    assert not failures, "example config uses keys no module accepts:\n" + "\n".join(failures)


def test_no_module_block_is_named_something_the_pipeline_ignores(example):
    """A block under a wrong name is silently skipped, which is worse."""
    known = set(get_pipeline_steps("membrane-bilayer"))
    unknown = sorted(set(example["modules"]) - known)
    assert not unknown, f"blocks the bilayer pipeline never reads: {unknown}"


def test_simparams_is_nested_under_export_not_top_level(example):
    """It was top-level, where nothing reads it, so the settings did nothing."""
    assert "simparams" not in example
    assert "simparams" in example["modules"]["export"]


def test_the_membrane_composition_uses_the_accepted_shape(example):
    """composition_upper / composition_lower are not read by anything."""
    membrane = example["modules"]["membrane"]
    assert "composition_upper" not in membrane
    assert "composition_lower" not in membrane
    composition = membrane["lipid_composition"]
    for leaflet in ("upper", "lower"):
        assert composition[leaflet], leaflet
        for entry in composition[leaflet]:
            assert set(entry) >= {"name", "ratio"}


@requires_forcefield("charmm36")
def test_the_lipids_named_are_buildable_with_the_declared_force_field(example):
    """An example that names an unavailable lipid teaches the wrong thing."""
    from gmxbuilder.modules.forcefield.lipid_policy import charmm_lipid_capability

    force_field = example["modules"]["forcefield"]["lipid_ff"]
    composition = example["modules"]["membrane"]["lipid_composition"]
    for leaflet in ("upper", "lower"):
        for entry in composition[leaflet]:
            available, reason = charmm_lipid_capability(entry["name"], force_field)
            assert available, f"{entry['name']} unavailable for {force_field}: {reason}"
