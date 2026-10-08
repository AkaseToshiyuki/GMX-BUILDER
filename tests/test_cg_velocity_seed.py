"""The Martini workflows must derive a per-system velocity seed.

`normalize_protocol` falls back to `derive_velocity_seed(42)` when no seed is
passed, so an exporter that simply omits the argument writes the same
`gen-seed` into every package it ever produces. Both shipped Martini workflows
did exactly that, which silently undid the velocity-seed persistence added for
the atomistic exporter: changing a build's seed changed nothing about the
generated initial velocities.

These tests assert the property at the two levels where it can be lost -- the
call the exporter makes, and the value that reaches the MDP text.
"""

from __future__ import annotations

import inspect

import pytest

from gmxbuilder.io.mdp import derive_velocity_seed
from gmxbuilder.modules.coarse_grained.protocol import normalize_protocol, write_mdp_files


@pytest.mark.parametrize(
    "package",
    [
        "gmxbuilder.modules.martini3_bilayer",
        "gmxbuilder.modules.martini3_solvent",
        "gmxbuilder.modules.coarse_grained",
    ],
)
def test_every_cg_exporter_passes_a_derived_velocity_seed(package: str) -> None:
    """Checked through the class each workflow resolves to, not a file path.

    The workflows share one implementation, so asserting on a per-package copy
    would no longer prove anything about what a given workflow runs.
    """
    import importlib

    module = importlib.import_module(package)
    source = inspect.getsource(module.CGExportModule.run)
    normalized = "".join(source.split())

    assert "velocity_seed=derive_velocity_seed(" in normalized, (
        f"{package} calls normalize_protocol without a velocity seed, so every "
        "package it writes would share one constant gen-seed"
    )
    assert 'system.metadata.get("seed"' in source


def test_the_fallback_really_is_a_constant() -> None:
    """Guards the premise: if the fallback varied, the bug above would not exist."""
    first = normalize_protocol({}, has_membrane=True)["velocity_seed"]
    second = normalize_protocol({}, has_membrane=False)["velocity_seed"]
    assert first == second == derive_velocity_seed(42)


@pytest.mark.parametrize("seed", [0, 7, 42, 99, 2_147_483_647])
def test_distinct_build_seeds_produce_distinct_velocity_seeds(seed: int) -> None:
    derived = derive_velocity_seed(seed)
    assert 1 <= derived <= 2_147_483_647
    others = {derive_velocity_seed(other) for other in (0, 7, 42, 99) if other != seed}
    assert derived not in others


def test_the_derived_seed_reaches_the_generated_mdp(tmp_path) -> None:
    """The value must survive into the MDP text, not just the config dict."""
    config = normalize_protocol({}, has_membrane=True, velocity_seed=derive_velocity_seed(7))
    # write_mdp_files returns (stage, filename) pairs, not contents; read the
    # files it actually wrote so this checks the shipped text.
    written = write_mdp_files(tmp_path, config)

    seeded = []
    for _stage, filename in written:
        text = (tmp_path / filename).read_text(encoding="utf-8")
        if "gen-vel = yes" in text:
            seeded.append(text)

    assert seeded, "no stage generates velocities, so the seed could not be checked"
    for text in seeded:
        assert f"gen-seed = {derive_velocity_seed(7)}" in text
        assert "gen-seed = -1" not in text
