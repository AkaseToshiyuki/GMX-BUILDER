"""Version metadata that must agree across the repository.

`uv.lock` records the project's own version, and CI installs with
`uv sync --locked`, which refuses a lockfile that no longer matches
`pyproject.toml`. So bumping the version without re-running `uv lock` turns
every CI job red at the install step, before a single test runs -- which is
exactly what happened on the 0.9.43 -> 0.9.44 bump.

Running `uv lock` here instead would hide the problem rather than report it,
and it needs the network. This only checks that the files agree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the 3.10 floor declared in pyproject
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _declared_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_the_package_reports_the_version_pyproject_declares():
    from gmxbuilder import __version__

    assert __version__ == _declared_version()


def test_the_lockfile_pins_the_version_pyproject_declares():
    """`uv sync --locked` fails the whole CI run when these disagree."""
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    entry = re.search(r'^name = "gmxbuilder"\nversion = "([^"]+)"', lock, flags=re.MULTILINE)
    assert entry, "uv.lock has no gmxbuilder entry to check"
    assert entry.group(1) == _declared_version(), "run `uv lock` after bumping the version"


def test_no_default_run_quietly_skips_the_external_tool_checks():
    """`--fast` must stay something you type, never something you inherit.

    It deselects every check that drives GROMACS, AmberTools, RDKit or a
    browser -- roughly a tenth of the suite and two thirds of its wall time,
    and the only part that can see a topology GROMACS will not accept. A green
    run that skipped all of it is worse than no run, so the flag is barred from
    `addopts` and from CI, where it would be invisible.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pytest_options = tomllib.load(handle)["tool"]["pytest"]["ini_options"]
    assert "--fast" not in " ".join(pytest_options.get("addopts", [])), (
        "--fast in addopts would make every default run skip the external-tool checks"
    )

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--fast" not in workflow, "CI must run the external-tool checks it can run"


def test_the_slow_marker_is_registered():
    """An unregistered marker is a warning, not an error, and typos survive it."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        markers = tomllib.load(handle)["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("slow:") for marker in markers)
