"""The published tree must run without any private material present.

Private/ is excluded from the public repository, and it is about to hold
optional feature modules rather than only documents. That makes a new failure
possible: public code importing something that only exists privately. It would
never show up in development, because a maintainer's checkout has both halves
-- it would show up as a broken public deployment.

These tests build the exact tree that would be published and import the
application from it, so the failure surfaces here instead of downstream.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _export_public_tree(destination: Path) -> Path:
    """Materialise the candidate index without any private path, as a publish would."""
    tree = subprocess.check_output(["git", "write-tree"], cwd=ROOT, text=True).strip()
    archive = subprocess.run(
        ["git", "archive", tree],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive,
        check=True,
        capture_output=True,
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_public_export import classify  # noqa: PLC0415

    for path in sorted(destination.rglob("*"), reverse=True):
        relative = path.relative_to(destination).as_posix()
        if classify(relative) is None:
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    return destination


@pytest.fixture(scope="module")
def public_tree(tmp_path_factory):
    return _export_public_tree(tmp_path_factory.mktemp("public-tree"))


def test_the_exported_tree_contains_nothing_private(public_tree):
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_public_export import classify  # noqa: PLC0415

    offenders = [
        path.relative_to(public_tree).as_posix()
        for path in public_tree.rglob("*")
        if path.is_file() and classify(path.relative_to(public_tree).as_posix()) is not None
    ]
    assert offenders == []


def test_the_application_imports_and_builds_its_routes_without_private_material(public_tree):
    """A public deployment must start, not merely install."""
    probe = (
        "from gmxbuilder.web.server import app\n"
        "print('ROUTES', len(app.routes))\n"
        "import gmxbuilder.app  # the CLI entry point must import too\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=public_tree,
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(public_tree / "src"),
            "HOME": str(public_tree),
        },
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ROUTES" in completed.stdout

    routes = int(completed.stdout.split("ROUTES")[1].split()[0])
    assert routes > 40, f"only {routes} routes registered in the public build"


def test_no_public_module_imports_private_material():
    """Static counterpart: the import must not exist in the first place."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("from Private", "import Private", "Private."):
            if marker in text:
                offenders.append(f"{path.relative_to(ROOT)}: {marker}")
    assert offenders == []


def test_published_forcefields_include_their_parameter_closure(public_tree):
    # Independent release contract: these models are shipped, not installed.
    for name in ("amber14sb.ff", "amber99sb.ff", "amber99sb-ildn.ff", "oplsaa.ff"):
        directory = public_tree / "src/gmxbuilder/data/forcefields" / name
        for filename in ("forcefield.itp", "ffbonded.itp", "ffnonbonded.itp"):
            assert (directory / filename).is_file(), f"Missing {name}/{filename}"
