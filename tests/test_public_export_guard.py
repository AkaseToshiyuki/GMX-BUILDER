"""Contracts for the guard that keeps private material out of the public repository."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_public_export import (  # noqa: E402
    PRIVATE_PREFIXES,
    classify,
    private_paths,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_public_export.py"
HOOK = ROOT / ".githooks" / "pre-push"

PRIVATE_REMOTE = "https://github.com/AkaseToshiyuki/GMXBUILDER.git"
PUBLIC_REMOTE = "https://github.com/AkaseToshiyuki/GMX-BUILDER.git"


@pytest.mark.parametrize(
    "path",
    [
        "AGENTS.md",
        "docs/AGENTS.md",
        "CLAUDE.md",
        "docs/CODING_STYLE.md",
        "docs/charmm_compat/IMPLEMENTATION_PLAN.md",
        "docs/charmm_compat/VALIDATION.md",
        "Private/README.md",
        "Private/docs/preprint/MANUSCRIPT.md",
        "Private/docs/preprint/GMXBUILDER_PREPRINT_DRAFT_V0.5.pdf",
        "Private/internal_docs/LICENSE_AUDIT_2026-08-18.md",
        "src/gmxbuilder/data/forcefields/charmm36/forcefield.itp",
        "src/gmxbuilder/data/forcefields/charmm36m/aminoacids.rtp",
    ],
)
def test_private_paths_are_recognized(path: str) -> None:
    assert classify(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/gmxbuilder/app.py",
        "src/gmxbuilder/data/forcefields/amber14sb.ff/forcefield.itp",
        # A near-miss: the prefix must match a directory boundary, not a
        # substring, or an unrelated sibling would be excluded from release.
        "src/gmxbuilder/data/forcefields/charmm36-extra/notes.md",
        "docs/USER_MANUAL.md",
        "tests/test_public_export_guard.py",
    ],
)
def test_publishable_paths_are_not_flagged(path: str) -> None:
    assert classify(path) is None


def test_every_prefix_carries_a_reason() -> None:
    for prefix, reason in PRIVATE_PREFIXES:
        assert prefix.endswith("/")
        assert reason and not reason.endswith(".")


def test_private_paths_reports_each_offender_with_its_reason() -> None:
    found = private_paths(
        [
            "README.md",
            "Private/README.md",
            "src/gmxbuilder/data/forcefields/charmm36m/ions.itp",
        ]
    )
    assert [path for path, _ in found] == [
        "Private/README.md",
        "src/gmxbuilder/data/forcefields/charmm36m/ions.itp",
    ]
    assert all(reason for _, reason in found)


@pytest.fixture
def private_repository(tmp_path: Path) -> Path:
    for name in ("scripts/check_public_export.py", "scripts/external_assets.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (tmp_path / "Private").mkdir()
    (tmp_path / "Private/notes.txt").write_text("Internal fixture\n")
    for args in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "Private fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_checker_rejects_the_private_tree(private_repository: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "HEAD"],
        capture_output=True,
        text=True,
        cwd=private_repository,
        timeout=60,
    )
    assert completed.returncode == 1
    assert "must not reach the public repository" in completed.stderr


def _run_hook(remote_url: str, sha: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HOOK), "some-remote", remote_url],
        input=f"refs/heads/main {sha} refs/heads/main {'0' * 40}\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=60,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull},
    )


def test_hook_blocks_private_content_reaching_a_public_remote(private_repository: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=private_repository,
        check=True,
    ).stdout.strip()

    completed = _run_hook(PUBLIC_REMOTE, head, private_repository)
    assert completed.returncode == 1
    assert "refusing to push" in completed.stderr


def test_hook_allows_pushes_to_the_private_remote() -> None:
    """The private repository is where this material belongs."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout.strip()

    completed = _run_hook(PRIVATE_REMOTE, head)
    assert completed.returncode == 0, completed.stderr


def test_hook_ignores_branch_deletions() -> None:
    """An all-zero local sha deletes a branch and publishes nothing."""
    completed = _run_hook(PUBLIC_REMOTE, "0" * 40)
    assert completed.returncode == 0, completed.stderr


def test_hook_is_executable_and_documents_its_limits() -> None:
    assert os.access(HOOK, os.X_OK)
    text = HOOK.read_text(encoding="utf-8")
    # A guard trusted beyond its reach is worse than no guard, so the two ways
    # it can be bypassed must stay written down next to it.
    assert "--no-verify" in text
    assert "core.hooksPath" in text


# --------------------------------------------------------------------------
# The guard and the installer manifest must agree about what is not shipped


def test_every_installed_force_field_is_private_without_being_listed_twice():
    """An asset is in the manifest because we install it instead of shipping it.

    This used to be a hardcoded pair of CHARMM prefixes, which covered the two
    force fields that existed when it was written and nothing added later. The
    failure mode is publishing force-field data whose upstream terms do not
    clearly permit it -- the one thing this module exists to prevent -- so the
    list is derived from the manifest rather than maintained beside it.
    """
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "scripts" / "external_assets.json").read_text(
            encoding="utf-8"
        )
    )
    targets = [str(asset["target"]) for asset in manifest["assets"]]
    assert targets, "the manifest lists no assets; this test would prove nothing"

    for target in targets:
        path = f"src/gmxbuilder/data/forcefields/{target}/forcefield.itp"
        assert classify(path) is not None, f"{target} is installed but would be published"


def test_a_newly_added_asset_is_covered_without_touching_the_guard(tmp_path, monkeypatch):
    """The point of deriving it: coverage arrives with the asset, not after it."""
    import json

    from scripts import check_public_export as guard

    manifest = tmp_path / "external_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [{"name": "A future force field", "target": "amber14sb_ol24"}],
            }
        )
    )
    monkeypatch.setattr(guard, "_ASSET_MANIFEST", manifest)

    prefixes = dict(guard._installed_forcefield_prefixes())
    assert "src/gmxbuilder/data/forcefields/amber14sb_ol24/" in prefixes
    # The known trees stay private even when a manifest omits them.
    assert "src/gmxbuilder/data/forcefields/charmm36m/" in prefixes


def test_an_unreadable_manifest_keeps_the_known_trees_private(tmp_path, monkeypatch):
    """This gates a release. Not being able to tell must not mean "publish"."""
    from scripts import check_public_export as guard

    monkeypatch.setattr(guard, "_ASSET_MANIFEST", tmp_path / "absent.json")
    prefixes = dict(guard._installed_forcefield_prefixes())

    assert "src/gmxbuilder/data/forcefields/charmm36/" in prefixes
    assert "src/gmxbuilder/data/forcefields/charmm36m/" in prefixes


def test_a_manifest_target_cannot_escape_the_force_field_directory(tmp_path, monkeypatch):
    """A traversing target is a manifest bug; silently making it a prefix hides it."""
    import json

    from scripts import check_public_export as guard

    manifest = tmp_path / "external_assets.json"
    manifest.write_text(
        json.dumps({"assets": [{"target": "../../.."}, {"target": "a/b"}, {"target": "."}]})
    )
    monkeypatch.setattr(guard, "_ASSET_MANIFEST", manifest)

    prefixes = [prefix for prefix, _ in guard._installed_forcefield_prefixes()]
    assert all(prefix.startswith("src/gmxbuilder/data/forcefields/") for prefix in prefixes)
    assert all(".." not in prefix for prefix in prefixes)
