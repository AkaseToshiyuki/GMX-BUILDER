#!/usr/bin/env python3
"""The single definition of what must never reach the public repository.

Two classes of path are private, and both are already implied elsewhere in the
tree -- this module makes the rule executable so a hook, a release step and a
human can all consult one definition instead of three.

``Private/``
    Internal material: the preprint sources, internal decision records, and
    the preprint build script. ``Private/README.md`` states the policy in
    prose; this is the machine-checkable form of it.

Every force field the installer fetches
    Force-field data whose upstream terms do not clearly permit
    redistribution from here. These are **derived from**
    ``scripts/external_assets.json`` rather than listed again: an asset is in
    that manifest precisely because we chose to install it instead of shipping
    it, so the two cannot disagree about which those are.

    This was a hardcoded list of the two CHARMM trees. That worked only for as
    long as nobody added a third asset -- and the failure mode is publishing
    force-field data we have no clear right to publish, which is the one thing
    this module exists to prevent. ``.gitignore`` already documents that the
    private research repository may track these paths while public checkouts
    populate them during installation.

Note on scope: ``.gitattributes`` ``export-ignore`` does **not** help. Verified
directly -- it removes a path from ``git archive`` output while the objects stay
in the repository and are transferred normally by ``git push``.

Exit status is 0 when nothing private is present and 1 when something is, so
this can gate a hook or a release step directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_MANIFEST = _REPOSITORY_ROOT / "scripts" / "external_assets.json"
_FORCEFIELD_ROOT = "src/gmxbuilder/data/forcefields"

# Already published and reviewed on 2026-09-22. This immutable baseline includes
# the historical CHARMM releases explicitly accepted in Private/README.md.
# Never substitute a mutable remote-tracking ref or a push-supplied remote SHA.
APPROVED_PUBLIC_BASE = "c176d8f66913939f4abf884db1d9918ce2a8b255"


def is_private_remote(url: str) -> bool:
    """Recognize only the approved GitHub repository, not a URL substring."""
    if url.startswith("git@github.com:"):
        path = url.removeprefix("git@github.com:")
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
            return False
        if parsed.query or parsed.fragment or parsed.password:
            return False
        if parsed.port not in {None, 443 if parsed.scheme == "https" else 22}:
            return False
        if parsed.username not in {None, "git"}:
            return False
        path = parsed.path.removeprefix("/")
    return path.removesuffix(".git") == "AkaseToshiyuki/GMXBUILDER"


_INSTALLED_FORCEFIELD_REASON = "force-field data that may not be redistributed from this repository"


def _installed_forcefield_prefixes() -> tuple[tuple[str, str], ...]:
    """Return one private prefix per force field the installer fetches.

    A missing or unreadable manifest is not treated as "nothing is private".
    This module gates a release; the safe answer when it cannot tell is to
    keep the known trees private, so the CHARMM pair is the floor rather than
    the whole list.
    """
    targets = {"charmm36", "charmm36m"}
    try:
        manifest = json.loads(_ASSET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    else:
        for asset in manifest.get("assets", []):
            target = str(asset.get("target", "")).strip().strip("/")
            # A target that escapes the force-field directory is a manifest
            # bug, and turning it into a private prefix would hide it.
            if target and "/" not in target and target not in {".", ".."}:
                targets.add(target)
    return tuple(
        (f"{_FORCEFIELD_ROOT}/{target}/", _INSTALLED_FORCEFIELD_REASON)
        for target in sorted(targets)
    )


PRIVATE_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "Private/",
        "internal material that must stay in the private repository",
    ),
    *_installed_forcefield_prefixes(),
)

PRIVATE_DOCUMENTS = {
    "docs/CODING_STYLE.md",
    "docs/charmm_compat/IMPLEMENTATION_PLAN.md",
    "docs/charmm_compat/VALIDATION.md",
}


def classify(path: str) -> str | None:
    """Return why *path* is private, or None when it may be published."""
    if Path(path).name in {"AGENTS.md", "CLAUDE.md"} or path in PRIVATE_DOCUMENTS:
        return "internal agent guidance, implementation plans or historical engineering records"
    for prefix, reason in PRIVATE_PREFIXES:
        if path == prefix.rstrip("/") or path.startswith(prefix):
            return reason
        base = prefix.rstrip("/")
        if base.startswith(_FORCEFIELD_ROOT + "/") and any(
            path == base + suffix
            or path.startswith(base + suffix + "/")
            or path.startswith(base + suffix + "-")
            for suffix in (".installing", ".backup")
        ):
            return reason
    return None


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        check=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull},
    )
    return result.stdout.decode("utf-8", errors="surrogateescape")


def tracked_paths(ref: str) -> list[str]:
    """Return every path recorded in the tree at *ref*."""
    return [path for path in _git("ls-tree", "-rz", "--name-only", ref).split("\0") if path]


def history_private_paths(ref: str) -> list[tuple[str, str]]:
    """Check all new reachable commits, including merged side histories."""
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        raise ValueError("Cannot prove public history from a shallow repository")
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    arguments = ["rev-list", commit]
    if (
        subprocess.run(
            ["git", "cat-file", "-e", APPROVED_PUBLIC_BASE + "^{commit}"],
            capture_output=True,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull},
        ).returncode
        == 0
    ):
        arguments += ["--not", APPROVED_PUBLIC_BASE]
    for sha in _git(*arguments).splitlines():
        found = private_paths(tracked_paths(sha))
        if found:
            return [(path, f"{reason} (reachable commit {sha})") for path, reason in found]
    # A tag to an accepted historical commit must still have a clean tip:
    # acceptance of existing history is not permission to republish its tree.
    return private_paths(tracked_paths(commit))


def validate_public_parent(ref: str) -> str:
    """Require a clean continuation of the approved public history."""
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", APPROVED_PUBLIC_BASE, commit],
            capture_output=True,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull},
        ).returncode
        != 0
    ):
        raise ValueError("Parent is not a descendant of the approved public history")
    if history_private_paths(commit):
        raise ValueError("Parent introduces private material into public history")
    return commit


def private_paths(paths: Iterable[str]) -> list[tuple[str, str]]:
    """Return the (path, reason) pairs that must not be published."""
    found = []
    for path in paths:
        reason = classify(path)
        if reason is not None:
            found.append((path, reason))
    return found


def report(found: list[tuple[str, str]], *, ref: str, limit: int = 20) -> None:
    """Describe the offending paths on stderr, grouped by reason."""
    print(
        f"{len(found)} path(s) in {ref} must not reach the public repository:",
        file=sys.stderr,
    )
    by_reason: dict[str, list[str]] = {}
    for path, reason in found:
        by_reason.setdefault(reason, []).append(path)
    for reason, paths in by_reason.items():
        print(f"\n  {reason}:", file=sys.stderr)
        for path in paths[:limit]:
            print(f"    {path}", file=sys.stderr)
        if len(paths) > limit:
            print(f"    ... and {len(paths) - limit} more", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ref",
        nargs="?",
        default="HEAD",
        help="git ref whose tree is checked (default: HEAD)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="report only through the exit status",
    )
    parser.add_argument("--history", action="store_true", help="check new reachable history")
    parser.add_argument("--private-remote", help="exit zero only for the approved private URL")
    arguments = parser.parse_args(argv)
    if arguments.private_remote is not None:
        try:
            return 0 if is_private_remote(arguments.private_remote) else 1
        except ValueError:
            return 1

    try:
        paths = tracked_paths(arguments.ref)
    except subprocess.CalledProcessError as error:
        print(f"Cannot read tree at {arguments.ref!r}: {error.stderr.strip()}", file=sys.stderr)
        return 2

    try:
        found = history_private_paths(arguments.ref) if arguments.history else private_paths(paths)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f"Cannot inspect commit history: {error}", file=sys.stderr)
        return 2
    if not found:
        if not arguments.quiet:
            print(f"{arguments.ref}: no private paths present ({len(paths)} tracked files).")
        return 0
    if not arguments.quiet:
        report(found, ref=arguments.ref)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
