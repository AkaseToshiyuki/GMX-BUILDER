#!/usr/bin/env python3
"""Build the public tree by construction, so private material cannot slip in.

The public repository is not this branch with things deleted. It is a tree
assembled from the publishable paths only, committed on top of the existing
public history. Private objects therefore never enter the published history at
all -- which matters because ``git push`` transfers reachable objects, so a
branch whose tip merely omits ``Private/`` would still carry every historical
version of it.

The rule comes from :mod:`check_public_export`, the same definition the
pre-push hook uses.

This script deliberately stops before publishing. It writes a local branch and
prints the command to run; pushing to a public remote is irreversible and stays
a human decision.

Typical use::

    python scripts/prepare_public_release.py --parent public/main
    git log --stat public-export -1        # inspect what would be published
    git push public public-export:main     # only when it looks right
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_public_export import classify, private_paths, validate_public_parent  # noqa: E402

DEFAULT_BRANCH = "public-export"


def _git(*arguments: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        input=stdin.encode("utf-8", errors="surrogateescape") if stdin is not None else None,
        check=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull},
    )
    return result.stdout.decode("utf-8", errors="surrogateescape")


def _ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull},
        ).returncode
        == 0
    )


def build_public_tree(source: str) -> tuple[str, list[str], list[str]]:
    """Return (tree sha, published paths, withheld paths) for *source*.

    The tree is assembled from an explicit index rather than by deleting from a
    checkout, so a path is published only if it was positively selected.
    """
    entries = _git("ls-tree", "-rz", source).split("\0")
    published: list[str] = []
    withheld: list[str] = []
    index_lines: list[str] = []

    for entry in entries:
        if not entry:
            continue
        # "<mode> <type> <sha>\t<path>"
        metadata, _, path = entry.partition("\t")
        if classify(path) is not None:
            withheld.append(path)
            continue
        published.append(path)
        mode, _object_type, sha = metadata.split()
        index_lines.append(f"{mode} {sha}\t{path}")

    if not published:
        raise SystemExit(f"{source} contains no publishable paths; refusing to build a tree.")

    # A dedicated index keeps the working tree and the real index untouched.
    with _temporary_index() as index_file:
        subprocess.run(
            ["git", "update-index", "-z", "--index-info"],
            input=("\0".join(index_lines) + "\0").encode("utf-8", errors="surrogateescape"),
            check=True,
            capture_output=True,
            env=index_file,
        )
        tree = subprocess.run(
            ["git", "write-tree"],
            capture_output=True,
            text=True,
            check=True,
            env=index_file,
        ).stdout.strip()
    return tree, published, withheld


class _temporary_index:
    """Provide an environment pointing GIT_INDEX_FILE at a scratch index."""

    def __enter__(self) -> dict[str, str]:
        import os
        import tempfile

        self._directory = tempfile.TemporaryDirectory(prefix="gmxbuilder-public-index-")
        environment = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_GRAFT_FILE": os.devnull}
        environment["GIT_INDEX_FILE"] = str(Path(self._directory.name) / "index")
        return environment

    def __exit__(self, *_exception_info: object) -> None:
        self._directory.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="HEAD", help="ref to publish from (default: HEAD)")
    parser.add_argument(
        "--parent",
        default="public/main",
        help="existing public commit to build on (default: public/main)",
    )
    parser.add_argument(
        "--no-parent",
        action="store_true",
        help="start a new public history instead of continuing one",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"local branch to write (default: {DEFAULT_BRANCH})",
    )
    parser.add_argument("--message", help="commit message (default: derived from the version)")
    arguments = parser.parse_args(argv)

    if not _ref_exists(arguments.source):
        print(f"Source ref {arguments.source!r} does not exist.", file=sys.stderr)
        return 2

    parent: str | None = None
    if not arguments.no_parent:
        if not _ref_exists(arguments.parent):
            print(
                f"Parent ref {arguments.parent!r} does not exist. Fetch the public "
                "remote, or pass --no-parent to start a new history.",
                file=sys.stderr,
            )
            return 2
        try:
            parent = validate_public_parent(arguments.parent)
        except (ValueError, subprocess.CalledProcessError) as error:
            print(f"Refusing public parent: {error}", file=sys.stderr)
            return 2

    tree, published, withheld = build_public_tree(arguments.source)

    # Belt and braces: the constructed tree is re-checked with the same rule
    # that the pre-push hook applies, so a bug in the filter cannot slip past.
    residual = private_paths(
        path for path in _git("ls-tree", "-rz", "--name-only", tree).split("\0") if path
    )
    if residual:
        print(
            f"Refusing to continue: {len(residual)} private path(s) survived filtering, "
            "which means the filter is wrong.",
            file=sys.stderr,
        )
        for path, _reason in residual[:10]:
            print(f"  {path}", file=sys.stderr)
        return 1

    version = _read_version()
    message = arguments.message or f"release: publish GMXBUILDER v{version}"
    commit_arguments = ["commit-tree", tree]
    if parent is not None:
        commit_arguments += ["-p", parent]
    commit = _git(*commit_arguments, stdin=message + "\n").strip()
    _git("update-ref", f"refs/heads/{arguments.branch}", commit)

    withheld_summary: dict[str, int] = {}
    for path in withheld:
        prefix = path.split("/", 1)[0] if path.startswith("Private/") else "force-field data"
        withheld_summary[prefix] = withheld_summary.get(prefix, 0) + 1

    print(f"Built {arguments.branch} ({commit[:12]}) from {arguments.source}")
    print(f"  published: {len(published)} files")
    print(f"  withheld:  {len(withheld)} files")
    for prefix, count in sorted(withheld_summary.items()):
        print(f"      {count:4d}  {prefix}")
    print(f"  parent:    {parent[:12] if parent else 'none (new history)'}")
    print()
    print("Nothing has been pushed. Review, then publish deliberately:")
    print(f"  git log --stat {arguments.branch} -1")
    print(f"  git push public {arguments.branch}:main")
    return 0


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "src" / "gmxbuilder" / "__version__.py"
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
