#!/usr/bin/env bash
# Run the test suite the way CI sees it, from the working tree.
#
# CI checks out without Git LFS -- deliberately, because fetching large assets three
# times per push exhausts the monthly quota and a gate that fails
# unpredictably part-way through a month teaches people to ignore it. It also
# has no GROMACS, no AmberTools, and none of the force fields that
# ./install-local.sh puts into the tree. A developer machine has all of those,
# so a test that quietly depends on one passes here and fails there.
#
# That is not hypothetical. CI was red for five consecutive pushes because
# three coarse-grained tests reached the LFS-tracked Martini archive without
# declaring it, and nobody looked. One of them was invisible even in CI: it
# carried @requires_gaff_runtime, which skipped it there for an unrelated
# reason, so the missing LFS gate would only have surfaced the day CI gained
# an AmberTools install.
#
# This reproduces those absences locally:
#
#   * a clone with LFS smudge skipped, so the big assets are pointers;
#   * the working tree copied over it, minus the LFS paths, so it tests what
#     you are about to push rather than what you last committed;
#   * GROMACS off PATH, and the GAFF and lipid-library roots pointed at
#     nothing.
#
# Usage:
#     scripts/reproduce_ci.sh [pytest args...]
#
# Exit status is pytest's. It is slower than `pytest -q` and is meant for
# before a push, not for the edit loop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git lfs version >/dev/null 2>&1 || { echo "ERROR: git-lfs is required for faithful CI reproduction" >&2; exit 2; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gmxbuilder-ci-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

CLONE="$WORK/tree"

echo "Cloning without LFS content ..."
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO_ROOT" clone --quiet --no-hardlinks --shared \
    "$REPO_ROOT" "$CLONE"

# Overlay the working tree so uncommitted work is what gets tested. The LFS
# paths are excluded by name: copying them would defeat the entire point by
# restoring the very files CI does not have.
git -C "$REPO_ROOT" lfs ls-files -n > "$WORK/lfs-paths.txt"
mapfile -t LFS_PATHS < "$WORK/lfs-paths.txt"
EXCLUDES=(--exclude ".git/")
for path in "${LFS_PATHS[@]}"; do
    [[ -n "$path" ]] && EXCLUDES+=(--exclude "/$path")
done

# Honour .gitignore, or every run copies .venv (725 MB here), build/, dist/
# and the caches into a directory that is deleted seconds later. Ignored files
# are by definition not what is about to be pushed, so they are not what this
# should be testing.
echo "Overlaying the working tree (${#LFS_PATHS[@]} LFS paths left as pointers) ..."
rsync -a --delete --filter=':- .gitignore' "${EXCLUDES[@]}" "$REPO_ROOT/" "$CLONE/"

# Sanity: the point of the exercise is that these are pointers. If the overlay
# restored one, every result below would be a false pass.
for path in "${LFS_PATHS[@]}"; do
    [[ -z "$path" ]] && continue
    if [[ -f "$CLONE/$path" ]] && ! head -c 40 "$CLONE/$path" | grep -q "git-lfs.github.com"; then
        echo "ERROR: $path holds content, not a pointer; the overlay defeated the check" >&2
        exit 2
    fi
done

# Strip the tools CI does not have.
#
# HOME is redirected rather than only PATH, because GROMACS discovery globs
# $HOME/Software/Gromacs*/bin/gmx as well as PATH -- an earlier version of this
# script cleared PATH alone and ten GROMACS tests still ran, which is precisely
# the false negative it exists to prevent. A fresh HOME also removes the lipid
# library under ~/.cache and the GAFF runtime under ~/.local/share, so the
# environment matches a runner rather than a workstation.
export HOME="$WORK/home"
mkdir -p "$HOME"
export PATH="/usr/bin:/bin"
unset GMX_BIN
export GMXBUILDER_GAFF_ENV="$WORK/no-gaff-runtime"
export GMXBUILDER_LIPID_LIBRARY="$WORK/no-lipid-library"
export PYTHONPATH="$CLONE/src"

# Prove the strip worked before trusting anything the suite says. /opt and
# /usr/local are searched too and are not ours to move.
if "${PYTHON:-$REPO_ROOT/.venv/bin/python}" -c "
import sys
sys.path.insert(0, '$CLONE/src')
from gmxbuilder.runtime.hardware import find_gromacs_executable
sys.exit(0 if find_gromacs_executable() else 1)
" 2>/dev/null; then
    echo "ERROR: GROMACS is still discoverable; this is not the CI environment." >&2
    exit 2
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "No interpreter at $PYTHON; set PYTHON=/path/to/python" >&2
    exit 2
fi

echo "Running the suite as CI sees it ..."
cd "$CLONE"
set +e
"$PYTHON" -m pytest --construction-only "$@"
status=$?
set -e

if [[ $status -eq 0 ]]; then
    echo "PASS: the suite is green without LFS assets, GROMACS, AmberTools or"
    echo "      installed force fields -- which is what CI runs."
else
    echo "FAIL: something here depends on this machine rather than on the"
    echo "      repository. Gate it with a marker from tests/prerequisites.py"
    echo "      rather than making CI install more."
fi
exit $status
