"""Detection of externally distributed prerequisites the repository cannot ship.

Force-field parameter files (``ffbonded.itp``, ``ffnonbonded.itp``, the water
topologies, ``ions.itp``) carry upstream licences that keep them out of this
repository; ``install-local.sh`` fetches them from pinned sources and verifies
their digests. The GAFF2 runtime and GROMACS are installed the same way.

Without these, a fresh clone produced dozens of unrelated-looking errors --
"No equilibrium bond length for atom types P-ON2B", "Water model 'tip3p' is
unavailable" -- that say nothing about what is actually missing. The suite
already skipped cleanly when GROMACS was absent; these helpers extend the same
treatment to the other prerequisites, so an incomplete environment reports what
it lacks instead of failing.

Availability is probed once per session: these are installation facts, and they
do not change while the suite runs.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

_FORCEFIELD_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "gmxbuilder" / "data" / "forcefields"
)

# A force field is only usable once its bonded and non-bonded parameters are
# present; the directory itself is tracked, so its existence proves nothing.
_REQUIRED_PARAMETER_FILES = ("ffbonded.itp", "ffnonbonded.itp")

_MISSING_FORCEFIELD_REASON = (
    "force field {name!r} has no bundled parameters; its files are distributed "
    "separately under their upstream licence and are installed by "
    "./install-local.sh"
)


def _force_field_directory(name: str) -> Path | None:
    """Return the data directory for *name*, accepting both naming styles."""
    for candidate in (_FORCEFIELD_ROOT / name, _FORCEFIELD_ROOT / f"{name}.ff"):
        if candidate.is_dir():
            return candidate
    return None


@cache
def forcefield_parameters_available(name: str) -> bool:
    """Return whether *name* has its separately distributed parameter files."""
    directory = _force_field_directory(name)
    if directory is None:
        return False
    return all((directory / required).is_file() for required in _REQUIRED_PARAMETER_FILES)


@cache
def water_topology_available(force_field: str, water_model: str) -> bool:
    """Return whether one water topology is bundled for *force_field*."""
    directory = _force_field_directory(force_field)
    if directory is None:
        return False
    return (directory / f"{water_model.lower()}.itp").is_file()


#: Assets stored through Git LFS. A clone made without LFS, or one whose
#: objects were never fetched, holds a small text pointer in their place.
_LFS_ASSETS = (
    "src/gmxbuilder/data/martini3/v3.0.0/martini_v3.0.0.itp",
    "src/gmxbuilder/data/prebuilt_assets/gmxbuilder-lipid-assets-v7.tar.xz",
)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


@cache
def lfs_assets_materialized() -> bool:
    """Return whether the Git LFS assets hold content rather than pointers.

    These are repository content, not an external prerequisite, but fetching
    them costs bandwidth against a monthly LFS quota -- hundreds of MB per checkout,
    across three jobs per push. A gate that exhausts that quota starts failing
    unpredictably part-way through a month, which teaches people to ignore it,
    so CI checks out without LFS and the tests that need the real files skip
    with a reason rather than failing on a pointer.
    """
    root = Path(__file__).resolve().parents[1]
    for relative in _LFS_ASSETS:
        path = root / relative
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                return False
    return True


@cache
def gaff_runtime_available() -> bool:
    """Return whether the managed GAFF2/AM1-BCC runtime is installed."""
    from gmxbuilder.modules.forcefield.gaff_backend import gaff_available

    return bool(gaff_available())


@cache
def gromacs_available() -> bool:
    """Return whether a usable GROMACS frontend is installed."""
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    return find_gromacs_executable() is not None


def requires_forcefield(name: str):
    """Skip unless *name* has its separately distributed parameters."""
    return pytest.mark.skipif(
        not forcefield_parameters_available(name),
        reason=_MISSING_FORCEFIELD_REASON.format(name=name),
    )


def requires_water_topology(force_field: str, water_model: str):
    """Skip unless one force-field/water topology pair is bundled."""
    return pytest.mark.skipif(
        not water_topology_available(force_field, water_model),
        reason=(
            f"water topology {water_model!r} for force field {force_field!r} is "
            "distributed separately and is installed by ./install-local.sh"
        ),
    )


requires_gaff_runtime = pytest.mark.skipif(
    not gaff_runtime_available(),
    reason=(
        "the managed GAFF2/AM1-BCC runtime is not installed; run ./install-local.sh to provide it"
    ),
)

requires_lfs_assets = pytest.mark.skipif(
    not lfs_assets_materialized(),
    reason=(
        "Git LFS assets are unfetched pointers; run `git lfs pull` to "
        "materialize the Martini 3 and prebuilt lipid archives"
    ),
)

requires_gromacs = pytest.mark.skipif(
    not gromacs_available(),
    reason="no GROMACS executable is available; set GMX_BIN or run ./install-local.sh",
)


def skip_without_forcefield(name: str) -> None:
    """Skip the running test from inside a helper or fixture."""
    if not forcefield_parameters_available(name):
        pytest.skip(_MISSING_FORCEFIELD_REASON.format(name=name))


def require_v4_entries(names, force_field, lipid_ff=None):
    """Integration probes need accepted assets, never generated production fallbacks."""
    from gmxbuilder.modules.membrane.equilibrated_library import get_equilibrated_lipid_library
    from gmxbuilder.modules.membrane.v4_availability import accepted_entry

    try:
        library = get_equilibrated_lipid_library()
    except RuntimeError as exc:
        if "Git LFS pointer" in str(exc) or "Prebuilt lipid assets are absent" in str(exc):
            pytest.skip(str(exc))
        raise
    selected = lipid_ff or force_field
    missing = [
        name for name in names if accepted_entry(library, name, force_field, selected) is None
    ]
    if missing:
        pytest.skip(f"V4 assets not accepted for {force_field}/{selected}: {', '.join(missing)}")
