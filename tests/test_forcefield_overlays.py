"""What a fresh installation gets must be what this machine has.

The CHARMM trees are installed rather than shipped, and this repository also
tracks a working copy of them. Those two can drift, and when they do the suite
stays green while every fresh clone is broken -- which is exactly what
happened: ``forcefield_overlays/charmm36/gmxbuilder-plasmalogen-bonded.itp``
was missing one bond type and two Urey-Bradley angle types that the tracked
tree had, so ``install-local.sh`` produced a CHARMM36 that could not build a
plasmalogen bilayer while the tests passed against the good local copy.

The project's own rule covers this: *do not assert on what your machine
happens to have*. These assert that the installer's inputs say the same thing
as the installation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.prerequisites import forcefield_parameters_available

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ROOT / "src" / "gmxbuilder" / "data" / "forcefield_overlays"
FORCEFIELDS = ROOT / "src" / "gmxbuilder" / "data" / "forcefields"

#: Overlay files are installed under a ``gmxbuilder-`` prefix. A tree installed
#: before that convention holds the same content under the bare name, so both
#: spellings are accepted when locating the installed counterpart.
_PREFIX = "gmxbuilder-"


def _parameters(text: str) -> dict[str, list[tuple[str, ...]]]:
    """Parameter rows per section, with comments and spacing discarded."""
    out: dict[str, list[tuple[str, ...]]] = {}
    section = None
    for raw in text.splitlines():
        line = raw.split(";")[0].rstrip()
        header = re.match(r"^\[\s*(\S+)\s*\]", line)
        if header:
            section = header.group(1)
            out.setdefault(section, [])
            continue
        if section and line.strip():
            out[section].append(tuple(line.split()))
    return out


def _overlay_pairs() -> list[tuple[str, Path, Path]]:
    pairs = []
    for directory in sorted(OVERLAYS.iterdir()):
        if not directory.is_dir():
            continue
        installed_root = FORCEFIELDS / directory.name
        for overlay in sorted(directory.glob("*.itp")):
            if not overlay.name.startswith(_PREFIX):
                continue
            bare = installed_root / overlay.name[len(_PREFIX) :]
            prefixed = installed_root / overlay.name
            # Both spellings present means two copies of the same parameters
            # sit in one force field and only the one `forcefield.itp` names
            # is in effect. That is litter, and it is what a `--force`
            # reinstall over a tree installed under the older convention
            # leaves behind.
            if bare.is_file() and prefixed.is_file():
                pairs.append((f"{directory.name}/{overlay.name}", overlay, None))
                continue
            for candidate in (prefixed, bare):
                if candidate.is_file():
                    pairs.append((f"{directory.name}/{overlay.name}", overlay, candidate))
                    break
    return pairs


def test_there_are_overlay_files_to_check():
    """A silent zero would make every check below vacuous."""
    assert list(OVERLAYS.iterdir()), "no overlays are present at all"


@pytest.mark.parametrize(
    ("label", "overlay", "installed"),
    _overlay_pairs() or [pytest.param("none installed", None, None, marks=pytest.mark.skip)],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_the_overlay_and_the_installed_copy_carry_the_same_parameters(label, overlay, installed):
    """Drift here breaks fresh clones while leaving this machine green.

    Comments and column alignment are ignored -- only the parameters matter.
    """
    assert installed is not None, (
        f"{label}: the installed force field holds this file under both its bare and its "
        "gmxbuilder- name; only the one forcefield.itp includes is in effect, so remove "
        "the other rather than leaving two copies of the same parameters"
    )
    expected = _parameters(installed.read_text(encoding="utf-8"))
    actual = _parameters(overlay.read_text(encoding="utf-8"))

    assert set(expected) == set(actual), (
        f"{label}: sections differ between the overlay and the installed copy"
    )
    for section in sorted(expected):
        missing = [row for row in expected[section] if row not in actual[section]]
        extra = [row for row in actual[section] if row not in expected[section]]
        assert not missing, (
            f"{label} [{section}]: the overlay is missing {len(missing)} parameter(s) the "
            f"installed force field has, so a fresh install would be incomplete: "
            f"{[' '.join(row) for row in missing[:3]]}"
        )
        assert not extra, (
            f"{label} [{section}]: the overlay carries {len(extra)} parameter(s) the installed "
            f"force field does not: {[' '.join(row) for row in extra[:3]]}"
        )


def test_the_charmm36_overlay_supplies_the_ether_linkage_terms():
    """The three parameters whose absence produced "No default U-B types".

    Named explicitly because they are what a plasmalogen needs and what the
    March 2019 port does not carry natively; the equivalence check above only
    notices them while a good installed copy exists to compare against.
    """
    if not forcefield_parameters_available("charmm36"):
        pytest.skip("charmm36 is installed separately by ./install-local.sh")

    overlay = _parameters(
        (OVERLAYS / "charmm36" / "gmxbuilder-plasmalogen-bonded.itp").read_text(encoding="utf-8")
    )
    bonds = {row[:2] for row in overlay.get("bondtypes", [])}
    angles = {row[:3] for row in overlay.get("angletypes", [])}

    assert ("CTL2", "OG301") in bonds
    assert ("HAL2", "CTL2", "OG301") in angles
    assert ("OG301", "CTL2", "CTL1") in angles
