"""Authoritative force-field release and compatibility metadata.

The user-facing name alone is not a compatibility contract.  Each profile
records the parameter family, release, non-bonded convention, supported
parameter sources, and minimum GROMACS version required by the bundled port.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess


@dataclass(frozen=True)
class ForceFieldProfile:
    name: str
    label: str
    family: str
    release: str
    default_water: str
    ligand_backends: tuple[str, ...]
    lipid_backends: tuple[str, ...]
    defaults_signature: tuple[int, int, str, str, str]
    minimum_gromacs: tuple[int, int] = (2020, 0)
    cgenff_version: str | None = None
    legacy: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "release": self.release,
            "default_water": self.default_water,
            "ligand_backends": list(self.ligand_backends),
            "lipid_backends": list(self.lipid_backends),
            "minimum_gromacs": ".".join(map(str, self.minimum_gromacs)),
            "cgenff_version": self.cgenff_version,
            "legacy": self.legacy,
        }


# (nbfunc, comb-rule, gen-pairs, fudgeLJ, fudgeQQ).  String scaling values are
# intentional: releases must not compare equal after lossy float rounding.
_PROFILES = {
    "amber14sb": ForceFieldProfile(
        name="amber14sb",
        label="AMBER ff14SB (GROMACS 2026.3 port; validated with GROMACS 2025.4)",
        family="amber",
        release="GROMACS-2026.3",
        default_water="tip3p",
        ligand_backends=("gaff2",),
        lipid_backends=("gaff2",),
        defaults_signature=(1, 2, "yes", "0.5", "0.83333333333333333"),
        # The imported data and generated topology were regression-tested with
        # GROMACS 2025.4; the port uses no 2026-only input syntax.
        minimum_gromacs=(2025, 4),
    ),
    "amber99sb-ildn": ForceFieldProfile(
        name="amber99sb-ildn",
        label="AMBER ff99SB-ILDN (legacy)",
        family="amber",
        release="GROMACS-bundled",
        default_water="tip3p",
        ligand_backends=("gaff2",),
        lipid_backends=("gaff2",),
        defaults_signature=(1, 2, "yes", "0.5", "0.8333"),
        legacy=True,
    ),
    "amber99sb": ForceFieldProfile(
        name="amber99sb",
        label="AMBER ff99SB (legacy)",
        family="amber",
        release="GROMACS-bundled",
        default_water="tip3p",
        ligand_backends=("gaff2",),
        lipid_backends=("gaff2",),
        defaults_signature=(1, 2, "yes", "0.5", "0.8333"),
        legacy=True,
    ),
    "charmm36m": ForceFieldProfile(
        name="charmm36m",
        label="CHARMM36m (Jul2022 / CGenFF 4.6)",
        family="charmm",
        release="Jul2022",
        default_water="tip3p",
        ligand_backends=("cgenff-import",),
        lipid_backends=("charmm36m",),
        defaults_signature=(1, 2, "yes", "1.0", "1.0"),
        cgenff_version="4.6",
        legacy=True,
    ),
    "charmm36": ForceFieldProfile(
        name="charmm36",
        label="CHARMM36 (Mar2019 / legacy)",
        family="charmm",
        release="Mar2019",
        default_water="tip3p",
        ligand_backends=("cgenff-import",),
        lipid_backends=("charmm36",),
        defaults_signature=(1, 2, "yes", "1.0", "1.0"),
        cgenff_version="4.1",
        legacy=True,
    ),
    "oplsaa": ForceFieldProfile(
        name="oplsaa",
        label="OPLS-AA/L (2001, legacy)",
        family="opls",
        release="2001",
        default_water="tip4p",
        ligand_backends=(),
        lipid_backends=(),
        defaults_signature=(1, 3, "yes", "0.5", "0.5"),
        legacy=True,
    ),
}


def get_force_field_profile(name: str) -> ForceFieldProfile:
    key = str(name).strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"No force-field profile for {name!r}") from exc


def force_field_profiles() -> tuple[ForceFieldProfile, ...]:
    return tuple(_PROFILES[name] for name in _PROFILES)


def force_field_family(name: str) -> str:
    return get_force_field_profile(name).family


def detect_gromacs_version(executable: str | None = None) -> tuple[int, int] | None:
    """Return the local GROMACS major/minor version, or ``None`` if unavailable."""
    candidate = executable or os.environ.get("GMX_BIN") or shutil.which("gmx")
    if not candidate:
        return None
    try:
        completed = subprocess.run(
            [candidate, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(
        r"GROMACS version:\s*(?:VERSION\s*)?(\d{4})\.(\d+)",
        completed.stdout + completed.stderr,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def validate_local_gromacs(force_field: str) -> str:
    """Validate a detected local GROMACS against the selected force field."""
    profile = get_force_field_profile(force_field)
    version = detect_gromacs_version()
    required = profile.minimum_gromacs
    if version is None:
        return (
            "Local GROMACS version was not detected during packaging; run "
            f"`gmx --version` and use {required[0]}.{required[1]} or newer."
        )
    if version < required:
        raise RuntimeError(
            f"{profile.label} requires GROMACS {required[0]}.{required[1]} or "
            f"newer; detected {version[0]}.{version[1]}"
        )
    return (
        f"GROMACS compatibility: detected {version[0]}.{version[1]} "
        f"(minimum for {profile.name}: {required[0]}.{required[1]})"
    )
