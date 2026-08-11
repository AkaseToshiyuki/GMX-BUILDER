"""Ion properties and force-field compatibility helpers."""

from __future__ import annotations

from pathlib import Path


ION_PROPERTIES: dict[str, dict[str, float]] = {
    "NA": {"charge": 1.0, "mass": 22.99, "vdw": 0.116},
    "K": {"charge": 1.0, "mass": 39.10, "vdw": 0.152},
    "CS": {"charge": 1.0, "mass": 132.91, "vdw": 0.167},
    "LI": {"charge": 1.0, "mass": 6.94, "vdw": 0.090},
    "CA": {"charge": 2.0, "mass": 40.08, "vdw": 0.114},
    "ZN": {"charge": 2.0, "mass": 65.38, "vdw": 0.118},
    "MG": {"charge": 2.0, "mass": 24.31, "vdw": 0.086},
    "CL": {"charge": -1.0, "mass": 35.45, "vdw": 0.167},
    "BR": {"charge": -1.0, "mass": 79.90, "vdw": 0.183},
    "I": {"charge": -1.0, "mass": 126.90, "vdw": 0.198},
}

KNOWN_IONS = frozenset(ION_PROPERTIES)
KNOWN_CATIONS = frozenset(name for name, item in ION_PROPERTIES.items() if item["charge"] > 0)
KNOWN_ANIONS = frozenset(name for name, item in ION_PROPERTIES.items() if item["charge"] < 0)


def ion_charge(name: str) -> int:
    return int(ION_PROPERTIES[name.upper()]["charge"])


def force_field_ion_file(force_field: str, water_model: str) -> Path | None:
    """Return the bundled ion topology selected by :class:`TopologyWriter`."""
    base = Path(__file__).resolve().parents[2] / "data" / "forcefields"
    ff_name = force_field.strip().lower()
    directory = next(
        (item for item in (base / ff_name, base / f"{ff_name}.ff") if item.is_dir()),
        None,
    )
    if directory is None:
        return None
    per_water = directory / f"ions_{water_model.strip().lower()}.itp"
    generic = directory / "ions.itp"
    return per_water if per_water.is_file() else generic if generic.is_file() else None


def molecule_types_in_itp(path: Path) -> set[str]:
    """Read exact ``[ moleculetype ]`` names from a GROMACS ITP."""
    names: set[str] = set()
    section = ""
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            section = line.strip("[] ").lower()
            continue
        if section == "moleculetype":
            names.add(line.split()[0].upper())
            section = ""
    return names


def supported_ions(force_field: str, water_model: str) -> set[str]:
    path = force_field_ion_file(force_field, water_model)
    return molecule_types_in_itp(path) & set(KNOWN_IONS) if path else set()
