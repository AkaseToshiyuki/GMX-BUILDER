"""Strict import of ParamChem/CGenFF MOL2 and CHARMM stream output."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.modules.forcefield.catalog import get_force_field_profile


_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "BR": 35,
    "I": 53,
}


@dataclass(frozen=True)
class CGenFFTemplate:
    name: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    coordinates: np.ndarray
    net_charge: int
    itp_path: Path
    atomtypes_path: Path
    cgenff_version: str | None
    maximum_penalty: float | None


def _clean(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _parse_mol2(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    try:
        start = (
            next(i for i, line in enumerate(lines) if line.strip().upper() == "@<TRIPOS>ATOM") + 1
        )
    except StopIteration as exc:
        raise ModuleConfigError("CGenFF MOL2 file has no @<TRIPOS>ATOM section") from exc
    names: list[str] = []
    elements: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[start:]:
        if line.strip().startswith("@<TRIPOS>"):
            break
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 6:
            raise ModuleConfigError(f"Malformed MOL2 atom line: {line.strip()}")
        name = fields[1].strip()
        if name in names:
            raise ModuleConfigError(f"Duplicate MOL2 atom name {name!r}")
        raw_element = fields[5].split(".", 1)[0].upper()
        element = raw_element if raw_element in _ATOMIC_NUMBERS else raw_element[:1]
        if element not in _ATOMIC_NUMBERS:
            raise ModuleConfigError(f"Unsupported MOL2 element/type {fields[5]!r} for atom {name}")
        try:
            xyz = [float(fields[2]), float(fields[3]), float(fields[4])]
        except ValueError as exc:
            raise ModuleConfigError(f"Invalid MOL2 coordinates for atom {name}") from exc
        names.append(name)
        elements.append(element)
        coordinates.append(xyz)
    if not names:
        raise ModuleConfigError("CGenFF MOL2 file contains no atoms")
    return names, elements, np.asarray(coordinates, dtype=float) * 0.1


def _parameter_match(
    records: list[tuple[tuple[str, ...], tuple[float, ...]]],
    atom_types: tuple[str, ...],
    *,
    multiple: bool = False,
) -> list[tuple[float, ...]]:
    matches: list[tuple[int, tuple[float, ...]]] = []
    reverse = tuple(reversed(atom_types))
    for pattern, values in records:
        for candidate in (atom_types, reverse):
            if len(pattern) == len(candidate) and all(
                expected.upper() in {"X", actual.upper()}
                for expected, actual in zip(pattern, candidate)
            ):
                matches.append((sum(value.upper() != "X" for value in pattern), values))
                break
    if not matches:
        raise ModuleConfigError(
            "CGenFF stream lacks a bonded parameter for atom types " + "-".join(atom_types)
        )
    best = max(score for score, _values in matches)
    selected = [values for score, values in matches if score == best]
    return selected if multiple else selected[:1]


def _bundled_atom_types(force_field: str) -> set[str]:
    base = Path(__file__).resolve().parents[2] / "data" / "forcefields"
    directory = next(
        (item for item in (base / force_field, base / f"{force_field}.ff") if item.is_dir()),
        None,
    )
    if directory is None:
        return set()
    path = directory / "ffnonbonded.itp"
    if not path.is_file():
        return set()
    section = ""
    names: set[str] = set()
    for raw in path.read_text(errors="replace").splitlines():
        line = _clean(raw)
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section == "atomtypes" and line and not line.startswith("#"):
            names.add(line.split()[0])
    return names


def prepare_cgenff_molecule(
    ligand_name: str,
    mol2_path: str | Path,
    stream_path: str | Path,
    force_field: str,
    output_dir: str | Path,
) -> CGenFFTemplate:
    """Validate and convert one ParamChem package to explicit GROMACS ITPs."""
    name = ligand_name.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_]{1,8}", name):
        raise ModuleConfigError(f"Unsupported ligand residue name for CGenFF: {ligand_name!r}")
    mol2 = Path(mol2_path)
    stream = Path(stream_path)
    if not mol2.is_file() or not stream.is_file():
        raise ModuleConfigError(f"Both MOL2 and STR files are required for CGenFF molecule {name}")
    atom_names, elements, coordinates = _parse_mol2(mol2)
    text = stream.read_text(errors="replace")
    version_match = re.search(
        r"CGenFF(?:\s+program)?\s+version\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I
    )
    version = version_match.group(1) if version_match else None
    expected_version = get_force_field_profile(force_field).cgenff_version
    if expected_version and not version:
        raise ModuleConfigError(
            "CGenFF STR does not declare its program version; the package cannot "
            f"be verified against {force_field} (CGenFF {expected_version})"
        )
    if version and expected_version and version != expected_version:
        raise ModuleConfigError(
            f"CGenFF stream version {version} is incompatible with {force_field} "
            f"(expected CGenFF {expected_version})"
        )

    masses: dict[str, float] = {}
    residue_name = None
    residue_charge = None
    atoms: list[tuple[str, str, float]] = []
    bonds: list[tuple[str, str]] = []
    impropers: list[tuple[str, str, str, str]] = []
    bond_params: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    angle_params: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    dihedral_params: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    improper_params: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    nonbonded: dict[str, tuple[float, float]] = {}
    nbfix: list[tuple[str, str, float, float]] = []
    mode = ""
    section = ""

    for raw in text.splitlines():
        upper_raw = raw.strip().upper()
        if upper_raw.startswith("READ RTF"):
            mode, section = "rtf", ""
            continue
        if upper_raw.startswith("READ PARA"):
            mode, section = "param", ""
            continue
        line = _clean(raw)
        if not line or line.startswith("*") or line.startswith("-"):
            continue
        fields = line.split()
        keyword = fields[0].upper()
        if keyword == "MASS" and len(fields) >= 4:
            masses[fields[2]] = float(fields[3])
            continue
        if mode == "rtf":
            if keyword == "RESI" and len(fields) >= 3:
                residue_name, residue_charge = fields[1].upper(), float(fields[2])
            elif keyword == "ATOM" and len(fields) >= 4:
                atoms.append((fields[1], fields[2], float(fields[3])))
            elif keyword in {"BOND", "DOUBLE"}:
                values = fields[1:]
                if len(values) % 2:
                    raise ModuleConfigError(f"Malformed CGenFF BOND record: {line}")
                bonds.extend((values[i], values[i + 1]) for i in range(0, len(values), 2))
            elif keyword in {"IMPR", "IMPH"}:
                values = fields[1:]
                if len(values) % 4:
                    raise ModuleConfigError(f"Malformed CGenFF improper record: {line}")
                impropers.extend(tuple(values[i : i + 4]) for i in range(0, len(values), 4))
            elif keyword in {"LONEPAIR", "ANISOTROPY", "DRUDE"}:
                raise ModuleConfigError(
                    f"CGenFF directive {keyword} requires a virtual-site/polarizable "
                    "conversion that GMXBUILDER does not currently implement"
                )
            continue
        if mode != "param":
            continue
        if keyword in {
            "ATOMS",
            "BONDS",
            "ANGLES",
            "DIHEDRALS",
            "IMPROPERS",
            "NONBONDED",
            "NBFIX",
            "CMAP",
        }:
            section = keyword
            if keyword == "NONBONDED" and len(fields) > 1:
                continue
            continue
        try:
            if section == "BONDS" and len(fields) >= 4:
                bond_params.append(((fields[0], fields[1]), (float(fields[2]), float(fields[3]))))
            elif section == "ANGLES" and len(fields) >= 5:
                values = tuple(float(item) for item in fields[3:7])
                angle_params.append(((fields[0], fields[1], fields[2]), values))
            elif section == "DIHEDRALS" and len(fields) >= 7:
                dihedral_params.append(
                    (tuple(fields[:4]), (float(fields[4]), float(fields[5]), float(fields[6])))
                )
            elif section == "IMPROPERS" and len(fields) >= 6:
                improper_params.append((tuple(fields[:4]), (float(fields[4]), float(fields[5]))))
            elif section == "NONBONDED" and len(fields) >= 4 and fields[0] in masses:
                nonbonded[fields[0]] = (float(fields[2]), float(fields[3]))
            elif section == "NBFIX" and len(fields) >= 4:
                nbfix.append((fields[0], fields[1], float(fields[2]), float(fields[3])))
            elif section == "CMAP":
                raise ModuleConfigError(
                    "CGenFF CMAP parameters are not supported for uploaded small molecules"
                )
        except ValueError as exc:
            raise ModuleConfigError(f"Malformed CGenFF parameter record: {line}") from exc

    if residue_name is None or residue_charge is None or not atoms:
        raise ModuleConfigError("CGenFF STR file does not contain one complete RESI topology")
    if residue_name != name:
        raise ModuleConfigError(
            f"CGenFF STR residue is {residue_name}, but the retained molecule is {name}; "
            "use the same residue name on the ParamChem website"
        )
    stream_names = [item[0] for item in atoms]
    if stream_names != atom_names:
        raise ModuleConfigError(
            "CGenFF MOL2 and STR atom names/order do not match exactly; upload the "
            "MOL2 file used to generate this ParamChem STR file"
        )
    atom_index = {atom: index + 1 for index, atom in enumerate(atom_names)}
    if any(left not in atom_index or right not in atom_index for left, right in bonds):
        raise ModuleConfigError("CGenFF STR bond table references an unknown atom")
    charge_sum = sum(item[2] for item in atoms)
    if abs(charge_sum - residue_charge) > 1e-4:
        raise ModuleConfigError(
            f"CGenFF atom charges sum to {charge_sum:+.6f}, not RESI charge {residue_charge:+.6f}"
        )
    rounded_charge = round(residue_charge)
    if abs(residue_charge - rounded_charge) > 1e-6:
        raise ModuleConfigError(f"CGenFF RESI charge must be integral, got {residue_charge:+.6f}")

    type_by_atom = {atom: atom_type for atom, atom_type, _charge in atoms}
    graph_bonds = [(atom_index[left], atom_index[right]) for left, right in bonds]
    from gmxbuilder.io.top import TopologyWriter

    angles, dihedrals, pairs = TopologyWriter._generate_graph_terms(graph_bonds)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomtypes_path = output / f"{name}_cgenff_atomtypes.itp"
    itp_path = output / f"{name}_cgenff.itp"
    bundled = _bundled_atom_types(force_field)
    used_types = {item[1] for item in atoms}
    missing_types = sorted(used_types - bundled)
    atomtype_lines = [
        "; Imported CGenFF atom types",
        "[ atomtypes ]",
        "; name bond_type at.num mass charge ptype sigma epsilon",
    ]
    element_by_type: dict[str, str] = {}
    for (_atom, atom_type, _charge), element in zip(atoms, elements):
        element_by_type.setdefault(atom_type, element)
    for atom_type in missing_types:
        if atom_type not in masses or atom_type not in nonbonded:
            raise ModuleConfigError(
                f"CGenFF type {atom_type} is absent from {force_field} and lacks MASS/NONBONDED data"
            )
        epsilon_kcal, rmin_half_a = nonbonded[atom_type]
        sigma_nm = (2.0 * abs(rmin_half_a)) * 0.1 / (2.0 ** (1.0 / 6.0))
        epsilon_kj = abs(epsilon_kcal) * 4.184
        element = element_by_type[atom_type]
        atomtype_lines.append(
            f"{atom_type:<10s} {atom_type:<10s} {_ATOMIC_NUMBERS[element]:3d} "
            f"{masses[atom_type]:10.5f} 0.0 A {sigma_nm:.10e} {epsilon_kj:.10e}"
        )
    if nbfix:
        atomtype_lines.extend(["", "[ nonbond_params ]", "; i j funct sigma epsilon"])
        for left, right, epsilon_kcal, rmin_a in nbfix:
            sigma_nm = abs(rmin_a) * 0.1 / (2.0 ** (1.0 / 6.0))
            atomtype_lines.append(
                f"{left:<10s} {right:<10s} 1 {sigma_nm:.10e} {abs(epsilon_kcal) * 4.184:.10e}"
            )
    atomtypes_path.write_text("\n".join(atomtype_lines) + "\n")

    lines = [
        f"; {name} imported from ParamChem/CGenFF",
        "[ moleculetype ]",
        f"{name} 3",
        "",
        "[ atoms ]",
        "; nr type resnr residue atom cgnr charge mass",
    ]
    for index, ((atom, atom_type, charge), element) in enumerate(zip(atoms, elements), 1):
        mass = masses.get(atom_type)
        if mass is None:
            raise ModuleConfigError(f"CGenFF stream lacks MASS for type {atom_type}")
        lines.append(
            f"{index:6d} {atom_type:>10s} 1 {name:>8s} {atom:>8s} {index:6d} "
            f"{charge:12.7f} {mass:10.5f}"
        )
    lines.extend(["", "[ bonds ]", "; ai aj funct b0(nm) kb(kJ mol^-1 nm^-2)"])
    for left, right in graph_bonds:
        types = (type_by_atom[atom_names[left - 1]], type_by_atom[atom_names[right - 1]])
        k_charmm, r0_a = _parameter_match(bond_params, types)[0]
        lines.append(f"{left:6d} {right:6d} 1 {r0_a * 0.1:.7f} {k_charmm * 836.8:.5f}")
    if angles:
        lines.extend(["", "[ angles ]", "; ai aj ak funct theta ktheta [r13 kub]"])
        for left, center, right in angles:
            types = tuple(type_by_atom[atom_names[index - 1]] for index in (left, center, right))
            values = _parameter_match(angle_params, types)[0]
            k_angle, theta = values[:2]
            if len(values) >= 4:
                kub, s0_a = values[2:4]
                lines.append(
                    f"{left:6d} {center:6d} {right:6d} 5 {theta:.5f} "
                    f"{k_angle * 8.368:.5f} {s0_a * 0.1:.7f} {kub * 836.8:.5f}"
                )
            else:
                lines.append(
                    f"{left:6d} {center:6d} {right:6d} 1 {theta:.5f} {k_angle * 8.368:.5f}"
                )
    if dihedrals:
        lines.extend(["", "[ dihedrals ]", "; ai aj ak al funct phase k multiplicity"])
        for indices in dihedrals:
            types = tuple(type_by_atom[atom_names[index - 1]] for index in indices)
            for force_kcal, multiplicity, phase in _parameter_match(
                dihedral_params, types, multiple=True
            ):
                lines.append(
                    f"{indices[0]:6d} {indices[1]:6d} {indices[2]:6d} {indices[3]:6d} "
                    f"9 {phase:.5f} {force_kcal * 4.184:.7f} {int(multiplicity):d}"
                )
    if pairs:
        lines.extend(["", "[ pairs ]", "; ai aj funct; graph-derived 1-4 pairs"])
        lines.extend(f"{left:6d} {right:6d} 1" for left, right in pairs)
    if impropers:
        lines.extend(["", "[ dihedrals ]", "; impropers: ai aj ak al funct phi0 k"])
        for names in impropers:
            indices = tuple(atom_index[item] for item in names)
            types = tuple(type_by_atom[item] for item in names)
            force_kcal, phase = _parameter_match(improper_params, types)[0]
            lines.append(
                f"{indices[0]:6d} {indices[1]:6d} {indices[2]:6d} {indices[3]:6d} "
                f"2 {phase:.5f} {force_kcal * 8.368:.7f}"
            )
    itp_path.write_text("\n".join(lines) + "\n")

    penalty_values = [
        float(item) for item in re.findall(r"penalty\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    ]
    maximum_penalty = max(penalty_values) if penalty_values else None
    if not np.isfinite(coordinates).all() or not math.isfinite(charge_sum):
        raise ModuleConfigError("CGenFF package contains non-finite numeric data")
    return CGenFFTemplate(
        name=name,
        atom_names=tuple(atom_names),
        elements=tuple(elements),
        coordinates=coordinates,
        net_charge=int(rounded_charge),
        itp_path=itp_path,
        atomtypes_path=atomtypes_path,
        cgenff_version=version,
        maximum_penalty=maximum_penalty,
    )
