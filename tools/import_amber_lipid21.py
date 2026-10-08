#!/usr/bin/env python3
"""Regenerate bundled GROMACS Lipid21 assets from an AmberTools install.

Run with the Python interpreter from an environment containing ParmEd.  The
result is deterministic text data that can be audited and shipped without a
runtime AmberTools dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_sequence
from gmxbuilder.modules.membrane.lipids import LipidRegistry


def _mapped_name(name: str, residue_index: int, lipid_name: str) -> str:
    if lipid_name in {"PSM", "SSM"}:
        if residue_index == 0:
            carbon = re.fullmatch(r"C1(\d+)", name)
            hydrogen = re.fullmatch(r"H(\d+)([RST])", name)
            if carbon:
                return f"C{carbon.group(1)}F"
            if hydrogen:
                suffix = {"R": "F", "S": "G", "T": "H"}[hydrogen.group(2)]
                return f"H{hydrogen.group(1)}{suffix}"
        elif residue_index == 2:
            carbon = re.fullmatch(r"C1(\d+)", name)
            hydrogen = re.fullmatch(r"H(\d+)([RST])", name)
            if carbon:
                return f"C{int(carbon.group(1)) + 2}S"
            if hydrogen:
                suffix = {"R": "S", "S": "T", "T": "U"}[hydrogen.group(2)]
                return f"H{int(hydrogen.group(1)) + 2}{suffix}"
        return name
    if lipid_name == "CHOL":
        return name
    if residue_index == 0:
        carbon = re.fullmatch(r"C1(\d+)", name)
        hydrogen = re.fullmatch(r"H(\d+)([RST])", name)
        if carbon:
            return f"C{carbon.group(1)}X"
        if hydrogen:
            suffix = {"R": "X", "S": "Y", "T": "Z"}[hydrogen.group(2)]
            return f"H{hydrogen.group(1)}{suffix}"
    elif residue_index == 2:
        carbon = re.fullmatch(r"C1(\d+)", name)
        hydrogen = re.fullmatch(r"H(\d+)([RST])", name)
        if carbon:
            return f"C{carbon.group(1)}Y"
        if hydrogen:
            suffix = {"R": "U", "S": "V", "T": "W"}[hydrogen.group(2)]
            return f"H{hydrogen.group(1)}{suffix}"
    return name


def _sections(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current = "preamble"
    for line in text.splitlines():
        match = re.fullmatch(r"\s*\[\s*([^]]+)\s*]\s*", line)
        if match:
            current = match.group(1).strip().lower()
            result.setdefault(current, []).append(line)
        else:
            result.setdefault(current, []).append(line)
    return result


def _explicit_pairs(structure) -> list[str]:
    """Preserve Amber's per-pair LJ and charge scaling independently of defaults.

    Read the source A/B coefficients rather than recombining atom types. Only
    active proper-dihedral endpoints contribute 1-4 interactions; repeated
    endpoints must agree. GROMACS pair function 2 carries its own charge scale.
    """
    data = structure.parm_data
    ntypes = structure.ptr("NTYPES")
    pairs: dict[tuple[int, int], tuple[float, ...]] = {}
    for dihedral in structure.dihedrals:
        if dihedral.improper or dihedral.ignore_end:
            continue
        first, last = sorted((dihedral.atom1, dihedral.atom4), key=lambda atom: atom.idx)
        scnb, scee = float(dihedral.type.scnb), float(dihedral.type.scee)
        if not all(math.isfinite(value) and value > 0 for value in (scnb, scee)):
            raise ValueError("Invalid Amber 1-4 scaling")
        offset = ntypes * (first.nb_idx - 1) + last.nb_idx - 1
        coefficient = int(data["NONBONDED_PARM_INDEX"][offset]) - 1
        if coefficient < 0:
            raise ValueError("Unsupported Amber non-LJ 1-4 interaction")
        c12 = float(data["LENNARD_JONES_ACOEF"][coefficient]) * 4.184e-12 / scnb
        c6 = float(data["LENNARD_JONES_BCOEF"][coefficient]) * 4.184e-6 / scnb
        if c12 == c6 == 0:
            sigma = epsilon = 0.0
        elif c12 > 0 and c6 > 0:
            sigma = (c12 / c6) ** (1 / 6)
            epsilon = c6 * c6 / (4 * c12)
        else:
            raise ValueError("Unsupported Amber 1-4 Lennard-Jones coefficients")
        key = first.idx + 1, last.idx + 1
        values = (1 / scee, float(first.charge), float(last.charge), sigma, epsilon)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite Amber 1-4 parameters")
        if key in pairs and not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14) for a, b in zip(pairs[key], values)
        ):
            raise ValueError(f"Conflicting Amber 1-4 definitions for {key}")
        pairs[key] = values
    return [
        f"{first} {last} 2 " + " ".join(f"{value:.14g}" for value in values)
        for (first, last), values in sorted(pairs.items())
    ]


def _rewrite_itp(
    text: str, lipid_name: str, names: list[str], pair_parameters: list[str]
) -> tuple[str, dict[str, str]]:
    sections = _sections(text)
    atomtypes: dict[str, str] = {}
    for line in sections["atomtypes"][1:]:
        raw = line.split(";", 1)[0].split()
        if not raw:
            continue
        old = raw[0]
        raw[0] = f"L21_{old}"
        atomtypes[old] = " ".join(raw)

    atoms = sections["atoms"]
    atom_number = 0
    rewritten_atoms = [atoms[0], "; nr type resnr residue atom cgnr charge mass"]
    for line in atoms[1:]:
        raw = line.split(";", 1)[0].split()
        if len(raw) < 8 or not raw[0].isdigit():
            continue
        atom_number += 1
        raw[1] = f"L21_{raw[1]}"
        raw[2] = "1"
        raw[3] = lipid_name
        raw[4] = names[atom_number - 1]
        rewritten_atoms.append(" ".join(raw))
    if atom_number != len(names):
        raise RuntimeError(f"{lipid_name}: topology/coordinate atom-count mismatch")
    sections["atoms"] = rewritten_atoms
    exported_pairs = {
        tuple(sorted(map(int, fields[:2])))
        for line in sections.get("pairs", [])[1:]
        if (fields := line.split(";", 1)[0].split()) and fields[0].isdigit()
    }
    source_pairs = {tuple(map(int, line.split()[:2])) for line in pair_parameters}
    if exported_pairs != source_pairs:
        raise RuntimeError(f"{lipid_name}: converted 1-4 endpoints differ from Amber")
    sections["pairs"] = [
        "[ pairs ]",
        "; ai aj funct fudgeQQ qi qj sigma epsilon; source-specific LJ and charge scaling",
        *pair_parameters,
    ]
    sections["moleculetype"] = [
        "[ moleculetype ]",
        "; Name nrexcl",
        f"{lipid_name} 3",
    ]
    order = [
        "moleculetype",
        "atoms",
        "bonds",
        "pairs",
        "angles",
        "dihedrals",
        "cmap",
        "constraints",
        "settles",
        "exclusions",
        "position_restraints",
    ]
    output = [
        "; Exact Amber Lipid21 v1.0 topology converted by ParmEd",
        "; Explicit [pairs] preserve Lipid21-specific 1-4 scaling.",
        "; Conversion contract: amber-explicit-pairs-v2 (independent of parent defaults).",
    ]
    for section in order:
        if section in sections:
            output.extend(["", *sections[section]])
    return "\n".join(output).rstrip() + "\n", atomtypes


def main() -> None:
    import parmed

    parser = argparse.ArgumentParser()
    parser.add_argument("--tleap", default=shutil.which("tleap"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/gmxbuilder/data/lipid21",
    )
    args = parser.parse_args()
    if not args.tleap:
        raise SystemExit("tleap was not found; pass --tleap from an AmberTools installation")

    supported = [name for name in LipidRegistry.list() if lipid21_sequence(name)]
    templates = {}
    all_atomtypes: dict[str, str] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "itp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-lipid21-") as temp:
        work = Path(temp)
        for name in sorted(supported):
            sequence = " ".join(lipid21_sequence(name) or ())
            leap_input = work / "leap.in"
            leap_input.write_text(
                "source leaprc.lipid21\n"
                f"M = sequence {{ {sequence} }}\n"
                "saveAmberParm M molecule.prmtop molecule.inpcrd\nquit\n"
            )
            result = subprocess.run(
                [args.tleap, "-f", str(leap_input)],
                cwd=work,
                text=True,
                capture_output=True,
            )
            if result.returncode:
                raise RuntimeError(f"tleap failed for {name}:\n{result.stdout}\n{result.stderr}")
            structure = parmed.load_file(
                str(work / "molecule.prmtop"), str(work / "molecule.inpcrd")
            )
            residue_indices = {
                id(residue): index for index, residue in enumerate(structure.residues)
            }
            names = [
                _mapped_name(atom.name, residue_indices[id(atom.residue)], name)
                for atom in structure.atoms
            ]
            if len(names) != len(set(names)):
                duplicates = sorted({value for value in names if names.count(value) > 1})
                raise RuntimeError(f"{name}: mapped atom names are not unique: {duplicates}")
            total_charge = sum(float(atom.charge) for atom in structure.atoms)
            expected_charge = LipidRegistry.get(name).charge
            if abs(total_charge - expected_charge) > 1e-3:
                raise RuntimeError(
                    f"{name}: Lipid21 charge {total_charge:.6f} != registry {expected_charge}"
                )
            gromacs_top = work / "molecule.top"
            structure.save(str(gromacs_top), format="gromacs", overwrite=True)
            itp, atomtypes = _rewrite_itp(
                gromacs_top.read_text(), name, names, _explicit_pairs(structure)
            )
            conflict = {
                key
                for key, value in atomtypes.items()
                if key in all_atomtypes and all_atomtypes[key] != value
            }
            if conflict:
                raise RuntimeError(f"{name}: inconsistent Lipid21 atom types: {sorted(conflict)}")
            all_atomtypes.update(atomtypes)
            (args.output / "itp" / f"{name}.itp").write_text(itp)
            coordinates = structure.coordinates / 10.0
            coordinates -= coordinates.mean(axis=0)
            templates[name] = {
                "atom_names": names,
                "coordinates_nm": coordinates.round(7).tolist(),
                "charge": expected_charge,
                "source_modules": list(lipid21_sequence(name) or ()),
            }
    atomtype_lines = [
        "; Amber Lipid21 v1.0 non-bonded types (namespaced for safe composition)",
        "[ atomtypes ]",
        "; name at.num mass charge ptype sigma epsilon",
    ]
    atomtype_lines.extend(all_atomtypes[key] for key in sorted(all_atomtypes))
    (args.output / "lipid21_atomtypes.itp").write_text("\n".join(atomtype_lines) + "\n")
    (args.output / "templates.json").write_text(
        json.dumps(templates, indent=2, sort_keys=True) + "\n"
    )
    print(f"Generated {len(templates)} exact Lipid21 templates in {args.output}")


if __name__ == "__main__":
    main()
