"""PDB file reader and writer.

Handles standard PDB format (ATOM, HETATM, CRYST1, REMARK, TER, END, CONECT).
Coordinates are converted from Angstroms (PDB) to nanometers (internal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.exceptions import ParseError

# PDB column specifications (1-indexed)
# ATOM/HETATM format:
#  1-6   record name
#  7-11  serial
# 13-16  atom name
# 17     altLoc
# 18-20  resname
# 22     chainID
# 23-26  resSeq
# 27     iCode
# 31-38  x (Angstrom)
# 39-46  y (Angstrom)
# 47-54  z (Angstrom)
# 55-60  occupancy
# 61-66  tempFactor
# 77-78  element
# 79-80  charge

_ANGSTROM_TO_NM = 0.1


def format_pdb_atom_name(atom_name: str, element: str) -> str:
    """Return a standards-compliant four-column PDB atom-name field.

    One-letter elements use a leading space (for example ``" CA "``);
    two-letter elements and digit-prefixed hydrogen names are left aligned.
    Chemistry tools such as PROPKA use this alignment when inferring elements
    and the covalent graph.
    """
    name = (atom_name or "ATOM")[:4].strip()
    elem = (element or "").strip()
    if len(name) < 4 and len(elem) == 1 and not name[:1].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def _infer_element_from_atom_field(atom_field: str) -> str:
    """Infer an element without discarding PDB atom-name alignment.

    A leading blank or digit denotes a one-letter element (``" CA "`` is an
    alpha carbon and ``"1HG1"`` is hydrogen).  A left-aligned alphabetic name
    may denote a two-letter element such as ``"CL  "`` or ``"FE  "``.
    """
    field = str(atom_field)[:4].ljust(4)
    letters = "".join(character for character in field if character.isalpha())
    if not letters:
        return ""
    if field[0].isspace() or field[0].isdigit():
        return letters[0].upper()
    candidate = letters[:2].upper()
    return candidate if candidate in _ELEMENTS else letters[0].upper()


# Standard protein residue names (20 standard + common variants)
_PROTEIN_RESNAMES = PROTEIN_RESNAMES
# Common solvent / buffer / ion residue names
_SOLVENT_IONS = {
    "HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD",
    "NA", "CL", "K", "CA", "ZN", "MG", "CD", "BR", "I", "CS", "LI",
    "RB", "SR", "BA", "MN", "FE", "CO", "NI", "CU", "AU", "HG", "PT",
    "F", "NH4", "NO3", "PO4", "SO4", "ACT", "EDO", "GOL", "MPD",
}
# Common lipid / detergent residue names
_LIPID_DETERGENT = {
    "POPC", "DPPC", "DMPC", "DOPC", "POPE", "DOPE", "POPG", "POPS",
    "DLPC", "DSPC", "SOPC", "CHOL", "CHL1", "ERG", "LPPC", "LPPE",
    "PIP2", "PIP3", "CER", "DAG", "LPS", "LMN", "LMG", "LHG",
    "OGL", "BNG", "DMU", "LDA", "OCT", "C8E", "C10E", "C12E",
}


def _known_nucleic_resnames() -> set[str]:
    from gmxbuilder.modules.nucleic_acid.support import (
        CANONICAL_DNA_RESNAMES,
        CANONICAL_RNA_RESNAMES,
        KNOWN_MODIFIED_NUCLEOTIDES,
    )

    return set(
        CANONICAL_DNA_RESNAMES
        | CANONICAL_RNA_RESNAMES
        | KNOWN_MODIFIED_NUCLEOTIDES
    )


class PDBParser:
    """Parse PDB-format files into Structure objects."""

    def __init__(self):
        self._remarks: list[str] = []

    def parse(self, path: str | Path, **kwargs) -> Structure:
        """Read a PDB file and return a Structure.

        Parameters
        ----------
        path : str or Path
            Path to the PDB file.
        **kwargs
            model_index: int = 1 — which MODEL to read (1-indexed).

        Returns
        -------
        Structure
        """
        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")

        model_index = kwargs.get("model_index", 1)
        if not isinstance(model_index, int) or model_index < 1:
            raise ParseError("model_index must be a positive integer")
        return self._parse_file(path, model_index)

    def parse_remarks(self, path: str | Path) -> list[str]:
        """Return only the REMARK lines from a PDB file."""
        path = Path(path)
        lines = []
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            for raw in fh:
                if raw.startswith("REMARK"):
                    lines.append(raw.rstrip("\n"))
        return lines

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_file(self, path: Path, model_index: int) -> Structure:
        atoms: list[dict] = []
        box = np.eye(3) * 10.0  # default 10 nm box
        model_ordinal = 0
        saw_model = False
        active_model = model_index == 1
        self._remarks = []
        connectivity: list[tuple[int, int]] = []

        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            for line_number, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")
                if not line:
                    continue

                record = line[:6].strip()

                if record == "MODEL":
                    saw_model = True
                    model_ordinal += 1
                    active_model = model_ordinal == model_index
                elif record == "REMARK":
                    self._remarks.append(line)
                    # Try to parse CRYST1 from REMARK 290 (common in membrane-protein PDB files)
                elif record == "CRYST1":
                    box = self._parse_cryst1(line)
                elif record in ("ATOM", "HETATM"):
                    if (not saw_model and model_index == 1) or active_model:
                        atoms.append(self._parse_atom_line(line, record, line_number))
                elif record == "ENDMDL":
                    if active_model and atoms:
                        break
                    active_model = False
                elif record == "TER":
                    pass  # Chain terminator — could track chains here
                elif record == "CONECT":
                    # Store connectivity hints
                    entries = line[6:].split()
                    if len(entries) >= 2:
                        try:
                            src = int(entries[0])
                            for dst_str in entries[1:]:
                                connectivity.append((src, int(dst_str)))
                        except ValueError:
                            # Hybrid-36 identifiers are valid in large PDBs,
                            # but connectivity is only a non-authoritative hint.
                            pass
                elif record in ("END", "MASTER"):
                    break

        if not atoms:
            raise ParseError(f"No atoms found in PDB file: {path}")

        insertion = next((atom for atom in atoms if atom["icode"]), None)
        if insertion is not None:
            raise ParseError(
                "PDB insertion codes are not yet representable in the integer residue "
                f"model (chain {insertion['chain'] or '?'} residue "
                f"{insertion['resid']}{insertion['icode']}); renumber residues uniquely "
                "before upload"
            )

        # Resolve explicit alternate locations before constructing Structure.
        # Highest occupancy wins; blank and then A are deterministic tie-breaks.
        selected: dict[tuple[str, int, str, str], tuple[int, tuple[float, int]]] = {}
        for index, atom in enumerate(atoms):
            key = (atom["chain"], atom["resid"], atom["resname"], atom["name"])
            preference = 2 if not atom["altloc"] else 1 if atom["altloc"] == "A" else 0
            rank = (float(atom["occupancy"]), preference)
            previous = selected.get(key)
            if previous is None or rank > previous[1]:
                selected[key] = (index, rank)
        keep_indices = sorted(index for index, _rank in selected.values())
        atoms = [atoms[index] for index in keep_indices]

        n = len(atoms)
        coords = np.zeros((n, 3), dtype=np.float64)
        atom_names = [""] * n
        resnames = [""] * n
        resids = [0] * n
        chain_ids = [""] * n
        segids = [""] * n
        elements = [""] * n
        occupancies = [1.0] * n
        tempfactors = [0.0] * n

        for i, a in enumerate(atoms):
            coords[i] = a["xyz"]
            atom_names[i] = a["name"]
            resnames[i] = a["resname"]
            resids[i] = a["resid"]
            chain_ids[i] = a["chain"]
            segids[i] = a["segid"]
            elements[i] = a["element"]
            occupancies[i] = a["occupancy"]
            tempfactors[i] = a["tempfactor"]

        # Estimate box from coordinates when CRYST1 is absent OR physically
        # unreasonable (e.g. placeholder 1.0 Å cell in some CIF→PDB conversions).
        dims = np.sqrt((box ** 2).sum(axis=1))
        if np.allclose(box, np.eye(3) * 10.0) or np.any(dims < 1.0) or np.any(dims > 1000.0):
            cmin = coords.min(axis=0)
            cmax = coords.max(axis=0)
            extent = cmax - cmin
            # Use the max extent + 30% padding, minimum 3 nm
            box_size = max(extent.max() * 1.3, 3.0)
            # Build a cubic box
            box = np.eye(3) * box_size

        return Structure(
            coordinates=coords,
            box_vectors=box,
            atom_names=atom_names,
            resnames=resnames,
            resids=resids,
            chain_ids=chain_ids,
            segids=segids,
            elements=elements,
            occupancies=occupancies,
            tempfactors=tempfactors,
        )

    # ------------------------------------------------------------------
    # Line parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cryst1(line: str) -> np.ndarray:
        """Parse CRYST1 record into a (3,3) box matrix in nm."""
        try:
            a = float(line[6:15]) * _ANGSTROM_TO_NM
            b = float(line[15:24]) * _ANGSTROM_TO_NM
            c = float(line[24:33]) * _ANGSTROM_TO_NM
            alpha = float(line[33:40])
            beta = float(line[40:47])
            gamma = float(line[47:54])

            if abs(alpha - 90) < 1e-6 and abs(beta - 90) < 1e-6 and abs(gamma - 90) < 1e-6:
                return np.diag([a, b, c])

            # General triclinic box (GROMACS convention, angles in degrees)
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)

            cos_alpha = np.cos(alpha_rad)
            cos_beta = np.cos(beta_rad)
            cos_gamma = np.cos(gamma_rad)
            sin_gamma = np.sin(gamma_rad)

            box = np.zeros((3, 3))
            box[0, 0] = a
            box[1, 0] = b * cos_gamma
            box[1, 1] = b * sin_gamma
            box[2, 0] = c * cos_beta
            box[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
            box[2, 2] = np.sqrt(c * c - box[2, 0] ** 2 - box[2, 1] ** 2)
            return box

        except (ValueError, IndexError):
            return np.eye(3) * 10.0

    @staticmethod
    def _parse_atom_line(line: str, record: str, line_number: int | None = None) -> dict:
        """Parse ATOM or HETATM line into a dictionary."""
        try:
            if len(line) < 54:
                raise ValueError("atom record is shorter than 54 columns")
            serial_text = line[6:11].strip()
            try:
                serial = int(serial_text)
            except ValueError:
                # Atom serials are not used as coordinate-array identifiers.
                # Accept hybrid-36/overflow labels instead of rejecting an
                # otherwise valid large structure.
                serial = 0
            atom_field = line[12:16]
            name = atom_field.strip()
            altloc = line[16:17].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip() if len(line) > 21 else ""
            resid_str = line[22:26].strip()
            resid = int(resid_str) if resid_str else 0
            icode = line[26:27].strip() if len(line) > 26 else ""

            x = float(line[30:38].strip()) * _ANGSTROM_TO_NM
            y = float(line[38:46].strip()) * _ANGSTROM_TO_NM
            z = float(line[46:54].strip()) * _ANGSTROM_TO_NM
            if not np.isfinite([x, y, z]).all():
                raise ValueError("coordinates must be finite numbers")

            occupancy = 1.0
            if len(line) >= 60:
                occ_str = line[54:60].strip()
                if occ_str:
                    occupancy = float(occ_str)

            tempfactor = 0.0
            if len(line) >= 66:
                tf_str = line[60:66].strip()
                if tf_str:
                    tempfactor = float(tf_str)

            # Element and segid from the right side of the atom name
            element = ""
            segid = ""
            if len(line) >= 78:
                element = line[76:78].strip()
            if len(line) >= 80:
                segid = line[72:76].strip()

            # Fallback: derive element from atom name
            if not element and name:
                element = _infer_element_from_atom_field(atom_field)

            return {
                "serial": serial,
                "name": name,
                "altloc": altloc,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "icode": icode,
                "xyz": np.array([x, y, z], dtype=np.float64),
                "occupancy": occupancy,
                "tempfactor": tempfactor,
                "element": element,
                "segid": segid,
                "record": record,
            }
        except (ValueError, IndexError) as exc:
            location = f" at line {line_number}" if line_number is not None else ""
            raise ParseError(f"Malformed PDB atom record{location}: {line[:60]}...") from exc


# Set of standard element symbols for validation
_ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "BR", "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU",
    "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE",
    "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD",
    "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",
    "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO",
    "AT", "RN", "FR", "RA", "AC", "TH", "PA", "U", "NP", "PU",
    "AM", "CM", "BK", "CF", "ES", "FM", "MD", "NO", "LR", "RF",
    "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN", "NH", "FL",
    "MC", "LV", "TS", "OG",
}


class PDBWriter:
    """Write a Structure to PDB format."""

    @staticmethod
    def write(
        structure: Structure,
        path: str | Path,
        title: str = "",
        *,
        wrap_ids_for_viewer: bool = False,
    ) -> None:
        """Write structure to a PDB file.

        Parameters
        ----------
        structure : Structure
        path : str or Path
        title : str
            COMPND record text.
        wrap_ids_for_viewer : bool
            Keep atom/residue identifiers inside fixed-width PDB fields for
            systems beyond the legacy PDB limits. Intended only for visual
            viewers; simulation exports use GRO and checkpoint metadata.
        """
        path = Path(path)
        coords = structure.coordinates
        box = structure.box_vectors
        dims = np.sqrt((box ** 2).sum(axis=1))
        if not wrap_ids_for_viewer:
            invalid_resids = [
                int(resid) for resid in structure.resids
                if int(resid) < -999 or int(resid) > 9999
            ]
            if structure.num_atoms > 99999 or invalid_resids:
                raise ValueError(
                    "Structure exceeds fixed-width PDB atom/residue identifier limits; "
                    "use GRO/mmCIF for simulation data or explicit viewer wrapping"
                )

        with open(path, "w") as fh:
            # Title
            if title:
                fh.write(f"HEADER    {title[:50]}\n")
                fh.write(f"COMPND    {title[:50]}\n")

            # REMARK
            fh.write("REMARK    Generated by GMXBUILDER\n")
            if wrap_ids_for_viewer and structure.num_atoms > 99999:
                fh.write(
                    "REMARK    Atom/residue identifiers wrap at PDB field limits; "
                    "coordinates and ordering remain exact\n"
                )

            # CRYST1 — assume orthorhombic for simplicity.  nm → Å conversion
            fh.write(
                f"CRYST1{dims[0] * 10:9.3f}{dims[1] * 10:9.3f}"
                f"{dims[2] * 10:9.3f}  90.00  90.00  90.00 P 1           1\n"
            )

            for i in range(structure.num_atoms):
                x, y, z = coords[i] / _ANGSTROM_TO_NM  # back to Angstrom
                atom_name = structure.atom_names[i][:4] if structure.atom_names[i] else "ATOM"
                resname = structure.resnames[i][:3] if structure.resnames[i] else "UNK"
                resid = structure.resids[i] if i < len(structure.resids) else i + 1
                serial = i + 1
                if wrap_ids_for_viewer:
                    serial = (i % 99999) + 1
                    resid = ((int(resid) - 1) % 9999) + 1
                chain = structure.chain_ids[i] if i < len(structure.chain_ids) else " "
                element = structure.elements[i] if i < len(structure.elements) else "C"
                occupancy = structure.occupancies[i] if i < len(structure.occupancies) else 1.0
                tempfactor = structure.tempfactors[i] if i < len(structure.tempfactors) else 0.0

                record = "HETATM" if resname in ("HOH", "SOL", "NA", "CL", "K", "CA", "ZN", "MG") else "ATOM"

                fh.write(
                    f"{record:<6}{serial:5d} {format_pdb_atom_name(atom_name, element)}"
                    f" {resname:>3s} {chain:1s}{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{tempfactor:6.2f}"
                    f"          {element:>2s}\n"
                )

            fh.write("TER\nEND\n")


# =============================================================================
# PDB validation & small-molecule detection
# =============================================================================

class PDBValidator:
    """Validate a PDB file and report issues."""

    @staticmethod
    def validate(path: str | Path) -> dict:
        """Check a PDB file for common problems.

        Returns
        -------
        dict with keys:
            valid : bool           — True if no blocking errors
            errors : list[str]    — blocking issues (cannot parse)
            warnings : list[str]  — non-blocking issues
        """
        path = Path(path)
        errors: list[str] = []
        warnings: list[str] = []

        if not path.exists():
            return {"valid": False, "errors": [f"File not found: {path}"], "warnings": []}

        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as e:
            return {"valid": False, "errors": [f"Cannot read file: {e}"], "warnings": []}

        lines = [line for line in content.split("\n") if line.strip()]
        if not lines:
            return {"valid": False, "errors": ["File is empty"], "warnings": []}

        atom_hetatm_lines = [
            line for line in lines if line[:6].strip() in ("ATOM", "HETATM")
        ]
        if not atom_hetatm_lines:
            errors.append("No ATOM or HETATM records found — file contains no atomic coordinates")

        serials_seen: set[int] = set()
        chain_ids_seen: set[str] = set()
        resnames_seen: set[str] = set()
        coord_x: list[float] = []
        coord_y: list[float] = []
        coord_z: list[float] = []
        has_cryst1 = any(line[:6].strip() == "CRYST1" for line in lines)

        for line in atom_hetatm_lines:
            try:
                serial_text = line[6:11].strip()
                try:
                    serial = int(serial_text)
                except ValueError:
                    serial = len(serials_seen) + 1
                    warnings.append(
                        f"Non-decimal atom serial {serial_text!r} was accepted as an overflow label"
                    )
                if serial in serials_seen:
                    warnings.append(f"Duplicate atom serial {serial}")
                serials_seen.add(serial)

                resname = line[17:20].strip()
                if resname:
                    resnames_seen.add(resname)

                chain = line[21:22].strip() if len(line) > 21 else ""
                if chain:
                    chain_ids_seen.add(chain)

                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                if not np.isfinite([x, y, z]).all():
                    raise ValueError("coordinates must be finite numbers")
                coord_x.append(x)
                coord_y.append(y)
                coord_z.append(z)

                occupancy = 1.0
                if len(line) >= 60:
                    occ_str = line[54:60].strip()
                    if occ_str:
                        occupancy = float(occ_str)
                if not (0.0 <= occupancy <= 1.0):
                    warnings.append(f"Atom {serial} has unusual occupancy {occupancy}")

                tempfactor = 0.0
                if len(line) >= 66:
                    tf_str = line[60:66].strip()
                    if tf_str:
                        tempfactor = float(tf_str)
                if tempfactor > 100.0:
                    warnings.append(f"Atom {serial} has high temperature factor {tempfactor:.1f}")

            except (ValueError, IndexError):
                errors.append(f"Malformed coordinate line — cannot parse atom record around serial {serial if 'serial' in dir() else '?'}")
                break

        if not coord_x:
            errors.append("No parseable coordinates found")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Coordinate range checks
        x_range = max(coord_x) - min(coord_x)
        y_range = max(coord_y) - min(coord_y)
        z_range = max(coord_z) - min(coord_z)
        max_range = max(x_range, y_range, z_range)

        if max_range > 5000.0:
            warnings.append(f"Structure spans {max_range:.0f} Å — unusually large, check coordinate units")
        if max_range < 1.0:
            warnings.append(f"Structure spans only {max_range:.1f} Å — very small, check coordinate format")

        if not has_cryst1:
            warnings.append("No CRYST1 record — box size will be estimated from coordinates")

        # Unknown residue names
        unknown_res = (
            resnames_seen - _PROTEIN_RESNAMES - _SOLVENT_IONS
            - _LIPID_DETERGENT - _known_nucleic_resnames()
        )
        if unknown_res:
            warnings.append(f"Non-standard residues detected: {', '.join(sorted(unknown_res))}")

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings}

    @staticmethod
    def detect_small_molecules(path: str | Path) -> list[dict]:
        """Identify non-protein, non-solvent, non-ion small molecules in a PDB.

        Returns a list of dicts, each describing one small-molecule instance:
            resname, chain, resid, atom_count, elements (set of element symbols)
        """
        path = Path(path)
        from gmxbuilder.modules.nucleic_acid.support import nucleic_polymer_residues

        try:
            polymer_residues = nucleic_polymer_residues(PDBParser().parse(path))
        except (ParseError, OSError, ValueError):
            polymer_residues = {}
        content = path.read_text()
        lines = [
            line
            for line in content.split("\n")
            if line[:6].strip() in ("ATOM", "HETATM")
        ]

        # Group by (resname, chain, resid)
        groups: dict[tuple[str, str, int], list[tuple[str, str]]] = {}
        for line in lines:
            try:
                resname = line[17:20].strip()
                chain = line[21:22].strip() if len(line) > 21 else ""
                resid_str = line[22:26].strip()
                resid = int(resid_str) if resid_str else 0
                key = (resname, chain, resid)
                if key not in groups:
                    groups[key] = []
                # Extract element
                element = ""
                if len(line) >= 78:
                    element = line[76:78].strip()
                if not element:
                    aname = line[12:16].strip()
                    element = aname.lstrip("0123456789 ")[:2].strip()
                    if element.upper() not in _ELEMENTS:
                        element = element[0].upper()
                    else:
                        element = element.upper()
                else:
                    aname = line[12:16].strip()
                groups[key].append((element or "?", aname))
            except (ValueError, IndexError):
                continue

        # Report every non-protein molecule except water. Crystallographic
        # ions, buffers, detergents and lipids must be visible in Step 1 so the
        # user can explicitly retain or remove them.
        water_resnames = {"HOH", "SOL", "WAT", "TIP", "TIP3", "SPC", "SPCE", "DOD"}
        small_mols = []
        for (resname, chain, resid), atoms in sorted(groups.items()):
            if resname in _PROTEIN_RESNAMES or resname in water_resnames:
                continue
            if (chain, resid) in polymer_residues:
                continue
            # Count element frequencies
            from collections import Counter
            elements = [element for element, _atom_name in atoms]
            elem_counts = Counter(e for e in elements if e != "?")
            small_mols.append({
                "resname": resname,
                "chain": chain or " ",
                "resid": resid,
                "atom_count": len(elements),
                "formula": "".join(f"{el}{c if c > 1 else ''}" for el, c in sorted(elem_counts.items())) if elem_counts else "?",
                "category": (
                    "lipid_or_detergent" if resname in _LIPID_DETERGENT
                    else "ion_or_solvent_additive" if resname in _SOLVENT_IONS
                    else "small_molecule"
                ),
            })

        return small_mols
