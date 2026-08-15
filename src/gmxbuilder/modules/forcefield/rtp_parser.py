"""Minimal .rtp (Residue Topology Parameter) file parser.

Reads CHARMM36 merged.rtp format and provides per-residue atom types,
charges, and bonded connectivity.
"""

from __future__ import annotations

import copy
from pathlib import Path


class RTPParser:
    """Parse GROMACS .rtp residue topology files."""

    def __init__(self, path: str | Path | None = None):
        self._residues: dict[str, dict] = {}
        if path:
            self.parse(path)

    def parse(self, path: str | Path) -> None:
        """Parse a .rtp file into residue templates.

        Supports CHARMM36, Amber, and OPLS-AA RTP formats.
        """
        path = Path(path)
        if not path.exists():
            return

        current_res = None
        current_section = None

        with open(path) as fh:
            for line in fh:
                # Strip inline comments (Amber/OPLS use trailing ; comments)
                line = line.split(";")[0].strip()
                if not line:
                    continue

                # New section header
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip()
                    if section in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
                        current_section = section
                    elif section == "bondedtypes":
                        current_section = None
                    elif section == "bondedtypes":
                        current_res = None  # skip bondedtypes data
                        current_section = None
                    else:
                        # New residue name — skip non-standard residues (water, ions)
                        rn = section.upper().strip()
                        if rn in (
                            "HOH",
                            "HO4",
                            "SOL",
                            "WAT",
                            "NA",
                            "CL",
                            "K",
                            "CA",
                            "MG",
                            "ZN",
                            "NA+",
                            "CL-",
                            "K+",
                            "CA2+",
                            "MG2+",
                            "ZN2+",
                            "UREA",
                            "MOH",
                            "1PROPANOL",
                            "ETHANOL",
                            "METHANOL",
                        ):
                            current_res = rn
                            current_section = None
                            if rn not in self._residues:
                                self._residues[rn] = {
                                    "atoms": [],
                                    "bonds": [],
                                    "angles": [],
                                    "dihedrals": [],
                                    "impropers": [],
                                }
                        else:
                            current_res = rn
                            current_section = None
                            if current_res not in self._residues:
                                self._residues[current_res] = {
                                    "atoms": [],
                                    "bonds": [],
                                    "angles": [],
                                    "dihedrals": [],
                                    "impropers": [],
                                }
                    continue

                if current_res is None or current_section is None:
                    continue

                parts = line.split()
                if not parts:
                    continue

                res = self._residues[current_res]

                if current_section == "atoms" and len(parts) >= 4:
                    try:
                        name = parts[0]
                        atype = parts[1]
                        charge = float(parts[2])
                        group = int(parts[3]) if len(parts) > 3 else 0
                        res["atoms"].append((name, atype, charge, group))
                    except (ValueError, IndexError):
                        pass  # skip malformed atom lines

                elif current_section == "bonds" and len(parts) >= 2:
                    res["bonds"].append((parts[0], parts[1]))

                elif current_section in ("angles", "dihedrals", "impropers"):
                    if current_section == "angles" and len(parts) >= 3:
                        res["angles"].append(tuple(parts[:3]))
                    elif current_section == "dihedrals" and len(parts) >= 4:
                        res["dihedrals"].append(tuple(parts[:4]))
                    elif current_section == "impropers" and len(parts) >= 4:
                        # OPLS carries a fifth preprocessor macro such as
                        # improper_O_C_X_Y; preserve it for ITP output.
                        res["impropers"].append(tuple(parts))

    def get_residue(self, name: str) -> dict | None:
        """Return the residue template dict, or None."""
        return self._residues.get(name.strip().upper())

    def get_atom_type(self, resname: str, atom_name: str) -> tuple[str, float] | None:
        """Return (charmm_type, charge) for an atom in a residue."""
        res = self.get_residue(resname)
        if res is None:
            return None
        for an, atype, charge, _ in res["atoms"]:
            if an.strip() == atom_name.strip():
                return (atype, charge)
        return None

    @property
    def residue_names(self) -> list[str]:
        return sorted(self._residues.keys())

    def set_residue(self, name: str, template: dict) -> None:
        """Register a generated residue template."""
        self._residues[name.strip().upper()] = template


# Singleton — loaded once
_rtp: RTPParser | None = None
_rtp_by_force_field: dict[str, RTPParser] = {}


def _force_field_path(force_field: str) -> Path:
    base = Path(__file__).resolve().parent.parent.parent / "data" / "forcefields"
    for candidate in (base / force_field, base / f"{force_field}.ff"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Bundled force field directory not found: {force_field}")


def load_force_field_rtp(force_field: str) -> RTPParser:
    """Load all RTP files belonging to the selected bundled force field."""
    force_field = force_field.strip().lower()
    cached = _rtp_by_force_field.get(force_field)
    if cached is not None:
        return cached
    ff_path = _force_field_path(force_field)

    rtp_files = sorted(ff_path.glob("*.rtp"))
    if not rtp_files:
        raise FileNotFoundError(f"No RTP files found for force field: {force_field}")

    parser = RTPParser()
    for rtp_file in rtp_files:
        parser.parse(rtp_file)
    _rtp_by_force_field[force_field] = parser
    return parser


def _tdb_path(force_field: str, end: str) -> Path:
    ff_path = _force_field_path(force_field)
    suffix = "n.tdb" if end == "N" else "c.tdb"
    preferred = "merged" if force_field == "charmm36" else "aminoacids"
    path = ff_path / f"{preferred}.{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"{end}-terminal database not found for force field: {force_field}")
    return path


def _parse_tdb(path: Path) -> dict[str, dict]:
    """Parse the replace/add/delete subset used by amino-acid TDB files."""
    operation_names = {"replace", "add", "delete", "bonds", "impropers"}
    patches: dict[str, dict] = {}
    current_patch: dict | None = None
    operation = ""
    pending_add: list[str] | None = None

    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            lowered = section.lower()
            if lowered in operation_names:
                operation = lowered
            else:
                current_patch = {
                    "replace": [],
                    "add": [],
                    "delete": [],
                    "bonds": [],
                    "impropers": [],
                }
                patches[section] = current_patch
                operation = ""
            pending_add = None
            continue
        if current_patch is None or not operation:
            continue

        parts = line.split()
        if operation == "delete":
            current_patch["delete"].extend(parts)
        elif operation == "replace":
            try:
                if len(parts) >= 5:
                    old_name, new_name, atom_type = parts[:3]
                elif len(parts) >= 4:
                    old_name, atom_type = parts[:2]
                    new_name = old_name
                else:
                    continue
                current_patch["replace"].append((old_name, new_name, atom_type, float(parts[-1])))
            except ValueError:
                continue
        elif operation == "add":
            if parts[0].lstrip("+-").isdigit():
                pending_add = parts
                continue
            if pending_add is None or len(parts) < 3:
                continue
            try:
                count = int(pending_add[0])
                base_name = pending_add[2]
                control = pending_add[3]
                atom_type = parts[0]
                # CHARMM includes a trailing charge-group marker (-1), while
                # OPLS ends directly with the charge.
                charge = float(parts[-2] if len(parts) >= 4 else parts[-1])
            except (ValueError, IndexError):
                pending_add = None
                continue
            names = (
                [base_name]
                if count == 1
                else [f"{base_name}{number}" for number in range(1, count + 1)]
            )
            for name in names:
                current_patch["add"].append((name, atom_type, charge, control))
            pending_add = None
        elif operation in ("bonds", "impropers"):
            current_patch[operation].append(tuple(parts))
    return patches


def _apply_tdb_patch(base_template: dict, patch: dict) -> dict:
    """Return a residue template with one terminal TDB patch applied."""
    result = copy.deepcopy(base_template)
    deleted = set(patch["delete"])
    replacements = {
        old: (new, atom_type, charge) for old, new, atom_type, charge in patch["replace"]
    }
    rename = {old: replacement[0] for old, replacement in replacements.items()}

    atoms = []
    for name, atom_type, charge, group in result["atoms"]:
        if name in deleted:
            continue
        if name in replacements:
            new_name, atom_type, charge = replacements[name]
            name = new_name
        atoms.append((name, atom_type, charge, group))
    existing = {atom[0] for atom in atoms}
    for name, atom_type, charge, _control in patch["add"]:
        if name not in existing:
            atoms.append((name, atom_type, charge, 0))
            existing.add(name)
    result["atoms"] = atoms

    def update_terms(terms: list[tuple]) -> list[tuple]:
        updated = []
        for term in terms:
            if any(name in deleted for name in term):
                continue
            updated.append(tuple(rename.get(name, name) for name in term))
        return updated

    for section in ("bonds", "angles", "dihedrals", "impropers"):
        result[section] = update_terms(result.get(section, []))

    bonds = list(result["bonds"])
    for name, _atom_type, _charge, control in patch["add"]:
        bond = (rename.get(control, control), name)
        if bond not in bonds and tuple(reversed(bond)) not in bonds:
            bonds.append(bond)
    for term in patch["bonds"]:
        if len(term) >= 2:
            bond = (rename.get(term[0], term[0]), rename.get(term[1], term[1]))
            if bond not in bonds and tuple(reversed(bond)) not in bonds:
                bonds.append(bond)
    result["bonds"] = bonds
    result["impropers"].extend(
        tuple(rename.get(name, name) for name in term[:4]) + tuple(term[4:])
        for term in patch["impropers"]
        if len(term) >= 4
    )
    return result


def get_terminal_residue(force_field: str, base_resname: str, end: str) -> tuple[str, dict]:
    """Return the force-field-specific standard terminal residue template."""
    force_field = force_field.strip().lower()
    base_resname = base_resname.strip().upper()
    end = end.strip().upper()
    if end not in {"N", "C"}:
        raise ValueError(f"Invalid terminus: {end!r}")

    parser = load_force_field_rtp(force_field)
    variant_name = f"{end}{base_resname}"
    existing = parser.get_residue(variant_name)
    if existing is not None:
        return variant_name, existing

    base_template = parser.get_residue(base_resname)
    if base_template is None:
        raise KeyError(f"RTP residue {base_resname} not found for {force_field}")

    patches = _parse_tdb(_tdb_path(force_field, end))
    if end == "N":
        candidates = (
            "GLY-NH3+" if base_resname == "GLY" else "",
            "PRO-NH2+" if base_resname == "PRO" else "",
            "NH3+",
        )
    else:
        candidates = (
            "GLY-COO-" if base_resname == "GLY" else "",
            "PRO-COO-" if base_resname == "PRO" else "",
            "COO-",
        )
    patch_name = next((name for name in candidates if name and name in patches), None)
    if patch_name is None:
        raise KeyError(f"No standard {end}-terminal patch for {base_resname} in {force_field}")
    generated = _apply_tdb_patch(base_template, patches[patch_name])
    parser.set_residue(variant_name, generated)
    return variant_name, generated


def get_rtp() -> RTPParser:
    """Return the global RTP parser, loading charmm36 if needed."""
    global _rtp
    if _rtp is None:
        _rtp = load_force_field_rtp("charmm36")
    return _rtp
