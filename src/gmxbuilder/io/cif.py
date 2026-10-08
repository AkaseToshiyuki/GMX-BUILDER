"""CIF (Crystallographic Information File) reader.

Supports the mmCIF format used by the wwPDB.  Extracts atom
coordinates, residue metadata, and unit-cell parameters into the
standard :class:`Structure` container.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ParseError
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.cell import classify_cell, display_envelope
from gmxbuilder.io.residue_identity import clean_identifier, working_residue_ids
from gmxbuilder.io.source_document import source_bytes, source_text


class CIFParser:
    """Parse mmCIF files into Structure objects."""

    def parse(self, path: str | Path) -> Structure:
        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")

        raw = source_text(path)
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")

        # ---- 1. Extract _atom_site loop ----
        atom_fields, atom_data = self._extract_loop(raw, "_atom_site.")

        if not atom_fields:
            raise ParseError(f"No _atom_site data found in {path}")

        # Map field names to column indices
        col: dict[str, int] = {}
        for i, f in enumerate(atom_fields):
            name = f.split(".", 1)[1] if "." in f else f
            col[name] = i

        if len(atom_data) % len(atom_fields) != 0:
            raise ParseError(
                "Incomplete _atom_site loop: value count is not divisible by column count"
            )
        for required in ("Cartn_x", "Cartn_y", "Cartn_z"):
            if required not in col:
                raise ParseError(f"_atom_site.{required} is required")

        all_rows = list(range(len(atom_data) // len(atom_fields)))
        model_col = col.get("pdbx_PDB_model_num")
        if model_col is not None and all_rows:
            first_model = self._str(
                atom_data,
                all_rows[0] * len(atom_fields),
                col,
                "pdbx_PDB_model_num",
                "1",
            )
            rows = [
                row
                for row in all_rows
                if self._str(
                    atom_data,
                    row * len(atom_fields),
                    col,
                    "pdbx_PDB_model_num",
                    first_model,
                )
                == first_model
            ]
        else:
            rows = all_rows

        conformer_records = []
        for row_idx in rows:
            base = row_idx * len(atom_fields)
            insertion = self._str(atom_data, base, col, "pdbx_PDB_ins_code", "").strip()
            atom_name = self._preferred_str(
                atom_data, base, col, ("auth_atom_id", "label_atom_id"), ""
            )
            resname = self._preferred_str(
                atom_data, base, col, ("auth_comp_id", "label_comp_id"), "UNK"
            )
            chain = self._preferred_str(atom_data, base, col, ("auth_asym_id", "label_asym_id"), "")
            resid = self._preferred_str(atom_data, base, col, ("auth_seq_id", "label_seq_id"), "")
            altloc = self._str(atom_data, base, col, "label_alt_id", "").strip()
            if altloc in {".", "?"}:
                altloc = ""
            occupancy = self._float(atom_data, base, col, "occupancy", 1.0)
            conformer_records.append(
                (
                    row_idx,
                    (
                        chain,
                        self._str(atom_data, base, col, "label_asym_id", chain),
                        resid,
                        insertion,
                    ),
                    resname,
                    atom_name,
                    altloc,
                    occupancy,
                )
            )
        from gmxbuilder.io.altloc import select_residue_conformers

        try:
            rows = select_residue_conformers(conformer_records)
        except ParseError as exc:
            for issue in exc.issues:
                base = issue["record"] * len(atom_fields)
                issue.update(
                    {
                        "chain": self._preferred_str(
                            atom_data, base, col, ("auth_asym_id", "label_asym_id"), ""
                        ),
                        "resid": self._preferred_str(
                            atom_data, base, col, ("auth_seq_id", "label_seq_id"), ""
                        ),
                        "insertion_code": self._str(atom_data, base, col, "pdbx_PDB_ins_code", ""),
                        "source_atom_id": self._str(atom_data, base, col, "id", ""),
                    }
                )
            raise
        # Atom-site rows need not be grouped by polymer sequence. Use the
        # deposited label sequence where available, never author numbering.
        chain_order = {}
        ordering = {}
        for row_idx in rows:
            base = row_idx * len(atom_fields)
            chain = self._preferred_str(atom_data, base, col, ("label_asym_id", "auth_asym_id"), "")
            chain_order.setdefault(chain, len(chain_order))
            sequence = self._str(atom_data, base, col, "label_seq_id", "")
            if sequence and (not sequence.isdigit() or int(sequence) <= 0):
                raise ParseError("label_seq_id must be a positive sequence position or missing")
            ordering[row_idx] = (chain_order[chain], int(sequence) if sequence else 0, row_idx)
        rows.sort(key=ordering.__getitem__)
        n_atoms = len(rows)
        if n_atoms == 0:
            raise ParseError("Empty _atom_site loop")

        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        atom_names: list[str] = []
        resnames: list[str] = []
        resids: list[int] = []
        chain_ids: list[str] = []
        elements: list[str] = []
        occupancies: list[float] = []
        tempfactors: list[float] = []

        residue_keys = []
        unknown_elements = []
        for output_idx, row_idx in enumerate(rows):
            base = row_idx * len(atom_fields)

            # Coordinates (Å → nm)
            x = self._required_float(atom_data, base, col, "Cartn_x") / 10.0
            y = self._required_float(atom_data, base, col, "Cartn_y") / 10.0
            z = self._required_float(atom_data, base, col, "Cartn_z") / 10.0
            if not np.isfinite([x, y, z]).all():
                raise ParseError(f"Non-finite coordinates in _atom_site row {row_idx + 1}")
            coords[output_idx] = [x, y, z]

            atom_names.append(
                self._preferred_str(atom_data, base, col, ("auth_atom_id", "label_atom_id"), "")
            )
            resnames.append(
                self._preferred_str(atom_data, base, col, ("auth_comp_id", "label_comp_id"), "UNK")
            )
            chain_ids.append(
                self._preferred_str(atom_data, base, col, ("auth_asym_id", "label_asym_id"), "")
            )
            occupancies.append(self._float(atom_data, base, col, "occupancy", 1.0))
            tempfactors.append(self._float(atom_data, base, col, "B_iso_or_equiv", 0.0))

            # Residue ID: try auth_seq_id first, then label_seq_id
            rid = self._int(atom_data, base, col, "auth_seq_id")
            if rid is None:
                rid = self._int(atom_data, base, col, "label_seq_id")
            if rid is None:
                raise ParseError("Residue identity requires an integer auth_seq_id or label_seq_id")
            if not np.iinfo(np.int64).min <= rid <= np.iinfo(np.int64).max:
                raise ParseError("Residue identifier exceeds the supported signed 64-bit range")
            residue_keys.append(
                (
                    chain_ids[-1],
                    self._str(atom_data, base, col, "label_asym_id", chain_ids[-1]),
                    str(rid),
                    clean_identifier(self._str(atom_data, base, col, "pdbx_PDB_ins_code", "")),
                )
            )

            # Element
            elem = self._str(atom_data, base, col, "type_symbol", "").strip()
            if not elem:
                raise ParseError(
                    "Missing _atom_site.type_symbol: atom names alone cannot establish elements"
                )
            from gemmi import Element

            if Element(elem).atomic_number == 0:
                unknown_elements.append(
                    {
                        "code": "unknown_element",
                        "record": row_idx,
                        "element": elem,
                        "chain": chain_ids[-1],
                        "resid": str(rid),
                        "insertion_code": residue_keys[-1][-1],
                        "resname": resnames[-1],
                        "atom": atom_names[-1],
                    }
                )
            elements.append(elem.upper())

        if unknown_elements:
            first = unknown_elements[0]
            raise ParseError(
                f"Unknown element in {len(unknown_elements)} atom(s); first is "
                f"{first['element']!r} at {first['chain']}:{first['resid']} "
                f"{first['resname']} {first['atom']}. Identify these sites from an "
                "authoritative source or explicitly exclude them before re-uploading; "
                "elements are not guessed.",
                issues=unknown_elements,
            )
        resids, residue_mapping = working_residue_ids(residue_keys)

        # ---- 2. Unit cell → box vectors ----
        box_vectors = self._parse_cell(raw)
        deposited_cell = None if box_vectors is None else box_vectors.tolist()

        cell_status = classify_cell(box_vectors)
        if cell_status in {"missing", "placeholder"}:
            box_vectors = display_envelope(coords)

        from gmxbuilder.io.cif_document import category

        digest = hashlib.sha256(source_bytes(path)).hexdigest()
        records = {}
        for row_idx in rows:
            records[f"{digest}:{row_idx}"] = dict(
                zip(
                    atom_fields,
                    atom_data[row_idx * len(atom_fields) : (row_idx + 1) * len(atom_fields)],
                    strict=True,
                )
            )
        connection_fields, connection_values = category(raw, "_struct_conn.")
        connections = [
            dict(
                zip(
                    connection_fields,
                    connection_values[i : i + len(connection_fields)],
                    strict=True,
                )
            )
            for i in range(0, len(connection_values), len(connection_fields) or 1)
        ]
        return Structure(
            source_ids=[f"{digest}:{i}" for i in rows],
            source_info={
                "schema": 1,
                "format": "mmcif",
                "sha256": digest,
                "selected_model": first_model if model_col is not None else "1",
                "selected_model_number": first_model if model_col is not None else "1",
                "selected_model_ordinal": 1,
                "model_count": len(
                    {
                        self._str(atom_data, r * len(atom_fields), col, "pdbx_PDB_model_num", "1")
                        for r in all_rows
                    }
                ),
                "input_rows": len(all_rows),
                "selected_rows": rows,
                "atoms": records,
                "connections": connections,
                "cell_vectors_nm": deposited_cell,
                "box_source": "deposited" if cell_status == "deposited" else "estimated",
                "cell_status": cell_status,
                "residue_mapping": residue_mapping,
            },
            coordinates=coords,
            box_vectors=box_vectors,
            atom_names=atom_names,
            resnames=resnames,
            resids=resids,
            chain_ids=chain_ids,
            elements=elements,
            occupancies=occupancies,
            tempfactors=tempfactors,
        )

    # ------------------------------------------------------------------
    # CIF parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_loop(raw: str, prefix: str) -> tuple[list[str], list[str]]:
        """Extract a CIF loop whose fields start with *prefix*.

        Returns (field_names, values) where values is a flat list.
        """
        from gmxbuilder.io.cif_document import category

        return category(raw, prefix)

    @staticmethod
    def _parse_cell(raw: str) -> np.ndarray | None:
        """Parse _cell.length_* and _cell.angle_* into a (3,3) box matrix (nm)."""

        from gmxbuilder.io.cif_document import structure_block

        block = structure_block(raw)

        def _get(tag: str) -> float | None:
            value = block.find_value(tag)
            if value is None or value in {".", "?"}:
                return None
            try:
                return CIFParser._number(value)
            except ValueError as exc:
                raise ParseError(f"Invalid {tag}: {value!r}") from exc

        a = _get("_cell.length_a")
        b = _get("_cell.length_b")
        c = _get("_cell.length_c")
        alpha = _get("_cell.angle_alpha")
        beta = _get("_cell.angle_beta")
        gamma = _get("_cell.angle_gamma")

        if all(v is None for v in (a, b, c, alpha, beta, gamma)):
            return None
        if any(v is None for v in (a, b, c, alpha, beta, gamma)):
            raise ParseError("Incomplete CIF unit cell: supply all lengths and angles")
        if not all(np.isfinite(v) for v in (a, b, c, alpha, beta, gamma)) or min(a, b, c) <= 0:
            raise ParseError("Unit-cell lengths must be finite and positive")
        if not all(0 < v < 180 for v in (alpha, beta, gamma)):
            raise ParseError("Unit-cell angles must be between 0 and 180 degrees")
        cosines = np.cos(np.radians([alpha, beta, gamma]))
        if 1 - np.dot(cosines, cosines) + 2 * np.prod(cosines) <= 0:
            raise ParseError("Unit-cell angles do not define a positive-volume cell")

        # Convert Å → nm and degrees → radians
        a_nm = a / 10.0
        b_nm = b / 10.0 if b else a_nm
        c_nm = c / 10.0 if c else a_nm
        al = np.radians(alpha or 90.0)
        be = np.radians(beta or 90.0)
        ga = np.radians(gamma or 90.0)

        # Convert to triclinic box vectors
        cos_al = np.cos(al)
        cos_be = np.cos(be)
        cos_ga, sin_ga = np.cos(ga), np.sin(ga)

        v1 = np.array([a_nm, 0.0, 0.0])
        v2 = np.array([b_nm * cos_ga, b_nm * sin_ga, 0.0])
        v3 = np.array(
            [
                c_nm * cos_be,
                c_nm * (cos_al - cos_be * cos_ga) / max(sin_ga, 1e-8),
                c_nm
                * np.sqrt(
                    max(1.0 - cos_al**2 - cos_be**2 - cos_ga**2 + 2 * cos_al * cos_be * cos_ga, 0)
                )
                / max(sin_ga, 1e-8),
            ]
        )

        return np.array([v1, v2, v3])

    # ------------------------------------------------------------------
    # Typed field access
    # ------------------------------------------------------------------

    @staticmethod
    def _str(data: list[str], base: int, col: dict[str, int], key: str, default: str) -> str:
        idx = col.get(key)
        if idx is None:
            return default
        pos = base + idx
        if pos >= len(data):
            return default
        v = data[pos]
        return v if v not in {".", "?"} else default

    @staticmethod
    def _preferred_str(
        data: list[str],
        base: int,
        col: dict[str, int],
        keys: tuple[str, ...],
        default: str,
    ) -> str:
        for key in keys:
            value = CIFParser._str(data, base, col, key, "")
            if value:
                return value
        return default

    @staticmethod
    def _number(value: str) -> float:
        """Parse a CIF number, including an optional uncertainty suffix."""
        match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?",
            value.strip(),
        )
        if not match:
            raise ValueError(value)
        return float(match.group(1))

    @staticmethod
    def _float(data: list[str], base: int, col: dict[str, int], key: str, default: float) -> float:
        v = CIFParser._str(data, base, col, key, "")
        if v == "":
            return default
        try:
            return CIFParser._number(v)
        except (ValueError, TypeError) as exc:
            raise ParseError(f"Invalid _atom_site.{key} value: {v!r}") from exc

    @staticmethod
    def _required_float(data: list[str], base: int, col: dict[str, int], key: str) -> float:
        value = CIFParser._str(data, base, col, key, "")
        if not value:
            raise ParseError(f"Missing _atom_site.{key} value")
        try:
            return CIFParser._number(value)
        except ValueError as exc:
            raise ParseError(f"Invalid _atom_site.{key} value: {value!r}") from exc

    @staticmethod
    def _int(data: list[str], base: int, col: dict[str, int], key: str) -> int | None:
        v = CIFParser._str(data, base, col, key, "")
        if v == "":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
