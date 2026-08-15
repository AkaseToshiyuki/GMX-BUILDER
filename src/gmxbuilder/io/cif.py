"""CIF (Crystallographic Information File) reader.

Supports the mmCIF format used by the wwPDB.  Extracts atom
coordinates, residue metadata, and unit-cell parameters into the
standard :class:`Structure` container.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.exceptions import ParseError


class CIFParser:
    """Parse mmCIF files into Structure objects."""

    def parse(self, path: str | Path) -> Structure:
        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")

        raw = path.read_text(encoding="utf-8-sig", errors="replace")
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

        selected: dict[tuple[str, str, str, str], tuple[int, tuple[float, int]]] = {}
        for row_idx in rows:
            base = row_idx * len(atom_fields)
            insertion = self._str(atom_data, base, col, "pdbx_PDB_ins_code", "").strip()
            if insertion not in {"", ".", "?"}:
                chain = self._preferred_str(
                    atom_data, base, col, ("auth_asym_id", "label_asym_id"), "?"
                )
                resid = self._preferred_str(
                    atom_data, base, col, ("auth_seq_id", "label_seq_id"), "?"
                )
                raise ParseError(
                    "mmCIF insertion codes are not yet representable in the integer "
                    f"residue model (chain {chain} residue {resid}{insertion}); "
                    "renumber residues uniquely before upload"
                )
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
            preference = 2 if not altloc else 1 if altloc == "A" else 0
            key = (chain, resid, resname, atom_name)
            rank = (occupancy, preference)
            previous = selected.get(key)
            if previous is None or rank > previous[1]:
                selected[key] = (row_idx, rank)
        rows = sorted(row_idx for row_idx, _rank in selected.values())
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
            resids.append(rid if rid is not None else output_idx + 1)

            # Element
            elem = self._str(atom_data, base, col, "type_symbol", "").strip()
            if not elem:
                # Guess from atom name
                an = atom_names[-1].strip()
                if an:
                    e = an[0].upper()
                    if len(an) >= 2 and an[1].islower():
                        elem = an[:2].title()
                    else:
                        elem = e
            elements.append(elem.upper())

        # ---- 2. Unit cell → box vectors ----
        box_vectors = self._parse_cell(raw)

        # Estimate from coordinates when cell parameters are absent or
        # physically unreasonable (e.g. placeholder 1.0 Å cell).
        dims = np.sqrt((box_vectors**2).sum(axis=1)) if box_vectors is not None else np.zeros(3)
        if box_vectors is None or np.any(dims < 1.0) or np.any(dims > 1000.0):
            cmin = coords.min(axis=0)
            cmax = coords.max(axis=0)
            extent = cmax - cmin
            box_size = max(extent.max() * 1.3, 3.0)
            box_vectors = np.eye(3) * box_size

        return Structure(
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
        # Find loop headers tolerantly (trailing spaces are legal and common).
        loop_pattern = re.compile(r"(?m)^loop_[ \t]*\n")
        loop_match = loop_pattern.search(raw)
        fields: list[str] = []
        values_start: int | None = None
        found = False

        while loop_match is not None:
            # Collect field names after loop_
            pos = loop_match.end()
            fields = []
            while pos < len(raw):
                line = raw[pos:].split("\n", 1)[0].strip()
                if not line or line.startswith("#"):
                    pos += len(raw[pos:].split("\n", 1)[0]) + 1
                    continue
                if not line.startswith("_"):
                    break
                # Only collect fields matching prefix
                fields.append(line)
                pos += len(raw[pos:].split("\n", 1)[0]) + 1
            if any(f.startswith(prefix) for f in fields):
                values_start = pos
                found = True
                break
            loop_match = loop_pattern.search(raw, pos)

        if not found or values_start is None:
            return [], []

        # Parse values until the next CIF control/tag line or loop terminator.
        terminator = re.search(
            r"(?m)^(?:loop_|data_|save_|_)[^\n]*$|^#[ \t]*$",
            raw[values_start:],
        )
        end = values_start + terminator.start() if terminator else len(raw)
        value_text = raw[values_start:end]

        # Tokenize CIF values: handle quoted strings
        tokens: list[str] = []
        i = 0
        while i < len(value_text):
            # Skip whitespace and comments
            if value_text[i] in " \t\r\n":
                i += 1
                continue
            if value_text[i] == "#":
                j = value_text.find("\n", i)
                i = (j + 1) if j != -1 else len(value_text)
                continue

            # Quoted values
            if value_text[i] == "'":
                j = value_text.find("'", i + 1)
                if j != -1:
                    tokens.append(value_text[i + 1 : j])
                    i = j + 1
                else:
                    i += 1
            elif value_text[i] == '"':
                j = value_text.find('"', i + 1)
                if j != -1:
                    tokens.append(value_text[i + 1 : j])
                    i = j + 1
                else:
                    i += 1
            elif value_text[i] == ";":
                # Multi-line quote
                j = value_text.find("\n;", i + 1)
                if j != -1:
                    tokens.append(value_text[i + 1 : j].strip())
                    i = j + 2
                else:
                    i += 1
            else:
                # Unquoted token — read until whitespace
                j = i
                while j < len(value_text) and value_text[j] not in " \t\r\n":
                    j += 1
                token = value_text[i:j].strip()
                if token and token != ".":
                    tokens.append(token)
                elif token == ".":
                    tokens.append(".")  # missing value placeholder
                i = j

        # Filter: only keep tokens that correspond to our prefix fields
        # (there may be other fields we don't care about; keep all tokens
        #  since they all belong to the loop in order)
        return fields, tokens

    @staticmethod
    def _parse_cell(raw: str) -> np.ndarray | None:
        """Parse _cell.length_* and _cell.angle_* into a (3,3) box matrix (nm)."""

        def _get(tag: str) -> float | None:
            m = re.search(rf"^{re.escape(tag)}\s+(\S+)", raw, re.MULTILINE)
            if m:
                try:
                    return CIFParser._number(m.group(1))
                except ValueError:
                    return None
            return None

        a = _get("_cell.length_a")
        b = _get("_cell.length_b")
        c = _get("_cell.length_c")
        alpha = _get("_cell.angle_alpha")
        beta = _get("_cell.angle_beta")
        gamma = _get("_cell.angle_gamma")

        if a is None:
            return None

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
        except (ValueError, TypeError):
            return default

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
