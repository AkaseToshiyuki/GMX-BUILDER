"""GRO file reader and writer.

GRO is GROMACS' primary coordinate format. Uses nanometers internally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.exceptions import ParseError


def _validate_gro_name(value: object, label: str, atom_index: int) -> str:
    """Validate one identifier against the fixed five-column GRO contract."""
    text = str(value)
    if not text:
        raise ValueError(f"GRO {label} at atom {atom_index + 1} is empty")
    if len(text) > 5:
        raise ValueError(
            f"GRO {label} {text!r} at atom {atom_index + 1} exceeds 5 characters"
        )
    if not text.isascii() or any(character.isspace() for character in text):
        raise ValueError(
            f"GRO {label} {text!r} at atom {atom_index + 1} must contain "
            "printable non-whitespace ASCII characters"
        )
    if any(ord(character) < 33 or ord(character) > 126 for character in text):
        raise ValueError(
            f"GRO {label} {text!r} at atom {atom_index + 1} contains "
            "unsupported characters"
        )
    return text


class GROReader:
    """Read GROMACS .gro files into Structure objects."""

    def read(self, path: str | Path) -> Structure:
        path = Path(path)
        if not path.exists():
            raise ParseError(f"File not found: {path}")

        with open(path) as fh:
            lines = fh.readlines()

        if len(lines) < 3:
            raise ParseError(f"GRO file too short: {path}")

        title = lines[0].strip()
        try:
            n_atoms = int(lines[1].strip())
        except ValueError as exc:
            raise ParseError(f"Invalid atom count in GRO file: {lines[1].strip()!r}") from exc
        atom_lines = lines[2:2 + n_atoms]

        if len(lines) < 2 + n_atoms + 1:
            raise ParseError(
                f"GRO file truncated: expected {n_atoms} atoms + box line, got {len(lines)} lines"
            )
        box_line = lines[2 + n_atoms].strip()

        if len(atom_lines) != n_atoms:
            raise ParseError(
                f"Expected {n_atoms} atom lines, got {len(atom_lines)}"
            )

        # Parse atoms
        coords = np.zeros((n_atoms, 3), dtype=np.float64)
        atom_names = [""] * n_atoms
        resnames = [""] * n_atoms
        resids = [0] * n_atoms

        for i, line in enumerate(atom_lines):
            try:
                resids[i] = int(line[:5].strip())
                resnames[i] = line[5:10].strip()
                atom_names[i] = line[10:15].strip()
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                coords[i] = [x, y, z]
            except (ValueError, IndexError) as exc:
                # Fallback: try free-format parsing
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        resids[i] = int(parts[0])
                        resnames[i] = parts[1] if len(parts) > 1 else ""
                        atom_names[i] = parts[2] if len(parts) > 2 else ""
                        coords[i] = [float(parts[3]), float(parts[4]), float(parts[5])]
                    except (ValueError, IndexError):
                        raise ParseError(f"Malformed atom line {i+1}: {line[:60]}") from exc
                else:
                    raise ParseError(f"Malformed atom line {i+1}: {line[:60]}") from exc

        # Parse box vectors (GROMACS formats: 1/3/5/9 values).  The official
        # nine-field order is v1(x), v2(y), v3(z), v1(y), v1(z), v2(x),
        # v2(z), v3(x), v3(y) — it is not row-major matrix order.
        box_parts = [float(x) for x in box_line.split()]
        if len(box_parts) >= 9:
            values = box_parts[:9]
            box = np.array([
                [values[0], values[3], values[4]],
                [values[5], values[1], values[6]],
                [values[7], values[8], values[2]],
            ], dtype=np.float64)
        elif len(box_parts) >= 5:
            # 5-value: v1(x) v2(y) v3(z) v1(y) v1(z) — triclinic with v2,v3 diagonal-only
            v1x, v2y, v3z, v1y, v1z = box_parts[:5]
            box = np.array([
                [v1x, v1y, v1z],
                [0.0, v2y, 0.0],
                [0.0, 0.0, v3z],
            ], dtype=np.float64)
        elif len(box_parts) >= 3:
            # Diagonal-only box
            box = np.diag([box_parts[0], box_parts[1], box_parts[2]])
        else:
            box = np.eye(3) * 10.0

        return Structure(
            coordinates=coords,
            box_vectors=box,
            atom_names=atom_names,
            resnames=resnames,
            resids=resids,
        )


class GROWriter:
    """Write a Structure to GROMACS .gro format."""

    @staticmethod
    def write(structure: Structure, path: str | Path, title: str = "GMXBUILDER") -> None:
        path = Path(path)
        coords = structure.coordinates
        box = structure.box_vectors
        n = structure.num_atoms

        with open(path, "w") as fh:
            fh.write(f"{title}\n")
            fh.write(f"{n:5d}\n")

            # Determine box line format
            diag = np.diag(box)
            off_diag = box - np.diag(diag)
            is_diagonal = np.allclose(off_diag, 0)

            for i in range(n):
                resid = structure.resids[i] if i < len(structure.resids) else i + 1
                resname = structure.resnames[i] if i < len(structure.resnames) else "UNK"
                aname = structure.atom_names[i] if i < len(structure.atom_names) else "X"
                resname = _validate_gro_name(resname or "UNK", "residue name", i)
                aname = _validate_gro_name(aname or "X", "atom name", i)
                x, y, z = coords[i]

                # GRO format: 5-digit fields, wrap at 100000 (GROMACS convention)
                wrapped_resid = (resid - 1) % 99999 + 1
                wrapped_atom = i % 99999 + 1
                fh.write(f"{wrapped_resid:5d}{resname:<5s}{aname:>5s}{wrapped_atom:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")

            # Box line
            if is_diagonal:
                fh.write(f" {diag[0]:10.5f} {diag[1]:10.5f} {diag[2]:10.5f}\n")
            else:
                values = (
                    box[0, 0], box[1, 1], box[2, 2],
                    box[0, 1], box[0, 2], box[1, 0],
                    box[1, 2], box[2, 0], box[2, 1],
                )
                fh.write(" " + " ".join(f"{v:10.5f}" for v in values) + "\n")
