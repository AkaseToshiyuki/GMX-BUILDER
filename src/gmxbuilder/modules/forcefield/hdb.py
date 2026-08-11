"""Hydrogen Database (.hdb) parser — adds missing hydrogens to protein structures.

Uses the bundled CHARMM36 merged.hdb to reconstruct hydrogen positions
after the PDB cleaning step removed them.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class HDBHydrogenAdder:
    """Add missing hydrogen atoms using .hdb rules."""

    def __init__(self, hdb_path: str | Path | None = None):
        self._rules: dict[str, list[dict]] = {}  # residue → [rule]
        if hdb_path:
            self.parse(hdb_path)

    def parse(self, path: str | Path) -> None:
        """Parse a GROMACS .hdb file.

        Format (CHARMM36 / GROMACS convention)::

            ALA          3
            1   1   HN   N    -C   CA
            1   5   HA   CA   N    C    CB
            3   4   HB   CB   CA   N

        The **header line** names the residue and the number of rules
        that follow (informational — we ignore it).  **Data lines**
        carry *nH* hydrogens to add, a *method* code, *nH* hydrogen
        names (a single base name is expanded with 1-based suffixes),
        then the control atom and its bonded neighbours for geometry.
        """
        path = Path(path)
        if not path.exists():
            return

        current_res = None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(";") or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                # ---- header line:  RESNAME  [n_rules  ...] ----------
                if not parts[0].lstrip("+-").isdigit() and len(parts[0]) <= 8:
                    current_res = parts[0].upper()
                    if current_res not in self._rules:
                        self._rules[current_res] = []
                    continue

                # ---- data line:  nH  method  H_names...  ctrl  bonded... -
                if current_res is None:
                    continue
                try:
                    n_h_line = int(parts[0])
                except ValueError:
                    continue

                if len(parts) < 3:
                    continue

                try:
                    method = int(parts[1])
                except ValueError:
                    continue

                # Consume hydrogen names — they always start with 'H'
                # (or digit-H for CHARMM convention, e.g. '1H', '2H').
                # Non-H atom names like CB/CA/N signal the end of H names.
                h_names: list[str] = []
                i = 2
                while i < len(parts):
                    name = parts[i]
                    is_h = (
                        name.upper().startswith("H")
                        or (len(name) >= 2 and name[0].isdigit() and name[1] == "H")
                    )
                    if not is_h:
                        break
                    h_names.append(name)
                    i += 1

                # Auto-expand: "HB" with nH=3 → HB1, HB2, HB3
                if n_h_line > len(h_names) and len(h_names) == 1:
                    base = h_names[0]
                    h_names = [f"{base}{j}" for j in range(1, n_h_line + 1)]

                if i >= len(parts):
                    continue  # no control atom — malformed line

                control_atom = parts[i]
                i += 1
                bonded_atoms = parts[i:] if i < len(parts) else []

                self._rules[current_res].append({
                    "control": control_atom,
                    "n_h": n_h_line,
                    "method": method,
                    "h_names": h_names,
                    "bonded_atoms": bonded_atoms,
                })

    def add_hydrogens(
        self,
        atom_names: list[str],
        atom_coords: np.ndarray,
        resnames: list[str],
        resids: list[int],
        chain_ids: list[str],
    ) -> tuple[list[str], np.ndarray, list[str], list[int], list[str]]:
        """Add missing hydrogen atoms and return all updated arrays.

        Returns (new_names, new_coords, new_resnames, new_resids, new_chains).
        """
        # Build per-residue index map
        residue_atoms: dict[tuple[str, str, int], list[int]] = {}
        for i, (rn, rid, chain) in enumerate(zip(resnames, resids, chain_ids)):
            key = (str(chain), rn, rid)
            if key not in residue_atoms:
                residue_atoms[key] = []
            residue_atoms[key].append(i)

        new_names = list(atom_names)
        new_resnames = list(resnames)
        new_resids = list(resids)
        new_chains = list(chain_ids)
        new_coords = atom_coords.copy() if isinstance(atom_coords, np.ndarray) else np.array(atom_coords)

        for (_chain, rn, rid), indices in residue_atoms.items():
            rules = self._rules.get(rn, [])
            if not rules:
                continue

            for rule in rules:
                control = rule["control"]
                # Find the control atom
                ctrl_idx = None
                for idx in indices:
                    if new_names[idx].strip() == control:
                        ctrl_idx = idx
                        break
                if ctrl_idx is None:
                    continue

                # Check if hydrogens already exist
                existing_h = set()
                for hname in rule["h_names"]:
                    for idx in indices:
                        if new_names[idx].strip() == hname:
                            existing_h.add(hname)

                # Add missing hydrogens
                ctrl_pos = new_coords[ctrl_idx]
                bonded_atom_names = rule["bonded_atoms"]
                bonded_positions = []
                for ba in bonded_atom_names:
                    for idx in indices:
                        if new_names[idx].strip() == ba:
                            bonded_positions.append(new_coords[idx])
                            break

                missing = [h for h in rule["h_names"] if h not in existing_h]
                if not missing:
                    continue

                # Compute hydrogen positions (bond length depends on control atom element)
                h_positions = _compute_h_positions(
                    ctrl_pos, bonded_positions, len(missing),
                    atom_name=new_names[ctrl_idx],
                    method=rule.get("method"),
                )

                for i, hname in enumerate(missing):
                    if i < len(h_positions):
                        new_names.append(hname)
                        new_resnames.append(rn)
                        new_resids.append(rid)
                        new_chains.append(new_chains[ctrl_idx])
                        new_coords = np.vstack([new_coords, h_positions[i]])
                        # Later HDB rules for the same atom may reference an
                        # earlier hydrogen to define methyl geometry.
                        indices.append(len(new_names) - 1)

        return new_names, new_coords, new_resnames, new_resids, new_chains


# Approximate X-H bond lengths (nm) by element — CHARMM36 force field
_H_BOND_LENGTHS: dict[str, float] = {
    "C": 0.109,   # C-H
    "N": 0.101,   # N-H (amine/amide)
    "O": 0.096,   # O-H (hydroxyl)
    "S": 0.134,   # S-H (thiol)
}


def _compute_h_positions(
    ctrl_pos: np.ndarray,
    bonded_positions: list[np.ndarray],
    n_h: int,
    atom_name: str = "C",
    bond_length: float | None = None,
    method: int | None = None,
) -> list[np.ndarray]:
    """Estimate hydrogen positions based on geometry of bonded neighbours.

    Bond length is inferred from the control atom's element if not specified.
    """
    if bond_length is None:
        elem = atom_name.strip()[0] if atom_name else "C"
        bond_length = _H_BOND_LENGTHS.get(elem, 0.109)

    positions = []

    if n_h == 1 and method == 6 and len(bonded_positions) >= 2:
        # GROMACS HDB method 6 encodes the two tetrahedral CH2 hydrogens
        # as separate rules with reversed neighbour order.  Preserve that
        # order through the cross product so the pair cannot coincide.
        v1 = bonded_positions[0] - ctrl_pos
        v2 = bonded_positions[1] - ctrl_pos
        avg = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
        normal = np.cross(v1, v2)
        if np.linalg.norm(avg) > 1e-6 and np.linalg.norm(normal) > 1e-6:
            direction = -avg / np.linalg.norm(avg) + 0.8 * normal / np.linalg.norm(normal)
            direction /= np.linalg.norm(direction)
            positions.append(ctrl_pos + direction * bond_length)

    elif n_h == 1 and len(bonded_positions) >= 1:
        # One H: opposite direction from bonded atoms
        direction = ctrl_pos - np.mean(bonded_positions, axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            positions.append(ctrl_pos + direction / norm * bond_length)

    elif n_h == 2 and len(bonded_positions) >= 2:
        # Two H (e.g. CH2): tetrahedral
        v1 = bonded_positions[0] - ctrl_pos
        v2 = bonded_positions[1] - ctrl_pos
        avg = (v1 + v2) / 2.0
        norm_avg = np.linalg.norm(avg)
        if norm_avg > 1e-6:
            avg_dir = avg / norm_avg
            # Two H positions
            perp = np.cross(v1, v2)
            if np.linalg.norm(perp) > 1e-6:
                perp = perp / np.linalg.norm(perp)
                h1_dir = -avg_dir + perp * 0.8
                h2_dir = -avg_dir - perp * 0.8
                h1_dir /= np.linalg.norm(h1_dir)
                h2_dir /= np.linalg.norm(h2_dir)
                positions.append(ctrl_pos + h1_dir * bond_length)
                positions.append(ctrl_pos + h2_dir * bond_length)

    elif n_h == 3 and len(bonded_positions) >= 1:
        # Three H (e.g. CH3, NH3): tetrahedral opposite
        direction = ctrl_pos - np.mean(bonded_positions, axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            main_dir = direction / norm
            # Generate 3 directions tetrahedrally from main_dir
            for angle in [0, 120, 240]:
                rad = np.radians(angle)
                perp1 = np.array([1.0, 0.0, 0.0]) if abs(main_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                perp1 = perp1 - main_dir * np.dot(perp1, main_dir)
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(main_dir, perp1)
                h_dir = main_dir * np.cos(np.radians(109.5)) + perp1 * np.sin(np.radians(109.5)) * np.cos(rad) + perp2 * np.sin(np.radians(109.5)) * np.sin(rad)
                h_dir /= np.linalg.norm(h_dir)
                positions.append(ctrl_pos + h_dir * bond_length)

    if len(positions) < n_h:
        # Degenerate or nearly collinear uploaded coordinates are common in
        # repaired/constructed side chains.  Hydrogen construction must remain
        # deterministic instead of returning a partial atom set.  Build a
        # stable local frame from the available neighbours and use idealized
        # directions as a fallback.
        if bonded_positions:
            vectors = [position - ctrl_pos for position in bonded_positions]
            unit_vectors = [
                vector / np.linalg.norm(vector) for vector in vectors
                if np.linalg.norm(vector) > 1e-8
            ]
            opposite = -np.sum(unit_vectors, axis=0) if unit_vectors else np.array([1.0, 0.0, 0.0])
        else:
            opposite = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(opposite) < 1e-8:
            opposite = np.array([1.0, 0.0, 0.0])
        main_dir = opposite / np.linalg.norm(opposite)
        trial = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(main_dir, trial))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        perp1 = trial - main_dir * np.dot(trial, main_dir)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(main_dir, perp1)
        if n_h == 1:
            directions = [main_dir]
        elif n_h == 2:
            directions = [main_dir + 0.8 * perp1, main_dir - 0.8 * perp1]
        else:
            directions = [
                main_dir * np.cos(np.radians(109.5))
                + np.sin(np.radians(109.5))
                * (perp1 * np.cos(np.radians(angle)) + perp2 * np.sin(np.radians(angle)))
                for angle in np.linspace(0.0, 360.0, n_h, endpoint=False)
            ]
        positions = [
            ctrl_pos + direction / np.linalg.norm(direction) * bond_length
            for direction in directions
        ]

    return positions[:n_h]
