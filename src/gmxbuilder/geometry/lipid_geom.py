"""Full-atom lipid geometry builder.

Generates approximate all-atom 3D coordinates for phospholipids based
on standard bond lengths, angles, and headgroup templates.  Bond
lengths match CHARMM36 equilibrium values so initial geometries are
physically reasonable for any force field.

Coordinates are constructed from bond lengths and angles rather than
hardcoded absolute positions, ensuring:
  - C-C single bonds:  0.153 nm
  - C-O ester:         0.143 nm
  - C=O carbonyl:      0.123 nm
  - P-O phosphate:     0.161 nm
  - C-N choline:       0.147 nm
  - Tetrahedral angles: 109.5°
  - Trigonal angles:    120.0°
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Standard bond lengths (nm) — CHARMM36 equilibrium values
# ---------------------------------------------------------------------------
_BOND_CC = 0.153   # C-C single bond (sp3-sp3)
_BOND_CO = 0.143   # C-O ester / alcohol (sp3-O)
_BOND_CD = 0.123   # C=O carbonyl double bond
_BOND_PO = 0.161   # P-O phosphate
_BOND_CN = 0.147   # C-N (sp3-N)
_BOND_CH = 0.109   # C-H (not placed explicitly, used for geometry)

# ---------------------------------------------------------------------------
# Bond angles (radians)
# ---------------------------------------------------------------------------
_ANGLE_TET = np.radians(109.5)   # sp3 tetrahedral
_ANGLE_TRIG = np.radians(120.0)  # sp2 trigonal planar
_ANGLE_PHOS = np.radians(104.0)  # phosphate O-P-O

# ---------------------------------------------------------------------------
# Glycerol backbone constants
# ---------------------------------------------------------------------------
# In phosphoglycerol, C1 and C3 attach to C2 (central CH).  The
# C1–C2–C3 angle is close to tetrahedral.
_GLYCEROL_CC_ANGLE = np.radians(112.0)  # C1–C2–C3 angle (slightly wider than tet)


def _place_atom(origin: np.ndarray, bond_to: np.ndarray,
                bond_length: float, angle: float,
                dihedral_ref: np.ndarray | None = None,
                dihedral_angle: float = 0.0) -> np.ndarray:
    """Place a new atom given bond length and angle relative to existing atoms.

    Parameters
    ----------
    origin : (3,) ndarray
        Position of the atom being connected TO.
    bond_to : (3,) ndarray
        Position of the atom already bonded to *origin* (defines the
        bond direction for the angle reference).
    bond_length : float
        Distance from *origin* to the new atom (nm).
    angle : float
        Bond angle at *origin* between *bond_to*→*origin*→new_atom (radians).
    dihedral_ref : (3,) ndarray or None
        Third reference atom for dihedral control.  If None, a random
        orientation is chosen orthogonal to the bond_to–origin axis.
    dihedral_angle : float
        Dihedral angle (radians) around the bond_to→origin axis.

    Returns
    -------
    pos : (3,) ndarray
    """
    # Bond axis: origin → bond_to (reversed — we're placing FROM origin)
    v1 = bond_to - origin
    v1 = v1 / np.linalg.norm(v1)

    # Build a local coordinate system
    if dihedral_ref is not None:
        v_ref = dihedral_ref - origin
        # Project out v1 component
        v_ref = v_ref - np.dot(v_ref, v1) * v1
        nref = np.linalg.norm(v_ref)
        if nref > 1e-6:
            v_ref = v_ref / nref
        else:
            v_ref = _orthogonal(v1)
    else:
        v_ref = _orthogonal(v1)

    # Rotate v1 by -angle around v1×v_ref to get the direction
    axis = np.cross(v1, v_ref)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues rotation: rotate -v1 by angle around axis
    cos_a, sin_a = np.cos(np.pi - angle), np.sin(np.pi - angle)
    v_new = (-v1) * cos_a + np.cross(axis, -v1) * sin_a + axis * np.dot(axis, -v1) * (1 - cos_a)
    v_new = v_new / np.linalg.norm(v_new)

    # Now apply dihedral rotation around v1
    cos_d, sin_d = np.cos(dihedral_angle), np.sin(dihedral_angle)
    v_final = v_new * cos_d + np.cross(v1, v_new) * sin_d + v1 * np.dot(v1, v_new) * (1 - cos_d)

    return origin + v_final * bond_length


def _orthogonal(v: np.ndarray) -> np.ndarray:
    """Return a unit vector orthogonal to *v*."""
    if abs(v[2]) < 0.9:
        u = np.array([v[1], -v[0], 0.0])
    else:
        u = np.array([1.0, 0.0, 0.0])
    return u / np.linalg.norm(u)


def _build_alkane_chain(n_carbon: int, start: np.ndarray, direction: np.ndarray,
                        atom_prefix: str, start_idx: int,
                        rng: np.random.Generator | None = None,
                        gauche_prob: float = 0.20) -> tuple[np.ndarray, list[str]]:
    """Build an alkane chain extending along *direction* from *start*.

    The **first** carbon is placed exactly at *start* — the caller is
    responsible for placing it correctly (e.g. via _place_atom).
    Subsequent carbons follow a tetrahedral zigzag pattern.

    Returns (coords, atom_names).
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = []
    names = []
    d = direction / np.linalg.norm(direction)

    if abs(d[2]) < 0.9:
        perp = np.array([d[1], -d[0], 0.0])
    else:
        perp = np.array([1.0, 0.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    binormal = np.cross(d, perp)
    binormal = binormal / np.linalg.norm(binormal)

    forward = _BOND_CC * np.cos(np.radians(54.75))
    lateral = _BOND_CC * np.sin(np.radians(54.75))

    gauche_angles = []
    for i in range(n_carbon - 1):
        if rng.random() < gauche_prob:
            gauche_angles.append(rng.choice([-1, 1]) * np.pi / 3.0)
        else:
            gauche_angles.append(0.0)

    pos = start.copy()

    # First carbon AT the start position (caller placed it)
    coords.append(pos.copy())
    names.append(f"{atom_prefix}{start_idx}")

    for i in range(1, n_carbon):
        offset = d * forward
        if i % 2 == 1:
            offset += perp * lateral
        else:
            offset -= perp * lateral
        pos = pos + offset
        coords.append(pos.copy())
        names.append(f"{atom_prefix}{start_idx + i}")

        if i - 1 < len(gauche_angles) and abs(gauche_angles[i - 1]) > 0.01:
            ang = gauche_angles[i - 1]
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            d_new = d * cos_a + np.cross(binormal, d) * sin_a \
                    + binormal * np.dot(binormal, d) * (1 - cos_a)
            perp_new = perp * cos_a + np.cross(binormal, perp) * sin_a \
                       + binormal * np.dot(binormal, perp) * (1 - cos_a)
            d = d_new / np.linalg.norm(d_new)
            perp = perp_new / np.linalg.norm(perp_new)
            binormal = np.cross(d, perp)
            binormal = binormal / np.linalg.norm(binormal)

    return np.array(coords), names


def build_lipid_geometry(lipid_name: str, tail1: tuple[int, int],
                         tail2: tuple[int, int], category: str,
                         rng: np.random.Generator | None = None,
                         gauche_prob: float = 0.20) -> tuple[np.ndarray, list[str]]:
    """Generate approximate all-atom coordinates for a phospholipid.

    Lipid is oriented vertically along Z:
      - Headgroup at +Z (choline, ethanolamine, etc.)
      - Glycerol backbone near Z=0
      - Two acyl tails extending toward −Z

    If *rng* is provided, random gauche defects are introduced in the
    acyl chains with probability *gauche_prob*, producing more compact
    and realistic conformations.

    Returns (coords_nm, atom_names).  Coordinates are centered so the
    glycerol C2 is near the origin and the tails point toward −Z.
    """
    atoms = []
    names = []

    # ---- 1. Glycerol backbone ----
    # Place C2 at origin.  C1 and C3 are bonded to C2 with tetrahedral
    # geometry (~112° C1-C2-C3 angle).
    c2 = np.array([0.0, 0.0, 0.0])

    # C1: to the "left" and slightly up
    half_angle = _GLYCEROL_CC_ANGLE / 2.0
    c1_offset = np.array([-_BOND_CC * np.sin(half_angle), 0.0,
                           _BOND_CC * np.cos(half_angle)])
    c1 = c2 + c1_offset

    # C3: to the "right" and slightly up
    c3_offset = np.array([_BOND_CC * np.sin(half_angle), 0.0,
                          _BOND_CC * np.cos(half_angle)])
    c3 = c2 + c3_offset

    atoms.extend([c1, c2, c3])
    names.extend(["C1", "C2", "C3"])

    # ---- 2. sn-1 ester (from C1) ----
    # C1–O11 ester bond: tetrahedral angle with C1–C2 bond
    o11 = _place_atom(c1, c2, _BOND_CO, _ANGLE_TET,
                       dihedral_ref=None, dihedral_angle=np.pi * 0.8)
    atoms.append(o11); names.append("O11")

    # O11–C11 carbonyl: trigonal planar
    c11 = _place_atom(o11, c1, _BOND_CD, _ANGLE_TRIG,
                       dihedral_ref=c2, dihedral_angle=0.0)
    atoms.append(c11); names.append("C11")

    # C11–O12 carbonyl oxygen: opposite side, trigonal
    o12 = _place_atom(c11, o11, _BOND_CD, _ANGLE_TRIG,
                       dihedral_ref=c1, dihedral_angle=np.pi)
    atoms.append(o12); names.append("O12")

    # O12–C12: ester linkage to first chain carbon.  Place C12 so
    # the chain naturally extends downward (toward −Z).
    c12_start = _place_atom(o12, c11, _BOND_CO, _ANGLE_TRIG,
                             dihedral_ref=o11, dihedral_angle=0.0)
    tail_dir = np.array([0.0, 0.0, -1.0])
    chain1, chain1_names = _build_alkane_chain(
        tail1[0], c12_start, tail_dir, "C1", 2, rng=rng, gauche_prob=gauche_prob)
    atoms.extend(chain1); names.extend(chain1_names)

    # ---- 3. sn-2 ester (from C2) ----
    # C2–O21 ester: the sn-2 ester comes off C2 (replaces the H)
    o21 = _place_atom(c2, c1, _BOND_CO, _ANGLE_TET,
                       dihedral_ref=c3, dihedral_angle=np.pi * 1.2)
    atoms.append(o21); names.append("O21")

    # O21–C21 carbonyl
    c21 = _place_atom(o21, c2, _BOND_CD, _ANGLE_TRIG,
                       dihedral_ref=c1, dihedral_angle=0.0)
    atoms.append(c21); names.append("C21")

    # C21–O22 carbonyl oxygen
    o22 = _place_atom(c21, o21, _BOND_CD, _ANGLE_TRIG,
                       dihedral_ref=c2, dihedral_angle=np.pi)
    atoms.append(o22); names.append("O22")

    # O22–C22: ester linkage to first chain carbon.
    c22_start = _place_atom(o22, c21, _BOND_CO, _ANGLE_TRIG,
                             dihedral_ref=o21, dihedral_angle=0.0)
    tail_dir2 = np.array([0.0, 0.0, -1.0])
    chain2, chain2_names = _build_alkane_chain(
        tail2[0], c22_start, tail_dir2, "C2", 2, rng=rng, gauche_prob=gauche_prob)
    atoms.extend(chain2); names.extend(chain2_names)

    # ---- 4. Phosphate group (from C3) ----
    # C3–O31: bridging ester oxygen connecting C3 to P
    o31 = _place_atom(c3, c2, _BOND_CO, _ANGLE_TET,
                       dihedral_ref=c1, dihedral_angle=np.pi * 0.7)
    atoms.append(o31); names.append("O31")

    # O31–P: phosphate
    p_pos = _place_atom(o31, c3, _BOND_PO, _ANGLE_TRIG,
                         dihedral_ref=c2, dihedral_angle=0.0)
    atoms.append(p_pos); names.append("P")

    # Non-bridging oxygens O32, O33 — tetrahedral around P
    o32 = _place_atom(p_pos, o31, _BOND_PO * 0.93, _ANGLE_PHOS,
                       dihedral_ref=c3, dihedral_angle=np.pi * 0.6)
    atoms.append(o32); names.append("O32")

    o33 = _place_atom(p_pos, o31, _BOND_PO * 0.93, _ANGLE_PHOS,
                       dihedral_ref=c3, dihedral_angle=-np.pi * 0.6)
    atoms.append(o33); names.append("O33")

    # Bridging oxygen O34 — connects P to headgroup
    o34 = _place_atom(p_pos, o31, _BOND_PO, _ANGLE_PHOS,
                       dihedral_ref=c3, dihedral_angle=np.pi)
    atoms.append(o34); names.append("O34")

    # ---- 5. Headgroup (from O34) ----
    hg_start = o34

    if category in ("PC",):
        # Choline: -O-CH2-CH2-N(CH3)3+
        c_a = _place_atom(hg_start, p_pos, _BOND_CO, _ANGLE_TRIG,
                           dihedral_ref=o31, dihedral_angle=0.0)
        atoms.append(c_a); names.append("C4")
        c_b = _place_atom(c_a, hg_start, _BOND_CC, _ANGLE_TET,
                           dihedral_ref=p_pos, dihedral_angle=0.0)
        atoms.append(c_b); names.append("C5")
        n_pos = _place_atom(c_b, c_a, _BOND_CN, _ANGLE_TET,
                             dihedral_ref=hg_start, dihedral_angle=0.0)
        atoms.append(n_pos); names.append("N")
        # Three methyl groups (tetrahedral around N)
        for j, dih in enumerate([0.0, np.pi * 2 / 3, -np.pi * 2 / 3]):
            c_met = _place_atom(n_pos, c_b, _BOND_CN, _ANGLE_TET,
                                 dihedral_ref=c_a, dihedral_angle=dih)
            atoms.append(c_met); names.append(f"C6{j+1}")

    elif category in ("PE",):
        # Ethanolamine: -O-CH2-CH2-NH3+
        c_a = _place_atom(hg_start, p_pos, _BOND_CO, _ANGLE_TRIG,
                           dihedral_ref=o31, dihedral_angle=0.0)
        atoms.append(c_a); names.append("C4")
        c_b = _place_atom(c_a, hg_start, _BOND_CC, _ANGLE_TET,
                           dihedral_ref=p_pos, dihedral_angle=0.0)
        atoms.append(c_b); names.append("C5")
        n_pos = _place_atom(c_b, c_a, _BOND_CN, _ANGLE_TET,
                             dihedral_ref=hg_start, dihedral_angle=0.0)
        atoms.append(n_pos); names.append("N")

    elif category in ("PG",):
        # Phosphoglycerol: -O-CH2-CHOH-CH2OH
        gc1 = _place_atom(hg_start, p_pos, _BOND_CO, _ANGLE_TRIG,
                           dihedral_ref=o31, dihedral_angle=0.0)
        atoms.append(gc1); names.append("GC1")
        gc2 = _place_atom(gc1, hg_start, _BOND_CC, _ANGLE_TET,
                           dihedral_ref=p_pos, dihedral_angle=0.0)
        atoms.append(gc2); names.append("GC2")
        go1 = _place_atom(gc2, gc1, _BOND_CO, _ANGLE_TET,
                           dihedral_ref=hg_start, dihedral_angle=np.pi * 0.6)
        atoms.append(go1); names.append("GO1")
        gc3 = _place_atom(gc1, hg_start, _BOND_CC, _ANGLE_TET,
                           dihedral_ref=p_pos, dihedral_angle=np.pi * 0.8)
        atoms.append(gc3); names.append("GC3")
        go2 = _place_atom(gc3, gc1, _BOND_CO, _ANGLE_TET,
                           dihedral_ref=hg_start, dihedral_angle=np.pi * 0.6)
        atoms.append(go2); names.append("GO2")

    elif category in ("PS",):
        # Phosphoserine: -O-CH2-CH(NH3+)-COO-
        c_a = _place_atom(hg_start, p_pos, _BOND_CO, _ANGLE_TRIG,
                           dihedral_ref=o31, dihedral_angle=0.0)
        atoms.append(c_a); names.append("C4")
        ca = _place_atom(c_a, hg_start, _BOND_CC, _ANGLE_TET,
                          dihedral_ref=p_pos, dihedral_angle=0.0)
        atoms.append(ca); names.append("CA")
        n_ser = _place_atom(ca, c_a, _BOND_CN, _ANGLE_TET,
                             dihedral_ref=hg_start, dihedral_angle=np.pi * 0.6)
        atoms.append(n_ser); names.append("N")
        c_coo = _place_atom(ca, c_a, _BOND_CC, _ANGLE_TET,
                             dihedral_ref=hg_start, dihedral_angle=-np.pi * 0.6)
        atoms.append(c_coo); names.append("C")
        o_coo1 = _place_atom(c_coo, ca, _BOND_CD, _ANGLE_TRIG,
                              dihedral_ref=c_a, dihedral_angle=np.pi * 0.5)
        atoms.append(o_coo1); names.append("O1")
        o_coo2 = _place_atom(c_coo, ca, _BOND_CD, _ANGLE_TRIG,
                              dihedral_ref=c_a, dihedral_angle=-np.pi * 0.5)
        atoms.append(o_coo2); names.append("O2")

    elif category in ("PA",):
        # Phosphatidic acid — no headgroup beyond phosphate
        pass

    elif category in ("SM",):
        # Sphingomyelin — PC-like headgroup
        c_a = _place_atom(hg_start, p_pos, _BOND_CO, _ANGLE_TRIG,
                           dihedral_ref=o31, dihedral_angle=0.0)
        atoms.append(c_a); names.append("C4")
        c_b = _place_atom(c_a, hg_start, _BOND_CC, _ANGLE_TET,
                           dihedral_ref=p_pos, dihedral_angle=0.0)
        atoms.append(c_b); names.append("C5")
        n_pos = _place_atom(c_b, c_a, _BOND_CN, _ANGLE_TET,
                             dihedral_ref=hg_start, dihedral_angle=0.0)
        atoms.append(n_pos); names.append("N")
        for j, dih in enumerate([0.0, np.pi * 2 / 3, -np.pi * 2 / 3]):
            c_met = _place_atom(n_pos, c_b, _BOND_CN, _ANGLE_TET,
                                 dihedral_ref=c_a, dihedral_angle=dih)
            atoms.append(c_met); names.append(f"C6{j+1}")

    elif category in ("DG",):
        # Diacylglycerol — no headgroup, just glycerol + two tails
        pass

    elif category in ("ST",):
        # Sterol — simplified planar ring placeholder
        for i in range(17):
            ring_y = -0.30 + i * 0.0375
            atoms.append(np.array([0.15, ring_y, 0.0]))
            names.append(f"C{i+1:02d}")

    coords = np.array(atoms, dtype=np.float64)
    coords -= coords.mean(axis=0)
    return coords, names
