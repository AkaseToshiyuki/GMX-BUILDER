"""Module 2: Aqueous-phase solvation builder.

Adds a water box around the existing system, removing overlapping
water molecules.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.component import Component
from gmxbuilder.core.chemistry import WATER_VOLUME_NM3
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.geometry.overlap import find_overlapping_atoms
from gmxbuilder.modules import register_module
from gmxbuilder.modules.solvation.water_models import WaterRegistry


# Approximate volume per water molecule (nm^3)
_WATER_VOLUME_PER_MOLECULE = WATER_VOLUME_NM3


@register_module
class SolvationBuilder(BaseModule):
    """Add water molecules to solvate the system."""

    name = "solvation"
    description = "Add water box around system with overlap removal"

    _DEFAULT_PADDING = 1.2   # nm
    _WATER_SPACING = 0.31    # nm, approximate spacing between waters

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {"water_model", "box_padding", "overlap_scale", "box_size",
             "remove_overlap", "use_prebuilt_water", "seed"},
        )
        water_model = str(config.get("water_model", "tip3p")).strip().lower()
        try:
            WaterRegistry.get(water_model)
        except KeyError as exc:
            raise ModuleConfigError(str(exc))
        for key, default, minimum, maximum in (
            ("box_padding", self._DEFAULT_PADDING, 0.0, 20.0),
            ("overlap_scale", 0.8, 0.1, 1.0),
        ):
            try:
                value = float(config.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError(f"{key} must be a finite number") from exc
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ModuleConfigError(
                    f"{key} must be between {minimum} and {maximum}, got {value}"
                )
        box_size = config.get("box_size")
        if box_size is not None:
            try:
                dims = np.asarray(box_size, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError("box_size must contain three finite dimensions") from exc
            if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
                raise ModuleConfigError("box_size must contain three positive finite dimensions")
        for flag in ("remove_overlap", "use_prebuilt_water"):
            if flag in config and not isinstance(config[flag], bool):
                raise ModuleConfigError(f"{flag} must be a boolean")
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        locked_water_model = system.metadata.get("water_model")
        requested_water_model = config.get("water_model")
        if (
            locked_water_model is not None
            and requested_water_model is not None
            and str(locked_water_model).lower() != str(requested_water_model).lower()
        ):
            raise ModuleConfigError(
                "Water model is locked by Step 2 force-field selection; "
                "return to Step 2 to change it"
            )
        water_model_name = str(
            locked_water_model or requested_water_model or "tip3p"
        ).strip().lower()
        water_model = WaterRegistry.get(water_model_name)
        box_padding = float(config.get("box_padding", self._DEFAULT_PADDING))
        remove_overlap = bool(config.get("remove_overlap", True))
        overlap_scale = float(config.get("overlap_scale", 0.8))

        if system.component_by_kind(ComponentKind.SOLVENT):
            raise ModuleConfigError(
                "System is already solvated; rerun Step 6 from the previous checkpoint"
            )

        coords = system.coordinates
        log = []

        # ---- 1. Determine box dimensions ----
        # If a MEMBRANE component exists (MembraneBuilder already ran), use
        # its box as the XY base.  Otherwise (Solvator / empty system)
        # compute the box from solute + padding.
        membrane_comps = system.component_by_kind(ComponentKind.MEMBRANE)
        membrane_box_set = bool(membrane_comps)

        if membrane_box_set:
            # XY comes from the membrane checkpoint.  Z padding is measured
            # from the two outer lipid molecular surfaces,
            # not from an asymmetric protein/solute bounding box.
            existing_dims = system.structure.dimensions()
            if not np.isfinite(existing_dims).all() or np.any(existing_dims <= 0):
                raise ModuleConfigError("Membrane checkpoint has invalid box dimensions")
            box_x, box_y = map(float, existing_dims[:2])

            membrane_coords = np.concatenate([
                coords[component.atom_indices] for component in membrane_comps
            ])
            membrane_mid_z = float(
                (membrane_coords[:, 2].min() + membrane_coords[:, 2].max()) / 2.0
            )
            interface_lower = float(membrane_coords[:, 2].min())
            interface_upper = float(membrane_coords[:, 2].max())
            interface_thickness = interface_upper - interface_lower
            if not np.isfinite(interface_thickness) or interface_thickness <= 0.0:
                raise ModuleConfigError("Membrane checkpoint has invalid lipid Z surfaces")

            cmin = coords.min(axis=0)
            cmax = coords.max(axis=0)
            nonmembrane_mask = np.ones(len(coords), dtype=bool)
            for component in membrane_comps:
                nonmembrane_mask[component.atom_indices] = False
            nonmembrane_z = coords[nonmembrane_mask, 2]
            required_padding = max(
                interface_lower - float(nonmembrane_z.min()) if len(nonmembrane_z) else 0.0,
                float(nonmembrane_z.max()) - interface_upper if len(nonmembrane_z) else 0.0,
                0.0,
            )
            if box_padding + 1e-6 < required_padding:
                raise ModuleConfigError(
                    f"Z Padding is measured from the lipid-water interfaces, but "
                    f"the protein/solute extends {required_padding:.2f} nm beyond an "
                    f"interface. Increase Z Padding to at least "
                    f"{required_padding:.2f} nm to keep every atom inside the box"
                )

            box_z = interface_thickness + 2.0 * box_padding
            box_dims = np.array([box_x, box_y, box_z])
            box_vectors = np.diag(box_dims)
            # Shift the membrane midplane to the box centre.  Protein
            # asymmetry must never change the two requested solvent layers.
            if len(coords) > 0:
                shift = np.array([
                    box_x / 2.0 - (cmax[0] + cmin[0]) / 2.0,
                    box_y / 2.0 - (cmax[1] + cmin[1]) / 2.0,
                    box_z / 2.0 - membrane_mid_z,
                ])
                if np.any(np.abs(shift) > 0.01):
                    system.structure.translate(shift)
                    coords = system.coordinates
            min_coords = np.zeros(3)
            max_coords = box_dims
            log.append(f"Solvation box: {box_x:.1f}×{box_y:.1f}×{box_z:.1f} nm "
                       f"(XY from membrane, lipid Z span {interface_thickness:.2f} "
                       f"+ 2×{box_padding:.1f} nm interface padding)")
        elif len(coords) == 0:
            # Empty system — use explicit box_size or existing dimensions
            box_size = config.get("box_size")
            if box_size is not None:
                dims = np.array(box_size, dtype=float)
                system.structure.box_vectors = np.diag(dims)
            else:
                dims = system.structure.dimensions()
            if not np.isfinite(dims).all() or np.any(dims <= 0):
                raise ModuleConfigError(
                    "Empty-system solvation requires a positive box_size"
                )
            min_coords = np.zeros(3)
            max_coords = dims
            box_dims = max_coords - min_coords
            box_vectors = np.diag(box_dims)
        else:
            # Solvator / no membrane box — compute from solute + padding
            cmin = coords.min(axis=0)
            cmax = coords.max(axis=0)
            box_dims = cmax - cmin + 2.0 * box_padding
            # Keep coordinates and the orthogonal box in the same [0, L]
            # frame. This makes CLI output, checkpoints and the Web viewer
            # agree and guarantees the requested padding on all six faces.
            system.structure.translate(box_padding - cmin)
            coords = system.coordinates
            min_coords = np.zeros(3)
            max_coords = box_dims
            box_vectors = np.diag(box_dims)
            log.append(f"Box from solute + {box_padding:.1f} nm padding: "
                       f"{box_dims[0]:.1f}×{box_dims[1]:.1f}×{box_dims[2]:.1f} nm")
        if not np.isfinite(box_dims).all() or np.any(box_dims <= 0):
            raise ModuleConfigError("Solvation produced invalid box dimensions")

        # ---- 2. Fill with water (pre-built box if available, else grid) ----
        seed = int(system.metadata.get("seed", config.get("seed", 42)))
        use_prebuilt = config.get("use_prebuilt_water", True)
        water_coords = None
        n_molecules = 0

        if use_prebuilt:
            water_coords, n_molecules = self._fill_from_prebuilt(
                box_dims, water_model_name, water_model, seed,
            )
            # _fill_from_prebuilt tiles from origin [0,0,0];
            # shift to align with solute region [min_coords, max_coords]
            if water_coords is not None and n_molecules > 0:
                water_coords += min_coords.reshape(1, 3)
        if water_coords is None:
            water_coords, n_molecules = self._generate_water_grid(
                min_coords, max_coords, water_model,
                spacing=self._WATER_SPACING, seed=seed,
            )
            log.append(f"Generated {n_molecules} water molecules (grid method)")
        else:
            log.append(f"Filled {n_molecules} water molecules from pre-built box")

        # ---- 3. Remove overlaps (per-molecule — avoid orphan hydrogens) ----
        if remove_overlap and n_molecules > 0 and len(coords) > 0:
            n_atoms_per_water = water_model.n_atoms
            # Reshape to (N_mol, n_atoms_per_water, 3) for per-molecule overlap check
            water_mols = water_coords.reshape(n_molecules, n_atoms_per_water, 3)
            # Per-element VDW radii for overlap detection (nm)
            _ELEM_VDW = {
                "H": 0.12, "C": 0.17, "N": 0.16, "O": 0.15, "S": 0.18,
                "P": 0.18, "F": 0.15, "CL": 0.18, "BR": 0.19, "I": 0.20,
                "NA": 0.23, "K": 0.28, "CA": 0.23, "MG": 0.17, "ZN": 0.16,
            }
            solute_vdw = np.array([
                _ELEM_VDW.get(
                    (system.structure.elements[i] if i < len(system.structure.elements) else "C").upper()[:2],
                    0.15)
                for i in range(len(coords))
            ])
            per_atom_vdw = [water_model.approximate_radius, 0.05, 0.05]
            if water_model.n_atoms == 4:
                per_atom_vdw.append(0.0)
            water_vdw = np.tile(per_atom_vdw, n_molecules)
            overlap = find_overlapping_atoms(
                water_coords, coords,
                vdw_radii_mobile=water_vdw,
                vdw_radii_fixed=solute_vdw,
                scale=overlap_scale,
                box_dimensions=box_dims,
            )
            # Per-molecule overlap: remove if ANY atom of the water overlaps
            mol_overlap = overlap.reshape(n_molecules, n_atoms_per_water).any(axis=1)
            water_mols = water_mols[~mol_overlap]
            n_removed = mol_overlap.sum()
            n_molecules = len(water_mols)
            water_coords = water_mols.reshape(-1, 3)
            log.append(f"Removed {n_removed} overlapping water molecules, {n_molecules} water molecules kept")

            # ---- 3b. Exclude water from membrane hydrophobic core ----
            if membrane_box_set and n_molecules > 0:
                # Find membrane Z boundaries (tail region, excluding headgroups)
                # Head groups are at ±dh/2, tails extend inward. We exclude
                # water from the tail-only region to prevent water penetration.
                memb_coords_list = []
                for comp in membrane_comps:
                    memb_coords_list.append(coords[comp.atom_indices])
                if memb_coords_list:
                    memb_coords = np.concatenate(memb_coords_list)
                    # Get Z midpoint of membrane, exclude inner core
                    # Water exclusion uses Z-range heuristic rather than per-atom headgroup mask.
                    memb_z = memb_coords[:, 2]
                    z_mid = (memb_z.min() + memb_z.max()) / 2.0
                    z_half_range = (memb_z.max() - memb_z.min()) * 0.30  # inner 60% of membrane
                    z_lo = z_mid - z_half_range
                    z_hi = z_mid + z_half_range
                    # Remove water molecules whose oxygen is in the membrane core
                    water_ow_z = water_mols[:, 0, 2]  # O is first atom of water
                    in_core = (water_ow_z > z_lo) & (water_ow_z < z_hi)
                    n_core_removed = in_core.sum()
                    if n_core_removed > 0:
                        water_mols = water_mols[~in_core]
                        n_molecules = len(water_mols)
                        water_coords = water_mols.reshape(-1, 3)
                        log.append(f"Membrane core exclusion: removed {n_core_removed} water molecules "
                                   f"(Z={z_lo:.1f}–{z_hi:.1f} nm)")

        # ---- 4. Build water Structure ----
        n_water_atoms = len(water_coords)
        atom_names = []
        resnames = []
        resids = []
        elements = []

        for m in range(n_molecules):
            for a in range(water_model.n_atoms):
                atom_names.append(water_model.atom_names[a])
                resnames.append("SOL")
                resids.append(m + 1)
                # Derive element from atom name: "OW"→"O", "HW1"→"H", "MW"→""
                aname = water_model.atom_names[a]
                if aname.startswith("O"):
                    elem = "O"
                elif aname.startswith("H"):
                    elem = "H"
                elif aname.upper().startswith("M") and "W" in aname.upper():
                    elem = ""  # virtual site — no real element
                else:
                    # Strip digits for multi-letter elements (e.g. "Na1"→"Na")
                    elem = "".join(ch for ch in aname if not ch.isdigit())
                elements.append(elem)

        water_structure = Structure(
            coordinates=water_coords,
            box_vectors=box_vectors,
            atom_names=atom_names,
            resnames=resnames,
            resids=resids,
            elements=elements,
        )

        water_system = System(structure=water_structure)

        # ---- 5. Merge into main system ----
        n_before = system.num_atoms
        merged = system.merge(water_system)

        merged.add_component(Component(
            name=f"SOLVENT_{water_model_name.upper()}",
            kind=ComponentKind.SOLVENT,
            atom_indices=np.arange(n_before, merged.num_atoms),
            metadata={
                "water_model": water_model_name,
                "n_molecules": n_molecules,
                "volume_nm3": float(np.prod(box_dims)),
            },
        ))

        # Update box
        merged.structure.box_vectors = box_vectors
        merged.metadata["water_model"] = water_model_name
        merged.metadata["solvation"] = {
            "water_model": water_model_name,
            "n_molecules": int(n_molecules),
            "box_padding": box_padding,
            "overlap_scale": overlap_scale,
            "box_dimensions_nm": box_dims.tolist(),
        }
        if membrane_box_set:
            merged.metadata["solvation"]["membrane_interface_z_nm"] = [
                box_padding,
                box_padding + interface_thickness,
            ]

        log.append(f"Total water atoms: {n_water_atoms} ({n_molecules} molecules)")

        return ModuleResult(
            success=True,
            system=merged,
            log=log,
        )

    def _generate_water_grid(
        self,
        min_coords: np.ndarray,
        max_coords: np.ndarray,
        water_model,
        spacing: float = 0.31,
        seed: int = 42,
    ) -> tuple[np.ndarray, int]:
        """Generate water oxygen positions on a 3D grid, then place hydrogens.

        Returns
        -------
        water_coords : (N*n_sites, 3) ndarray
        n_molecules : int
        """
        x_range = np.arange(min_coords[0] + spacing / 2, max_coords[0], spacing)
        y_range = np.arange(min_coords[1] + spacing / 2, max_coords[1], spacing)
        z_range = np.arange(min_coords[2] + spacing / 2, max_coords[2], spacing)

        # Oxygen positions
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        o_positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        n_molecules = len(o_positions)

        # Callers normally use pre-built boxes. The fallback still needs to
        # honor the selected model geometry and site count.
        oh_bond = water_model.oh_bond
        half_angle = np.radians(water_model.hoh_angle / 2.0)

        # Local H vectors in water molecule frame
        h1_local = np.array([oh_bond * np.sin(half_angle), 0.0, oh_bond * np.cos(half_angle)])
        h2_local = np.array([-oh_bond * np.sin(half_angle), 0.0, oh_bond * np.cos(half_angle)])

        all_coords = []
        rng = np.random.default_rng(seed)

        for o_pos in o_positions:
            # Full 3D rotation (random quaternion) for each water molecule
            # Avoids artificial orientational ordering from 2D-only rotation
            phi = rng.uniform(0, 2 * np.pi)
            theta = np.arccos(rng.uniform(-1, 1))
            psi = rng.uniform(0, 2 * np.pi)
            c1, s1 = np.cos(phi), np.sin(phi)
            c2, s2 = np.cos(theta), np.sin(theta)
            c3, s3 = np.cos(psi), np.sin(psi)
            rot = np.array([
                [c1*c3 - s1*c2*s3, -c1*s3 - s1*c2*c3, s1*s2],
                [s1*c3 + c1*c2*s3, -s1*s3 + c1*c2*c3, -c1*s2],
                [s2*s3, s2*c3, c2],
            ])
            h1 = o_pos + rot @ h1_local
            h2 = o_pos + rot @ h2_local

            all_coords.append(o_pos)
            all_coords.append(h1)
            all_coords.append(h2)
            if water_model.n_atoms == 4:
                m_local = np.array([0.0, 0.0, water_model.virtual_site_distance])
                all_coords.append(o_pos + rot @ m_local)

        return np.asarray(all_coords, dtype=np.float64).reshape(-1, 3), n_molecules

    def _fill_from_prebuilt(
        self,
        box_dims: np.ndarray,
        water_model_name: str,
        water_model,
        seed: int,
    ) -> tuple[np.ndarray | None, int]:
        """Fill the target box by tiling a pre-built water box.

        Falls back to None if the pre-built box file is not found, so the
        caller can use the grid method instead.

        Returns (coords, n_molecules) or (None, 0) on fallback.
        """
        from pathlib import Path

        # Locate bundled water box
        # Locate bundled water box relative to package data directory
        import gmxbuilder.data.water_boxes as _wb_pkg
        box_path = Path(_wb_pkg.__path__[0]) / f"{water_model_name}_water.gro"
        if not box_path.exists():
            return None, 0

        # Parse pre-built GRO
        raw = box_path.read_text()
        lines = raw.split("\n")
        if len(lines) < 3:
            return None, 0

        try:
            n_atoms_total = int(lines[1].strip())
        except ValueError:
            return None, 0
        if len(lines) < n_atoms_total + 3:
            return None, 0
        # Box line is the last non-empty line (handle trailing newline variance)
        box_line_str = None
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                box_line_str = stripped
                break
        if box_line_str is None:
            return None, 0
        box_line = box_line_str.split()
        if len(box_line) >= 3:
            wb_x = float(box_line[0])
            wb_y = float(box_line[1])
            wb_z = float(box_line[2])
        else:
            return None, 0
        if not np.isfinite([wb_x, wb_y, wb_z]).all() or min(wb_x, wb_y, wb_z) <= 0:
            return None, 0

        n_atoms_per_water = water_model.n_atoms
        if n_atoms_total <= 0 or n_atoms_total % n_atoms_per_water:
            return None, 0
        n_waters_per_box = n_atoms_total // n_atoms_per_water

        # Read coordinates
        wb_coords = np.zeros((n_atoms_total, 3), dtype=np.float64)
        for i in range(n_atoms_total):
            line = lines[2 + i]
            # GRO format: 5resid + 5resname + 5atom + 5atomid = 20 cols, then 8+8+8 coords
            try:
                wb_coords[i, 0] = float(line[20:28])
                wb_coords[i, 1] = float(line[28:36])
                wb_coords[i, 2] = float(line[36:44])
            except (ValueError, IndexError):
                return None, 0

        # ---- Tile the water box to fill target dimensions ----
        nx = max(1, int(np.ceil(box_dims[0] / wb_x)))
        ny = max(1, int(np.ceil(box_dims[1] / wb_y)))
        nz = max(1, int(np.ceil(box_dims[2] / wb_z)))

        # Reshape to (N_water, n_atoms, 3) for tiling
        wb_mols = wb_coords.reshape(n_waters_per_box, n_atoms_per_water, 3)
        tiled_mols = []
        rng = np.random.default_rng(seed)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    offset = np.array([ix * wb_x, iy * wb_y, iz * wb_z])
                    shifted = wb_mols + offset
                    tiled_mols.append(shifted)

        tiled = np.concatenate(tiled_mols, axis=0)  # (N_total, n_atoms, 3)
        n_tiled = len(tiled)

        # ---- Trim complete molecules to the target box ----
        keep = np.ones(n_tiled, dtype=bool)
        for a in range(n_atoms_per_water):
            for d in range(3):
                keep &= tiled[:, a, d] >= 0.0
                keep &= tiled[:, a, d] < box_dims[d]

        tiled = tiled[keep]
        n_kept = len(tiled)
        coords = tiled.reshape(-1, 3)

        # ---- Shuffle to break tiling seams ----
        perm = rng.permutation(n_kept)
        coords = coords.reshape(n_kept, n_atoms_per_water, 3)[perm].reshape(-1, 3)

        return coords, n_kept
