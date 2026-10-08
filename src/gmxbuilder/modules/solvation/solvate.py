"""Module 2: Aqueous-phase solvation builder.

Adds a water box around the existing system, removing overlapping
water molecules.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.chemistry import WATER_VOLUME_NM3
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.overlap import find_overlapping_atoms
from gmxbuilder.modules import register_module
from gmxbuilder.modules.solvation.membrane_exclusion import (
    GRO_ROUNDING_GUARD_NM,
    SURFACE_MARGIN_NM,
    assert_membrane_water_free,
    atoms_in_membrane,
)
from gmxbuilder.modules.solvation.water_models import WaterRegistry
from gmxbuilder.pipeline.base import BaseModule, ModuleResult

# Approximate volume per water molecule (nm^3)
_WATER_VOLUME_PER_MOLECULE = WATER_VOLUME_NM3


@register_module
class SolvationBuilder(BaseModule):
    """Add water molecules to solvate the system."""

    name = "solvation"
    description = "Add water box around system with overlap removal"

    _DEFAULT_PADDING = 1.5  # nm; consistent with the Solvator task default
    _WATER_SPACING = 0.31  # nm, approximate spacing between waters
    _MAX_BOX_DIMENSION_NM = 100.0
    _MAX_BOX_VOLUME_NM3 = 50_000.0
    _MAX_ESTIMATED_WATER_MOLECULES = 1_500_000
    _MAX_TRANSIENT_WATER_MOLECULES = 2_000_000

    @classmethod
    def _validate_box_budget(cls, dimensions, *, label: str) -> np.ndarray:
        try:
            dims = np.asarray(dimensions, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError(f"{label} must contain three finite dimensions") from exc
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0.0):
            raise ModuleConfigError(f"{label} must contain three positive finite dimensions")
        if np.any(dims > cls._MAX_BOX_DIMENSION_NM):
            raise ModuleConfigError(
                f"{label} dimensions must not exceed {cls._MAX_BOX_DIMENSION_NM:g} nm"
            )
        volume = float(np.prod(dims))
        if not np.isfinite(volume) or volume > cls._MAX_BOX_VOLUME_NM3:
            raise ModuleConfigError(
                f"{label} volume must not exceed {cls._MAX_BOX_VOLUME_NM3:g} nm³"
            )
        estimated_waters = int(np.ceil(volume / _WATER_VOLUME_PER_MOLECULE))
        if estimated_waters > cls._MAX_ESTIMATED_WATER_MOLECULES:
            raise ModuleConfigError(
                f"{label} would require approximately {estimated_waters} water molecules, "
                f"exceeding the supported maximum of {cls._MAX_ESTIMATED_WATER_MOLECULES}"
            )
        return dims

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "water_model",
                "box_padding",
                "overlap_scale",
                "box_size",
                "remove_overlap",
                "use_prebuilt_water",
                "seed",
            },
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
            self._validate_box_budget(box_size, label="box_size")
        for flag in ("remove_overlap", "use_prebuilt_water"):
            if flag in config and not isinstance(config[flag], bool):
                raise ModuleConfigError(f"{flag} must be a boolean")
        return True

    def run(self, system: System, config: dict) -> ModuleResult:
        # Commit geometry only through the returned result; errors preserve input.
        system = system.copy()
        from gmxbuilder.pipeline.progress import report_progress

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
        water_model_name = (
            str(locked_water_model or requested_water_model or "tip3p").strip().lower()
        )
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
        report_progress(0.05, "Sizing the solvation box")
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

            membrane_coords = np.concatenate(
                [coords[component.atom_indices] for component in membrane_comps]
            )
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
            solute_z_image_gap = (
                box_z - float(np.ptp(nonmembrane_z)) if len(nonmembrane_z) else None
            )
            if solute_z_image_gap is not None and solute_z_image_gap <= 1e-6:
                raise ModuleConfigError(
                    "Protein/solute Z envelopes touch their periodic images. "
                    "Increase Z Padding to leave solvent between periodic copies."
                )
            box_dims = np.array([box_x, box_y, box_z])
            box_vectors = np.diag(box_dims)
            # Shift the membrane midplane to the box centre.  Protein
            # asymmetry must never change the two requested solvent layers.
            if len(coords) > 0:
                # Whole lipids may extend across XY periodic boundaries.
                # Their raw bounds must not move an already centred solute.
                centre_xy = (cmax[:2] + cmin[:2]) / 2.0
                if nonmembrane_mask.any():
                    solute_xy = coords[nonmembrane_mask, :2]
                    centre_xy = (solute_xy.max(axis=0) + solute_xy.min(axis=0)) / 2.0
                shift = np.array(
                    [
                        box_x / 2.0 - centre_xy[0],
                        box_y / 2.0 - centre_xy[1],
                        box_z / 2.0 - membrane_mid_z,
                    ]
                )
                system.structure.translate(shift)
                coords = system.coordinates
            min_coords = np.zeros(3)
            max_coords = box_dims
            log.append(
                f"Solvation box: {box_x:.1f}×{box_y:.1f}×{box_z:.1f} nm "
                f"(XY from membrane, lipid Z span {interface_thickness:.2f} "
                f"+ 2×{box_padding:.1f} nm interface padding)"
            )
        elif len(coords) == 0:
            # Empty system — use explicit box_size or existing dimensions
            box_size = config.get("box_size")
            if box_size is not None:
                dims = np.array(box_size, dtype=float)
                system.structure.box_vectors = np.diag(dims)
            else:
                dims = system.structure.dimensions()
            if not np.isfinite(dims).all() or np.any(dims <= 0):
                raise ModuleConfigError("Empty-system solvation requires a positive box_size")
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
            log.append(
                f"Box from solute + {box_padding:.1f} nm padding: "
                f"{box_dims[0]:.1f}×{box_dims[1]:.1f}×{box_dims[2]:.1f} nm"
            )
        box_dims = self._validate_box_budget(box_dims, label="Solvation box")

        # ---- 2. Fill with water (pre-built box if available, else grid) ----
        report_progress(0.2, "Filling the box with water")
        seed = int(system.metadata.get("seed", config.get("seed", 42)))
        use_prebuilt = config.get("use_prebuilt_water", True)
        water_coords = None
        n_molecules = 0

        if use_prebuilt:
            water_coords, n_molecules = self._fill_from_prebuilt(
                box_dims,
                water_model_name,
                water_model,
                seed,
            )
            # _fill_from_prebuilt tiles from origin [0,0,0];
            # shift to align with solute region [min_coords, max_coords]
            if water_coords is not None and n_molecules > 0:
                water_coords += min_coords.reshape(1, 3)
        if water_coords is None:
            water_coords, n_molecules = self._generate_water_grid(
                min_coords,
                max_coords,
                water_model,
                spacing=self._WATER_SPACING,
                seed=seed,
            )
            log.append(f"Generated {n_molecules} water molecules (grid method)")
        else:
            log.append(f"Filled {n_molecules} water molecules from pre-built box")

        # ---- 3. Remove overlaps (per-molecule — avoid orphan hydrogens) ----
        report_progress(0.45, "Removing waters that overlap the solute")
        if remove_overlap and n_molecules > 0 and len(coords) > 0:
            n_atoms_per_water = water_model.n_atoms
            # Reshape to (N_mol, n_atoms_per_water, 3) for per-molecule overlap check
            water_mols = water_coords.reshape(n_molecules, n_atoms_per_water, 3)
            # Per-element VDW radii for overlap detection (nm)
            _ELEM_VDW = {
                "H": 0.12,
                "C": 0.17,
                "N": 0.16,
                "O": 0.15,
                "S": 0.18,
                "P": 0.18,
                "F": 0.15,
                "CL": 0.18,
                "BR": 0.19,
                "I": 0.20,
                "NA": 0.23,
                "K": 0.28,
                "CA": 0.23,
                "MG": 0.17,
                "ZN": 0.16,
            }
            solute_vdw = np.array(
                [
                    _ELEM_VDW.get(
                        (
                            system.structure.elements[i]
                            if i < len(system.structure.elements)
                            else "C"
                        ).upper()[:2],
                        0.15,
                    )
                    for i in range(len(coords))
                ]
            )
            per_atom_vdw = [water_model.approximate_radius, 0.05, 0.05]
            if water_model.n_atoms == 4:
                per_atom_vdw.append(0.0)
            water_vdw = np.tile(per_atom_vdw, n_molecules)
            overlap = find_overlapping_atoms(
                water_coords,
                coords,
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
            log.append(
                f"Removed {n_removed} overlapping water molecules, {n_molecules} water "
                f"molecules kept"
            )

        # This hard construction policy also applies when overlap removal is
        # disabled. Remove whole molecules if any site enters the lipid slab.
        slab_removed = 0
        if membrane_box_set and n_molecules:
            slab = (box_padding, box_padding + interface_thickness, box_dims[2])
            excluded = (
                atoms_in_membrane(
                    water_coords, slab, margin_nm=SURFACE_MARGIN_NM + GRO_ROUNDING_GUARD_NM
                )
                .reshape(n_molecules, water_model.n_atoms)
                .any(axis=1)
            )
            slab_removed = int(excluded.sum())
            water_coords = water_coords.reshape(-1, water_model.n_atoms, 3)[~excluded].reshape(
                -1, 3
            )
            n_molecules -= slab_removed
            log.append(
                f"Removed {slab_removed} waters from the complete lipid Z envelope "
                "(all XY positions, including pores; construction-only exclusion)"
            )

        # ---- 4. Build water Structure ----
        report_progress(0.8, "Building the solvent structure")
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
        report_progress(0.92, "Merging solvent into the system")
        n_before = system.num_atoms
        merged = system.merge(water_system)

        merged.add_component(
            Component(
                name=f"SOLVENT_{water_model_name.upper()}",
                kind=ComponentKind.SOLVENT,
                atom_indices=np.arange(n_before, merged.num_atoms),
                metadata={
                    "water_model": water_model_name,
                    "n_molecules": n_molecules,
                    "volume_nm3": float(np.prod(box_dims)),
                },
            )
        )

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
            merged.metadata["solvation"]["membrane_water_exclusion"] = {
                "policy": "whole_lipid_z_envelope_all_water_sites_v1",
                "removed_molecules": slab_removed,
            }
            merged.metadata["solvation"]["membrane_interface_z_nm"] = [
                box_padding,
                box_padding + interface_thickness,
            ]
            merged.metadata["solvation"]["solute_z_envelope_image_gap_nm"] = solute_z_image_gap
            if solute_z_image_gap is not None:
                log.append(
                    f"Solute Z-envelope separation from periodic copies: "
                    f"{solute_z_image_gap:.3f} nm; check against the simulation interaction range"
                )

        assert_membrane_water_free(merged)
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
        try:
            minimum = np.asarray(min_coords, dtype=float)
            maximum = np.asarray(max_coords, dtype=float)
            spacing = float(spacing)
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError("Water-grid bounds and spacing must be finite numbers") from exc
        if (
            minimum.shape != (3,)
            or maximum.shape != (3,)
            or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all()
            or not np.isfinite(spacing)
            or spacing <= 0.0
        ):
            raise ModuleConfigError("Water-grid bounds and spacing must be finite and positive")
        dimensions = maximum - minimum
        self._validate_box_budget(dimensions, label="Water-grid box")
        axis_counts = []
        for length in dimensions:
            ratio = float(length) / spacing
            if not np.isfinite(ratio) or ratio > self._MAX_ESTIMATED_WATER_MOLECULES:
                raise ModuleConfigError(
                    "Water-grid spacing would exceed the supported molecule budget"
                )
            axis_counts.append(int(np.ceil(ratio)))
        candidate_count = axis_counts[0] * axis_counts[1] * axis_counts[2]
        if candidate_count > self._MAX_ESTIMATED_WATER_MOLECULES:
            raise ModuleConfigError(
                f"Water grid would allocate up to {candidate_count} molecules, exceeding "
                f"the supported maximum of {self._MAX_ESTIMATED_WATER_MOLECULES}"
            )

        x_range = np.arange(minimum[0] + spacing / 2, maximum[0], spacing)
        y_range = np.arange(minimum[1] + spacing / 2, maximum[1], spacing)
        z_range = np.arange(minimum[2] + spacing / 2, maximum[2], spacing)

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

        all_coords = np.empty((n_molecules, water_model.n_atoms, 3), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for start in range(0, n_molecules, 65536):
            end = min(start + 65536, n_molecules)
            # The original loop consumes phi, cos(theta), psi in that order.
            draws = rng.random((end - start, 3))
            phi = draws[:, 0] * (2 * np.pi)
            theta = np.arccos(draws[:, 1] * 2 - 1)
            psi = draws[:, 2] * (2 * np.pi)
            c1, s1 = np.cos(phi), np.sin(phi)
            c2, s2 = np.cos(theta), np.sin(theta)
            c3, s3 = np.cos(psi), np.sin(psi)
            rotations = np.empty((end - start, 3, 3))
            rotations[:, 0, :] = np.column_stack(
                (c1 * c3 - s1 * c2 * s3, -c1 * s3 - s1 * c2 * c3, s1 * s2)
            )
            rotations[:, 1, :] = np.column_stack(
                (s1 * c3 + c1 * c2 * s3, -s1 * s3 + c1 * c2 * c3, -c1 * s2)
            )
            rotations[:, 2, :] = np.column_stack((s2 * s3, s2 * c3, c2))
            oxygen = o_positions[start:end]
            all_coords[start:end, 0] = oxygen
            all_coords[start:end, 1] = oxygen + rotations @ h1_local
            all_coords[start:end, 2] = oxygen + rotations @ h2_local
            if water_model.n_atoms == 4:
                m_local = np.array([0.0, 0.0, water_model.virtual_site_distance])
                all_coords[start:end, 3] = oxygen + rotations @ m_local
        return all_coords.reshape(-1, 3), n_molecules

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

        box_dims = self._validate_box_budget(box_dims, label="Pre-built water box")

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
        transient_waters = nx * ny * nz * n_waters_per_box
        if transient_waters > self._MAX_TRANSIENT_WATER_MOLECULES:
            raise ModuleConfigError(
                f"Pre-built water tiling would allocate {transient_waters} molecules, "
                f"exceeding the transient limit of {self._MAX_TRANSIENT_WATER_MOLECULES}"
            )

        # Reshape to (N_water, n_atoms, 3) for tiling
        wb_mols = wb_coords.reshape(n_waters_per_box, n_atoms_per_water, 3)
        tiled_mols = []
        rng = np.random.default_rng(seed)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    offset = np.array([ix * wb_x, iy * wb_y, iz * wb_z])
                    shifted = wb_mols + offset
                    keep = np.all((shifted >= 0.0) & (shifted < box_dims), axis=(1, 2))
                    tiled_mols.append(shifted[keep])

        tiled = np.concatenate(tiled_mols, axis=0)  # (N_total, n_atoms, 3)
        n_kept = len(tiled)
        coords = tiled.reshape(-1, 3)

        # ---- Shuffle to break tiling seams ----
        perm = rng.permutation(n_kept)
        coords = coords.reshape(n_kept, n_atoms_per_water, 3)[perm].reshape(-1, 3)

        return coords, n_kept
