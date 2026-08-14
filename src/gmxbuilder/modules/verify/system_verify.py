"""SystemVerificationModule — validate built system against 3D viewer preview.

Compares key geometric properties between the frontend 3D viewer's
representation and the actual built GRO file:
  - Box dimensions (X, Y, Z)
  - Protein center of mass
  - Membrane midplane Z position
  - Protein extent (min/max)

Also generates a preview PDB from the final system for visual comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.modules import register_module


# ---------------------------------------------------------------------------
# Tolerance thresholds
# ---------------------------------------------------------------------------

BOX_TOLERANCE_NM = 0.5          # nm — box dimensions must match within this
MEMBRANE_Z_TOLERANCE_NM = 0.5   # nm — membrane midplane Z within this
PROTEIN_COM_TOLERANCE_NM = 0.5  # nm — protein COM within this
PROTEIN_EXTENT_TOLERANCE_NM = 0.6  # nm — per-axis protein CA extent


@register_module
class SystemVerificationModule(BaseModule):
    """Compare the built system with the frontend 3D viewer preview.

    Runs **after** the Export module.  Reads the generated GRO file,
    extracts the protein + membrane geometry, and optionally compares
    with a frontend-supplied preview specification.

    When ``preview_config`` is present in the module config, a detailed
    metric comparison is performed and mismatches are reported as errors.
    When absent (e.g. CLI usage), the module still generates a ``preview.pdb``
    of the final system for visual inspection.
    """

    name = "verify"
    description = "Verify built system geometry matches frontend preview"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def validate_config(self, config: dict) -> bool:
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, system: System, config: dict) -> ModuleResult:
        output_dir = Path(config.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        log: list[str] = []
        errors: list[str] = []

        # ---- 1. Write preview PDB of the final system ----
        preview_path = output_dir / "preview.pdb"
        try:
            from gmxbuilder.io.pdb import PDBWriter
            PDBWriter.write(system.structure, preview_path,
                            title="GMXBUILDER System Verification Preview")
            log.append(f"Wrote preview PDB: {preview_path}")
        except Exception as exc:
            log.append(f"Warning: could not write preview PDB: {exc}")

        # ---- 2. Write preview geometry metadata ----
        built_metrics = self._compute_metrics(system)
        metrics_path = output_dir / "verification_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(built_metrics, fh, indent=2, default=_json_default)
        log.append(f"Wrote verification metrics: {metrics_path}")

        # ---- 3. Compare with frontend preview (if provided) ----
        preview_config = config.get("preview_config")
        if preview_config:
            comparison_errors = self._compare_metrics(built_metrics, preview_config)
            if comparison_errors:
                errors.extend(comparison_errors)
                log.append("✗ Verification FAILED — geometry mismatch detected")
            else:
                log.append("✓ Verification PASSED — built system matches 3D viewer preview")
        else:
            log.append("No frontend preview config — skipping comparison (CLI mode)")

        # ---- 4. Cross-check: read GRO and compare with system structure ----
        gro_path = output_dir / "input.gro"
        if gro_path.exists():
            try:
                gro_metrics = self._compute_metrics_from_gro(gro_path, system)
                if gro_metrics:
                    gro_errors = self._compare_gro_vs_system(built_metrics, gro_metrics)
                    if gro_errors:
                        errors.extend(gro_errors)
                        log.append("✗ GRO cross-check FAILED")
                    else:
                        log.append("✓ GRO cross-check PASSED — GRO matches in-memory system")
                else:
                    log.append("Warning: could not parse GRO for cross-check")
            except (IndexError, TypeError, ValueError) as exc:
                errors.append(f"GRO cross-check could not classify atoms safely: {exc}")
                log.append("✗ GRO cross-check FAILED")
        else:
            log.append("Note: input.gro not found — skipping GRO cross-check")

        if errors:
            log.append(f"\\n{len(errors)} verification error(s):")
            for e in errors:
                log.append(f"  • {e}")
            return ModuleResult(
                success=False,
                system=system,
                log=log,
            )

        return ModuleResult(success=True, system=system, log=log)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(system: System) -> dict:
        """Extract key geometric properties from the built system.

        Uses only CA (backbone) atoms for protein metrics so that comparison
        is robust against protonation / hydrogen addition / ACE capping,
        which change side-chain atom identities but not backbone geometry.
        """
        struct = system.structure
        coords = struct.coordinates
        box_dims = struct.dimensions().tolist()  # [a, b, c]

        # ---- Protein (CA atoms only) ----
        prot_indices: list[int] = []
        for comp in system.components:
            if comp.kind == ComponentKind.PROTEIN:
                for idx in comp.atom_indices:
                    idx_i = int(idx)
                    if idx_i < len(struct.atom_names) and struct.atom_names[idx_i] == "CA":
                        prot_indices.append(idx_i)

        prot_coords = coords[prot_indices] if prot_indices else None

        protein = None
        if prot_coords is not None and len(prot_coords) > 0:
            com = prot_coords.mean(axis=0).tolist()
            pmin = prot_coords.min(axis=0).tolist()
            pmax = prot_coords.max(axis=0).tolist()
            protein = {
                "center_of_mass_nm": [round(v, 3) for v in com],
                "min_nm": [round(v, 3) for v in pmin],
                "max_nm": [round(v, 3) for v in pmax],
                "extent_nm": [round(pmax[i] - pmin[i], 3) for i in range(3)],
                "n_atoms": len(prot_indices),
                "note": "CA atoms only",
            }

        # ---- Membrane ----
        memb_indices: list[int] = []
        for comp in system.components:
            if comp.kind == ComponentKind.MEMBRANE:
                memb_indices.extend(comp.atom_indices)

        memb_coords = coords[memb_indices] if memb_indices else None

        membrane = None
        if memb_coords is not None and len(memb_coords) > 0:
            z_vals = memb_coords[:, 2]
            membrane = {
                "midplane_z_nm": round(float(z_vals.mean()), 3),
                "min_z_nm": round(float(z_vals.min()), 3),
                "max_z_nm": round(float(z_vals.max()), 3),
                "thickness_nm": round(float(z_vals.max() - z_vals.min()), 3),
                "n_atoms": len(memb_indices),
            }

        return {
            "box_dimensions_nm": [round(v, 3) for v in box_dims],
            "total_atoms": int(system.num_atoms),
            "protein": protein,
            "membrane": membrane,
        }

    @staticmethod
    def _compute_metrics_from_gro(gro_path: Path, system: System) -> dict | None:
        """Read GRO file and compute metrics, using system components for atom classification."""
        try:
            from gmxbuilder.io.gro import GROReader
            gro_struct = GROReader().read(gro_path)
        except Exception:
            return None

        coords = gro_struct.coordinates
        box_dims = gro_struct.dimensions().tolist()
        if gro_struct.num_atoms != system.num_atoms:
            return {
                "box_dimensions_nm": [round(v, 3) for v in box_dims],
                "total_atoms": int(gro_struct.num_atoms),
                "protein": None,
                "membrane": None,
            }

        # Classify atoms by component using System metadata
        prot_indices: list[int] = []
        memb_indices: list[int] = []
        for comp in system.components:
            indices = [int(i) for i in comp.atom_indices]
            if any(index < 0 or index >= gro_struct.num_atoms for index in indices):
                raise ValueError(
                    f"component {comp.name!r} contains an atom index outside the GRO file"
                )
            if comp.kind == ComponentKind.PROTEIN:
                prot_indices.extend(indices)
            elif comp.kind == ComponentKind.MEMBRANE:
                memb_indices.extend(indices)

        prot_coords = coords[prot_indices] if prot_indices else None
        memb_coords = coords[memb_indices] if memb_indices else None

        protein = None
        if prot_coords is not None and len(prot_coords) > 0:
            com = prot_coords.mean(axis=0).tolist()
            pmin = prot_coords.min(axis=0).tolist()
            pmax = prot_coords.max(axis=0).tolist()
            protein = {
                "center_of_mass_nm": [round(v, 3) for v in com],
                "min_nm": [round(v, 3) for v in pmin],
                "max_nm": [round(v, 3) for v in pmax],
                "extent_nm": [round(pmax[i] - pmin[i], 3) for i in range(3)],
                "n_atoms": len(prot_indices),
            }

        membrane = None
        if memb_coords is not None and len(memb_coords) > 0:
            z_vals = memb_coords[:, 2]
            membrane = {
                "midplane_z_nm": round(float(z_vals.mean()), 3),
                "min_z_nm": round(float(z_vals.min()), 3),
                "max_z_nm": round(float(z_vals.max()), 3),
                "thickness_nm": round(float(z_vals.max() - z_vals.min()), 3),
                "n_atoms": len(memb_indices),
            }

        return {
            "box_dimensions_nm": [round(v, 3) for v in box_dims],
            "total_atoms": int(gro_struct.num_atoms),
            "protein": protein,
            "membrane": membrane,
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_metrics(built: dict, preview: dict) -> list[str]:
        """Compare built metrics against frontend preview metrics.

        Because the frontend viewer renders the protein at/near the origin while
        the builder shifts everything to the box center, we compare:
          - Box dimensions: absolute (should match)
          - Protein extent: absolute (should match — shape is invariant)
          - Protein COM relative to membrane midplane: relative (should be ~0)
          - Membrane midplane relative to box center: relative
        """
        errors: list[str] = []

        # ---- Box dimensions (absolute) ----
        built_box = built.get("box_dimensions_nm", [0, 0, 0])
        preview_box = preview.get("box_dimensions_nm", [0, 0, 0])
        for i, axis in enumerate(["X", "Y", "Z"]):
            bv = built_box[i] if i < len(built_box) else 0
            pv = preview_box[i] if i < len(preview_box) else 0
            if abs(bv - pv) > BOX_TOLERANCE_NM:
                errors.append(
                    f"Box {axis}: built={bv:.3f} nm, preview={pv:.3f} nm "
                    f"(diff={abs(bv - pv):.3f} nm > tolerance {BOX_TOLERANCE_NM} nm)"
                )

        built_prot = built.get("protein")
        preview_prot = preview.get("protein")
        built_memb = built.get("membrane")
        preview_memb = preview.get("membrane")

        # ---- Protein COM Z relative to membrane midplane ----
        if built_prot and preview_prot and built_memb and preview_memb:
            built_com = built_prot.get("center_of_mass_nm", [0, 0, 0])
            preview_com = preview_prot.get("center_of_mass_nm", [0, 0, 0])

            built_memb_z = built_memb.get("midplane_z_nm", 0)
            preview_memb_z = preview_memb.get("midplane_z_nm", 0)

            # Protein COM Z relative to membrane midplane (should be near 0)
            built_rel_z = built_com[2] - built_memb_z
            preview_rel_z = preview_com[2] - preview_memb_z
            if abs(built_rel_z - preview_rel_z) > MEMBRANE_Z_TOLERANCE_NM:
                errors.append(
                    f"Protein COM Z (relative to membrane): built={built_rel_z:.3f} nm, "
                    f"preview={preview_rel_z:.3f} nm "
                    f"(diff={abs(built_rel_z - preview_rel_z):.3f} nm > tolerance {MEMBRANE_Z_TOLERANCE_NM} nm)"
                )

        # ---- Protein extent (absolute — invariant under translation) ----
        if built_prot and preview_prot:
            built_ext = built_prot.get("extent_nm", [0, 0, 0])
            preview_ext = preview_prot.get("extent_nm", [0, 0, 0])
            for i, axis in enumerate(["X", "Y", "Z"]):
                bv = built_ext[i] if i < len(built_ext) else 0
                pv = preview_ext[i] if i < len(preview_ext) else 0
                if abs(bv - pv) > PROTEIN_EXTENT_TOLERANCE_NM:
                    errors.append(
                        f"Protein extent {axis}: built={bv:.3f} nm, preview={pv:.3f} nm "
                        f"(diff={abs(bv - pv):.3f} nm)"
                    )

        # ---- Membrane thickness (range check — actual lipid Z extent exceeds DHH model) ----
        if built_memb and preview_memb:
            built_thick = built_memb.get("thickness_nm", 0)
            preview_half = preview_memb.get("half_thickness_nm")
            if preview_half is not None and built_thick > 0:
                preview_dhh = preview_half * 2.0  # DHH from frontend
                # Actual lipid geometry spans ~1.5–3× DHH (headgroups + tails)
                min_expected = preview_dhh * 0.8
                max_expected = preview_dhh * 3.0
                if built_thick < min_expected:
                    errors.append(
                        f"Membrane too thin: built={built_thick:.3f} nm, expected >{min_expected:.3f} nm "
                        f"(DHH={preview_dhh:.1f} nm)"
                    )
                elif built_thick > max_expected:
                    errors.append(
                        f"Membrane too thick: built={built_thick:.3f} nm, expected <{max_expected:.3f} nm "
                        f"(DHH={preview_dhh:.1f} nm)"
                    )

        return errors

    @staticmethod
    def _compare_gro_vs_system(sys_metrics: dict, gro_metrics: dict) -> list[str]:
        """Cross-check: verify GRO file matches in-memory system structure."""
        errors: list[str] = []

        if sys_metrics.get("total_atoms") != gro_metrics.get("total_atoms"):
            errors.append(
                "GRO atom count differs from the checked system: "
                f"system={sys_metrics.get('total_atoms')}, "
                f"gro={gro_metrics.get('total_atoms')}"
            )

        sys_box = sys_metrics.get("box_dimensions_nm", [0, 0, 0])
        gro_box = gro_metrics.get("box_dimensions_nm", [0, 0, 0])
        # GRO box vs system box should be near-identical (same data, different serialization)
        tight_tol = 0.01  # nm — should match exactly
        for i, axis in enumerate(["X", "Y", "Z"]):
            sv = sys_box[i] if i < len(sys_box) else 0
            gv = gro_box[i] if i < len(gro_box) else 0
            if abs(sv - gv) > tight_tol:
                errors.append(
                    f"GRO vs System box {axis}: system={sv:.3f} nm, gro={gv:.3f} nm"
                )

        # Protein COM
        sys_prot = sys_metrics.get("protein")
        gro_prot = gro_metrics.get("protein")
        if sys_prot and gro_prot:
            sys_com = sys_prot.get("center_of_mass_nm", [0, 0, 0])
            gro_com = gro_prot.get("center_of_mass_nm", [0, 0, 0])
            for i, axis in enumerate(["X", "Y", "Z"]):
                sv = sys_com[i] if i < len(sys_com) else 0
                gv = gro_com[i] if i < len(gro_com) else 0
                if abs(sv - gv) > tight_tol:
                    errors.append(
                        f"GRO vs System protein COM {axis}: system={sv:.3f} nm, gro={gv:.3f} nm"
                    )

        # Membrane midplane
        sys_memb = sys_metrics.get("membrane")
        gro_memb = gro_metrics.get("membrane")
        if sys_memb and gro_memb:
            sz = sys_memb.get("midplane_z_nm", 0)
            gz = gro_memb.get("midplane_z_nm", 0)
            if abs(sz - gz) > tight_tol:
                errors.append(
                    f"GRO vs System membrane midplane Z: system={sz:.3f} nm, gro={gz:.3f} nm"
                )

        return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
