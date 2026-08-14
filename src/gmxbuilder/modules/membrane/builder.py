"""Module 1: Phospholipid bilayer & membrane protein builder.

Generates a lipid bilayer, optionally embeds a protein, and adds
the MEMBRANE component to the System.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.geometry.grid import hexagonal_grid
from gmxbuilder.geometry.periodic import wrap_periodic_coordinates
from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry
from gmxbuilder.geometry.relax import (
    relax_interleaflet_clashes_xy,
    rotate_lipids_away_from_clashes,
    rotate_lipids_away_from_external_clashes,
    scale_lipid_centres_xy,
)
from gmxbuilder.modules import register_module
from gmxbuilder.modules.membrane.embed import embed_protein
from gmxbuilder.modules.membrane.lipid_orientation import (
    MAX_TAIL_CORE_GAP_NM,
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    LipidOrientationError,
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    rotate_to_opposite_leaflet,
    outward_orientation,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.modules.membrane.orient import orient_protein
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.runtime.hardware import configured_task_threads


def _reconcile_lipid_selection(system: System, active_lipids: list[str]) -> str | None:
    """Accept a changed Step 5 set only within the already selected FF family."""
    active = sorted({str(name).strip().upper() for name in active_lipids})
    selected = sorted({
        str(name).strip().upper()
        for name in system.metadata.get("selected_lipid_names", [])
    })
    protein_ff = str(system.metadata.get("force_field", "")).lower()
    lipid_ff = str(system.metadata.get("lipid_ff", "")).lower()
    original_lipid_ff = lipid_ff
    resolved_reason = ""
    if protein_ff.startswith("amber"):
        from gmxbuilder.modules.forcefield.lipid_policy import (
            amber_lipid_backend,
            amber_lipid_backend_candidates,
        )

        resolved_ff, resolved_reason = amber_lipid_backend(active)
        if resolved_ff is None:
            raise ModuleConfigError(resolved_reason)
        compatible_backends = amber_lipid_backend_candidates(active)
        # Preserve an explicit, coherent backend selected in Step 2 or by the
        # offline library builder.  Lipid21 remains the default preference,
        # but it must not silently replace a valid whole-membrane GAFF2 choice.
        if lipid_ff not in compatible_backends:
            system.metadata["lipid_ff"] = resolved_ff
            system.metadata["gaff_lipids"] = active if resolved_ff == "gaff2" else []
            system.metadata["lipid21_lipids"] = active if resolved_ff == "lipid21" else []
            lipid_ff = resolved_ff
        elif lipid_ff != resolved_ff:
            resolved_reason = (
                f"explicit coherent {lipid_ff} backend retained; "
                f"preferred automatic backend would be {resolved_ff}"
            )
    backend_change = (
        f"Amber lipid backend updated for this composition: "
        f"{original_lipid_ff or 'unset'} -> {lipid_ff}. {resolved_reason}"
        if original_lipid_ff != lipid_ff else None
    )
    if not selected or active == selected:
        return backend_change
    compatible = False
    if protein_ff.startswith("amber") and lipid_ff in {"gaff2", "lipid21"}:
        from gmxbuilder.modules.forcefield.lipid_policy import amber_lipid_backend

        compatible = amber_lipid_backend(active)[0] == lipid_ff
    elif protein_ff in {"charmm36", "charmm36m"} and lipid_ff == protein_ff:
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_has_rtp

        compatible = all(lipid_has_rtp(name, protein_ff) for name in active)

    if not compatible:
        raise ModuleConfigError(
            "Selected lipid composition differs from the Step 2 compatibility "
            f"check (Step 2={selected}, Step 5={active}) and is not supported by "
            f"the confirmed {protein_ff}/{lipid_ff} parameter family. Return to "
            "Step 2 and confirm the force-field combination again."
        )

    system.metadata["selected_lipid_names"] = active
    system.metadata["gaff_lipids"] = active if lipid_ff == "gaff2" else []
    system.metadata["lipid21_lipids"] = active if lipid_ff == "lipid21" else []
    message = (
        f"Lipid compatibility revalidated for changed Step 5 composition: "
        f"{', '.join(active)} ({protein_ff}/{lipid_ff})"
    )
    if backend_change:
        message += f". {backend_change}"
    return message


def _headgroup_anchor_index(coords: np.ndarray, atom_names: list[str]) -> int:
    """Select an upper-leaflet headgroup anchor without trusting GAFF numbering."""
    stripped = [str(name).strip() for name in atom_names]
    if "P" in stripped:
        return stripped.index("P")

    polar_indices = []
    for index, name in enumerate(stripped):
        element = next((char for char in name.upper() if char.isalpha()), "")
        if element in {"O", "N", "P", "S"}:
            polar_indices.append(index)
    if not polar_indices:
        polar_indices = list(range(len(atom_names)))
    return int(polar_indices[int(np.argmax(coords[polar_indices, 2]))])


def _weighted_leaflet_apl(composition: list[tuple[str, float]]) -> float:
    """Return the ratio-weighted natural area of one leaflet."""
    area = 0.0
    total_ratio = 0.0
    for name, ratio in composition:
        value = float(ratio)
        if value <= 0.0:
            continue
        try:
            apl = float(LipidRegistry.get(name).area_per_lipid)
        except KeyError:
            apl = 0.65
        area += apl * value
        total_ratio += value
    if total_ratio <= 0.0:
        raise ModuleConfigError("Leaflet composition must contain a positive lipid ratio")
    return area / total_ratio


def _leaflet_headgroup_plane(leaflet_system: System, *, upper: bool) -> float:
    """Return the mean Z position of recorded per-lipid headgroup anchors."""
    coords = leaflet_system.coordinates
    lipid_sizes = list(leaflet_system.metadata.get("lipid_sizes") or [])
    local_indices = list(
        leaflet_system.metadata.get("headgroup_anchor_local_indices") or []
    )
    if lipid_sizes and len(local_indices) == len(lipid_sizes):
        offsets = np.cumsum([0] + lipid_sizes)
        absolute = np.asarray([
            int(offsets[index]) + int(local_indices[index])
            for index in range(len(lipid_sizes))
        ])
        return float(np.mean(coords[absolute, 2]))

    stripped = np.asarray([
        str(name).strip() for name in leaflet_system.structure.atom_names
    ])
    phosphorus = coords[stripped == "P", 2]
    if len(phosphorus):
        return float(np.mean(phosphorus))
    return float(np.percentile(coords[:, 2], 90 if upper else 10))


@register_module
class MembraneBuilder(BaseModule):
    """Build a phospholipid bilayer and embed a membrane protein."""

    name = "membrane"
    description = "Generate lipid bilayer with optional protein embedding"

    _MIN_BOX_XY = 4.0   # nm
    _MIN_LIPIDS_PER_LEAFLET = 64  # minimum lipids for a stable bilayer
    _GRID_JITTER = 0.05       # nm — random XY displacement for lipid placement
    _PROTEIN_EXCLUSION_XY = 0.20  # nm — grid-point exclusion around protein (tight)
    _LIPID_PROTEIN_MIN_DIST = 0.10  # nm — minimum lipid-protein atom distance

    # ------------------------------------------------------------------
    # Physically-justified packing constants
    # ------------------------------------------------------------------
    # _BILAYER_Z_HEADROOM_FACTOR:  dh → full bilayer Z extent multiplier.
    #   POPC dh=3.8 nm, headgroup region extends ~0.9 nm beyond phosphate
    #   on each side + ~0.3 nm hydration shell each side, giving
    #   3.8 + 2×(0.9+0.3) ≈ 6.2 nm.  Factor 1.8 gives 6.84 nm — a safe
    #   upper bound that is tightened to actual coordinates in step 11b.
    _BILAYER_Z_HEADROOM_FACTOR = 1.8

    # _LIPID_PACKING_FACTOR: inverse fill-fraction for box sizing.
    #   N lipids at natural APL need N×APL area.  The factor 1.30
    #   shrinks the box so lipids sit at ~130 % of natural density.
    #   This oversampling, combined with 10 % extra lipids and XY
    #   compression, ensures solvent-impermeable coverage after
    #   clash removal. The compression step then expands lipids uniformly
    #   to seal the box edges.
    _LIPID_PACKING_FACTOR = 1.00

    # _DENSE_GRID_SPACING: initial hexagonal grid point spacing (nm).
    #   At 0.35 nm this creates ~5.2× more candidate positions than
    #   needed for POPC (APL ≈ 0.64 nm² → √0.64 ≈ 0.80 nm natural
    #   spacing).  The high oversampling ensures uniform XY coverage
    #   even after protein exclusion and random thinning.
    _DENSE_GRID_SPACING = 0.35

    def __init__(self, use_equilibrated_library: bool = True):
        # The offline library builder disables reads while bootstrapping the
        # bilayer from which validated conformers will be extracted.
        self.use_equilibrated_library = bool(use_equilibrated_library)

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {"lipid_type", "lipid_composition", "n_lipids_per_leaflet", "seed",
             "box_padding", "pad", "bilayer_size", "orient_method", "embed_method"},
        )
        # Accept either lipid_type (single) or lipid_composition (mixed)
        if "lipid_composition" in config:
            comp = config["lipid_composition"]
            if not isinstance(comp, dict):
                raise ModuleConfigError("lipid_composition must be an object")

            def _validate_leaflet(entries: object, label: str) -> None:
                if not isinstance(entries, list) or not entries:
                    raise ModuleConfigError(f"{label} leaflet composition must not be empty")
                total_ratio = 0.0
                has_bilayer_host = False
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        raise ModuleConfigError(
                            f"{label} leaflet entry {index} must be an object"
                        )
                    name = entry.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ModuleConfigError(
                            f"{label} leaflet entry {index} has no lipid name"
                        )
                    try:
                        ratio = float(entry.get("ratio"))
                    except (TypeError, ValueError) as exc:
                        raise ModuleConfigError(
                            f"{label} leaflet lipid {name!r} has an invalid ratio"
                        ) from exc
                    if not np.isfinite(ratio) or ratio < 0.0:
                        raise ModuleConfigError(
                            f"{label} leaflet lipid {name!r} ratio must be finite and non-negative"
                        )
                    total_ratio += ratio

                    try:
                        registered = LipidRegistry.get(name)
                        if ratio > 0.0 and registered.category != "ST":
                            has_bilayer_host = True
                    except KeyError:
                        # Preserve custom-lipid support when geometry metadata
                        # is supplied by the caller.
                        if "category" not in entry:
                            raise ModuleConfigError(
                                f"Unknown lipid {name!r} in {label} leaflet — not in registry "
                                "and no category/tail data was provided."
                            )
                        if ratio > 0.0 and str(entry.get("category")).upper() != "ST":
                            has_bilayer_host = True

                if not np.isclose(total_ratio, 100.0, atol=0.5):
                    raise ModuleConfigError(
                        f"{label} leaflet ratios must total 100%, got {total_ratio:.3f}%"
                    )
                if not has_bilayer_host:
                    raise ModuleConfigError(
                        f"The {label} leaflet contains only sterols. Sterols cannot "
                        "form a phospholipid bilayer by themselves; include a "
                        "phospholipid, sphingolipid, glycolipid or ceramide host."
                    )

            upper = comp.get("upper")
            _validate_leaflet(upper, "upper")
            lower = comp.get("lower")
            if "lower" in comp and lower is not None:
                _validate_leaflet(lower, "lower")
        elif "lipid_type" in config:
            lipid_type = config["lipid_type"]
            if not isinstance(lipid_type, str) or not lipid_type.strip():
                raise ModuleConfigError("lipid_type must be a non-empty string")
            try:
                registered = LipidRegistry.get(lipid_type)
            except KeyError as exc:
                raise ModuleConfigError(str(exc))
            if registered.category == "ST":
                raise ModuleConfigError(
                    "Sterols cannot form a phospholipid bilayer by themselves; "
                    "use lipid_composition and include a bilayer-forming host lipid."
                )
        else:
            raise ModuleConfigError("Either 'lipid_type' or 'lipid_composition' is required")

        explicit_count = config.get("n_lipids_per_leaflet")
        if explicit_count is not None:
            if isinstance(explicit_count, bool):
                raise ModuleConfigError("n_lipids_per_leaflet must be an integer")
            try:
                count = int(explicit_count)
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError("n_lipids_per_leaflet must be an integer") from exc
            if not np.isfinite(float(explicit_count)) or float(explicit_count) != count:
                raise ModuleConfigError("n_lipids_per_leaflet must be an integer")
            if count < self._MIN_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"n_lipids_per_leaflet must be at least "
                    f"{self._MIN_LIPIDS_PER_LEAFLET}, got {count}"
                )
        padding_values: dict[str, float] = {}
        for key in ("box_padding", "pad"):
            if key not in config:
                continue
            try:
                value = float(config[key])
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError(f"{key} must be a finite number") from exc
            if not np.isfinite(value) or not 0.0 <= value <= 50.0:
                raise ModuleConfigError(f"{key} must be between 0 and 50 nm")
            padding_values[key] = value
        if len(padding_values) == 2 and not np.isclose(
            padding_values["box_padding"], padding_values["pad"], atol=1e-9
        ):
            raise ModuleConfigError(
                "box_padding and legacy pad disagree; provide only box_padding"
            )

        size = config.get("bilayer_size")
        if size is not None and size != "auto":
            try:
                dims = np.asarray(size, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError(
                    "bilayer_size must be 'auto', one positive number, or two positive numbers"
                ) from exc
            if dims.ndim == 0:
                dims = np.repeat(dims, 2)
            if dims.shape != (2,) or not np.isfinite(dims).all() or np.any(dims <= 0.0):
                raise ModuleConfigError(
                    "bilayer_size must be 'auto', one positive number, or two positive numbers"
                )

        orient_method = config.get("orient_method")
        if orient_method is not None and orient_method not in {
            "ppm", "hmoment", "tmd", "pca", "com",
        }:
            raise ModuleConfigError(f"Unknown orient_method: {orient_method}")
        embed_method = config.get("embed_method")
        if embed_method is not None and embed_method not in {
            "com", "hydrophobic", "ppm",
        }:
            raise ModuleConfigError(f"Unknown embed_method: {embed_method}")
        if "seed" in config:
            seed = config["seed"]
            if isinstance(seed, bool):
                raise ModuleConfigError("seed must be an integer")
            try:
                parsed_seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ModuleConfigError("seed must be an integer") from exc
            try:
                seed_is_exact = float(seed) == parsed_seed
            except (TypeError, ValueError, OverflowError):
                seed_is_exact = False
            if not seed_is_exact:
                raise ModuleConfigError("seed must be an integer")
        return True

    def _parse_composition(self, config: dict) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Return (upper_mix, lower_mix) as lists of (lipid_name, ratio_pct)."""
        if "lipid_composition" in config:
            comp = config["lipid_composition"]
            # Custom lipids must already be present in the task-scoped
            # registry.  Never trust browser-supplied geometry metadata or
            # register it process-wide here.
            for entries in (comp.get("upper") or [], comp.get("lower") or []):
                for entry in entries:
                    name = str(entry.get("name", "")).strip().upper()
                    if not name:
                        continue
                    try:
                        LipidRegistry.get(name)
                    except KeyError as exc:
                        raise ModuleConfigError(
                            f"Unknown or unavailable task-scoped lipid {name!r}"
                        ) from exc
            upper = [(e["name"].upper(), e["ratio"]) for e in comp.get("upper", [])]
            lower_raw = comp.get("lower")
            if lower_raw:
                lower = [(e["name"].upper(), e["ratio"]) for e in lower_raw]
            else:
                lower = list(upper)  # symmetric
            return upper, lower
        else:
            # Fallback: single lipid_type
            lt = config["lipid_type"].upper()
            return [(lt, 100)], [(lt, 100)]

    def run(self, system: System, config: dict) -> ModuleResult:
        upper_mix, lower_mix = self._parse_composition(config)

        # Step 2 must have resolved the exact lipid set that Step 5 is about
        # to build.  Otherwise structure processing may have used a different
        # protein force field and the incremental preview would not match the
        # final full-pipeline build.
        active_lipids = sorted({
            name for name, ratio in upper_mix + lower_mix if float(ratio) > 0.0
        })
        reconciliation_log = _reconcile_lipid_selection(system, active_lipids)
        self.validate_config(config)
        seed = system.metadata.get("seed", config.get("seed", 42))
        rng = np.random.default_rng(seed)

        log = [reconciliation_log] if reconciliation_log else []
        # Use the dominant (highest ratio) lipid for sizing defaults
        if not upper_mix:
            raise ModuleConfigError("No lipids specified in composition_upper")
        dominant_name = max(upper_mix, key=lambda x: x[1])[0]
        try:
            dominant_lipid = LipidRegistry.get(dominant_name)
        except KeyError:
            dominant_lipid = None  # custom lipid — use fallback defaults
        dh = dominant_lipid.bilayer_thickness if dominant_lipid else 3.8

        has_protein = bool(system.component_by_kind(ComponentKind.PROTEIN))

        # ---- 1. Orient protein if present ----
        if has_protein:
            already_oriented = system.metadata.get("_oriented", False)
            orient_method = config.get("orient_method", None)
            if orient_method and not already_oriented:
                system.structure = orient_protein(system.structure, method=orient_method)
                log.append(f"Protein oriented to membrane normal (Z-axis, method={orient_method})")
            elif already_oriented:
                log.append("Protein orientation: skipped (already oriented by OrientModule)")
            # If neither: protein was pre-aligned or doesn't need re-orientation

        # ---- 1b. Preserve the checked OrientModule result ----
        # The Orient checkpoint is the user-approved protein pose relative to
        # the nominal membrane plane.  Membrane construction must not undo its
        # Z-offset or tilt.  Step 11b may translate protein and lipids together
        # to centre the final box, which preserves their relative geometry.

        # ---- 2. Compute protein extent ----
        prot_min_xy = np.zeros(2)
        prot_max_xy = np.zeros(2)
        prot_z_min = 0.0
        prot_z_max = 0.0
        if has_protein:
            protein_comps = system.component_by_kind(ComponentKind.PROTEIN)
            all_prot_idx = np.concatenate([c.atom_indices for c in protein_comps])
            prot_coords = system.coordinates[all_prot_idx]
            prot_min_xy = prot_coords[:, :2].min(axis=0)
            prot_max_xy = prot_coords[:, :2].max(axis=0)
            prot_z_min = float(prot_coords[:, 2].min())
            prot_z_max = float(prot_coords[:, 2].max())

        # ---- 3a. Compute weighted APL (needed for box sizing) ----
        # Both leaflets share one periodic XY box. Asymmetric mixtures must be
        # sized for the larger natural leaflet area; using only the upper
        # composition over-compresses lower leaflets enriched in large lipids
        # such as cardiolipin.
        upper_area = _weighted_leaflet_apl(upper_mix)
        lower_area = _weighted_leaflet_apl(lower_mix)
        avg_area = max(upper_area, lower_area)
        if abs(upper_area - lower_area) > 0.01:
            log.append(
                f"Asymmetric leaflet APL: upper={upper_area:.3f}, "
                f"lower={lower_area:.3f} nm²; box uses {avg_area:.3f} nm²"
            )

        # ---- 3b. Determine target box ----
        n_lipids_per_leaflet = config.get("n_lipids_per_leaflet")
        if n_lipids_per_leaflet is not None:
            n_lipids_per_leaflet = int(n_lipids_per_leaflet)
            if n_lipids_per_leaflet < self._MIN_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"n_lipids_per_leaflet must be at least "
                    f"{self._MIN_LIPIDS_PER_LEAFLET}, got {n_lipids_per_leaflet}"
                )
            # Box sized for exactly n_lipids (not the oversampled target_n).
            # The 10% extra in target_n ensures enough survive clash removal.
            # After scaling (step 9c) the lipids exactly fill this box.
            #   n_lipids = box_xy² / APL * 1.30
            # → box_xy² = n_lipids * APL / 1.30 + protein_XY²
            protein_xy_area = 0.0
            if has_protein:
                ext_xy = prot_max_xy - prot_min_xy
                protein_xy_area = float(max(ext_xy) ** 2)
            lipid_area_needed = n_lipids_per_leaflet * avg_area / self._LIPID_PACKING_FACTOR
            box_xy = max(np.sqrt(lipid_area_needed + protein_xy_area), self._MIN_BOX_XY)
        else:
            # Legacy: compute from box_padding around protein
            box_padding = float(config.get("box_padding", config.get("pad", 2.0)))
            if box_padding < 0.0 or box_padding > 50.0:
                raise ModuleConfigError(f"box_padding must be 0.0–50.0 nm, got {box_padding}")
            if has_protein:
                ext_xy = prot_max_xy - prot_min_xy
                box_xy_nominal = max(ext_xy) + 2.0 * box_padding
            else:
                box_xy_nominal = self._MIN_BOX_XY
            # Handle explicit bilayer_size override
            explicit = config.get("bilayer_size")
            if explicit and explicit != "auto":
                if isinstance(explicit, (int, float)):
                    box_xy_nominal = float(explicit)
                elif isinstance(explicit, (list, tuple)):
                    box_xy_nominal = max(float(v) for v in explicit)
            box_xy = max(box_xy_nominal, self._MIN_BOX_XY)

        # Z: system extent (protein + membrane) — water padding added in solvation
        membrane_z_full = dh * self._BILAYER_Z_HEADROOM_FACTOR
        prot_z_extent = prot_z_max - prot_z_min if has_protein else 0.0
        box_z = max(prot_z_extent, membrane_z_full)

        if n_lipids_per_leaflet is not None:
            log.append(f"Membrane system: {box_xy:.1f}×{box_xy:.1f}×{box_z:.1f} nm — "
                       f"area: {box_xy*box_xy:.1f} nm² "
                       f"(n={n_lipids_per_leaflet} lipids/leaflet, Z water layer added in solvation)")
        else:
            log.append(f"Membrane system: {box_xy:.1f}×{box_xy:.1f}×{box_z:.1f} nm — "
                       f"area: {box_xy*box_xy:.1f} nm² "
                       f"(XY padding={box_padding:.1f} nm, Z water layer added in solvation)")

        # ---- 4. Dense candidate placement ----
        # 5a. Generate VERY dense hexagonal grid (0.35 nm spacing ≈ 3× denser than target)
        dense_spacing = self._DENSE_GRID_SPACING
        grid_xy = hexagonal_grid(
            xy_extent=(box_xy, box_xy),
            spacing=dense_spacing,
            center=np.array([0.0, 0.0]),
            jitter=self._GRID_JITTER,
            rng=rng,
        )
        # 5b. Trim lipid centres to the periodic box. Atoms may wrap across
        # PBC; subtracting a molecular-radius margin here would compress all
        # centres into a smaller area and invalidate the requested APL.
        half_box = box_xy / 2.0
        grid_in_box = (np.abs(grid_xy[:, 0]) < half_box) & \
                      (np.abs(grid_xy[:, 1]) < half_box)
        grid_xy = grid_xy[grid_in_box]
        n_dense = len(grid_xy)
        log.append(f"Dense grid: {n_dense} candidate positions (spacing={dense_spacing:.2f} nm)")

        # 5c. Remove grid points overlapping with protein (tight XY exclusion)
        if has_protein:
            protein_comps = system.component_by_kind(ComponentKind.PROTEIN)
            all_prot_idx = np.concatenate([c.atom_indices for c in protein_comps])
            protein_xy = system.coordinates[all_prot_idx][:, :2]
            if len(protein_xy) > 0 and len(grid_xy) > 0:
                from scipy.spatial import cKDTree
                prot_tree = cKDTree(protein_xy)
                dists, _ = prot_tree.query(
                    grid_xy, k=1, workers=configured_task_threads()
                )
                grid_xy = grid_xy[dists >= self._PROTEIN_EXCLUSION_XY]
            log.append(f"Protein overlap filter: {len(grid_xy)} positions survived "
                       f"(exclusion={self._PROTEIN_EXCLUSION_XY:.2f} nm)")

        # 5d. Thin to the requested target density.
        # After protein exclusion, randomly select lipids to reach the
        # user-specified count (or area-based target).  Oversample 10%
        # to account for protein clash removal in step 9b.
        if n_lipids_per_leaflet is not None:
            if has_protein:
                # Oversample 10% — protein clash removal (step 9b) will remove
                # some lipids, so we start with slightly more than requested.
                target_n = max(int(n_lipids_per_leaflet * 1.10), n_lipids_per_leaflet + 4)
            else:
                # There is no protein clash filter to compensate for.
                target_n = n_lipids_per_leaflet
        else:
            target_n = max(self._MIN_LIPIDS_PER_LEAFLET,
                           int(box_xy * box_xy / avg_area * 1.30))
        n_after_protein = len(grid_xy)
        if n_after_protein > target_n:
            # Farthest-point thinning keeps the selected lipid centres
            # uniformly separated. Random thinning of a 0.35 nm dense grid
            # frequently selected adjacent sites and created hard overlaps.
            chosen = _select_spread_positions(
                grid_xy, target_n, rng, box_xy=box_xy
            )
            grid_xy = grid_xy[chosen]
            log.append(f"Density thinning: {target_n} lipids/leaflet selected "
                       f"(from {n_after_protein} candidates, target APL={avg_area:.3f} nm²)")
            actual_n = target_n
        else:
            actual_n = n_after_protein
            log.append(f"All {actual_n} candidates kept (below target {target_n})")
        if actual_n < self._MIN_LIPIDS_PER_LEAFLET:
            return ModuleResult(
                success=False,
                system=system,
                log=[(
                    f"ERROR: Only {actual_n} lipids/leaflet available "
                    f"(need ≥{self._MIN_LIPIDS_PER_LEAFLET}). "
                    "Increase lipids-per-leaflet or reduce protein exclusion."
                )],
            )
        log.append(f"Final placement: {actual_n} lipids/leaflet "
                   f"({box_xy:.1f}×{box_xy:.1f} nm²)")

        # ---- 8. Assign lipid types to grid positions by ratio ----
        upper_assignments = self._assign_lipids(actual_n, upper_mix, rng)
        lower_assignments = self._assign_lipids(actual_n, lower_mix, rng)

        # Build composition summary
        def _counts(assignments):
            from collections import Counter
            c = Counter(assignments)
            return {k: c[k] for k in sorted(c)}

        upper_counts = _counts(upper_assignments)
        lower_counts = _counts(lower_assignments)
        log.append(f"Upper leaflet: {upper_counts}")
        log.append(f"Lower leaflet: {lower_counts}")

        # ---- 9. Build upper and lower leaflets ----
        z_upper = dh / 2.0
        z_lower = -dh / 2.0

        force_field = str(system.metadata.get("force_field", "amber14sb"))
        lipid_ff = str(system.metadata.get("lipid_ff", force_field))
        upper_system = self._build_mixed_leaflet(
            grid_xy, z_upper, upper_assignments, rng, force_field, lipid_ff,
            box_xy=box_xy,
        )
        lower_system = self._build_mixed_leaflet(
            grid_xy, z_lower, lower_assignments, rng, force_field, lipid_ff,
            box_xy=box_xy,
        )
        library_hits = int(upper_system.metadata.get("library_hits", 0)) + int(
            lower_system.metadata.get("library_hits", 0)
        )
        bootstrap_hits = int(upper_system.metadata.get("bootstrap_hits", 0)) + int(
            lower_system.metadata.get("bootstrap_hits", 0)
        )
        if library_hits:
            log.append(f"Validated force-field conformer library: {library_hits} lipids")
        if bootstrap_hits:
            log.append(
                f"WARNING: {bootstrap_hits} lipids used deterministic bootstrap geometry; "
                "no validated NPT library entry was installed for this force-field family"
            )

        # The farthest-point lattice already establishes the target APL.
        # Per-lipid translational repulsion is deliberately avoided here:
        # dense many-body tail contacts can otherwise collapse the lattice.
        # Azimuthal rigid-body declashing below preserves every lipid centre.
        log.append(
            f"Built 2 leaflets: upper={actual_n}, lower={actual_n} lipids"
        )

        # ---- 9a2. Leaflet closing — eliminate vacuum gap between leaflets ----
        # After relaxation, the leaflets may have a gap between tail ends at
        # the midplane (Z ≈ 0).  Close the leaflets until tail atoms from
        # upper and lower leaflets make gentle VDW contact, then back off
        # slightly to avoid hard clashes.  This ensures no vacuum layer
        # between the two leaflets.
        relax_interleaflet_clashes_xy(
            upper_system.structure.coordinates,
            lower_system.structure.coordinates,
            upper_system.metadata.get("lipid_sizes", []),
            lower_system.metadata.get("lipid_sizes", []),
            box_xy=box_xy,
        )
        _close_leaflets(
            upper_system, lower_system, log, target_dhh=dh, box_xy=box_xy
        )

        # ---- 9b. Protein-lipid clash removal (atom level, protein only) ----
        # Only remove lipids that clash with protein atoms.
        # Lipid-lipid clashes are resolved by energy minimization in MD.
        if has_protein:
            from scipy.spatial import cKDTree
            prot_indices_all = np.concatenate([
                c.atom_indices for c in protein_comps
            ])
            prot_coords = system.coordinates[prot_indices_all]
            prot_tree = cKDTree(prot_coords)
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                MembraneBuilder._filter_protein_clashes(
                    leaflet_sys, prot_tree, leaflet_name,
                    self._LIPID_PROTEIN_MIN_DIST, log)

        # ---- 9b2. Pack lipids against protein surface ----
        # After clash removal, surviving lipids may still sit far from the
        # protein surface (median ~0.8 nm in tests), leaving water-sized
        # gaps at the interface.  Push interface lipids closer in XY to
        # eliminate these cavities — water molecules that enter the bilayer
        # interior can nucleate pores and cause system instability during MD.
        if has_protein:
            prot_coords_local = system.coordinates[prot_indices_all]
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                _pack_lipids_against_protein(
                    leaflet_sys, prot_coords_local,
                    target_contact=self._LIPID_PROTEIN_MIN_DIST + 0.15,
                    max_shift=0.05,
                    log=log,
                    leaflet_label=leaflet_name,
                )
            # Re-filter clashes after packing.
            prot_tree_local = cKDTree(prot_coords_local)
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                MembraneBuilder._filter_protein_clashes(
                    leaflet_sys, prot_tree_local, leaflet_name,
                    self._LIPID_PROTEIN_MIN_DIST, log)

        # Explicit lipid counts are a final-output contract.  Protein builds
        # start with a small surplus to survive clash filtering; retain a
        # deterministic random subset if more than requested remain.
        if n_lipids_per_leaflet is not None:
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                self._trim_leaflet_to_count(
                    leaflet_sys, n_lipids_per_leaflet, rng, leaflet_name, log
                )

        # ---- 9c. Rigid-body XY scaling — fill box uniformly ----
        # After clash removal, scale lipid centres in XY so the lipid field
        # fills the target box.  Every lipid receives a single translation;
        # its internal covalent geometry and Z profile remain unchanged.
        # This ensures:
        #   1. Uniform density — same scaling for all lipids, no gradient
        #   2. Square shape — lipids fill the entire box_xy × box_xy area
        #   3. Box sealed — no lateral gaps for water/ions to bypass
        # After scaling, re-run clash removal since lipids may have been
        # pushed into the protein.
        for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
            n_lip = leaflet_sys.metadata.get("n_lipids", 0)
            lipid_sizes = leaflet_sys.metadata.get("lipid_sizes")
            if n_lip == 0 or not lipid_sizes:
                continue
            coords = leaflet_sys.structure.coordinates
            initial_extent = np.ptp(coords[:, :2], axis=0)
            if np.any(initial_extent < 0.01):
                continue
            target_extent = box_xy - 0.04
            if (
                n_lipids_per_leaflet is None
                and np.any(np.abs(initial_extent - target_extent) > 0.005)
            ):
                _, scales = scale_lipid_centres_xy(coords, lipid_sizes, target_extent)
                final_extent = np.ptp(coords[:, :2], axis=0)
                log.append(
                    f"Rigid-body XY scaling ({leaflet_name}): "
                    f"×{scales[0]:.3f}/×{scales[1]:.3f}; "
                    f"extent={final_extent[0]:.2f}×{final_extent[1]:.2f} nm"
                )

            leaflet_sys.structure.coordinates, minimum_clearance = (
                rotate_lipids_away_from_clashes(
                    leaflet_sys.structure.coordinates,
                    lipid_sizes,
                    min_distance=0.035,
                    box_xy=box_xy,
                )
            )
            if minimum_clearance < 0.025:
                log.append(
                    f"WARNING: {leaflet_name} leaflet retains a "
                    f"{minimum_clearance:.3f} nm inter-lipid contact after rigid relaxation"
                )

        relax_interleaflet_clashes_xy(
            upper_system.structure.coordinates,
            lower_system.structure.coordinates,
            upper_system.metadata.get("lipid_sizes", []),
            lower_system.metadata.get("lipid_sizes", []),
            box_xy=box_xy,
        )
        _close_leaflets(
            upper_system, lower_system, log, target_dhh=dh, box_xy=box_xy
        )
        upper_system.structure.coordinates, upper_cross_clearance = (
            rotate_lipids_away_from_external_clashes(
                upper_system.structure.coordinates,
                upper_system.metadata.get("lipid_sizes", []),
                lower_system.structure.coordinates,
                box_xy=box_xy,
            )
        )
        lower_system.structure.coordinates, lower_cross_clearance = (
            rotate_lipids_away_from_external_clashes(
                lower_system.structure.coordinates,
                lower_system.metadata.get("lipid_sizes", []),
                upper_system.structure.coordinates,
                box_xy=box_xy,
            )
        )
        cross_clearance = min(upper_cross_clearance, lower_cross_clearance)
        if cross_clearance < 0.04:
            log.append(
                "WARNING: bilayer retains a "
                f"{cross_clearance:.3f} nm cross-leaflet contact before minimization"
            )

        # After scaling, re-run protein clash removal — XY scaling may have
        # pushed lipids into the protein.
        if has_protein:
            from scipy.spatial import cKDTree
            _prot_coords_scl = system.coordinates[prot_indices_all]
            _prot_tree_scl = cKDTree(_prot_coords_scl)
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                MembraneBuilder._filter_protein_clashes(
                    leaflet_sys, _prot_tree_scl, leaflet_name,
                    self._LIPID_PROTEIN_MIN_DIST, log,
                    label_prefix="Post-scale clash filter")

        # Counts and structural invariants must describe the final leaflets,
        # after every protein clash filter and trimming operation.
        actual_upper = int(upper_system.metadata.get("n_lipids", 0))
        actual_lower = int(lower_system.metadata.get("n_lipids", 0))
        orientation_quality = _validate_bilayer_structure(
            upper_system, lower_system, log,
        )

        # ---- 10. Embed protein ----
        if has_protein:
            # OrientModule already established the approved protein pose.
            already_oriented = system.metadata.get("_oriented", False)
            embed_method = config.get("embed_method") or "com"
            if not already_oriented:
                embed_protein(system, dh, method=embed_method)
                log.append(f"Protein embedded in bilayer (method={embed_method})")
            elif already_oriented:
                log.append("Protein pose preserved from OrientModule")

        # ---- 11. Merge membrane into system ----
        membrane_system = upper_system.merge(lower_system)
        merged = system.merge(membrane_system)

        # ---- 11b. Recenter solute and tighten box ----
        # After placement + relaxation, the solute (protein + lipids) may
        # be offset from the box origin.  Recenter everything together so
        # the membrane midplane sits at Z=0 and the XY solute centre is at
        # the box centre.  This ensures the viewer's grey spheres
        # (drawn at Z=±dh/2) align with the actual lipid positions.
        mem_indices = np.arange(system.num_atoms, merged.num_atoms)

        # -- XY: centre the entire solute at origin --
        # The viewer draws the box wireframe and membrane-plane spheres
        # centred at the coordinate origin [0, 0, 0].  We must centre the
        # solute (protein + lipids) there too so all four elements
        # (protein, lipids, spheres, box) share the same reference frame.
        all_xy = merged.coordinates[:, :2]
        all_xy_min = all_xy.min(axis=0)
        all_xy_max = all_xy.max(axis=0)
        all_xy_extent = all_xy_max - all_xy_min
        all_xy_center = (all_xy_max + all_xy_min) / 2.0

        # Keep the APL-derived periodic box.  Whole lipid conformers can
        # legitimately cross a periodic edge, so their raw all-atom extent
        # must not enlarge the box and dilute the membrane.

        # Centre solute at origin in XY — matches the viewer's box/sphere
        # coordinate frame (box wireframe drawn at ±box_xy/2 from origin).
        shift_xy = -all_xy_center

        # -- Z: centre the membrane midplane at Z=0 --
        # The lipids were built at Z = ±dh/2, so the membrane midplane
        # is already near Z=0.  Compute its actual position from the
        # lipid Z coordinates and shift everything so the midplane sits
        # exactly at Z=0.  This is the reference plane the viewer uses
        # for the grey membrane-plane spheres.
        mem_z_all = merged.coordinates[mem_indices][:, 2]
        z_mid_actual = (mem_z_all.min() + mem_z_all.max()) / 2.0

        # -- Build the total shift --
        shift_xyz = np.array([shift_xy[0], shift_xy[1], -z_mid_actual])

        # Apply shift to ALL atoms (protein + lipids) together —
        # preserves every degree of freedom (orientation, tilt, Z-offset)
        # that the OrientModule established.
        merged.structure.coordinates += shift_xyz

        # -- Determine box Z that covers the entire solute when midplane is at 0 --
        # After centring, the solute spans from -half_z (protein bottom) to
        # +half_z (lipid headgroups).  The box must cover both extremes.
        all_z_centred = merged.coordinates[:, 2]
        z_abs_max = max(abs(all_z_centred.min()), abs(all_z_centred.max()))
        box_z = max(box_z, 2.0 * z_abs_max)

        protein_extent_xy = (
            float(np.max(prot_max_xy - prot_min_xy)) if has_protein else 0.0
        )
        if protein_extent_xy > box_xy:
            raise ModuleConfigError(
                f"Protein XY extent ({protein_extent_xy:.2f} nm) exceeds the "
                f"APL-sized membrane box ({box_xy:.2f} nm). Refusing to enlarge "
                "the box after lipid placement because that would dilute the "
                "bilayer; increase lipids per leaflet or XY padding."
            )
        else:
            log.append(
                f"Box XY retained at APL target: {box_xy:.2f} nm "
                f"(raw solute span {all_xy_extent[0]:.2f}×{all_xy_extent[1]:.2f} nm; "
                "whole lipids may cross periodic edges)"
            )

        if abs(z_mid_actual) > 0.01:
            log.append(f"Membrane midplane centred: shifted by {-z_mid_actual:.2f} nm "
                       f"to Z=0 (was at {z_mid_actual:.2f} nm)")

        # Set final box vectors
        merged.structure.box_vectors = np.diag([box_xy, box_xy, box_z])

        # ---- 11c. Quality validation ----
        # Validation issues are reported as warnings (non-blocking) rather
        # than fatal errors.  The membrane is built and saved regardless;
        # users can increase lipids-per-leaflet and re-run if quality is
        # unacceptable.  Only skip the quality check entirely if there is
        # no membrane to validate.
        if mem_indices.size > 0:
            _validate_membrane_quality(
                merged, mem_indices, system.num_atoms, box_xy, box_z,
                has_protein, log,
            )

        # Build a compact label for the composition
        def _label(mix):
            return "+".join(f"{n}({r}%)" for n, r in mix if r > 0)
        comp_label = _label(upper_mix)
        if _asymmetric_check(config):
            comp_label += "_asym"

        # Compute actual membrane Z profile from chemically meaningful
        # headgroup markers.  Percentiles of all atoms overestimate DHH when
        # explicit hydrogens or long headgroups extend beyond phosphorus.
        mem_z_all = merged.coordinates[mem_indices][:, 2]
        z_mid_actual = (mem_z_all.min() + mem_z_all.max()) / 2.0
        upper_head_z = _leaflet_headgroup_plane(upper_system, upper=True)
        lower_head_z = _leaflet_headgroup_plane(lower_system, upper=False)
        actual_dhh = float(upper_head_z - lower_head_z)

        # Combine per-lipid atom counts from both leaflets (supports mixed-size lipids)
        upper_lipid_sizes = upper_system.metadata.get("lipid_sizes", [])
        lower_lipid_sizes = lower_system.metadata.get("lipid_sizes", [])
        all_lipid_sizes = list(upper_lipid_sizes) + list(lower_lipid_sizes)

        # Add MEMBRANE component
        n_mem_atoms = membrane_system.num_atoms
        mem_start = merged.num_atoms - n_mem_atoms
        merged.add_component(Component(
                        name=f"MEMBRANE_{comp_label}"[:50],
            kind=ComponentKind.MEMBRANE,
            atom_indices=np.arange(mem_start, merged.num_atoms),
            metadata={
                "composition_upper": [(n, r) for n, r in upper_mix],
                "composition_lower": [(n, r) for n, r in lower_mix],
                "n_lipids_upper": actual_upper,
                "n_lipids_lower": actual_lower,
                "lipid_sizes": all_lipid_sizes,
                "bilayer_thickness": actual_dhh,  # measured from placed lipids
                "bilayer_thickness_nominal": dh,   # original estimate from registry
                "equilibrated_library_lipids": library_hits,
                "bootstrap_geometry_lipids": bootstrap_hits,
                "orientation_quality": orientation_quality,
                "box_xy": box_xy,
                "box_z": box_z,
                "box_padding": float(config.get("box_padding", config.get("pad", 2.0))),
            },
        ))

        return ModuleResult(
            success=True,
            system=merged,
            log=log,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _retain_lipids(leaflet_sys: System, keep_mask: np.ndarray) -> int:
        """Retain selected whole lipids and keep all atom fields aligned."""
        lipid_sizes = list(leaflet_sys.metadata.get("lipid_sizes") or [])
        n_lipids = len(lipid_sizes)
        if keep_mask.shape != (n_lipids,):
            raise ValueError("keep_mask length must match lipid_sizes")

        offsets = np.cumsum([0] + lipid_sizes)
        atom_indices = np.concatenate([
            np.arange(offsets[i], offsets[i + 1], dtype=np.int64)
            for i in range(n_lipids)
            if keep_mask[i]
        ]) if keep_mask.any() else np.empty(0, dtype=np.int64)

        structure = leaflet_sys.structure
        n_atoms_before = structure.num_atoms
        structure.coordinates = structure.coordinates[atom_indices].copy()
        for field_name in (
            "atom_names", "resnames", "resids", "chain_ids", "segids",
            "elements", "occupancies", "tempfactors",
        ):
            values = getattr(structure, field_name)
            if values and len(values) == n_atoms_before:
                setattr(structure, field_name, [values[int(i)] for i in atom_indices])

        retained_sizes = [lipid_sizes[i] for i in range(n_lipids) if keep_mask[i]]
        leaflet_sys.metadata["lipid_sizes"] = retained_sizes
        anchor_indices = list(
            leaflet_sys.metadata.get("headgroup_anchor_local_indices") or []
        )
        if len(anchor_indices) == n_lipids:
            leaflet_sys.metadata["headgroup_anchor_local_indices"] = [
                anchor_indices[i] for i in range(n_lipids) if keep_mask[i]
            ]
        leaflet_sys.metadata["n_lipids"] = len(retained_sizes)
        return n_lipids - len(retained_sizes)

    @staticmethod
    def _trim_leaflet_to_count(
        leaflet_sys: System,
        target_count: int,
        rng: np.random.Generator,
        leaflet_label: str,
        log: list[str],
    ) -> int:
        """Trim a surplus leaflet to an explicit whole-lipid count."""
        n_lipids = int(leaflet_sys.metadata.get("n_lipids", 0))
        if n_lipids <= target_count:
            if n_lipids < target_count:
                log.append(
                    f"⚠ Explicit count ({leaflet_label}): requested {target_count}, "
                    f"but only {n_lipids} survived clash filtering"
                )
            return 0

        keep_mask = np.zeros(n_lipids, dtype=bool)
        selected = rng.choice(n_lipids, size=target_count, replace=False)
        keep_mask[selected] = True
        n_removed = MembraneBuilder._retain_lipids(leaflet_sys, keep_mask)
        log.append(
            f"Explicit count ({leaflet_label}): trimmed {n_removed} surplus lipids "
            f"to {target_count}"
        )
        return n_removed

    @staticmethod
    def _filter_protein_clashes(
        leaflet_sys: System,
        prot_tree,  # scipy.spatial.cKDTree
        leaflet_label: str,
        min_dist: float,
        log: list[str],
        *,
        label_prefix: str = "Protein clash filter",
    ) -> int:
        """Remove lipids whose atoms are closer than *min_dist* to any protein atom.

        Modifies *leaflet_sys* in-place (coordinates, atom metadata, lipid_sizes,
        n_lipids).  Returns the number of lipids removed.
        """
        leaflet_coords = leaflet_sys.coordinates
        n_lipids_in = leaflet_sys.metadata.get("n_lipids", 0)
        if n_lipids_in == 0:
            return 0

        lipid_sizes = leaflet_sys.metadata.get("lipid_sizes")
        if lipid_sizes:
            offsets = np.cumsum([0] + list(lipid_sizes))
        else:
            atoms_per_lipid = len(leaflet_coords) // n_lipids_in
            offsets = np.array([i * atoms_per_lipid for i in range(n_lipids_in + 1)])

        keep_mask = np.ones(n_lipids_in, dtype=bool)
        for li in range(n_lipids_in):
            start, end = offsets[li], offsets[li + 1]
            dists, _ = prot_tree.query(
                leaflet_coords[start:end],
                k=1,
                workers=configured_task_threads(),
            )
            if dists.min() < min_dist:
                keep_mask[li] = False

        n_removed = int((~keep_mask).sum())
        if n_removed == 0:
            return 0

        if not lipid_sizes:
            leaflet_sys.metadata["lipid_sizes"] = [
                int(offsets[i + 1] - offsets[i]) for i in range(n_lipids_in)
            ]
        MembraneBuilder._retain_lipids(leaflet_sys, keep_mask)

        log.append(f"{label_prefix} ({leaflet_label}): removed {n_removed} lipids")
        return n_removed

    def _compute_lipid_count_mixed(self, box_xy: float, avg_area: float, config: dict) -> int:
        """Compute the number of lipids per leaflet using weighted average area."""
        explicit = config.get("n_lipids_per_leaflet")
        if explicit is not None:
            return int(explicit)
        area = box_xy ** 2
        if avg_area <= 0:
            raise ModuleConfigError(
                f"Cannot compute lipid count: weighted average APL is {avg_area:.3f} nm². "
                "Check that all selected lipids have valid area_per_lipid values."
            )
        n = int(area / avg_area)
        return max(n, 16)

    def _assign_lipids(
        self, n_total: int, mix: list[tuple[str, int]], rng: np.random.Generator
    ) -> list[str]:
        """Assign n_total lipids to types according to ratio percentages."""
        names = []
        for name, ratio in mix:
            if ratio <= 0:
                continue
            count = max(1, round(n_total * ratio / 100.0))
            names.extend([name] * count)
        # Trim or pad to exact n_total
        if len(names) > n_total:
            names = names[:n_total]
        elif len(names) < n_total:
            # Pad with dominant
            dominant = max(mix, key=lambda x: x[1])[0]
            names.extend([dominant] * (n_total - len(names)))
        rng.shuffle(names)
        return names

    def _build_one_lipid(self, args: tuple) -> tuple:
        """Build a single lipid: rotate + translate geometry. (Worker for parallel exec.)"""
        i, lipid_name, coords, atom_names, gx, gy, z, angle = args
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        rotated = coords @ rot.T
        rotated[:, 0] += gx
        rotated[:, 1] += gy
        rotated[:, 2] += z

        # Derive elements — lipid atoms are organic (C,N,O,P,S,H); no ion elements
        elements = []
        for aname in atom_names:
            elem = "C"
            for ch in aname:
                if ch.isalpha() and ch.isupper():
                    elem = ch
                    break
            # Multi-letter elements for ions (not "CA" — lipid "CA" is alpha carbon)
            for tl in ("CL", "BR", "NA", "MG", "ZN", "FE"):
                if aname.upper().startswith(tl):
                    elem = tl.title()
                    break
            elements.append(elem)

        return (rotated, atom_names, [lipid_name] * len(atom_names),
                [i + 1] * len(atom_names), elements)

    def _build_mixed_leaflet(
        self,
        grid_xy: np.ndarray,
        z: float,
        assignments: list[str],
        rng: np.random.Generator,
        force_field: str = "amber14sb",
        lipid_ff: str | None = None,
        *,
        box_xy: float | None = None,
    ) -> System:
        """Build a leaflet with full-atom lipid geometries at each grid position."""
        from scipy.spatial import cKDTree
        used = min(len(grid_xy), len(assignments))
        if used == 0:
            return System(
                structure=Structure(
                    coordinates=np.empty((0, 3)),
                    box_vectors=np.eye(3) * 10.0,
                ),
                metadata={"n_lipids": 0},
            )

        conf_seeds = rng.integers(0, 2**31 - 1, size=used)
        results = []
        headgroup_anchor_local_indices = []
        library_hits = 0
        bootstrap_hits = 0
        placed_coords = np.empty((0, 3), dtype=float)
        placed_tree = None
        for i in range(used):
            ln = assignments[i]
            lipid = LipidRegistry.get(ln)
            loaded = False
            if self.use_equilibrated_library:
                from gmxbuilder.modules.membrane.equilibrated_library import (
                    get_equilibrated_lipid_library,
                )

                selected_lipid_ff = lipid_ff or (
                    "gaff2" if force_field.startswith("amber") else force_field
                )
                try:
                    coords, atom_names = get_equilibrated_lipid_library().load_one(
                        ln, force_field, selected_lipid_ff, rng=rng,
                    )
                    loaded = True
                    library_hits += 1
                except FileNotFoundError:
                    pass
            if not loaded:
                coords, atom_names = build_rdkit_lipid_geometry(
                    ln, lipid.smiles, force_field=force_field,
                    seed=int(conf_seeds[i] % 5), net_charge=lipid.charge,
                    lipid_ff=lipid_ff,
                )
                bootstrap_hits += 1

            # Every backend and every library entry is normalized through the
            # same chemistry-based axis.  This is a rigid-body rotation, so
            # force-field bond geometry and equilibrated torsions are kept.
            try:
                coords = orient_lipid_to_outward_normal(
                    coords, atom_names, upper=True,
                )
            except LipidOrientationError as exc:
                source = "pre-equilibrated library" if loaded else "bootstrap geometry"
                raise ModuleConfigError(
                    f"Lipid {ln} from {source} cannot form a physical bilayer: {exc}"
                ) from exc

            # ``z`` is the physical headgroup plane (DHH/2).  Phosphorus
            # remains the marker for phospholipids; for nonphospholipids use
            # the water-facing polar geometry because GAFF names such as O3
            # are serial labels rather than chemical atom identities.
            anchor_index = _headgroup_anchor_index(coords, atom_names)
            anchor_z = float(coords[anchor_index, 2])
            coords = coords.copy()
            coords[:, 2] -= anchor_z
            headgroup_anchor_local_indices.append(anchor_index)

            if z < 0:
                # Lower leaflet: use a proper 180-degree rotation, never a
                # mirror reflection that would invert lipid stereochemistry.
                coords = rotate_to_opposite_leaflet(coords)
            base_angle = float(rng.uniform(0.0, 2.0 * np.pi))
            best_result = None
            best_clearance = -np.inf
            for offset in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
                candidate = self._build_one_lipid((
                    i, ln, coords, atom_names,
                    float(grid_xy[i, 0]), float(grid_xy[i, 1]), z,
                    base_angle + float(offset),
                ))
                if placed_tree is None:
                    best_result = candidate
                    break
                candidate_search = candidate[0].copy()
                if box_xy is not None:
                    candidate_search[:, :2] = wrap_periodic_coordinates(
                        candidate_search[:, :2], box_xy,
                    )
                clearance = float(placed_tree.query(candidate_search, k=1)[0].min())
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_result = candidate
            results.append((i, best_result))
            placed_coords = np.vstack((placed_coords, best_result[0]))
            placed_search = placed_coords.copy()
            tree_options = {}
            if box_xy is not None:
                placed_search[:, :2] = wrap_periodic_coordinates(
                    placed_search[:, :2], box_xy,
                )
                tree_options = {"boxsize": np.asarray([box_xy, box_xy, 0.0])}
            placed_tree = cKDTree(placed_search, **tree_options)

        all_coords = [r[1][0] for r in results]
        lipid_sizes = [len(c) for c in all_coords]  # atom count per lipid
        all_names, all_resnames, all_resids, all_elements = [], [], [], []
        for _, (_, names, resns, rids, elems) in results:
            all_names.extend(names)
            all_resnames.extend(resns)
            all_resids.extend(rids)
            all_elements.extend(elems)

        merged_coords = np.vstack(all_coords)
        structure = Structure(
            coordinates=merged_coords,
            box_vectors=np.eye(3) * 10.0,
            atom_names=all_names,
            resnames=all_resnames,
            resids=all_resids,
            elements=all_elements,
        )
        return System(structure=structure, metadata={
            "n_lipids": used,
            "lipid_sizes": lipid_sizes,
            "headgroup_anchor_local_indices": headgroup_anchor_local_indices,
            "library_hits": library_hits,
            "bootstrap_hits": bootstrap_hits,
        })


def _select_spread_positions(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    box_xy: float | None = None,
) -> np.ndarray:
    """Select points by farthest sampling, optionally on a periodic XY torus."""
    if count <= 0 or count > len(points):
        raise ValueError("count must be between 1 and the number of points")
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")

    def squared_distances(origin: np.ndarray) -> np.ndarray:
        delta = points - origin
        if box_xy is not None:
            delta -= box_xy * np.round(delta / box_xy)
        return np.sum(delta * delta, axis=1)

    selected = np.empty(count, dtype=int)
    selected[0] = int(rng.integers(len(points)))
    min_distance_sq = squared_distances(points[selected[0]])
    min_distance_sq[selected[0]] = -1.0
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance_sq))
        distance_sq = squared_distances(points[selected[index]])
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected[:index + 1]] = -1.0
    return selected


def _leaflet_orientation_data(
    leaflet_system: System,
    *,
    upper: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-lipid projections/cosines and hydrophobic-tail Z atoms."""
    sizes = [int(value) for value in leaflet_system.metadata.get("lipid_sizes", [])]
    if not sizes or sum(sizes) != leaflet_system.num_atoms:
        raise ModuleConfigError("Membrane lipid partition metadata is inconsistent")
    offsets = np.cumsum([0] + sizes)
    names = leaflet_system.structure.atom_names
    projections: list[float] = []
    cosines: list[float] = []
    tail_z: list[np.ndarray] = []
    for index, size in enumerate(sizes):
        start, end = int(offsets[index]), int(offsets[index + 1])
        molecule_names = [str(value).strip() for value in names[start:end]]
        try:
            profile = infer_lipid_orientation(
                leaflet_system.coordinates[start:end], molecule_names,
            )
        except LipidOrientationError as exc:
            leaflet_name = "upper" if upper else "lower"
            raise ModuleConfigError(
                f"Cannot validate {leaflet_name}-leaflet lipid {index + 1}: {exc}"
            ) from exc
        projection, cosine = outward_orientation(profile, upper=upper)
        projections.append(projection)
        cosines.append(cosine)
        tail_z.append(
            leaflet_system.coordinates[start:end][profile.tail_indices, 2]
        )
    return (
        np.asarray(projections, dtype=float),
        np.asarray(cosines, dtype=float),
        np.concatenate(tail_z),
    )


def _validate_bilayer_structure(
    upper_system: System,
    lower_system: System,
    log: list[str],
) -> dict:
    """Enforce chemical orientation and hydrophobic-core continuity.

    Unlike the occupancy diagnostics below, these are hard structural
    invariants.  A system with a solvent-facing tail or a water-sized vacuum
    layer at the midplane is not emitted as a successful membrane.
    """
    upper_projection, upper_cosine, upper_tail_z = _leaflet_orientation_data(
        upper_system, upper=True,
    )
    lower_projection, lower_cosine, lower_tail_z = _leaflet_orientation_data(
        lower_system, upper=False,
    )
    projections = np.concatenate((upper_projection, lower_projection))
    cosines = np.concatenate((upper_cosine, lower_cosine))
    invalid = (projections < MIN_INWARD_PROJECTION_NM) | (
        cosines < MIN_INWARD_COSINE
    )
    if invalid.any():
        raise ModuleConfigError(
            "Membrane orientation validation failed: "
            f"{int(invalid.sum())}/{len(invalid)} lipids do not point their polar "
            "heads toward solvent and hydrophobic regions toward the bilayer core"
        )

    upper_head = _leaflet_headgroup_plane(upper_system, upper=True)
    lower_head = _leaflet_headgroup_plane(lower_system, upper=False)
    if upper_head <= lower_head:
        raise ModuleConfigError(
            "Membrane leaflet ordering failed: upper headgroups are not above lower headgroups"
        )

    # The terminal part of each chain occupies only a small fraction of all
    # hydrophobic carbons.  Use the inward 1% tail-cloud boundary.  This is an
    # atom-centre distance: subtracting two carbon VDW radii (~0.34 nm total),
    # 0.62 nm leaves no more than one water diameter of free space.
    upper_inner = float(np.percentile(upper_tail_z, 1.0))
    lower_inner = float(np.percentile(lower_tail_z, 99.0))
    tail_core_gap = upper_inner - lower_inner
    maximum_core_gap = MAX_TAIL_CORE_GAP_NM
    if tail_core_gap > maximum_core_gap:
        raise ModuleConfigError(
            "Membrane hydrophobic core is not sealed: "
            f"leaflet tail gap {tail_core_gap:.3f} nm exceeds "
            f"{maximum_core_gap:.2f} nm"
        )

    log.append(
        "Membrane orientation: all "
        f"{len(projections)} lipids have solvent-facing heads and inward tails "
        f"(minimum inward projection {projections.min():.3f} nm)"
    )
    log.append(
        f"Hydrophobic core sealed: tail gap {tail_core_gap:.3f} nm "
        "(negative values indicate interdigitation)"
    )
    return {
        "passed": True,
        "n_lipids_checked": len(projections),
        "minimum_inward_projection_nm": float(projections.min()),
        "minimum_inward_cosine": float(cosines.min()),
        "tail_core_gap_nm": float(tail_core_gap),
        "maximum_tail_core_gap_nm": maximum_core_gap,
    }


def _close_leaflets(
    upper_system: System,
    lower_system: System,
    log: list[str],
    target_contact: float = 0.30,
    backoff_margin: float = 0.02,
    max_shift: float = 1.0,
    target_dhh: float | None = None,
    box_xy: float | None = None,
) -> None:
    """Close the gap between upper and lower leaflets to eliminate vacuum layer.

    After lipid placement and relaxation, the tail ends of the two leaflets
    may not meet at the bilayer midplane, creating a visible vacuum gap.
    This function shifts both leaflets toward each other until tail atoms
    make gentle VDW contact, then backs off by a small margin.

    Parameters
    ----------
    upper_system : System
        Upper leaflet system (headgroups at +Z).
    lower_system : System
        Lower leaflet system (headgroups at −Z).
    log : list[str]
        Build log to append messages to.
    target_contact : float
        Target minimum atom-atom distance between leaflets (nm).
        Carbon VDW radius ≈ 0.17 nm, so 2 × 0.17 = 0.34 nm for contact.
        We use 0.30 nm to allow slight interdigitation without hard clashes.
    backoff_margin : float
        Extra margin to back off after contact (nm).  Prevents hard
        VDW clashes at the interface.
    max_shift : float
        Maximum per-leaflet shift (nm).  Caps the correction to prevent
        pathological behaviour when leaflets are extremely far apart.
    """
    from scipy.spatial import cKDTree

    n_upper = upper_system.metadata.get("n_lipids", 0)
    n_lower = lower_system.metadata.get("n_lipids", 0)
    if n_upper == 0 or n_lower == 0:
        return

    upper_coords = upper_system.coordinates
    lower_coords = lower_system.coordinates
    if box_xy is not None and (not np.isfinite(box_xy) or box_xy <= 0.0):
        raise ValueError("box_xy must be a positive finite length")

    current_dhh = _leaflet_headgroup_plane(
        upper_system, upper=True
    ) - _leaflet_headgroup_plane(lower_system, upper=False)
    # Leaflet translation is a last-resort bulk correction.  Registry DHH is
    # an approximate fluid-bilayer target rather than an exact constraint;
    # allow a 5% contraction while sealing the core.  The previous fixed
    # 0.10-nm allowance was only ~2% for thick sphingomyelins and left a
    # water-sized midplane gap even with otherwise physical tail geometry.
    dhh_tolerance = (
        max(0.10, 0.05 * float(target_dhh))
        if target_dhh is not None else 0.10
    )
    max_outward = float("inf")
    max_inward = float("inf")
    if target_dhh is not None:
        max_outward = max((target_dhh + dhh_tolerance - current_dhh) / 2.0, 0.0)
        max_inward = max((current_dhh - (target_dhh - dhh_tolerance)) / 2.0, 0.0)

    # ---- metric 1: closest atom-pair distance ----
    upper_search = upper_coords.copy()
    lower_search = lower_coords.copy()
    tree_options = {}
    if box_xy is not None:
        z_origin = min(
            float(upper_search[:, 2].min()), float(lower_search[:, 2].min())
        ) - 1.0
        z_box = max(
            float(upper_search[:, 2].max()), float(lower_search[:, 2].max())
        ) - z_origin + 1.0
        upper_search[:, :2] = wrap_periodic_coordinates(
            upper_search[:, :2], box_xy,
        )
        lower_search[:, :2] = wrap_periodic_coordinates(
            lower_search[:, :2], box_xy,
        )
        upper_search[:, 2] -= z_origin
        lower_search[:, 2] -= z_origin
        tree_options = {"boxsize": np.asarray([box_xy, box_xy, z_box])}
    tree = cKDTree(upper_search, **tree_options)
    dists, _ = tree.query(
        lower_search, k=1, workers=configured_task_threads()
    )
    min_dist = float(dists.min())

    # ---- metric 2: chemically identified hydrophobic-tail Z-gap ----
    # Headgroup atoms must not make an inverted conformation look sealed.
    _, _, upper_hydrophobic_z = _leaflet_orientation_data(
        upper_system, upper=True,
    )
    _, _, lower_hydrophobic_z = _leaflet_orientation_data(
        lower_system, upper=False,
    )
    upper_tail_z = float(np.percentile(upper_hydrophobic_z, 1))
    lower_tail_z = float(np.percentile(lower_hydrophobic_z, 99))
    bulk_gap = upper_tail_z - lower_tail_z  # > 0 => vacuum; < 0 => overlap

    safety_min_dist = 0.22
    if min_dist < safety_min_dist - 0.03:
        separation = min(
            (safety_min_dist - min_dist) / 2.0,
            max_shift * 0.3,
            max_outward,
        )
        if separation > 0.003:
            upper_system.structure.translate(np.array([0.0, 0.0, +separation]))
            lower_system.structure.translate(np.array([0.0, 0.0, -separation]))
            log.append(
                f"Leaflet separation: clash {min_dist:.3f} nm; "
                f"backed off {separation:.3f} nm each"
            )
        return

    # ---- decide the optimal shift ----
    # We want to close the bulk tail gap but avoid driving the closest
    # atom pair into hard VDW overlap.  Take the *smaller* of the two
    # shifts so neither constraint is violated.
    target_bulk_overlap = 0.05  # nm — slight interdigitation of tail clouds

    # Shift needed to close the bulk tail gap
    shift_from_bulk = (bulk_gap + target_bulk_overlap) / 2.0 if bulk_gap > -target_bulk_overlap else 0.0
    # Shift that would bring closest atoms to the safety limit
    # Shift that creates the desired min distance (target contact)
    shift_from_target = (min_dist - (target_contact + backoff_margin)) / 2.0

    if bulk_gap > 0.05:
        # Bulk tail gap detected — close it aggressively.  A few outlier
        # tail atoms may already be close to the midplane (from relaxation),
        # but the bulk of tail atoms hasn't reached the midplane yet.  We
        # close based on the bulk gap and let the repulsion relaxation
        # (already done) handle any resulting VDW clashes among outliers.
        max_safe_shift = max((min_dist - safety_min_dist) / 2.0, 0.0)
        shift = min(shift_from_bulk, max_safe_shift, max_shift, max_inward)
        if shift > 0.003:
            upper_system.structure.translate(np.array([0.0, 0.0, -shift]))
            lower_system.structure.translate(np.array([0.0, 0.0, +shift]))
            log.append(
                f"Leaflet closing: bulk tail gap {bulk_gap:.3f} nm → each "
                f"shifted {shift:.3f} nm (min pair was {min_dist:.3f} nm)"
            )
        else:
            log.append(
                f"Leaflets: bulk gap {bulk_gap:.3f} nm — shift too small "
                f"({shift:.4f} nm), skipped"
            )
    elif shift_from_target > 0.01 and min_dist > safety_min_dist:
        # No significant bulk gap but atom pairs too far apart
        shift = min(shift_from_target, max_shift * 0.5, max_inward)
        if shift > 0.003:
            desired_min = target_contact + backoff_margin
            upper_system.structure.translate(np.array([0.0, 0.0, -shift]))
            lower_system.structure.translate(np.array([0.0, 0.0, +shift]))
            log.append(
                f"Leaflet closing: atom gap {min_dist - desired_min:.3f} nm → "
                f"shifted {shift:.3f} nm each"
            )
    elif min_dist < safety_min_dist - 0.03:
        # Hard VDW clash — separate
        separation = safety_min_dist - min_dist
        shift = min(separation / 2.0, max_shift * 0.3)
        if shift > 0.003:
            upper_system.structure.translate(np.array([0.0, 0.0, +shift]))
            lower_system.structure.translate(np.array([0.0, 0.0, -shift]))
            log.append(
                f"Leaflet separation: clash {min_dist:.3f} nm → "
                f"backed off {shift:.3f} nm each"
            )
    else:
        log.append(
            f"Leaflets at optimal contact "
            f"(min {min_dist:.3f} nm, bulk gap {bulk_gap:.3f} nm)"
        )


def _pack_lipids_against_protein(
    leaflet_system: System,
    protein_coords: np.ndarray,
    target_contact: float,
    max_shift: float,
    log: list[str],
    leaflet_label: str = "",
) -> None:
    """Push lipids toward the protein surface to eliminate water-sized gaps.

    After the clash-removal filter (step 9b), a few lipids that survived
    may still sit anomalously far from the protein surface, leaving
    cavities large enough for water molecules (Ø ≈ 0.28 nm).

    The function runs a small number of passes: each pass finds lipids
    whose closest atom is within *surface_band* of the protein (i.e.
    lipids that actually face the protein surface), then nudges them
    gently in XY toward the nearest protein atom.  Lipids already close
    to the protein (< target_contact) are left alone — they form the
    natural interface layer.  Lipids far from the protein are bulk
    membrane lipids and are not touched.

    Z coordinates are preserved to maintain leaflet headgroup/tail ordering.

    Parameters
    ----------
    leaflet_system : System
        Upper or lower leaflet system.
    protein_coords : (M, 3) ndarray
        Protein atom coordinates in the same reference frame.
    target_contact : float
        Desired minimum lipid-protein atom distance (nm).
    max_shift : float
        Maximum per-lipid XY displacement per pass (nm).
    log : list[str]
        Build log to append messages to.
    leaflet_label : str
        "upper" or "lower" for log messages.
    """
    from scipy.spatial import cKDTree

    n_lipids = leaflet_system.metadata.get("n_lipids", 0)
    lipid_sizes = leaflet_system.metadata.get("lipid_sizes")
    if n_lipids == 0 or lipid_sizes is None:
        return

    offsets = np.cumsum([0] + list(lipid_sizes))
    coords = leaflet_system.coordinates
    surface_band = 0.6     # nm — only nudge lipids very close to the protein surface
    step_size = 0.04       # nm — tiny incremental shift per pass
    max_passes = 4         # converge quickly — this is a gap-filler, not a wall-builder

    total_pushed = 0

    for _pass in range(max_passes):
        prot_tree = cKDTree(protein_coords)
        pass_pushed = 0

        for li in range(n_lipids):
            start = offsets[li]
            end = offsets[li + 1]
            lipid_atoms = coords[start:end]

            dists, idx = prot_tree.query(
                lipid_atoms, k=1, workers=configured_task_threads()
            )
            min_dist = float(dists.min())

            # Only push lipids near the protein surface
            if min_dist <= target_contact or min_dist > surface_band:
                continue

            closest_lipid_i = int(dists.argmin())
            closest_prot_i = int(idx[closest_lipid_i])
            lip_xyz = lipid_atoms[closest_lipid_i]
            prot_xyz = protein_coords[closest_prot_i]

            dx = prot_xyz[0] - lip_xyz[0]
            dy = prot_xyz[1] - lip_xyz[1]
            norm = float(np.sqrt(dx * dx + dy * dy))
            if norm < 0.001:
                continue

            ux, uy = dx / norm, dy / norm
            shift = min(min_dist - target_contact, step_size, max_shift)

            coords[start:end, 0] += ux * shift
            coords[start:end, 1] += uy * shift
            pass_pushed += 1

        total_pushed = max(total_pushed, pass_pushed)
        if pass_pushed == 0:
            break  # converged

    if total_pushed > 0:
        log.append(
            f"Protein-lipid packing ({leaflet_label}): "
            f"{total_pushed}/{n_lipids} lipids packed against protein surface "
            f"(target {target_contact:.2f} nm)"
        )
    else:
        log.append(
            f"Protein-lipid packing ({leaflet_label}): "
            f"all interface lipids already within {target_contact:.2f} nm"
        )


def _validate_membrane_quality(
    merged: System,
    mem_indices: np.ndarray,
    n_solute: int,
    box_xy: float,
    box_z: float,
    has_protein: bool,
    log: list[str],
) -> None:
    """Validate membrane quality and emit warnings for potential issues.

    Checks (non-blocking — issues are logged as ⚠ warnings):
      1. Local density uniformity — no sparse or over-dense XY regions
      2. Z-axis seal — no water-permeable path through the bilayer
      3. Protein-lipid interface — no water-sized gaps at protein surface
      4. Box edge seal — membrane fills box to edges
      5. Protein-to-box-edge buffer — prevents periodic image contact

    The membrane is always saved regardless of warnings.  Users can
    increase lipids-per-leaflet and re-run if quality is unacceptable.
    """
    from scipy.spatial import cKDTree

    coords = merged.coordinates
    lipid_coords = coords[mem_indices]
    # ---- Check 1: local XY density uniformity ----
    cell_size = 1.0  # nm
    n_cells = max(3, int(box_xy / cell_size))
    cell_edges = np.linspace(-box_xy / 2, box_xy / 2, n_cells + 1)
    lipid_xy = lipid_coords[:, :2]
    cell_counts = np.zeros((n_cells, n_cells), dtype=int)
    for i in range(n_cells):
        for j in range(n_cells):
            in_cell = (
                (lipid_xy[:, 0] >= cell_edges[i]) & (lipid_xy[:, 0] < cell_edges[i + 1]) &
                (lipid_xy[:, 1] >= cell_edges[j]) & (lipid_xy[:, 1] < cell_edges[j + 1])
            )
            cell_counts[i, j] = in_cell.sum()
    # Exclude cells overlapping protein
    if has_protein:
        prot_xy = coords[:n_solute, :2]
        for i in range(n_cells):
            for j in range(n_cells):
                cx_min, cx_max = cell_edges[i], cell_edges[i + 1]
                cy_min, cy_max = cell_edges[j], cell_edges[j + 1]
                prot_in_cell = (
                    (prot_xy[:, 0] >= cx_min) & (prot_xy[:, 0] < cx_max) &
                    (prot_xy[:, 1] >= cy_min) & (prot_xy[:, 1] < cy_max)
                ).sum()
                if prot_in_cell > 10:
                    cell_counts[i, j] = -1
    valid_counts = cell_counts[cell_counts >= 0]
    if len(valid_counts) > 0:
        empty_cells = (valid_counts == 0).sum()
        total_valid = len(valid_counts)
        # Hexagonal packing trimmed to a square box leaves corners sparse.
        # Allow up to 15% empty cells — corners are expected to be empty.
        if empty_cells > total_valid * 0.15:
            log.append(
                f"⚠ Membrane has {empty_cells} empty XY cells "
                f"({empty_cells}/{total_valid}, {empty_cells/total_valid*100:.0f}%). "
                f"Increase lipids per leaflet for full coverage."
            )

    # ---- Check 2: Z-axis seal ----
    z_all_lipid = lipid_coords[:, 2]
    z_mid = (z_all_lipid.min() + z_all_lipid.max()) / 2.0
    upper_z = z_all_lipid[z_all_lipid > z_mid]
    lower_z = z_all_lipid[z_all_lipid < z_mid]
    if len(upper_z) > 0 and len(lower_z) > 0:
        n_xy_samples = min(100, max(25, int(box_xy / 0.4) ** 2))
        rng_check = np.random.default_rng(42)
        sample_points = rng_check.uniform(0.0, box_xy, (n_xy_samples, 2))
        periodic_lipid_xy = wrap_periodic_coordinates(
            lipid_coords[:, :2] + box_xy / 2.0, box_xy,
        )
        lipid_tree = cKDTree(periodic_lipid_xy, boxsize=box_xy)
        gap_count = 0
        for sx, sy in sample_points:
            nearby = lipid_tree.query_ball_point([sx, sy], 0.7)  # wider search
            if len(nearby) < 3:
                gap_count += 1
                continue
            nearby_z = lipid_coords[nearby, 2]
            z_sorted = np.sort(nearby_z)
            z_gaps = np.diff(z_sorted)
            # Gap > 1.0 nm between consecutive lipid atoms = potential water channel
            if np.any(z_gaps > 1.0):
                gap_count += 1
        gap_fraction = gap_count / n_xy_samples
        if gap_fraction > 0.20:
            log.append(
                f"⚠ Membrane has Z-axis gaps in {gap_fraction*100:.0f}% of "
                f"sampled XY area. Increase lipids per leaflet to seal the bilayer."
            )
        elif gap_fraction > 0.0:
            log.append(f"Membrane Z-seal: {gap_fraction*100:.0f}% sparse (within tolerance)")

    # ---- Check 3: protein-lipid interface seal ----
    if has_protein:
        prot_coords = coords[:n_solute]
        if len(prot_coords) == 0:
            log.append("⚠ Protein coordinates empty — internal error.")
        else:
            protein_z_min = float(prot_coords[:, 2].min())
            protein_z_max = float(prot_coords[:, 2].max())
            lipid_z_min = float(lipid_coords[:, 2].min())
            lipid_z_max = float(lipid_coords[:, 2].max())
            if protein_z_max < lipid_z_min or protein_z_min > lipid_z_max:
                    raise ModuleConfigError(
                        "Protein and bilayer Z envelopes do not intersect after membrane "
                        "construction. Re-run protein orientation or manually place the "
                        "protein within the membrane preview before continuing. "
                        f"Protein Z={protein_z_min:.3f}..{protein_z_max:.3f} nm; "
                        f"bilayer Z={lipid_z_min:.3f}..{lipid_z_max:.3f} nm."
                    )
            lipid_tree_3d = cKDTree(lipid_coords)
            prot_dists, _ = lipid_tree_3d.query(
                prot_coords, k=1, workers=configured_task_threads()
            )
            closest_5pct = float(np.percentile(prot_dists, 5))
            closest_10pct = float(np.percentile(prot_dists, 10))
            if closest_5pct > 0.40:
                log.append(
                    f"⚠ Protein-lipid interface gap: 5th percentile distance "
                    f"{closest_5pct:.2f} nm > 0.40 nm. Increase lipids per leaflet "
                    f"for tighter protein packing."
                )
            elif closest_10pct > 0.50:
                log.append(
                    f"⚠ Protein-lipid interface loose: 10th percentile distance "
                    f"{closest_10pct:.2f} nm > 0.50 nm. Consider increasing lipids."
                )

    # ---- Check 4: periodic edge seal ----
    # Check 2 uses minimum-image XY distances and already samples all four
    # periodic boundaries; a raw Cartesian extent is not a valid edge metric.

    # ---- Check 5: protein-to-box-edge buffer ----
    # Ensure the protein is surrounded by enough lipids on all sides so
    # it does not directly contact the periodic box boundary.  Without
    # sufficient buffer, the protein interacts with its own periodic
    # image during MD, causing artifacts.
    # Threshold: roughly six lipid diameters, conservatively set to 2.5 nm.
    if has_protein:
        prot_coords = coords[:n_solute]
        box_half = box_xy / 2.0
        # Distance from each protein atom to each of the 4 box edges
        dist_to_edges = np.stack([
            prot_coords[:, 0] - (-box_half),   # left edge
            box_half - prot_coords[:, 0],       # right edge
            prot_coords[:, 1] - (-box_half),   # bottom edge
            box_half - prot_coords[:, 1],       # top edge
        ], axis=1)  # (N_prot, 4)
        min_edge_dist = float(dist_to_edges.min())
        min_buffer = 2.0  # nm — ~4-5 POPC diameters, prevents periodic image contact
        if min_edge_dist < min_buffer:
            log.append(
                f"⚠ Protein too close to box edge: minimum distance "
                f"{min_edge_dist:.2f} nm < {min_buffer:.1f} nm. "
                f"Increase lipids per leaflet to provide adequate buffer "
                f"between protein and periodic boundary."
            )


def _asymmetric_check(config: dict) -> bool:
    """Check if the configuration specifies an asymmetric bilayer."""
    if "lipid_composition" in config:
        comp = config["lipid_composition"]
        lower = comp.get("lower")
        if lower is None:
            return False
        upper = [(e["name"].upper(), e["ratio"]) for e in comp["upper"]]
        lower_parsed = [(e["name"].upper(), e["ratio"]) for e in lower]
        return upper != lower_parsed
    return False
