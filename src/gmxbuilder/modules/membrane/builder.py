"""Module 1: Phospholipid bilayer & membrane protein builder.

Generates a lipid bilayer, optionally embeds a protein, and adds
the MEMBRANE component to the System.
"""

from __future__ import annotations

from functools import cache

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
from gmxbuilder.modules.membrane.area_model import leaflet_area_per_lipid
from gmxbuilder.modules.membrane.conformer_reader import conformer_read_scope
from gmxbuilder.modules.membrane.embed import embed_protein
from gmxbuilder.modules.membrane.lipid_orientation import (
    MAX_TAIL_CORE_GAP_NM,
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    LipidOrientationError,
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
    rotate_to_opposite_leaflet,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.modules.membrane.orient import orient_protein
from gmxbuilder.pipeline.base import BaseModule, ModuleResult
from gmxbuilder.runtime.hardware import current_task_threads


def _reconcile_lipid_selection(system: System, active_lipids: list[str]) -> str | None:
    """Accept a changed Step 5 set only within the already selected FF family."""
    active = sorted({str(name).strip().upper() for name in active_lipids})
    selected = sorted(
        {str(name).strip().upper() for name in system.metadata.get("selected_lipid_names", [])}
    )
    protein_ff = str(system.metadata.get("force_field", "")).lower()
    lipid_ff = str(system.metadata.get("lipid_ff", "")).lower()
    original_lipid_ff = lipid_ff
    resolved_reason = ""
    if protein_ff.startswith("amber"):
        from gmxbuilder.modules.forcefield.lipid_policy import (
            amber_lipid_backend,
            amber_lipid_backend_candidates,
            lipid_backend_for,
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
            lipid_ff = resolved_ff
        elif lipid_ff != resolved_ff:
            resolved_reason = (
                f"explicit coherent {lipid_ff} backend retained; "
                f"preferred automatic backend would be {resolved_ff}"
            )
        if lipid_ff == "amber-mixed":
            from gmxbuilder.modules.forcefield.lipid_policy import amber_mixed_validation_reason

            validation_reason = amber_mixed_validation_reason(active)
            if validation_reason:
                raise ModuleConfigError(validation_reason)
        system.metadata["gaff_lipids"] = [
            n for n in active if lipid_backend_for(n, lipid_ff) == "gaff2"
        ]
        system.metadata["lipid21_lipids"] = [
            n for n in active if lipid_backend_for(n, lipid_ff) == "lipid21"
        ]
    backend_change = (
        f"Amber lipid backend updated for this composition: "
        f"{original_lipid_ff or 'unset'} -> {lipid_ff}. {resolved_reason}"
        if original_lipid_ff != lipid_ff
        else None
    )
    if not selected or active == selected:
        return backend_change
    compatible = False
    if protein_ff.startswith("amber") and lipid_ff in {"gaff2", "lipid21", "amber-mixed"}:
        from gmxbuilder.modules.forcefield.lipid_policy import amber_lipid_backend

        compatible = lipid_ff in amber_lipid_backend_candidates(active)
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


def _leaflet_headgroup_plane(leaflet_system: System, *, upper: bool) -> float:
    """Return the mean Z position of recorded per-lipid headgroup anchors."""
    coords = leaflet_system.coordinates
    lipid_sizes = list(leaflet_system.metadata.get("lipid_sizes") or [])
    local_indices = list(leaflet_system.metadata.get("headgroup_anchor_local_indices") or [])
    if lipid_sizes and len(local_indices) == len(lipid_sizes):
        offsets = np.cumsum([0] + lipid_sizes)
        absolute = np.asarray(
            [int(offsets[index]) + int(local_indices[index]) for index in range(len(lipid_sizes))]
        )
        return float(np.mean(coords[absolute, 2]))

    stripped = np.asarray([str(name).strip() for name in leaflet_system.structure.atom_names])
    phosphorus = coords[stripped == "P", 2]
    if len(phosphorus):
        return float(np.mean(phosphorus))
    return float(np.percentile(coords[:, 2], 90 if upper else 10))


# Two-letter element symbols that can begin a lipid atom name. "CA" is
# deliberately absent: in a lipid it is an alpha carbon, not calcium.
_TWO_LETTER_ELEMENTS = ("CL", "BR", "NA", "MG", "ZN", "FE")


@cache
def _elements_for_atom_names(atom_names: tuple[str, ...]) -> tuple[str, ...]:
    """Derive element symbols from atom names.

    Cached because the answer depends on nothing else: every instance of a
    lipid type carries the same atom names, so this is computed once per type
    rather than once per placed molecule. It was previously inline in
    `_build_one_lipid`, which a profile showed doing 14.2 million `str.upper()`
    and 14.3 million `str.startswith()` calls in a single membrane build --
    three seconds, and about a sixth of that step.
    """
    elements = []
    for name in atom_names:
        upper = name.upper()
        element = "C"
        for character in name:
            if character.isalpha() and character.isupper():
                element = character
                break
        for symbol in _TWO_LETTER_ELEMENTS:
            if upper.startswith(symbol):
                element = symbol.title()
                break
        elements.append(element)
    return tuple(elements)


@register_module
class MembraneBuilder(BaseModule):
    """Build a phospholipid bilayer and embed a membrane protein."""

    name = "membrane"
    description = "Generate lipid bilayer with optional protein embedding"

    _MIN_BOX_XY = 4.0  # nm
    # Minimum leaflet size accepted by this construction implementation.  This
    # is a supported-input bound, not a claim of thermodynamic stability.
    _MIN_LIPIDS_PER_LEAFLET = 64
    _MAX_LIPIDS_PER_LEAFLET = 5_000
    _MAX_BOX_XY_NM = 100.0
    _MAX_BOX_Z_NM = 100.0
    _MAX_DENSE_GRID_CANDIDATES = 500_000
    _GRID_JITTER = 0.05  # nm — random XY displacement for lipid placement
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

    # _LIPID_PACKING_FACTOR: construction fill factor for box sizing.
    #   N lipids at the selected construction APL need N×APL area when the
    #   factor is 1.00.  The separately oversampled placement pool supplies
    #   candidates that can survive protein-clash removal; it does not change
    #   the requested final areal density.
    _LIPID_PACKING_FACTOR = 1.00

    # _DENSE_GRID_SPACING: initial hexagonal grid point spacing (nm).
    #   At 0.35 nm this creates ~5.2× more candidate positions than
    #   needed for POPC (APL ≈ 0.64 nm² → √0.64 ≈ 0.80 nm natural
    #   spacing).  The high oversampling ensures uniform XY coverage
    #   even after protein exclusion and random thinning.
    _DENSE_GRID_SPACING = 0.35

    @staticmethod
    def _slab_atoms(solute_coords: np.ndarray, half_thickness: float) -> np.ndarray:
        """The solute atoms that lie inside the bilayer's own Z extent.

        A membrane protein is not a cylinder. Its extracellular and cytoplasmic
        domains can be far wider than the part that crosses the bilayer, and
        they sit above and below the lipids rather than among them -- so they
        displace no lipid and deny no area. Everything about lipid placement
        that asks "where is the protein" has to ask it of this subset, or it
        answers with a shape the membrane never sees.

        The bilayer is built with its leaflets at ``z = +-dh/2`` about the
        origin, so its own thickness is the slab. Atoms outside it are still
        checked in three dimensions by the clash filters, which is where a
        domain that genuinely dips into the headgroups gets caught.
        """
        if len(solute_coords) == 0:
            return solute_coords[:0, :2]
        inside = np.abs(solute_coords[:, 2]) <= half_thickness
        return solute_coords[inside, :2]

    @classmethod
    def _slab_footprint_area(
        cls,
        slab_xy: np.ndarray,
        exclusion_radius: float,
        spacing: float,
    ) -> float:
        """The XY area the solute denies to lipid centres, measured not assumed.

        Sizing the box needs a number for "how much of the membrane plane is
        not available to lipids". That number used to be the square of the
        whole solute's larger XY span -- a bounding square around every atom
        at every height, which for an elongated protein with a large
        extracellular domain is several times the area the membrane actually
        loses. The box grew by that surplus, the lipid count did not, and the
        difference was filled by the solvator: a slab of water lying in the
        hydrophobic core, tens of square nanometres of it, which no
        one-dimensional seal test can see.

        So it is measured with the criterion that will actually be applied.
        Lipid candidates are rejected within ``exclusion_radius`` of a slab
        atom; the area denied is the area of the cells that rejection covers,
        rasterised at the placement grid's own spacing. The box budget and the
        placement filter cannot disagree, because they are the same question
        asked twice.
        """
        if len(slab_xy) == 0:
            return 0.0
        from scipy.spatial import cKDTree

        margin = exclusion_radius + spacing
        low = slab_xy.min(axis=0) - margin
        high = slab_xy.max(axis=0) + margin
        extent = high - low
        if (
            not np.isfinite(extent).all()
            or not np.isfinite(spacing)
            or spacing <= 0
            or np.any(extent > 100.0)
        ):
            raise ModuleConfigError("Membrane footprint exceeds the supported 100 nm XY extent")
        cell_counts = np.maximum(np.ceil(extent / spacing), 1)
        if np.prod(cell_counts) > 4_000_000:
            raise ModuleConfigError("Membrane footprint exceeds the 4 million cell budget")
        counts = cell_counts.astype(int)
        axes = [low[axis] + (np.arange(counts[axis]) + 0.5) * spacing for axis in (0, 1)]
        cells = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        distances, _ = cKDTree(slab_xy).query(cells, k=1, workers=current_task_threads())
        return float((distances < exclusion_radius).sum()) * spacing * spacing

    @classmethod
    def _area_balanced_counts(cls, baseline, leaflet_apls, footprints, minimum_xy):
        """One box, independently filled leaflets; preserve the requested minimum."""
        areas = [
            baseline * apl / cls._LIPID_PACKING_FACTOR + occupied
            for apl, occupied in zip(leaflet_apls, footprints, strict=True)
        ]
        box_area = max(*areas, minimum_xy**2)
        counts = [
            max(
                baseline,
                int(np.ceil((box_area - occupied) * cls._LIPID_PACKING_FACTOR / apl - 1e-9)),
            )
            for apl, occupied in zip(leaflet_apls, footprints, strict=True)
        ]
        return float(np.sqrt(box_area)), counts

    def __init__(
        self,
        use_equilibrated_library: bool = True,
        *,
        allow_repairable_core_gap: bool = False,
    ):
        # The offline library builder disables reads while bootstrapping the
        # bilayer from which validated conformers will be extracted.
        self.use_equilibrated_library = bool(use_equilibrated_library)
        # Only the offline library workflow may accept an initially open core:
        # it immediately regrids whole lipids, performs gradual vacuum
        # precompression, and then runs explicit-solvent NVT/NPT.  Interactive
        # membrane construction keeps the strict sealed-core invariant.
        self.allow_repairable_core_gap = bool(allow_repairable_core_gap)

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(
            config,
            {
                "lipid_type",
                "lipid_composition",
                "n_lipids_per_leaflet",
                "seed",
                "box_padding",
                "pad",
                "bilayer_size",
                "orient_method",
                "embed_method",
            },
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
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        raise ModuleConfigError(f"{label} leaflet entry {index} must be an object")
                    name = entry.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ModuleConfigError(f"{label} leaflet entry {index} has no lipid name")
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
                        LipidRegistry.get(name)
                    except KeyError:
                        # Preserve custom-lipid support when geometry metadata
                        # is supplied by the caller.
                        if "category" not in entry:
                            raise ModuleConfigError(
                                f"Unknown lipid {name!r} in {label} leaflet — not in registry "
                                "and no category/tail data was provided."
                            )

                if not np.isclose(total_ratio, 100.0, atol=0.5):
                    raise ModuleConfigError(
                        f"{label} leaflet ratios must total 100%, got {total_ratio:.3f}%"
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
                LipidRegistry.get(lipid_type)
            except KeyError as exc:
                raise ModuleConfigError(str(exc))
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
            if count > self._MAX_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"n_lipids_per_leaflet must be at most "
                    f"{self._MAX_LIPIDS_PER_LEAFLET}, got {count}"
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
            raise ModuleConfigError("box_padding and legacy pad disagree; provide only box_padding")

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
            if np.any(dims > self._MAX_BOX_XY_NM):
                raise ModuleConfigError(
                    f"bilayer_size dimensions must not exceed {self._MAX_BOX_XY_NM:g} nm"
                )

        orient_method = config.get("orient_method")
        if orient_method is not None and orient_method not in {
            "ppm",
            "hmoment",
            "tmd",
            "pca",
            "com",
        }:
            raise ModuleConfigError(f"Unknown orient_method: {orient_method}")
        embed_method = config.get("embed_method")
        if embed_method is not None and embed_method not in {
            "com",
            "hydrophobic",
            "ppm",
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

    def _parse_composition(
        self, config: dict
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
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

    @conformer_read_scope()
    def run(self, system: System, config: dict) -> ModuleResult:
        from gmxbuilder.pipeline.progress import report_progress

        upper_mix, lower_mix = self._parse_composition(config)

        # Step 2 must have resolved the exact lipid set that Step 5 is about
        # to build.  Otherwise structure processing may have used a different
        # protein force field and the incremental preview would not match the
        # final full-pipeline build.
        active_lipids = sorted(
            {name for name, ratio in upper_mix + lower_mix if float(ratio) > 0.0}
        )
        reconciliation_log = _reconcile_lipid_selection(system, active_lipids)
        self.validate_config(config)
        if self.use_equilibrated_library:
            from gmxbuilder.modules.membrane.v4_availability import require_v4_lipids

            require_v4_lipids(
                active_lipids,
                str(system.metadata.get("force_field", "amber14sb")),
                str(
                    system.metadata.get("lipid_ff", system.metadata.get("force_field", "amber14sb"))
                ),
            )
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

        # Everything already in the system is something the lipids must not be
        # built through. Defined by *presence*, not by kind: the membrane step
        # runs before solvation and ions, so whatever is here is solute --
        # protein, nucleic acid, ligand, cofactor -- and a ComponentKind added
        # later is respected without this file being touched. Selecting only
        # PROTEIN meant a ligand or a nucleic acid was invisible to the
        # placement grid, the clash filter and the box, and lipids were built
        # straight through it.
        solute_coords = np.asarray(system.coordinates, dtype=float)
        has_solute = len(solute_coords) > 0

        # Orientation and embedding are a different question and stay
        # protein-specific: they compute a membrane normal from a protein, and
        # a ligand has no such thing.
        has_protein = bool(system.component_by_kind(ComponentKind.PROTEIN))

        # ---- 1. Orient protein if present ----
        report_progress(0.02, "Sizing the bilayer and periodic box")
        if has_protein:
            already_oriented = system.metadata.get("_oriented", False)
            orient_method = config.get("orient_method")
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
        # Re-read rather than reuse the array captured before step 1: orienting
        # the protein replaces ``system.structure``, and every measurement from
        # here on -- the box, the excluded area, the packing obstacles, the
        # clash filters -- would otherwise describe the pose the user replaced.
        solute_coords = np.asarray(system.coordinates, dtype=float)
        prot_min_xy = np.zeros(2)
        prot_max_xy = np.zeros(2)
        prot_z_min = 0.0
        prot_z_max = 0.0
        if has_solute:
            prot_min_xy = solute_coords[:, :2].min(axis=0)
            prot_max_xy = solute_coords[:, :2].max(axis=0)
            prot_z_min = float(solute_coords[:, 2].min())
            prot_z_max = float(solute_coords[:, 2].max())

        # ---- 3a. Compute weighted APL (needed for box sizing) ----
        # Both leaflets share one periodic XY box. Asymmetric mixtures must be
        # sized for the larger natural leaflet area; using only the upper
        # composition over-compresses lower leaflets enriched in large lipids
        # such as cardiolipin.
        force_field = str(system.metadata.get("force_field", "amber14sb"))
        lipid_ff = str(system.metadata.get("lipid_ff", force_field))
        try:
            upper_model = leaflet_area_per_lipid(upper_mix, force_field, lipid_ff)
            lower_model = leaflet_area_per_lipid(lower_mix, force_field, lipid_ff)
        except ValueError as exc:
            raise ModuleConfigError(str(exc)) from exc
        upper_area = upper_model.area_per_lipid_nm2
        lower_area = lower_model.area_per_lipid_nm2
        avg_area = max(upper_area, lower_area)
        if not np.isfinite(avg_area) or avg_area <= 0.0:
            raise ModuleConfigError("Weighted area per lipid must be a positive finite value")
        # Where each area came from is part of the result, not a debugging
        # aid: a sterol's number is a literature partial molar area and a
        # phospholipid's is an experimental one, and a reviewer reproducing
        # the box needs to see which was used.
        log.append(f"Upper leaflet {upper_model.summary()}")
        if lower_mix != upper_mix:
            log.append(f"Lower leaflet {lower_model.summary()}")
        if abs(upper_area - lower_area) > 0.01:
            log.append(
                f"Asymmetric leaflet APL: upper={upper_area:.3f}, "
                f"lower={lower_area:.3f} nm²; box uses {avg_area:.3f} nm²"
            )

        leaflet_slabs = (
            [
                solute_coords[(solute_coords[:, 2] >= 0) & (solute_coords[:, 2] <= dh / 2), :2],
                solute_coords[(solute_coords[:, 2] <= 0) & (solute_coords[:, 2] >= -dh / 2), :2],
            ]
            if has_solute
            else [np.empty((0, 2)), np.empty((0, 2))]
        )
        leaflet_footprints = [
            self._slab_footprint_area(atoms, self._PROTEIN_EXCLUSION_XY, self._DENSE_GRID_SPACING)
            for atoms in leaflet_slabs
        ]
        leaflet_apls = [upper_area, lower_area]
        leaflet_counts = None

        # ---- 3b. Determine target box ----
        n_lipids_per_leaflet = config.get("n_lipids_per_leaflet")
        if n_lipids_per_leaflet is not None:
            n_lipids_per_leaflet = int(n_lipids_per_leaflet)
            if n_lipids_per_leaflet < self._MIN_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"n_lipids_per_leaflet must be at least "
                    f"{self._MIN_LIPIDS_PER_LEAFLET}, got {n_lipids_per_leaflet}"
                )
            if n_lipids_per_leaflet > self._MAX_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"n_lipids_per_leaflet must be at most "
                    f"{self._MAX_LIPIDS_PER_LEAFLET}, got {n_lipids_per_leaflet}"
                )
            minimum_xy = self._MIN_BOX_XY
            if has_solute:
                clearance = float(config.get("box_padding", config.get("pad", 2.0)))
                if not np.isfinite(clearance) or not 0 <= clearance <= 50:
                    raise ModuleConfigError("box_padding must be finite and 0.0–50.0 nm")
                minimum_xy = max(
                    minimum_xy, float(np.max(prot_max_xy - prot_min_xy)) + 2 * clearance
                )
            box_xy, leaflet_counts = self._area_balanced_counts(
                n_lipids_per_leaflet, leaflet_apls, leaflet_footprints, minimum_xy
            )
            if max(leaflet_counts) > self._MAX_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"Area-balanced leaflets need {leaflet_counts[0]} upper / "
                    f"{leaflet_counts[1]} lower lipids, "
                    f"exceeding the {self._MAX_LIPIDS_PER_LEAFLET} per-leaflet limit. "
                    "Reduce the starting count, protein XY extent or XY padding."
                )
            log.append(
                f"Area-balanced lipid counts: requested baseline {n_lipids_per_leaflet}; "
                f"upper {leaflet_counts[0]}, lower {leaflet_counts[1]}; "
                f"solute footprints {leaflet_footprints[0]:.2f} / {leaflet_footprints[1]:.2f} nm²"
            )
        else:
            # Legacy: compute from box_padding around protein
            box_padding = float(config.get("box_padding", config.get("pad", 2.0)))
            if box_padding < 0.0 or box_padding > 50.0:
                raise ModuleConfigError(f"box_padding must be 0.0–50.0 nm, got {box_padding}")
            if has_solute:
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
        prot_z_extent = prot_z_max - prot_z_min if has_solute else 0.0
        box_z = max(prot_z_extent, membrane_z_full)
        if not np.isfinite(box_xy) or box_xy <= 0.0 or box_xy > self._MAX_BOX_XY_NM:
            raise ModuleConfigError(
                "Derived membrane XY box must be finite, positive, and no larger than "
                f"{self._MAX_BOX_XY_NM:g} nm; reduce the lipid count, protein size, or padding"
            )
        if not np.isfinite(box_z) or box_z <= 0.0 or box_z > self._MAX_BOX_Z_NM:
            raise ModuleConfigError(
                "Derived membrane Z box must be finite, positive, and no larger than "
                f"{self._MAX_BOX_Z_NM:g} nm"
            )
        if n_lipids_per_leaflet is None:
            # The requested density determines final counts. Extra placement
            # candidates are a separate clash-recovery budget and are trimmed.
            leaflet_counts = tuple(
                max(self._MIN_LIPIDS_PER_LEAFLET, int(max(box_xy**2 - footprint, 0) / apl))
                for footprint, apl in zip(leaflet_footprints, leaflet_apls, strict=True)
            )
            if max(leaflet_counts) > self._MAX_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError("Automatic leaflet count exceeds the supported maximum")

        if n_lipids_per_leaflet is not None:
            log.append(
                f"Membrane system: {box_xy:.1f}×{box_xy:.1f}×{box_z:.1f} nm — "
                f"area: {box_xy * box_xy:.1f} nm² "
                f"(upper={leaflet_counts[0]}, lower={leaflet_counts[1]}, "
                "Z water layer added in solvation)"
            )
        else:
            log.append(
                f"Membrane system: {box_xy:.1f}×{box_xy:.1f}×{box_z:.1f} nm — "
                f"area: {box_xy * box_xy:.1f} nm² "
                f"(XY padding={box_padding:.1f} nm, Z water layer added in solvation)"
            )

        # ---- 4. Dense candidate placement ----
        report_progress(0.1, "Generating candidate lipid positions")
        # 5a. Generate VERY dense hexagonal grid (0.35 nm spacing ≈ 3× denser than target)
        dense_spacing = self._DENSE_GRID_SPACING
        try:
            grid_xy = hexagonal_grid(
                xy_extent=(box_xy, box_xy),
                spacing=dense_spacing,
                center=np.array([0.0, 0.0]),
                jitter=self._GRID_JITTER,
                rng=rng,
                max_points=self._MAX_DENSE_GRID_CANDIDATES,
            )
        except ValueError as exc:
            raise ModuleConfigError(
                f"Membrane placement grid is outside supported limits: {exc}"
            ) from exc
        # 5b. Trim lipid centres to the periodic box. Atoms may wrap across
        # PBC; subtracting a molecular-radius margin here would compress all
        # centres into a smaller area and invalidate the requested APL.
        half_box = box_xy / 2.0
        grid_in_box = (np.abs(grid_xy[:, 0]) < half_box) & (np.abs(grid_xy[:, 1]) < half_box)
        grid_xy = grid_xy[grid_in_box]
        n_dense = len(grid_xy)
        log.append(f"Dense grid: {n_dense} candidate positions (spacing={dense_spacing:.2f} nm)")

        # Each leaflet sees only the solute atoms inside its own half-slab.
        # Keep the original shared lattice for symmetric solute-free systems.
        leaflet_grids = []
        first_candidates = None
        for side, slab in enumerate(leaflet_slabs):
            candidates = grid_xy.copy()
            if len(slab) and len(candidates):
                from scipy.spatial import cKDTree

                # Two candidate batches; avoid creating a thread pool in this loop.
                distances, _ = cKDTree(slab).query(candidates, k=1)
                candidates = candidates[distances >= self._PROTEIN_EXCLUSION_XY]
            target_n = leaflet_counts[side]
            if has_solute:
                target_n = max(int(target_n * 1.10), target_n + 4)
            same_candidates = (
                side == 1
                and np.array_equal(candidates, first_candidates)
                and (leaflet_counts is None or leaflet_counts[0] == leaflet_counts[1])
            )
            if side == 0:
                first_candidates = candidates.copy()
            if same_candidates:
                candidates = leaflet_grids[0].copy()
            elif len(candidates) > target_n:
                chosen = _select_spread_positions(candidates, target_n, rng, box_xy=box_xy)
                candidates = candidates[chosen]
            if len(candidates) < self._MIN_LIPIDS_PER_LEAFLET:
                raise ModuleConfigError(
                    f"Only {len(candidates)} positions available in the "
                    f"{'upper' if side == 0 else 'lower'} leaflet; "
                    f"need at least {self._MIN_LIPIDS_PER_LEAFLET}."
                )
            leaflet_grids.append(candidates)
        log.append(
            f"Candidate placement: upper={len(leaflet_grids[0])}, lower={len(leaflet_grids[1])}"
        )

        # ---- 8. Assign lipid types to grid positions by ratio ----
        report_progress(0.25, "Assigning lipid types across the grid")
        upper_assignments = self._assign_lipids(len(leaflet_grids[0]), upper_mix, rng)
        lower_assignments = self._assign_lipids(len(leaflet_grids[1]), lower_mix, rng)

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
        report_progress(0.32, "Building the upper and lower leaflets")
        z_upper = dh / 2.0
        z_lower = -dh / 2.0

        upper_system = self._build_mixed_leaflet(
            leaflet_grids[0],
            z_upper,
            upper_assignments,
            rng,
            force_field,
            lipid_ff,
            box_xy=box_xy,
        )
        lower_system = self._build_mixed_leaflet(
            leaflet_grids[1],
            z_lower,
            lower_assignments,
            rng,
            force_field,
            lipid_ff,
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
            f"Built 2 leaflets: upper={len(upper_assignments)}, "
            f"lower={len(lower_assignments)} lipids"
        )

        # ---- 9a2. Leaflet closing — eliminate vacuum gap between leaflets ----
        report_progress(0.6, "Closing the gap between leaflets")
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
            workers=current_task_threads(),
            obstacles=solute_coords if has_solute else None,
        )
        _close_leaflets(upper_system, lower_system, log, target_dhh=dh, box_xy=box_xy)

        # ---- 9b. Seat lipids against the solute ----
        report_progress(0.68, "Seating lipids against the solute surface")
        # Moving beats deleting. A lipid whose atoms reach into the solute was
        # placed on an admissible grid point and is a perfectly good lipid; it
        # is simply a few hundredths of a nanometre out of position, and one
        # rigid translation fixes that. Deleting it instead was how the
        # explicit count came to be violated.
        #
        # Seating also closes the water-sized cavities at the interface that
        # this step originally existed for: the same operation draws distant
        # interface lipids in. Water in the bilayer interior nucleates pores.
        if has_solute:
            from scipy.spatial import cKDTree

            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                _seat_lipids_against_solute(
                    leaflet_sys,
                    solute_coords,
                    target_contact=self._LIPID_PROTEIN_MIN_DIST + 0.15,
                    max_shift=0.05,
                    log=log,
                    leaflet_label=leaflet_name,
                    min_distance=self._LIPID_PROTEIN_MIN_DIST,
                )
            # Whatever seating could not resolve -- a lipid centred inside the
            # solute has no radial direction to move along -- is removed here.
            # The count contract is settled after the last such removal, not
            # before it.
            prot_tree = cKDTree(solute_coords)
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                MembraneBuilder._filter_protein_clashes(
                    leaflet_sys,
                    prot_tree,
                    leaflet_name,
                    self._LIPID_PROTEIN_MIN_DIST,
                    log,
                    label_prefix="Unseatable clash filter",
                )

        # ---- 9c. Rigid-body XY scaling — fill box uniformly ----
        report_progress(0.85, "Scaling the bilayer to fill the box")
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
            if n_lipids_per_leaflet is None and np.any(
                np.abs(initial_extent - target_extent) > 0.005
            ):
                _, scales = scale_lipid_centres_xy(coords, lipid_sizes, target_extent)
                final_extent = np.ptp(coords[:, :2], axis=0)
                log.append(
                    f"Rigid-body XY scaling ({leaflet_name}): "
                    f"×{scales[0]:.3f}/×{scales[1]:.3f}; "
                    f"extent={final_extent[0]:.2f}×{final_extent[1]:.2f} nm"
                )

            leaflet_sys.structure.coordinates, minimum_clearance = rotate_lipids_away_from_clashes(
                leaflet_sys.structure.coordinates,
                lipid_sizes,
                min_distance=0.035,
                box_xy=box_xy,
            )
            if minimum_clearance < 0.025:
                log.append(
                    f"WARNING: {leaflet_name} leaflet retains a "
                    f"{minimum_clearance:.3f} nm inter-lipid contact after rigid relaxation"
                )

            if has_solute:
                # The rotations above score against other lipids only, so they
                # can turn a lipid into the solute. Undo that before anything
                # measures the interface.
                leaflet_sys.structure.coordinates, _ = rotate_lipids_away_from_external_clashes(
                    leaflet_sys.structure.coordinates,
                    lipid_sizes,
                    solute_coords,
                    min_distance=self._LIPID_PROTEIN_MIN_DIST,
                    box_xy=box_xy,
                    workers=current_task_threads(),
                )

        relax_interleaflet_clashes_xy(
            upper_system.structure.coordinates,
            lower_system.structure.coordinates,
            upper_system.metadata.get("lipid_sizes", []),
            lower_system.metadata.get("lipid_sizes", []),
            box_xy=box_xy,
            workers=current_task_threads(),
            obstacles=solute_coords if has_solute else None,
        )
        _close_leaflets(upper_system, lower_system, log, target_dhh=dh, box_xy=box_xy)
        upper_system.structure.coordinates, upper_cross_clearance = (
            rotate_lipids_away_from_external_clashes(
                upper_system.structure.coordinates,
                upper_system.metadata.get("lipid_sizes", []),
                lower_system.structure.coordinates,
                box_xy=box_xy,
                workers=current_task_threads(),
            )
        )
        lower_system.structure.coordinates, lower_cross_clearance = (
            rotate_lipids_away_from_external_clashes(
                lower_system.structure.coordinates,
                lower_system.metadata.get("lipid_sizes", []),
                upper_system.structure.coordinates,
                box_xy=box_xy,
                workers=current_task_threads(),
            )
        )
        cross_clearance = min(upper_cross_clearance, lower_cross_clearance)
        if cross_clearance < 0.04:
            log.append(
                "WARNING: bilayer retains a "
                f"{cross_clearance:.3f} nm cross-leaflet contact before minimization"
            )

        # Scaling and the rotations above move lipids relative to the solute,
        # so seat them once more -- again by moving, not by deleting.
        if has_solute:
            from scipy.spatial import cKDTree

            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                _seat_lipids_against_solute(
                    leaflet_sys,
                    solute_coords,
                    target_contact=self._LIPID_PROTEIN_MIN_DIST + 0.15,
                    max_shift=0.05,
                    log=log,
                    leaflet_label=f"post-scale {leaflet_name}",
                    min_distance=self._LIPID_PROTEIN_MIN_DIST,
                )
            _prot_tree_scl = cKDTree(solute_coords)
            for leaflet_name, leaflet_sys in [("upper", upper_system), ("lower", lower_system)]:
                MembraneBuilder._filter_protein_clashes(
                    leaflet_sys,
                    _prot_tree_scl,
                    leaflet_name,
                    self._LIPID_PROTEIN_MIN_DIST,
                    log,
                    label_prefix="Post-scale clash filter",
                )

        # ---- The explicit lipid count, settled last ----
        #
        # This is the last point at which a lipid can be removed, so it is the
        # only correct place to enforce the count. Enforcing it earlier, as
        # this did, meant the log truthfully said "trimmed to 128" and the
        # package then shipped 127: two force fields, two independent builds,
        # the same deterministic shortfall.
        #
        # A shortfall is now an error rather than a quiet difference. The
        # request cannot be met with this solute in this box, and the user is
        # the one who gets to decide what to change about that.
        if leaflet_counts is not None:
            for side, (leaflet_name, leaflet_sys) in enumerate(
                [("upper", upper_system), ("lower", lower_system)]
            ):
                target_count = leaflet_counts[side]
                available = int(leaflet_sys.metadata.get("n_lipids", 0))
                if available < target_count:
                    raise ModuleConfigError(
                        f"Area-balanced {leaflet_name} leaflet requires {target_count} lipids, "
                        f"but only {available} survive the solute clash checks. "
                        "Review protein orientation and membrane composition; "
                        "a sparse leaflet cannot be accepted."
                    )
                self._trim_leaflet_to_count(leaflet_sys, target_count, rng, leaflet_name, log)

        # Counts and structural invariants must describe the final leaflets,
        # after every protein clash filter and trimming operation.
        actual_upper = int(upper_system.metadata.get("n_lipids", 0))
        actual_lower = int(lower_system.metadata.get("n_lipids", 0))
        orientation_quality = _validate_bilayer_structure(
            upper_system,
            lower_system,
            log,
            allow_repairable_core_gap=self.allow_repairable_core_gap,
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
        report_progress(0.92, "Merging the membrane into the system")
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
        # Use the solute envelope for the displayed XY origin. Whole lipid
        # tails can cross periodic edges and should not displace the protein
        # in the displayed box. This common translation preserves all pairwise
        # minimum-image distances; it does not change physical image clearance.
        all_xy_center = (all_xy_max + all_xy_min) / 2.0
        if has_solute:
            solute_xy = merged.coordinates[: system.num_atoms, :2]
            all_xy_center = (solute_xy.max(axis=0) + solute_xy.min(axis=0)) / 2.0

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

        protein_extent_xy = float(np.max(prot_max_xy - prot_min_xy)) if has_solute else 0.0
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
            log.append(
                f"Membrane midplane centred: shifted by {-z_mid_actual:.2f} nm "
                f"to Z=0 (was at {z_mid_actual:.2f} nm)"
            )

        # Set final box vectors
        merged.structure.box_vectors = np.diag([box_xy, box_xy, box_z])

        # ---- 11c. Quality validation ----
        report_progress(0.97, "Validating bilayer quality")
        # Validation issues are reported as warnings (non-blocking) rather
        # than fatal errors.  The membrane is built and saved regardless;
        # users can increase lipids-per-leaflet and re-run if quality is
        # unacceptable.  Only skip the quality check entirely if there is
        # no membrane to validate.
        if mem_indices.size > 0:
            _validate_membrane_quality(
                merged,
                mem_indices,
                system.num_atoms,
                box_xy,
                box_z,
                has_solute,
                log,
                slab_half_thickness=dh / 2.0,
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

        def final_species_counts(leaflet, sizes):
            starts = np.cumsum([0] + list(sizes[:-1]))
            return _counts([str(leaflet.structure.resnames[int(i)]) for i in starts])

        # Add MEMBRANE component
        n_mem_atoms = membrane_system.num_atoms
        mem_start = merged.num_atoms - n_mem_atoms
        merged.add_component(
            Component(
                name=f"MEMBRANE_{comp_label}"[:50],
                kind=ComponentKind.MEMBRANE,
                atom_indices=np.arange(mem_start, merged.num_atoms),
                metadata={
                    "composition_upper": [(n, r) for n, r in upper_mix],
                    "composition_lower": [(n, r) for n, r in lower_mix],
                    "n_lipids_upper": actual_upper,
                    "n_lipids_lower": actual_lower,
                    "lipid_counts_upper": final_species_counts(upper_system, upper_lipid_sizes),
                    "lipid_counts_lower": final_species_counts(lower_system, lower_lipid_sizes),
                    "requested_lipids_per_leaflet": n_lipids_per_leaflet,
                    "leaflet_count_policy": "area-balanced"
                    if leaflet_counts is not None
                    else "legacy-area",
                    "solute_footprint_nm2": {
                        "upper": leaflet_footprints[0],
                        "lower": leaflet_footprints[1],
                    },
                    "target_area_per_lipid_nm2": {"upper": upper_area, "lower": lower_area},
                    "lipid_sizes": all_lipid_sizes,
                    "bilayer_thickness": actual_dhh,  # measured from placed lipids
                    "bilayer_thickness_nominal": dh,  # original estimate from registry
                    "equilibrated_library_lipids": library_hits,
                    "bootstrap_geometry_lipids": bootstrap_hits,
                    "orientation_quality": orientation_quality,
                    "box_xy": box_xy,
                    "box_z": box_z,
                    "box_padding": float(config.get("box_padding", config.get("pad", 2.0))),
                },
            )
        )

        from gmxbuilder.modules.membrane.composition_warnings import composition_warnings

        advice = composition_warnings({"upper": upper_mix, "lower": lower_mix})
        merged.metadata["membrane_composition_warnings"] = advice
        return ModuleResult(
            success=True,
            system=merged,
            log=log,
            warnings=[item["message"] for item in advice],
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
        atom_indices = (
            np.concatenate(
                [
                    np.arange(offsets[i], offsets[i + 1], dtype=np.int64)
                    for i in range(n_lipids)
                    if keep_mask[i]
                ]
            )
            if keep_mask.any()
            else np.empty(0, dtype=np.int64)
        )

        structure = leaflet_sys.structure
        structure.select_atoms(atom_indices)

        retained_sizes = [lipid_sizes[i] for i in range(n_lipids) if keep_mask[i]]
        leaflet_sys.metadata["lipid_sizes"] = retained_sizes
        anchor_indices = list(leaflet_sys.metadata.get("headgroup_anchor_local_indices") or [])
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
            # Deliberately single-threaded: this queries one lipid's ~130
            # atoms and runs once per lipid, so spawning a thread pool per
            # call costs far more than the query. Measured: asking for 48
            # workers here took the packing phase from 1.3 s to 7.2 s.
            dists, _ = prot_tree.query(leaflet_coords[start:end], k=1)
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
        area = box_xy**2
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

        elements = list(_elements_for_atom_names(tuple(atom_names)))

        return (
            rotated,
            atom_names,
            [lipid_name] * len(atom_names),
            [i + 1] * len(atom_names),
            elements,
        )

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
        from gmxbuilder.geometry.neighbor_index import AppendOnlyNeighbors

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
        bootstrap_conformer_retries = 0
        tree_options = {} if box_xy is None else {"boxsize": np.asarray([box_xy, box_xy, 0.0])}
        placed = AppendOnlyNeighbors(**tree_options)
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
                        ln,
                        force_field,
                        selected_lipid_ff,
                        rng=rng,
                    )
                    loaded = True
                    library_hits += 1
                except FileNotFoundError as exc:
                    if ln in LipidRegistry.list_builtin():
                        raise ModuleConfigError(
                            f"V4 conformers for {ln}/{selected_lipid_ff} became unavailable. "
                            "Refresh lipid availability and retry."
                        ) from exc
                    from gmxbuilder.modules.forcefield.lipid_policy import library_entry_superseded

                    superseded = library_entry_superseded(ln, force_field, selected_lipid_ff)
                    if superseded:
                        raise ModuleConfigError(
                            f"{ln} is temporarily unavailable: {superseded}"
                        ) from exc
            if not loaded:
                coords, atom_names = build_rdkit_lipid_geometry(
                    ln,
                    lipid.smiles,
                    force_field=force_field,
                    seed=int(conf_seeds[i] % 5),
                    net_charge=lipid.charge,
                    lipid_ff=lipid_ff,
                )
                bootstrap_hits += 1

            # Every backend and every library entry is normalized through the
            # same chemistry-based axis.  This is a rigid-body rotation, so
            # force-field bond geometry and equilibrated torsions are kept.
            try:
                coords = orient_lipid_to_outward_normal(
                    coords,
                    atom_names,
                    upper=True,
                )
            except LipidOrientationError as initial_exc:
                # RDKit embedding can occasionally return a compact folded
                # conformer whose head and tail centroids nearly cancel even
                # though other deterministic seeds give a valid amphiphile.
                # Retry only coordinate-only bootstrap backends.  GAFF2 and
                # Lipid21 carry authoritative cached coordinates and must not
                # be silently replaced by a different geometry source.
                retry_error = initial_exc
                recovered = False
                if loaded:
                    # The library preserves valid molecular shapes. Selecting a
                    # shape suitable for this placement belongs to the builder.
                    for _ in range(32):
                        retry_coords, retry_names = get_equilibrated_lipid_library().load_one(
                            ln,
                            force_field,
                            selected_lipid_ff,
                            rng=rng,
                        )
                        try:
                            coords = orient_lipid_to_outward_normal(
                                retry_coords, retry_names, upper=True
                            )
                        except LipidOrientationError as exc:
                            retry_error = exc
                            continue
                        atom_names = retry_names
                        recovered = True
                        break
                from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

                selected_lipid_ff = lipid_backend_for(ln, lipid_ff)
                if not loaded and selected_lipid_ff not in {"gaff2", "lipid21"}:
                    first_seed = int(conf_seeds[i] % 5)
                    for retry_seed in range(5):
                        if retry_seed == first_seed:
                            continue
                        retry_coords, retry_names = build_rdkit_lipid_geometry(
                            ln,
                            lipid.smiles,
                            force_field=force_field,
                            seed=retry_seed,
                            net_charge=lipid.charge,
                            lipid_ff=lipid_ff,
                        )
                        try:
                            coords = orient_lipid_to_outward_normal(
                                retry_coords,
                                retry_names,
                                upper=True,
                            )
                        except LipidOrientationError as exc:
                            retry_error = exc
                            continue
                        atom_names = retry_names
                        bootstrap_conformer_retries += 1
                        recovered = True
                        break
                if not recovered:
                    source = "pre-equilibrated library" if loaded else "bootstrap geometry"
                    raise ModuleConfigError(
                        f"Lipid {ln} from {source} cannot form a physical bilayer: {retry_error}"
                    ) from retry_error

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
            angles = base_angle + np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
            if placed.blocks:
                cosine, sine = np.cos(angles), np.sin(angles)
                rotations = np.zeros((len(angles), 3, 3))
                rotations[:, 0, 0] = rotations[:, 1, 1] = cosine
                rotations[:, 0, 1] = -sine
                rotations[:, 1, 0] = sine
                rotations[:, 2, 2] = 1
                candidates = coords @ rotations.transpose(0, 2, 1)
                candidates += np.asarray([grid_xy[i, 0], grid_xy[i, 1], z])
                search = candidates.reshape(-1, 3).copy()
                if box_xy is not None:
                    search[:, :2] = wrap_periodic_coordinates(search[:, :2], box_xy)
                clearance = placed.distances(search).reshape(len(angles), len(coords)).min(axis=1)
                # argmax preserves the old first-candidate tie rule.
                best_angle = angles[int(np.argmax(clearance))]
            else:
                best_angle = angles[0]
            best_result = self._build_one_lipid(
                (
                    i,
                    ln,
                    coords,
                    atom_names,
                    float(grid_xy[i, 0]),
                    float(grid_xy[i, 1]),
                    z,
                    float(best_angle),
                )
            )
            results.append((i, best_result))
            search = best_result[0].copy()
            if box_xy is not None:
                search[:, :2] = wrap_periodic_coordinates(search[:, :2], box_xy)
            placed.append(search)

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
        return System(
            structure=structure,
            metadata={
                "n_lipids": used,
                "lipid_sizes": lipid_sizes,
                "headgroup_anchor_local_indices": headgroup_anchor_local_indices,
                "library_hits": library_hits,
                "bootstrap_hits": bootstrap_hits,
                "bootstrap_conformer_retries": bootstrap_conformer_retries,
            },
        )


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
        min_distance_sq[selected[: index + 1]] = -1.0
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
                leaflet_system.coordinates[start:end],
                molecule_names,
            )
        except LipidOrientationError as exc:
            leaflet_name = "upper" if upper else "lower"
            raise ModuleConfigError(
                f"Cannot validate {leaflet_name}-leaflet lipid {index + 1}: {exc}"
            ) from exc
        projection, cosine = outward_orientation(profile, upper=upper)
        projections.append(projection)
        cosines.append(cosine)
        tail_z.append(leaflet_system.coordinates[start:end][profile.tail_indices, 2])
    return (
        np.asarray(projections, dtype=float),
        np.asarray(cosines, dtype=float),
        np.concatenate(tail_z),
    )


def _validate_bilayer_structure(
    upper_system: System,
    lower_system: System,
    log: list[str],
    *,
    allow_repairable_core_gap: bool = False,
) -> dict:
    """Enforce chemical orientation and hydrophobic-core continuity.

    Unlike the occupancy diagnostics below, these are hard structural
    invariants.  A system with a solvent-facing tail or a water-sized vacuum
    layer at the midplane is not emitted as a successful membrane.
    """
    upper_projection, upper_cosine, upper_tail_z = _leaflet_orientation_data(
        upper_system,
        upper=True,
    )
    lower_projection, lower_cosine, lower_tail_z = _leaflet_orientation_data(
        lower_system,
        upper=False,
    )
    projections = np.concatenate((upper_projection, lower_projection))
    cosines = np.concatenate((upper_cosine, lower_cosine))
    invalid = (projections < MIN_INWARD_PROJECTION_NM) | (cosines < MIN_INWARD_COSINE)
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
    core_sealed = tail_core_gap <= maximum_core_gap
    if not core_sealed and not allow_repairable_core_gap:
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
    if core_sealed:
        log.append(
            f"Hydrophobic core sealed: tail gap {tail_core_gap:.3f} nm "
            "(negative values indicate interdigitation)"
        )
    else:
        log.append(
            "WARNING: offline bootstrap core gap "
            f"{tail_core_gap:.3f} nm requires rigid regridding, gradual "
            "precompression, and explicit-solvent equilibration"
        )
    return {
        "passed": bool(core_sealed),
        "repair_required": bool(not core_sealed),
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

    current_dhh = _leaflet_headgroup_plane(upper_system, upper=True) - _leaflet_headgroup_plane(
        lower_system, upper=False
    )
    # Leaflet translation is a last-resort bulk correction.  Registry DHH is
    # an approximate fluid-bilayer target rather than an exact constraint;
    # allow a 5% contraction while sealing the core.  The previous fixed
    # 0.10-nm allowance was only ~2% for thick sphingomyelins and left a
    # water-sized midplane gap even with otherwise physical tail geometry.
    dhh_tolerance = max(0.10, 0.05 * float(target_dhh)) if target_dhh is not None else 0.10
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
        z_origin = min(float(upper_search[:, 2].min()), float(lower_search[:, 2].min())) - 1.0
        z_box = (
            max(float(upper_search[:, 2].max()), float(lower_search[:, 2].max())) - z_origin + 1.0
        )
        upper_search[:, :2] = wrap_periodic_coordinates(
            upper_search[:, :2],
            box_xy,
        )
        lower_search[:, :2] = wrap_periodic_coordinates(
            lower_search[:, :2],
            box_xy,
        )
        upper_search[:, 2] -= z_origin
        lower_search[:, 2] -= z_origin
        tree_options = {"boxsize": np.asarray([box_xy, box_xy, z_box])}
    tree = cKDTree(upper_search, **tree_options)
    dists, _ = tree.query(lower_search, k=1, workers=current_task_threads())
    min_dist = float(dists.min())

    # ---- metric 2: chemically identified hydrophobic-tail Z-gap ----
    # Headgroup atoms must not make an inverted conformation look sealed.
    _, _, upper_hydrophobic_z = _leaflet_orientation_data(
        upper_system,
        upper=True,
    )
    _, _, lower_hydrophobic_z = _leaflet_orientation_data(
        lower_system,
        upper=False,
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
                f"Leaflet separation: clash {min_dist:.3f} nm; backed off {separation:.3f} nm each"
            )
        return

    # ---- decide the optimal shift ----
    # We want to close the bulk tail gap but avoid driving the closest
    # atom pair into hard VDW overlap.  Take the *smaller* of the two
    # shifts so neither constraint is violated.
    target_bulk_overlap = 0.05  # nm — slight interdigitation of tail clouds

    # Shift needed to close the bulk tail gap
    shift_from_bulk = (
        (bulk_gap + target_bulk_overlap) / 2.0 if bulk_gap > -target_bulk_overlap else 0.0
    )
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
                f"Leaflets: bulk gap {bulk_gap:.3f} nm — shift too small ({shift:.4f} nm), skipped"
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
                f"Leaflet separation: clash {min_dist:.3f} nm → backed off {shift:.3f} nm each"
            )
    else:
        log.append(
            f"Leaflets at optimal contact (min {min_dist:.3f} nm, bulk gap {bulk_gap:.3f} nm)"
        )


def _seat_lipids_against_solute(
    leaflet_system: System,
    protein_coords: np.ndarray,
    target_contact: float,
    max_shift: float,
    log: list[str],
    leaflet_label: str = "",
    *,
    min_distance: float = 0.0,
) -> int:
    """Adjust near-solute lipids using rigid XY translations.

    Pull distant interface lipids toward target_contact and push contacts closer
    than min_distance outward. Distances and displacements are in nm. Preserve
    internal coordinates and Z positions; max_shift bounds each lipid's movement
    per pass. A zero min_distance disables outward movement.

    Only lipids within the local surface band are considered. Append progress to
    log using leaflet_label, and return the unresolved-contact count, or None when
    there are no lipid records. The caller handles unresolved contacts."""
    from scipy.spatial import cKDTree

    n_lipids = leaflet_system.metadata.get("n_lipids", 0)
    lipid_sizes = leaflet_system.metadata.get("lipid_sizes")
    if n_lipids == 0 or lipid_sizes is None:
        return

    offsets = np.cumsum([0] + list(lipid_sizes))
    coords = leaflet_system.coordinates
    surface_band = 0.6  # nm — only nudge lipids very close to the protein surface
    step_size = 0.04  # nm — tiny incremental shift per pass
    max_passes = 4  # converge quickly — this is a gap-filler, not a wall-builder

    total_pushed = 0

    for _pass in range(max_passes):
        prot_tree = cKDTree(protein_coords)
        pass_pushed = 0

        for li in range(n_lipids):
            start = offsets[li]
            end = offsets[li + 1]
            lipid_atoms = coords[start:end]

            # Single-threaded for the same reason as _filter_protein_clashes:
            # one lipid per query, four passes over every lipid.
            dists, idx = prot_tree.query(lipid_atoms, k=1)
            min_dist = float(dists.min())

            too_close = min_dist < min_distance
            # Bulk lipids are not touched; nor are lipids already seated.
            if not too_close and (min_dist <= target_contact or min_dist > surface_band):
                continue

            closest_lipid_i = int(dists.argmin())
            closest_prot_i = int(idx[closest_lipid_i])
            lip_xyz = lipid_atoms[closest_lipid_i]
            prot_xyz = protein_coords[closest_prot_i]

            dx = prot_xyz[0] - lip_xyz[0]
            dy = prot_xyz[1] - lip_xyz[1]
            norm = float(np.sqrt(dx * dx + dy * dy))
            if norm < 0.001:
                # The lipid atom is directly above or below the solute atom, so
                # there is no radial direction to move along. Leave it; the
                # caller counts it as unseated rather than moving it blindly.
                continue

            ux, uy = dx / norm, dy / norm
            if too_close:
                # Outward, by the deficit, capped like every other shift here.
                shift = -min(target_contact - min_dist, step_size, max_shift)
            else:
                shift = min(min_dist - target_contact, step_size, max_shift)

            coords[start:end, 0] += ux * shift
            coords[start:end, 1] += uy * shift
            pass_pushed += 1

        total_pushed = max(total_pushed, pass_pushed)
        if pass_pushed == 0:
            break  # converged

    unseated = 0
    if min_distance > 0.0:
        final_tree = cKDTree(protein_coords)
        for li in range(n_lipids):
            nearest = float(final_tree.query(coords[offsets[li] : offsets[li + 1]], k=1)[0].min())
            if nearest < min_distance:
                unseated += 1

    if total_pushed > 0:
        log.append(
            f"Solute-lipid seating ({leaflet_label}): "
            f"{total_pushed}/{n_lipids} lipids moved into the contact band "
            f"(target {target_contact:.2f} nm)"
        )
    else:
        log.append(
            f"Solute-lipid seating ({leaflet_label}): "
            f"all interface lipids already within {target_contact:.2f} nm"
        )
    return unseated


def _validate_membrane_quality(
    merged: System,
    mem_indices: np.ndarray,
    n_solute: int,
    box_xy: float,
    box_z: float,
    has_solute: bool,
    log: list[str],
    *,
    slab_half_thickness: float,
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
    # Step 11b may translate the whole system to centre the box, so the bilayer
    # is not at the origin any more. Every slab test below is taken about the
    # lipids' own midplane rather than about z=0.
    z_membrane_centre = float(lipid_coords[:, 2].min() + lipid_coords[:, 2].max()) / 2.0
    # ---- Check 1: local XY density uniformity ----
    cell_size = 1.0  # nm
    n_cells = max(3, int(box_xy / cell_size))
    cell_edges = np.linspace(-box_xy / 2, box_xy / 2, n_cells + 1)
    lipid_xy = lipid_coords[:, :2]
    # One binning pass rather than a Python loop that tested every atom against
    # every cell: that was n_cells^2 full-array comparisons over ~110k atoms,
    # done twice. The bins are the same half-open intervals except for the last
    # one, which histogram2d closes -- so an atom exactly on the far box edge is
    # now counted instead of silently dropped.
    cell_counts = np.histogram2d(lipid_xy[:, 0], lipid_xy[:, 1], bins=[cell_edges, cell_edges])[
        0
    ].astype(int)
    # Exclude cells overlapping protein.
    #
    # Only the part of it inside the bilayer. Masking on the whole solute
    # projected over every height excused the cells beneath an extracellular
    # domain from the coverage check -- which is exactly where lipids were
    # missing, so the one test able to see a lateral hole was blind over the
    # region that had one.
    if has_solute:
        prot_xy = coords[:n_solute][
            np.abs(coords[:n_solute, 2] - z_membrane_centre) <= slab_half_thickness
        ][:, :2]
        if len(prot_xy):
            protein_counts = np.histogram2d(
                prot_xy[:, 0], prot_xy[:, 1], bins=[cell_edges, cell_edges]
            )[0]
            cell_counts[protein_counts > 10] = -1
    valid_counts = cell_counts[cell_counts >= 0]
    if len(valid_counts) > 0:
        empty_cells = (valid_counts == 0).sum()
        total_valid = len(valid_counts)
        # Hexagonal packing trimmed to a square box leaves corners sparse.
        # Allow up to 15% empty cells — corners are expected to be empty.
        if empty_cells > total_valid * 0.15:
            log.append(
                f"⚠ Membrane has {empty_cells} empty XY cells "
                f"({empty_cells}/{total_valid}, {empty_cells / total_valid * 100:.0f}%). "
                f"Increase lipids per leaflet for full coverage."
            )

    # ---- Check 2: Z-axis seal ----
    z_all_lipid = lipid_coords[:, 2]
    z_mid = z_membrane_centre
    upper_z = z_all_lipid[z_all_lipid > z_mid]
    lower_z = z_all_lipid[z_all_lipid < z_mid]
    if len(upper_z) > 0 and len(lower_z) > 0:
        # A regular lattice at the 0.4 nm pitch this already asked for, instead
        # of a hundred random draws. The cap made the pitch a fiction: any box
        # wider than 4 nm got the same hundred points however large it grew, so
        # a 15 nm system was judged on one sample per 2.3 nm2 -- coarser than
        # the holes being looked for. A lattice also removes the seeded draw,
        # so the answer no longer depends on which points chance supplied.
        pitch = 0.4
        n_side = max(5, int(box_xy / pitch))
        axis = (np.arange(n_side) + 0.5) * (box_xy / n_side)
        sample_points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        periodic_lipid_xy = wrap_periodic_coordinates(
            lipid_coords[:, :2] + box_xy / 2.0,
            box_xy,
        )
        # Samples standing on the solute are not membrane, and counting them as
        # holes would make a correctly built system look worse the larger its
        # protein is.
        if has_solute:
            solute_slab = coords[:n_solute][
                np.abs(coords[:n_solute, 2] - z_membrane_centre) <= slab_half_thickness
            ][:, :2]
            if len(solute_slab):
                solute_periodic = wrap_periodic_coordinates(solute_slab + box_xy / 2.0, box_xy)
                on_solute, _ = cKDTree(solute_periodic, boxsize=box_xy).query(
                    sample_points, k=1, workers=current_task_threads()
                )
                sample_points = sample_points[on_solute >= 0.7]
        n_xy_samples = max(len(sample_points), 1)
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
                f"⚠ Membrane has Z-axis gaps in {gap_fraction * 100:.0f}% of "
                f"sampled XY area. Increase lipids per leaflet to seal the bilayer."
            )
        elif gap_fraction > 0.0:
            log.append(f"Membrane Z-seal: {gap_fraction * 100:.0f}% sparse (within tolerance)")

    # ---- Check 3: protein-lipid interface seal ----
    if has_solute:
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
            prot_dists, _ = lipid_tree_3d.query(prot_coords, k=1, workers=current_task_threads())
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
    if has_solute:
        prot_coords = coords[:n_solute]
        box_half = box_xy / 2.0
        # Distance from each protein atom to each of the 4 box edges
        dist_to_edges = np.stack(
            [
                prot_coords[:, 0] - (-box_half),  # left edge
                box_half - prot_coords[:, 0],  # right edge
                prot_coords[:, 1] - (-box_half),  # bottom edge
                box_half - prot_coords[:, 1],  # top edge
            ],
            axis=1,
        )  # (N_prot, 4)
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
