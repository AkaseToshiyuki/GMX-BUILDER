"""Choose and record area estimates for initial membrane sizing.

Use a compatible, converged pure-bilayer library measurement when available.
Otherwise use a cholesterol-based partial-area estimate for sterols, the
registry area for other known lipids, or the unknown-lipid fallback.

Leaflet areas are ratio-weighted construction estimates. A hosted trajectory's
box area belongs to the mixture, not the guest lipid alone. Neither linear
mixing nor the sterol fallback certifies the equilibrium area of a new mixture."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from gmxbuilder.modules.membrane.lipids import LipidRegistry

#: Registry category for sterols.
STEROL_CATEGORY = "ST"

# Cholesterol partial area measured in monounsaturated PC bilayers, nm^2:
# Gallova et al., Chem Phys Lipids 163:765-770 (2010),
# doi:10.1016/j.chemphyslip.2010.08.002. Applying this value to other sterols
# or host classes is a construction approximation, not measured equivalence.
CHOLESTEROL_PARTIAL_AREA_NM2 = 0.24

CHOLESTEROL_PARTIAL_AREA_SOURCE = (
    "Gallova et al., Chem Phys Lipids 163(8):765-770 (2010), SANS, "
    "concentration-independent in monounsaturated PC"
)

#: Fallback for a lipid absent from the registry entirely.
UNKNOWN_LIPID_AREA_NM2 = 0.65


@dataclass(frozen=True)
class AreaTerm:
    """One lipid's contribution, with where its number came from."""

    lipid: str
    fraction: float
    area_nm2: float
    source: str

    def as_metadata(self) -> dict:
        return {
            "lipid": self.lipid,
            "fraction": self.fraction,
            "area_per_lipid_nm2": self.area_nm2,
            "source": self.source,
        }


@dataclass(frozen=True)
class LeafletArea:
    """A leaflet's construction area, and the terms that produced it."""

    area_per_lipid_nm2: float
    terms: tuple[AreaTerm, ...]

    def as_metadata(self) -> dict:
        return {
            "area_per_lipid_nm2": self.area_per_lipid_nm2,
            "rule": "ratio-weighted-partial-areas",
            "terms": [term.as_metadata() for term in self.terms],
        }

    def summary(self) -> str:
        parts = ", ".join(
            f"{term.lipid} {term.fraction * 100:.0f}% {term.area_nm2:.3f} ({term.source})"
            for term in self.terms
        )
        return f"area/lipid {self.area_per_lipid_nm2:.3f} nm^2 from {parts}"


def is_sterol(name: str) -> bool:
    try:
        return LipidRegistry.get(name).category == STEROL_CATEGORY
    except KeyError:
        return False


def sterol_partial_area(name: str) -> float:
    """Scale the registry cross-section using cholesterol's measured partial area.

    The same factor is applied to all sterols, preserving their registry size
    ordering. This is a construction approximation, not an experimental partial
    area measurement for each sterol and host composition."""
    reference = float(LipidRegistry.get("CHOL").area_per_lipid)
    own = float(LipidRegistry.get(name).area_per_lipid)
    return own * (CHOLESTEROL_PARTIAL_AREA_NM2 / reference)


def converged_library_area(
    name: str,
    force_field: str,
    lipid_ff: str | None = None,
    *,
    temperature_K: float | None = None,
) -> float | None:
    """Return a converged pure-bilayer area for this lipid, or None.

    Deliberately strict. It accepts only the V4 observables block, only when
    that block says it converged, and only when it measured a *pure* bilayer:
    a hosted lipid's entry is built in a mixture, so its area belongs to
    the mixture and using it as the sterol's own would repeat the error this
    module exists to correct.
    """
    from gmxbuilder.modules.membrane.equilibrated_library import (
        EquilibratedLipidLibrary,
        lipid_parameter_family,
    )

    try:
        family = lipid_parameter_family(force_field, lipid_ff)
    except ValueError:
        return None

    try:
        library = EquilibratedLipidLibrary()
    except (RuntimeError, OSError, ValueError):
        # Constructing the library materialises the prebuilt asset archive, and
        # that can fail for reasons this function has no business propagating:
        # a source checkout whose Git LFS pointers are unfetched, a read-only
        # cache, a partial download. "No measured area is available" is the
        # correct answer in every one of those cases, and the caller falls back
        # to the registry. Raising instead took the whole membrane step down
        # with it -- observed in CI, where the archive is an unfetched pointer.
        return None

    for root in library.roots:
        try:
            directory = library._contained_entry(root, family, name)
        except ValueError:
            continue
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        protocol = metadata.get("v4_protocol")
        if protocol:
            # A protocol-scoped measurement is not a temperature-independent
            # material constant. Callers without a target temperature keep the
            # explicitly labelled construction estimate instead.
            try:
                if temperature_K is None or not math.isclose(
                    float(protocol["temperature_K"]),
                    float(temperature_K),
                    abs_tol=1e-6,
                    rel_tol=0,
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
        observables = metadata.get("observables")
        if not isinstance(observables, dict):
            continue  # a V3 entry: its area is one frame of a 1 ns run
        if not observables.get("converged"):
            continue
        from gmxbuilder.modules.membrane.area_observable import AUTOCORRELATION_METHOD

        if observables.get("autocorrelation_method") != AUTOCORRELATION_METHOD:
            continue
        if observables.get("measures") != "pure-bilayer-area-per-lipid":
            continue
        block = observables.get("area_per_lipid_nm2")
        if not isinstance(block, dict):
            continue
        mean = block.get("mean")
        if isinstance(mean, (int, float)) and math.isfinite(mean) and mean > 0:
            return float(mean)
    return None


def species_area(
    name: str,
    force_field: str = "",
    lipid_ff: str | None = None,
) -> tuple[float, str]:
    """Return this lipid's construction area and the source it came from."""
    if force_field:
        measured = converged_library_area(name, force_field, lipid_ff)
        if measured is not None:
            return measured, "measured-converged-library"

    if is_sterol(name):
        return sterol_partial_area(name), "literature-partial-molar-area"

    try:
        return float(LipidRegistry.get(name).area_per_lipid), "registry-experimental"
    except KeyError:
        return UNKNOWN_LIPID_AREA_NM2, "default-unknown-lipid"


def leaflet_area_per_lipid(
    composition: list[tuple[str, float]],
    force_field: str = "",
    lipid_ff: str | None = None,
) -> LeafletArea:
    """Return the ratio-weighted construction area for one leaflet."""
    terms: list[AreaTerm] = []
    total_ratio = 0.0
    for name, ratio in composition:
        value = float(ratio)
        if value <= 0.0:
            continue
        area, source = species_area(name, force_field, lipid_ff)
        terms.append(AreaTerm(lipid=str(name), fraction=value, area_nm2=area, source=source))
        total_ratio += value

    if total_ratio <= 0.0:
        raise ValueError("Leaflet composition must contain a positive lipid ratio")

    normalised = tuple(
        AreaTerm(term.lipid, term.fraction / total_ratio, term.area_nm2, term.source)
        for term in terms
    )
    area = sum(term.fraction * term.area_nm2 for term in normalised)
    return LeafletArea(area_per_lipid_nm2=float(area), terms=normalised)


__all__ = [
    "CHOLESTEROL_PARTIAL_AREA_NM2",
    "CHOLESTEROL_PARTIAL_AREA_SOURCE",
    "AreaTerm",
    "LeafletArea",
    "converged_library_area",
    "is_sterol",
    "leaflet_area_per_lipid",
    "species_area",
    "sterol_partial_area",
]
