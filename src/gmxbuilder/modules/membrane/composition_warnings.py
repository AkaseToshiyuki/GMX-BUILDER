"""Nonblocking composition advice; never a substitute for construction gates."""

from collections import defaultdict


def composition_warnings(leaflets, *, resolution="atomistic"):
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    warnings = []
    for leaflet, entries in leaflets.items():
        counts = defaultdict(float)
        for entry in entries:
            if isinstance(entry, dict):
                name, ratio = entry["name"], entry["ratio"]
            else:
                name, ratio = entry
            if float(ratio) > 0:
                counts[str(name).upper()] += float(ratio)
        if not counts:
            continue  # Input validity is checked by the caller, not softened here.
        categories = []
        for name in counts:
            try:
                categories.append(LipidRegistry.get(name).category)
            except KeyError:
                categories.append(None)
        if all(category == "ST" for category in categories):
            warnings.append(
                {
                    "code": "sterol_only_leaflet",
                    "severity": "warning",
                    "blocking": False,
                    "affected_leaflets": [leaflet],
                    "composition": dict(counts),
                    "resolution": resolution,
                    "message": f"The {leaflet} leaflet contains only sterols. A stable hydrated "
                    "bilayer is not established for this composition. You may continue building; "
                    "the resulting structure requires physical validation.",
                    "evidence_refs": [],
                }
            )
        if resolution == "atomistic" and set(counts) == {"DOPE"}:
            # Experimental phase behaviour is a warning, not an assumed
            # simulation outcome or a transition assigned to a Martini model.
            warnings.append(
                {
                    "code": "dope_nonlamellar_phase_risk",
                    "severity": "warning",
                    "blocking": False,
                    "affected_leaflets": [leaflet],
                    "composition": dict(counts),
                    "resolution": resolution,
                    "message": f"The {leaflet} leaflet contains only DOPE. Experimental DOPE "
                    "data include a lamellar-to-inverted-hexagonal transition. A planar "
                    "bilayer may not be the equilibrium structure at your chosen temperature "
                    "and hydration. You may continue; validate the intended phase.",
                    "evidence_refs": [
                        "https://www.avantiresearch.com/en-gb/support-hub/physical-properties/"
                        "phase-transition-temps"
                    ],
                }
            )
    return warnings
