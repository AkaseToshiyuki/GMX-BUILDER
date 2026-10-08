"""UI option-catalog construction independent of FastAPI routing."""

from __future__ import annotations

from gmxbuilder.modules.forcefield.catalog import (
    force_field_installed,
    get_force_field_profile,
)
from gmxbuilder.modules.forcefield.gaff_backend import gaff_available
from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability
from gmxbuilder.modules.forcefield.lipid_policy import (
    gaff_lipid_capability,
    lipid_has_rtp,
)
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry
from gmxbuilder.modules.membrane.chain_identity import chain_identity as lipid_chain_identity
from gmxbuilder.modules.membrane.lipids import CATEGORY_NAMES, LipidRegistry
from gmxbuilder.modules.solvation.water_models import WaterRegistry, supported_force_fields


def build_ui_options(*, availability: dict | None = None) -> dict:
    """Return the available, locally executable choices for UI dropdowns."""
    from gmxbuilder.modules.membrane.v4_availability import refresh_availability_list

    if availability is None:
        try:
            availability = refresh_availability_list()
        except OSError:
            availability = {"entries": []}
    ready = {
        (entry["lipid_name"], entry["lipid_ff"]): entry
        for entry in availability["entries"]
        if entry["ready"]
    }
    gaff_ready = gaff_available()

    def lipid_option(name: str) -> dict:
        lipid = LipidRegistry.get(name)
        sources = []
        lipid21_supported, _lipid21_reason = lipid21_capability(name)
        if lipid21_supported:
            sources.append("lipid21")
        gaff_supported, gaff_reason = gaff_lipid_capability(name)
        if gaff_ready and gaff_supported:
            sources.append("gaff2")
        if lipid_has_rtp(name, "charmm36m"):
            sources.append("charmm36m")
        if lipid_has_rtp(name, "charmm36"):
            sources.append("charmm36")
        parameter_sources = list(sources)
        sources = [source for source in sources if (name, source) in ready]
        if any(ready[(name, source)]["amber_mixed_ready"] for source in sources):
            sources.append("amber-mixed")
        scopes = {
            source: ready[(name, source)].get("validation_scope")
            for source in sources
            if (name, source) in ready
        }
        if "amber-mixed" in sources:
            scopes["amber-mixed"] = next(
                ready[(name, source)].get("validation_scope")
                for source in sources
                if (name, source) in ready and ready[(name, source)]["amber_mixed_ready"]
            )
        source_labels = {
            "amber-mixed": "Lipid21 with GAFF2 for missing species",
            "lipid21": "Amber Lipid21 (native parameters)",
            "gaff2": "Amber/GAFF2",
            "charmm36m": "CHARMM36m RTP",
            "charmm36": "CHARMM36 RTP",
        }
        return {
            "name": name,
            "common_name": lipid.common_name,
            "category": lipid.category,
            "formula": lipid.formula,
            "headgroup": lipid.headgroup,
            "tail1": list(lipid.tail1),
            "tail2": list(lipid.tail2),
            **lipid_chain_identity(lipid.smiles),
            "area_per_lipid": lipid.area_per_lipid,
            "charge": lipid.charge,
            "thickness": lipid.bilayer_thickness,
            "smiles": lipid.smiles,
            "parameterizations": sources,
            "parameter_sources": parameter_sources,
            "library_version": 4,
            "validation_scopes": scopes,
            "parameterization": " + ".join(source_labels[item] for item in sources)
            if sources
            else "Unavailable",
            "gaff2_unavailable_reason": gaff_reason if not gaff_supported else "",
        }

    return {
        "lipids": [lipid_option(name) for name in LipidRegistry.list()],
        "lipid_categories": {
            category: {"label": CATEGORY_NAMES.get(category, category), "lipids": names}
            for category, names in LipidRegistry.list_by_category().items()
        },
        "water_models": [
            {
                "name": name,
                "full_name": WaterRegistry.get(name).full_name,
                "n_atoms": WaterRegistry.get(name).n_atoms,
                "supported_force_fields": supported_force_fields(name),
            }
            for name in WaterRegistry.list()
        ],
        "solvents": [
            {"name": "tip3p", "label": "TIP3P Water", "category": "water", "density": 0.998},
            {"name": "spc", "label": "SPC Water", "category": "water", "density": 0.978},
            {"name": "spce", "label": "SPC/E Water", "category": "water", "density": 0.998},
            {"name": "tip4p", "label": "TIP4P Water", "category": "water", "density": 0.997},
            {
                "name": "methanol",
                "label": "Methanol (MeOH)",
                "category": "organic",
                "density": 0.791,
            },
            {
                "name": "ethanol",
                "label": "Ethanol (EtOH)",
                "category": "organic",
                "density": 0.789,
            },
            {
                "name": "1propanol",
                "label": "1-Propanol (PrOH)",
                "category": "organic",
                "density": 0.803,
            },
        ],
        # `installed` is what greys an option out. It is recomputed on every
        # request rather than cached: the installer can run while the service
        # is up, and a stale "no" would keep a working force field disabled
        # until somebody restarted it.
        "force_fields": [
            {
                **get_force_field_profile(name).as_dict(),
                "version": ForceFieldRegistry.get(name).version,
                "water_model": ForceFieldRegistry.get(name).water_model,
                "installed": force_field_installed(name),
            }
            for name in ForceFieldRegistry.list()
        ],
        "gaff2_available": gaff_ready,
    }
