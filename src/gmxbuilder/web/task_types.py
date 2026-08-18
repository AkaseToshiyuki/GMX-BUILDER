"""Task type definitions.

Each task type defines which pipeline modules are relevant and
what default configuration is recommended. Adding a new task type
here automatically populates the frontend wizard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Task type registry
# ---------------------------------------------------------------------------


@dataclass
class TaskType:
    id: str  # unique slug, e.g. "membrane-bilayer"
    category: str  # "Membrane", "Solution", "Glycan", etc.
    title: str  # display name, e.g. "Bilayer Builder"
    description: str  # one-sentence summary
    icon: str  # emoji or SVG placeholder
    enabled: bool = True  # False = greyed-out with "Coming Soon"
    pipeline: str = "membrane"  # "membrane" | "solvator" — which pipeline to create
    requires_input: bool = True  # whether the workflow starts from an uploaded structure
    route_slug: str = ""  # stable browser URL segment
    required_modules: list[str] = field(default_factory=list)  # module names that MUST run
    visible_modules: list[str] = field(default_factory=list)  # module names shown in wizard
    default_config: dict = field(default_factory=dict)  # default module settings


_TASK_TYPES: list[TaskType] = [
    TaskType(
        id="martini3-bilayer",
        category="Coarse Grained",
        title="Martini 3 Bilayer Builder",
        description="Build a Martini 3 protein/bilayer complex or protein-free bilayer",
        icon="🧬",
        enabled=True,
        pipeline="martini_bilayer",
        requires_input=False,
        route_slug="Martini3BilayerBuilder",
        required_modules=[
            "input",
            "cg_model",
            "cg_mapping",
            "cg_orientation",
            "cg_environment",
            "cg_solvation",
            "cg_system",
        ],
        visible_modules=[
            "input",
            "cg_model",
            "cg_mapping",
            "cg_orientation",
            "cg_environment",
            "cg_solvation",
            "cg_system",
            "simparams",
        ],
        default_config={
            "input": {"include_protein": True, "environment": "bilayer"},
            "cg_model": {"model": "martini3", "water_model": "W"},
            "cg_mapping": {
                "protein_model": "folded",
                "secondary_structure": "auto",
                "elastic": True,
            },
            "cg_orientation": {"method": "ppm", "half_thickness": 1.4},
            "cg_environment": {
                "n_lipids_per_leaflet": 150,
                "upper_leaflet": [{"name": "POPC", "ratio": 100}],
                "lower_leaflet": [{"name": "POPC", "ratio": 100}],
                "asymmetric": False,
            },
            "cg_solvation": {"include_solvent": True, "padding_nm": 2.0},
            "cg_system": {"salt_molarity": 0.15},
        },
    ),
    TaskType(
        id="martini3-solvent",
        category="Coarse Grained",
        title="Martini 3 Solvent Builder",
        description="Map a standard protein and build a solvated Martini 3 system",
        icon="💧",
        enabled=True,
        pipeline="martini_solvent",
        requires_input=True,
        route_slug="Martini3SolventBuilder",
        required_modules=[
            "input",
            "cg_model",
            "cg_mapping",
            "cg_environment",
            "cg_solvation",
            "cg_system",
        ],
        visible_modules=[
            "input",
            "cg_model",
            "cg_mapping",
            "cg_environment",
            "cg_solvation",
            "cg_system",
            "simparams",
        ],
        default_config={
            "input": {"include_protein": True, "environment": "solution"},
            "cg_model": {"model": "martini3", "water_model": "W"},
            "cg_mapping": {
                "protein_model": "folded",
                "secondary_structure": "auto",
                "elastic": True,
            },
            "cg_environment": {},
            "cg_solvation": {"include_solvent": True, "padding_nm": 1.5},
            "cg_system": {"salt_molarity": 0.15},
        },
    ),
    TaskType(
        id="membrane-bilayer",
        category="Membrane",
        title="Bilayer Builder",
        description="Generate a protein/bilayer complex or bilayer-only system for molecular dynamics simulations",
        icon="🫧",
        enabled=True,
        route_slug="BilayerBuilder",
        required_modules=[
            "input",
            "forcefield",
            "structure",
            "orient",
            "membrane",
            "solvation",
            "ions",
        ],
        visible_modules=[
            "input",
            "forcefield",
            "structure",
            "orient",
            "membrane",
            "solvation",
            "ions",
            "simparams",
        ],
        default_config={
            "membrane": {"lipid_type": "POPC", "bilayer_size": "auto", "box_padding": 2.0},
            "solvation": {"box_padding": 2.0},
            "ions": {"concentration": 0.15, "neutralize": True, "cation": "NA", "anion": "CL"},
            "forcefield": {"name": "amber14sb", "water_model": "tip3p"},
            "export": {"write_mdp": True},
        },
    ),
    TaskType(
        id="pure-membrane",
        category="Membrane",
        title="Pure Bilayer System",
        description="Build a relaxed lipid-only bilayer, optionally with water and ions",
        icon="🟦",
        enabled=True,
        pipeline="pure_membrane",
        requires_input=False,
        route_slug="PureBilayerSystem",
        required_modules=["forcefield", "membrane"],
        visible_modules=["forcefield", "membrane", "solvation", "ions", "simparams"],
        default_config={
            "membrane": {"lipid_type": "POPC", "n_lipids_per_leaflet": 150},
            "solvation": {"enabled": True, "box_padding": 2.0},
            "ions": {"concentration": 0.15, "neutralize": True, "cation": "NA", "anion": "CL"},
            "forcefield": {"name": "amber14sb", "water_model": "tip3p"},
            "export": {"write_mdp": True},
        },
    ),
    TaskType(
        id="membrane-monolayer",
        category="Membrane",
        title="Monolayer Builder",
        description="Generate a protein/monolayer complex or monolayer-only system for molecular dynamics simulations",
        icon="🪞",
        enabled=False,
        required_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "topology",
        ],
        visible_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "simparams",
        ],
    ),
    TaskType(
        id="nanodisc-builder",
        category="Membrane",
        title="Nanodisc Builder",
        description="Generate a lipid-only or protein-embedded nanodisc system for molecular dynamics simulations",
        icon="🪙",
        enabled=False,
        required_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "topology",
        ],
        visible_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "simparams",
        ],
    ),
    TaskType(
        id="hmmm-builder",
        category="Membrane",
        title="HMMM Builder",
        description="Generate a bilayer simulation system with the Highly Mobile Membrane-Mimetic (HMMM) model",
        icon="🔬",
        enabled=False,
        required_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "topology",
        ],
        visible_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "simparams",
        ],
    ),
    TaskType(
        id="bicelle-builder",
        category="Membrane",
        title="Bicelle Builder",
        description="Generate a protein/bicelle complex or bicelle-only system for molecular dynamics simulations",
        icon="🫓",
        enabled=False,
        required_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "topology",
        ],
        visible_modules=[
            "input",
            "forcefield",
            "structure",
            "membrane",
            "solvation",
            "ions",
            "simparams",
        ],
    ),
    TaskType(
        id="solvator",
        category="Solution",
        title="Solvator",
        description="Solvate canonical protein, DNA/RNA, and compatible ligand systems",
        icon="💧",
        enabled=True,
        pipeline="solvator",
        route_slug="Solvator",
        required_modules=["input", "forcefield", "structure", "solvation", "ions"],
        visible_modules=["input", "forcefield", "structure", "solvation", "ions", "simparams"],
        default_config={
            "solvation": {"box_padding": 1.5},
            "ions": {"concentration": 0.15, "neutralize": True, "cation": "NA", "anion": "CL"},
            "forcefield": {"name": "amber14sb", "water_model": "tip3p"},
            "export": {"write_mdp": True},
            "simparams": {"pcoupl_type": "isotropic", "constraints": "h-bonds"},
        },
    ),
    TaskType(
        id="liquid-builder",
        category="Solution",
        title="Liquid Builder",
        description="Generate pure liquid or mixed-solvent boxes for diffusion/phase studies",
        icon="🧪",
        enabled=False,  # needs organic solvent geometry generation
        pipeline="liquid",
        required_modules=["solvation", "ions", "topology"],
        visible_modules=["solvation", "ions", "simparams"],
        default_config={
            "solvation": {"water_model": "tip3p", "box_padding": 0.0, "box_size": [5.0, 5.0, 5.0]},
            "ions": {"concentration": 0.0, "neutralize": False},
            "forcefield": {"name": "amber14sb"},
            "export": {"write_mdp": True},
            "simparams": {"pcoupl_type": "isotropic", "constraints": "h-bonds"},
        },
    ),
    # solution-builder: superseded by solvator (solution-phase pipeline).
    # Glycan / Ligand / Glycolipid / LPS: not yet implemented.
    #   Ligand parameterization is handled by the CGenFF module integrated
    #   in the existing pipeline (SMILES → 3D → charges → ITP).
    #   Carbohydrate/glycolipid/LPS support requires dedicated topology
    #   databases and will be re-added when the data is ready.
    # Nanomaterial: no implementation plan — very niche.
]

_CATEGORY_ORDER = {
    "Membrane": 0,
    "Solution": 1,
    "Coarse Grained": 2,
}


def get_all_task_types() -> list[dict]:
    """Return all task types as dictionaries for the API."""
    result = []
    for t in _TASK_TYPES:
        result.append(
            {
                "id": t.id,
                "category": t.category,
                "title": t.title,
                "description": t.description,
                "icon": t.icon,
                "enabled": t.enabled,
                "route_slug": t.route_slug,
            }
        )
    # Keep the public landing page organized by workflow scale while
    # preserving the registry order within each category.
    result.sort(
        key=lambda item: _CATEGORY_ORDER.get(item["category"], len(_CATEGORY_ORDER))
    )
    return result


def get_task_type(task_id: str) -> TaskType | None:
    """Look up a task type by ID."""
    for t in _TASK_TYPES:
        if t.id == task_id:
            return t
    return None


def get_task_type_detail(task_id: str) -> dict | None:
    """Return full detail for a task type (for the wizard)."""
    t = get_task_type(task_id)
    if t is None:
        return None
    return {
        "id": t.id,
        "category": t.category,
        "title": t.title,
        "description": t.description,
        "icon": t.icon,
        "enabled": t.enabled,
        "pipeline": t.pipeline,
        "requires_input": t.requires_input,
        "route_slug": t.route_slug,
        "required_modules": t.required_modules,
        "visible_modules": t.visible_modules,
        "default_config": t.default_config,
    }
