"""Water model registry — TIP3P, SPC/E, TIP4P."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WaterModel:
    """Metadata for a water model."""

    name: str               # "tip3p", "spce", "tip4p"
    full_name: str          # "TIP3P", "SPC/E", "TIP4P"
    n_atoms: int            # 3 or 4
    charge: float           # Total molecular charge
    atom_names: list[str]   # ["OW", "HW1", "HW2"] etc.
    atom_masses: list[float]
    default_density: float  # kg/L, used to compute box size
    approximate_radius: float  # nm, for overlap detection
    oh_bond: float          # nm
    hoh_angle: float        # degrees
    virtual_site_distance: float | None = None  # O-to-M distance, nm


class WaterRegistry:
    """Registry of supported water models."""

    _models: dict[str, WaterModel] = {}

    _DEFAULTS = {
        "tip3p": WaterModel(
            name="tip3p",
            full_name="TIP3P",
            n_atoms=3,
            charge=0.0,
            atom_names=["OW", "HW1", "HW2"],
            atom_masses=[15.9994, 1.008, 1.008],
            default_density=0.998,
            approximate_radius=0.14,
            oh_bond=0.09572,
            hoh_angle=104.52,
        ),
        "spc": WaterModel(
            name="spc",
            full_name="SPC",
            n_atoms=3,
            charge=0.0,
            atom_names=["OW", "HW1", "HW2"],
            atom_masses=[15.9994, 1.008, 1.008],
            default_density=0.978,
            approximate_radius=0.14,
            oh_bond=0.10000,
            hoh_angle=109.47,
        ),
        "spce": WaterModel(
            name="spce",
            full_name="SPC/E",
            n_atoms=3,
            charge=0.0,
            atom_names=["OW", "HW1", "HW2"],
            atom_masses=[15.9994, 1.008, 1.008],
            default_density=0.998,
            approximate_radius=0.14,
            oh_bond=0.10000,
            hoh_angle=109.47,
        ),
        "tip4p": WaterModel(
            name="tip4p",
            full_name="TIP4P",
            n_atoms=4,
            charge=0.0,
            atom_names=["OW", "HW1", "HW2", "MW"],
            atom_masses=[15.9994, 1.008, 1.008, 0.0],
            default_density=0.997,
            approximate_radius=0.14,
            oh_bond=0.09572,
            hoh_angle=104.52,
            virtual_site_distance=0.01546,
        ),
    }

    @classmethod
    def get(cls, name: str) -> WaterModel:
        name = name.lower()
        if name not in cls._models:
            if name in cls._DEFAULTS:
                return cls._DEFAULTS[name]
            raise KeyError(f"Unknown water model: {name!r}. Available: {cls.list()}")
        return cls._models[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(set(cls._DEFAULTS.keys()) | set(cls._models.keys()))

    @classmethod
    def register(cls, model: WaterModel) -> None:
        cls._models[model.name.lower()] = model


def water_model_supported(force_field: str, water_model: str) -> bool:
    """Return whether the bundled force field contains this water topology."""
    import gmxbuilder.data.forcefields as forcefield_data

    base = Path(forcefield_data.__path__[0])
    ff_name = force_field.strip().lower()
    model_name = water_model.strip().lower()
    for directory in (base / ff_name, base / f"{ff_name}.ff"):
        if directory.is_dir():
            return (directory / f"{model_name}.itp").is_file()
    return False


def supported_force_fields(water_model: str) -> list[str]:
    from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry

    return [
        name for name in ForceFieldRegistry.list()
        if water_model_supported(name, water_model)
    ]
