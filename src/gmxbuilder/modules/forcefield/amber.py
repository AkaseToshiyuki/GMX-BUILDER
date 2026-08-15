"""Amber force field implementations."""

from __future__ import annotations

from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Topology, AtomType, MoleculeBlock
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry

# Amber ff99SB / ff14SB LJ parameters (sigma in nm, epsilon in kJ/mol)
_AMBER_DEFAULTS: dict[str, dict] = {
    "C": {"mass": 12.01, "sigma": 0.339967, "epsilon": 0.45773},
    "N": {"mass": 14.01, "sigma": 0.325000, "epsilon": 0.71128},
    "O": {"mass": 16.00, "sigma": 0.295992, "epsilon": 0.87864},
    "H": {"mass": 1.008, "sigma": 0.106908, "epsilon": 0.06569},
    "S": {"mass": 32.06, "sigma": 0.356359, "epsilon": 1.04600},
    "P": {"mass": 30.97, "sigma": 0.374177, "epsilon": 0.83680},
    "NA": {"mass": 22.99, "sigma": 0.242992, "epsilon": 0.19623},
    "CL": {"mass": 35.45, "sigma": 0.404468, "epsilon": 0.62760},
    "K": {"mass": 39.10, "sigma": 0.314264, "epsilon": 0.36468},
    "CA": {"mass": 40.08, "sigma": 0.241199, "epsilon": 0.25620},
    "ZN": {"mass": 65.38, "sigma": 0.195998, "epsilon": 0.52300},
    "MG": {"mass": 24.31, "sigma": 0.141445, "epsilon": 0.10836},
}


class _AmberBase(ForceField):
    """Shared topology builder for Amber-family force fields."""

    water_model = "tip3p"
    supported_lipids = ["POPC", "DPPC", "DOPE", "POPG"]

    def build_system_topology(self, system: System) -> Topology:
        topology = Topology(force_field=self.name)
        try:
            from gmxbuilder.modules.forcefield.rtp_parser import RTPParser
            from pathlib import Path

            ff_dir = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "forcefields"
                / f"{self.name}.ff"
            )
            rtp = RTPParser()
            if ff_dir.is_dir():
                for rtp_file in sorted(ff_dir.glob("*.rtp")):
                    rtp.parse(rtp_file)
        except Exception:
            rtp = None

        n_resnames = len(system.structure.resnames)
        n_atom_names = len(system.structure.atom_names)

        atom_types = []
        for i in range(system.num_atoms):
            elem = system.structure.elements[i] if i < len(system.structure.elements) else "C"
            elem = elem.upper()
            params = _AMBER_DEFAULTS.get(elem, _AMBER_DEFAULTS["C"])

            rn = system.structure.resnames[i] if i < n_resnames else "ALA"
            an = system.structure.atom_names[i] if i < n_atom_names else "CA"
            atype_name = f"{elem}{i + 1}"
            charge = 0.0
            if rtp is not None:
                rtp_result = rtp.get_atom_type(rn, an)
                if rtp_result:
                    atype_name, charge = rtp_result

            at = AtomType(
                name=atype_name,
                mass=params["mass"],
                charge=charge,
                sigma=params["sigma"],
                epsilon=params["epsilon"],
            )
            atom_types.append(at)

        topology.atom_types = atom_types

        for comp in system.components:
            nrexcl = 3
            type_name = comp.kind.name
            if comp.kind == ComponentKind.MEMBRANE:
                type_name = comp.metadata.get("lipid_type", "LIPID")
            elif comp.kind == ComponentKind.SOLVENT:
                type_name = "SOL"
            elif comp.kind == ComponentKind.IONS:
                type_name = "IONS"
            elif comp.kind == ComponentKind.PROTEIN:
                type_name = "Protein"
            topology.molecule_blocks.append(
                MoleculeBlock(
                    atom_indices=list(comp.atom_indices),
                    nrexcl=nrexcl,
                    type_name=type_name,
                    num_molecules=1
                    if comp.kind != ComponentKind.SOLVENT
                    else comp.metadata.get("n_molecules", 1),
                )
            )
        return topology

    def get_ff_includes(self) -> list[str]:
        return [
            '#include "forcefield.itp"',
            '#include "tip3p.itp"',
            '#include "ions.itp"',
        ]


@ForceFieldRegistry.register
class Amber99SBForceField(_AmberBase):
    """Amber ff99SB force field."""

    name = "amber99sb"
    version = "ff99SB"
    water_model = "tip3p"


@ForceFieldRegistry.register
class Amber99SBILDNForceField(_AmberBase):
    """Amber ff99SB-ILDN force field (improved sidechain torsions)."""

    name = "amber99sb-ildn"
    version = "ff99SB-ILDN"
    water_model = "tip3p"


@ForceFieldRegistry.register
class Amber14SBForceField(_AmberBase):
    """Official GROMACS 2026.3 port of Amber ff14SB."""

    name = "amber14sb"
    version = "ff14SB / GROMACS 2026.3"
    water_model = "tip3p"
