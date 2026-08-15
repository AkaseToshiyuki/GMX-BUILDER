"""OPLS-AA force field implementation."""

from __future__ import annotations

from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Topology, AtomType, MoleculeBlock
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry

# OPLS-AA LJ parameters (sigma in nm, epsilon in kJ/mol)
_OPLS_DEFAULTS: dict[str, dict] = {
    "C": {"mass": 12.011, "sigma": 0.350000, "epsilon": 0.27614},
    "N": {"mass": 14.007, "sigma": 0.325000, "epsilon": 0.71128},
    "O": {"mass": 15.999, "sigma": 0.296000, "epsilon": 0.87864},
    "H": {"mass": 1.008, "sigma": 0.250000, "epsilon": 0.12552},
    "S": {"mass": 32.065, "sigma": 0.355000, "epsilon": 1.04600},
    "P": {"mass": 30.974, "sigma": 0.374177, "epsilon": 0.83680},
    "NA": {"mass": 22.990, "sigma": 0.333000, "epsilon": 0.01160},
    "CL": {"mass": 35.453, "sigma": 0.440000, "epsilon": 0.41840},
    "K": {"mass": 39.098, "sigma": 0.373000, "epsilon": 0.00377},
    "CA": {"mass": 40.078, "sigma": 0.302000, "epsilon": 0.23849},
    "ZN": {"mass": 65.380, "sigma": 0.196000, "epsilon": 0.52300},
    "MG": {"mass": 24.305, "sigma": 0.164447, "epsilon": 0.36903},
    "F": {"mass": 18.998, "sigma": 0.312000, "epsilon": 0.25522},
    "BR": {"mass": 79.904, "sigma": 0.462000, "epsilon": 0.37656},
    "I": {"mass": 126.90, "sigma": 0.540000, "epsilon": 0.41840},
}


@ForceFieldRegistry.register
class OPLSAAForceField(ForceField):
    """Legacy OPLS-AA/L (2001) all-atom force field."""

    name = "oplsaa"
    version = "OPLS-AA/L (2001)"
    water_model = "tip4p"
    supported_lipids = []

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
            params = _OPLS_DEFAULTS.get(elem, _OPLS_DEFAULTS["C"])

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
