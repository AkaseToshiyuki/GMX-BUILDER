"""OPLS-AA force field implementation."""

from __future__ import annotations

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import MoleculeBlock, Topology
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry


@ForceFieldRegistry.register
class OPLSAAForceField(ForceField):
    """Legacy OPLS-AA/L (2001) all-atom force field."""

    name = "oplsaa"
    version = "OPLS-AA/L (2001)"
    water_model = "tip4p"
    supported_lipids = []

    def build_system_topology(self, system: System) -> Topology:
        topology = Topology(force_field=self.name)
        from gmxbuilder.modules.forcefield.protein_templates import native_atom_types

        topology.atom_types = native_atom_types(system, self.name)

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
        from gmxbuilder.modules.forcefield.protein_templates import assign_protein_atoms

        assign_protein_atoms(system, topology, self.name)
        return topology

    def get_ff_includes(self) -> list[str]:
        return [
            '#include "forcefield.itp"',
            '#include "tip3p.itp"',
            '#include "ions.itp"',
        ]
