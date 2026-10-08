"""Amber force field implementations."""

from __future__ import annotations

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import MoleculeBlock, Topology
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry


class _AmberBase(ForceField):
    """Shared topology builder for Amber-family force fields."""

    water_model = "tip3p"
    supported_lipids = ["POPC", "DPPC", "DOPE", "POPG"]

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


@ForceFieldRegistry.register
class Amber14SBOL24ForceField(_AmberBase):
    """ff14SB with the Olomouc OL24 DNA and chiOL3 RNA parameters merged in.

    Assembled by ``install-local.sh`` rather than shipped, so the directory is
    absent on a fresh clone. It is registered regardless: the interface has to
    be able to say the force field exists and is not installed, which it
    cannot do for something it has never heard of.
    """

    name = "amber14sb_ol24"
    version = "ff14SB + OL24 DNA / chiOL3 RNA"
    water_model = "tip3p"
