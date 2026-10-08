"""CHARMM36 force field implementation.

Provides topology assignment for CHARMM36 force field parameters.
Full implementation (Phase 4) will include residue template parsing
and atom type mapping.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ForceFieldError
from gmxbuilder.core.system import System
from gmxbuilder.core.topology import MoleculeBlock, Topology
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry


@ForceFieldRegistry.register
class CHARMM36ForceField(ForceField):
    """CHARMM36 all-atom force field (Mar2019)."""

    name = "charmm36"
    version = "mar2019"
    water_model = "tip3p"
    supported_lipids = ["POPC", "DPPC", "POPE", "DOPE", "POPG", "POPS"]

    def build_system_topology(self, system: System) -> Topology:
        topology = Topology(force_field=self.name)
        from gmxbuilder.modules.forcefield.protein_templates import native_atom_types

        topology.atom_types = native_atom_types(system, self.name)

        # Build molecule blocks for each component
        for comp in system.components:
            nrexcl = 3
            type_name = comp.kind.name
            n_mol = 1

            if comp.kind == ComponentKind.MEMBRANE:
                # Split into individual lipid molecules using per-lipid atom counts
                # (supports mixed-size lipids like POPC+CHOL)
                n_upper = comp.metadata.get("n_lipids_upper", 0)
                n_lower = comp.metadata.get("n_lipids_lower", 0)
                n_lipids = n_upper + n_lower
                lipid_sizes = comp.metadata.get("lipid_sizes")
                if n_lipids > 0 and len(comp.atom_indices) > 0:
                    if lipid_sizes and len(lipid_sizes) == n_lipids:
                        # Use per-lipid sizes for mixed compositions
                        offsets = np.cumsum([0] + list(lipid_sizes))
                    else:
                        # Fallback: uniform size
                        atoms_per_lipid = len(comp.atom_indices) // n_lipids
                        offsets = np.array([i * atoms_per_lipid for i in range(n_lipids + 1)])
                    if offsets[-1] > 0:
                        for li in range(n_lipids):
                            start = offsets[li]
                            end = offsets[li + 1]
                            lipid_indices = list(comp.atom_indices[start:end])
                            residue_names = {
                                system.structure.resnames[int(index)].strip().upper()
                                for index in lipid_indices
                            }
                            if len(residue_names) != 1:
                                raise ForceFieldError(
                                    "A membrane molecule block contains mixed residue names"
                                )
                            topology.molecule_blocks.append(
                                MoleculeBlock(
                                    atom_indices=lipid_indices,
                                    nrexcl=nrexcl,
                                    type_name=residue_names.pop(),
                                    num_molecules=1,
                                )
                            )
                        continue  # skip default block creation
                # Fallback: single block if splitting fails
                n_mol = 1
            elif comp.kind == ComponentKind.SOLVENT:
                type_name = "SOL"
                n_mol = comp.metadata.get("n_molecules", 1)
            elif comp.kind == ComponentKind.IONS:
                type_name = "IONS"
            elif comp.kind == ComponentKind.PROTEIN:
                type_name = "Protein"

            topology.molecule_blocks.append(
                MoleculeBlock(
                    atom_indices=list(comp.atom_indices),
                    nrexcl=nrexcl,
                    type_name=type_name,
                    num_molecules=n_mol,
                )
            )

        from gmxbuilder.modules.forcefield.protein_templates import assign_protein_atoms

        assign_protein_atoms(system, topology, self.name)
        return topology

    def get_ff_includes(self) -> list[str]:
        return [
            '#include "charmm36.ff/forcefield.itp"',
            '#include "charmm36.ff/ions.itp"',
            '#include "charmm36.ff/tip3p.itp"',
        ]


@ForceFieldRegistry.register
class CHARMM36mForceField(CHARMM36ForceField):
    """CHARMM36m (Jul2022) — 2428 residues across 63 split RTP files.

    Extended coverage: carbohydrates, lipids, CGenFF, metals, nucleic
    acids, ethers, silicates, solvents.  Shares the same topology
    builder and FF include logic as CHARMM36.
    """

    name = "charmm36m"
    version = "jul2022"
    water_model = "tip3p"
    supported_lipids = [
        "POPC",
        "DPPC",
        "DMPC",
        "DOPC",
        "POPE",
        "DOPE",
        "POPG",
        "POPS",
        "POPA",
        "CHOL",
    ]
