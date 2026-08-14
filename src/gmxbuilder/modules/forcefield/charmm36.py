"""CHARMM36 force field implementation.

Provides topology assignment for CHARMM36 force field parameters.
Full implementation (Phase 4) will include residue template parsing
and atom type mapping.
"""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.system import System
from gmxbuilder.core.topology import Topology, AtomType, MoleculeBlock
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ForceFieldError
from gmxbuilder.modules.forcefield.base_ff import ForceField
from gmxbuilder.modules.forcefield.registry import ForceFieldRegistry


# Approximate CHARMM36 atom type parameters for common elements
# Used as fallback when no .rtp file is available
_CHARMM36_DEFAULTS: dict[str, dict] = {
    "C":  {"mass": 12.011, "sigma": 0.356359, "epsilon": 0.46024},
    "N":  {"mass": 14.007, "sigma": 0.329632, "epsilon": 0.83680},
    "O":  {"mass": 15.999, "sigma": 0.302905, "epsilon": 0.50208},
    "H":  {"mass": 1.008,  "sigma": 0.040001, "epsilon": 0.19246},
    "S":  {"mass": 32.065, "sigma": 0.356359, "epsilon": 1.04600},
    "P":  {"mass": 30.974, "sigma": 0.374177, "epsilon": 0.83680},
    "NA": {"mass": 22.990, "sigma": 0.242992, "epsilon": 0.19623},
    "CL": {"mass": 35.453, "sigma": 0.404468, "epsilon": 0.62760},
    "K":  {"mass": 39.098, "sigma": 0.314264, "epsilon": 0.36468},
    "CA": {"mass": 40.078, "sigma": 0.241199, "epsilon": 0.25620},
    "ZN": {"mass": 65.380, "sigma": 0.195998, "epsilon": 0.52300},
    "MG": {"mass": 24.305, "sigma": 0.141445, "epsilon": 0.10836},
}


@ForceFieldRegistry.register
class CHARMM36ForceField(ForceField):
    """CHARMM36 all-atom force field (Mar2019)."""

    name = "charmm36"
    version = "mar2019"
    water_model = "tip3p"
    supported_lipids = ["POPC", "DPPC", "POPE", "DOPE", "POPG", "POPS"]

    def build_system_topology(self, system: System) -> Topology:
        topology = Topology(force_field=self.name)

        # Load the RTP files for this exact force field.  CHARMM36m must not
        # silently receive atom types and charges from CHARMM36's singleton.
        from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
        rtp = load_force_field_rtp(self.name)

        n_resnames = len(system.structure.resnames)
        n_atom_names = len(system.structure.atom_names)

        # Generic fallback atom types by element (when RTP lookup fails)
        _ELEM_GENERIC_TYPE: dict[str, str] = {
            "C": "CT3", "N": "NH1", "O": "O", "H": "H",
            "S": "S", "P": "P", "NA": "NA", "CL": "CL",
            "K": "K", "CA": "CA", "ZN": "ZN", "MG": "MG",
        }

        atom_types = []
        for i in range(system.num_atoms):
            elem = system.structure.elements[i] if i < len(system.structure.elements) else "C"
            elem = elem.upper()
            params = _CHARMM36_DEFAULTS.get(elem, _CHARMM36_DEFAULTS["C"])

            # Look up residue-specific atom type and charge from RTP
            rn = system.structure.resnames[i] if i < n_resnames else "ALA"
            an = system.structure.atom_names[i] if i < n_atom_names else "CA"
            atype_name = _ELEM_GENERIC_TYPE.get(elem, "CT3")
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
                            topology.molecule_blocks.append(MoleculeBlock(
                                atom_indices=lipid_indices,
                                nrexcl=nrexcl,
                                type_name=residue_names.pop(),
                                num_molecules=1,
                            ))
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

            topology.molecule_blocks.append(MoleculeBlock(
                atom_indices=list(comp.atom_indices),
                nrexcl=nrexcl,
                type_name=type_name,
                num_molecules=n_mol,
            ))

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
    supported_lipids = ["POPC", "DPPC", "DMPC", "DOPC", "POPE", "DOPE",
                        "POPG", "POPS", "POPA", "CHOL"]
