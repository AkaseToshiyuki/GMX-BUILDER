"""System — the central data container that flows through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.topology import Topology
from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind


_CHECKPOINT_SCHEMA_VERSION = 2


def _unicode_array(values: list[str]) -> np.ndarray:
    """Return a pickle-free Unicode array sized to the actual data.

    NumPy fixed-width Unicode is safe to load with ``allow_pickle=False``, but
    a hard-coded width silently truncates longer GROMACS names.  Let NumPy
    derive the required width from the complete list instead.
    """
    return np.asarray([str(value) for value in values], dtype=np.str_)


@dataclass
class System:
    """Central data container representing a molecular system.

    All pipeline modules accept a System and return a ModuleResult
    containing a (possibly mutated) System.  The System owns the
    Structure, Topology, and a list of named Components that map
    atom-index ranges to semantic subsystems (protein, membrane, etc.).
    """

    structure: Structure
    topology: Topology | None = None
    components: list[Component] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_atoms(self) -> int:
        return self.structure.num_atoms

    @property
    def coordinates(self) -> np.ndarray:
        return self.structure.coordinates

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------

    def component_by_kind(self, kind: ComponentKind) -> list[Component]:
        return [c for c in self.components if c.kind == kind]

    def component_by_name(self, name: str) -> Component | None:
        for c in self.components:
            if c.name == name:
                return c
        return None

    def add_component(self, component: Component) -> None:
        self.components.append(component)

    # ------------------------------------------------------------------
    # Whole-system operations
    # ------------------------------------------------------------------

    def merge(self, other: System) -> System:
        """Return a new System created by appending *other*."""
        n = self.num_atoms
        merged_structure = self.structure.append(other.structure)

        merged_topology = None
        if self.topology and other.topology:
            # Topology.merge() mutates self in place — copy first
            merged_topology = self.topology.copy().merge(other.topology.copy())
        elif self.topology:
            merged_topology = self.topology.copy()
        elif other.topology:
            merged_topology = other.topology.copy()

        merged_components = list(self.components)
        for comp in other.components:
            shifted_indices = comp.atom_indices + n
            merged_components.append(Component(
                name=comp.name,
                kind=comp.kind,
                atom_indices=shifted_indices,
                metadata=dict(comp.metadata),
            ))

        merged_metadata = {**self.metadata, **other.metadata}

        return System(
            structure=merged_structure,
            topology=merged_topology,
            components=merged_components,
            metadata=merged_metadata,
        )

    def copy(self) -> System:
        import copy
        return copy.deepcopy(self)

    # Approximate per-residue charges (CHARMM convention at pH 7).
    # Shared with modules/ions/neutralize.py — single source of truth in core.
    _RESIDUE_CHARGES: ClassVar[dict[str, float]] = {
        "ARG": 1.0, "LYS": 1.0, "HIS": 0.0, "HID": 0.0, "HIE": 0.0, "HIP": 1.0,
        "HSD": 0.0, "HSE": 0.0, "HSP": 1.0,
        "ASP": -1.0, "GLU": -1.0, "ASH": 0.0, "GLH": 0.0,
        "CYM": -1.0, "LYN": 0.0,
        "ALA": 0.0, "ASN": 0.0, "CYS": 0.0, "CYX": 0.0, "GLN": 0.0,
        "GLY": 0.0, "ILE": 0.0, "LEU": 0.0, "MET": 0.0, "PHE": 0.0,
        "PRO": 0.0, "SER": 0.0, "THR": 0.0, "TRP": 0.0, "TYR": 0.0, "VAL": 0.0,
        "ACE": 0.0, "NME": 0.0, "NMA": 0.0, "NH2": 0.0,
        "SEP": -2.0, "TPO": -2.0, "PTR": -2.0,
        "S1P": -1.0, "T1P": -1.0, "Y1P": -1.0,
        "MSE": 0.0, "SEC": 0.0,
        "PYL": 0.0, "PCA": 0.0, "HYP": 0.0, "CIR": 0.0, "TYS": -1.0,
        "CSO": 0.0, "CSD": 0.0, "CSX": 0.0, "CSN": 0.0,
        "SAC": 0.0, "TAC": 0.0, "GCS": 0.0, "GCT": 0.0, "GPL": 0.0,
        "ALY": 0.0, "SLY": -1.0, "CLY": 0.0, "MLZ": 1.0, "MLY": 1.0,
        "M3L": 1.0, "KCX": -1.0,
        "CRY": 1.0, "BLY": 0.0, "PLY": 0.0, "GRY": -1.0,
        "KME": 1.0, "KM2": 1.0, "KM3": 1.0, "RME": 1.0, "RM2": 1.0,
        "2MR": 1.0, "DA2": 1.0, "SNC": 0.0, "SMC": 0.0, "OCS": -1.0,
        "OAS": 0.0, "NIY": 0.0, "SME": 0.0, "LYZ": 1.0,
        "MYR": 0.0, "PLC": 0.0, "WOH": 0.0, "FOR": 0.0, "UNK": 0.0,
    }

    def residue_formal_charge(self, resname: str) -> float:
        """Return the selected force field's formal charge for one residue.

        The legacy table remains a fallback for pre-force-field checkpoints,
        but force-field RTP charges are authoritative.  This matters for names
        such as SEP, whose bundled CHARMM36m state is monoanionic while the
        Amber14SB SEP state is dianionic.
        """
        name = str(resname).strip().upper()
        force_field = str(self.metadata.get("force_field", "")).strip().lower()
        if force_field:
            try:
                from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

                template = load_force_field_rtp(force_field).get_residue(name)
                if template is not None:
                    charge = sum(float(atom[2]) for atom in template["atoms"])
                    rounded = round(charge)
                    if abs(charge - rounded) <= 1e-3:
                        return float(rounded)
            except (FileNotFoundError, KeyError, ValueError):
                pass
        return float(self._RESIDUE_CHARGES.get(name, 0.0))

    def total_charge(self) -> float:
        """Net charge estimated from residue-level lookup.

        Per-atom partial charges are populated only at topology-write time
        (from .rtp data).  This method uses the same per-residue table as
        the neutralisation module so the ion budget stays consistent.

        Returns 0.0 when there are no components (empty system).
        """
        total = 0.0
        for comp in self.components:
            if comp.kind in (ComponentKind.SOLVENT, ComponentKind.IONS):
                continue
            if comp.kind == ComponentKind.PROTEIN:
                grouped: dict[str, list[tuple[tuple[str, int, str], list[int]]]] = {}
                lookup: dict[tuple[str, int, str], list[int]] = {}
                for raw_index in comp.atom_indices:
                    index = int(raw_index)
                    key = (
                        str(self.structure.resnames[index]).strip().upper(),
                        int(self.structure.resids[index]),
                        str(self.structure.chain_ids[index]),
                    )
                    if key not in lookup:
                        lookup[key] = []
                        grouped.setdefault(key[2], []).append((key, lookup[key]))
                    lookup[key].append(index)

                force_field = str(self.metadata.get("force_field", "")).strip().lower()
                for _chain, residues in grouped.items():
                    for position, (key, indices) in enumerate(residues):
                        resname = key[0]
                        charge = None
                        if force_field and len(residues) > 1:
                            end = "N" if position == 0 else (
                                "C" if position == len(residues) - 1 else ""
                            )
                            if (end == "N" and resname != "ACE") or (
                                end == "C" and resname not in {"NME", "NH2"}
                            ):
                                try:
                                    from gmxbuilder.modules.forcefield.rtp_parser import (
                                        get_terminal_residue,
                                    )

                                    _variant, template = get_terminal_residue(
                                        force_field, resname, end
                                    )
                                    observed = {
                                        str(self.structure.atom_names[index]).strip()
                                        for index in indices
                                    }
                                    expected = {str(atom[0]).strip() for atom in template["atoms"]}
                                    if observed == expected:
                                        value = sum(float(atom[2]) for atom in template["atoms"])
                                        if abs(value - round(value)) <= 1e-3:
                                            charge = float(round(value))
                                except (FileNotFoundError, KeyError, ValueError):
                                    pass
                        total += (
                            charge if charge is not None
                            else self.residue_formal_charge(resname)
                        )
            elif comp.kind == ComponentKind.MEMBRANE:
                from gmxbuilder.modules.membrane.lipids import LipidRegistry

                seen: set[tuple[str, int, str]] = set()
                for i in comp.atom_indices:
                    rn = str(self.structure.resnames[i]).strip().upper()
                    rid = int(self.structure.resids[i])
                    chain = self.structure.chain_ids[i] if i < len(self.structure.chain_ids) else ""
                    key = (rn, rid, chain)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        total += float(LipidRegistry.get(rn).charge)
                    except (KeyError, ValueError):
                        # Unknown membrane residues must not be assigned an
                        # invented formal charge here; force-field validation
                        # reports them separately.
                        continue
            elif comp.kind == ComponentKind.NUCLEIC_ACID:
                if not comp.metadata.get("prepared"):
                    raise ValueError(
                        "Nucleic-acid net charge is unavailable before native "
                        "polymer topology preparation"
                    )
                charge = comp.metadata.get("net_charge")
                if isinstance(charge, bool) or not isinstance(charge, (int, float)):
                    raise ValueError("Prepared nucleic-acid component lacks an exact net charge")
                rounded = round(float(charge))
                if abs(float(charge) - rounded) > 1e-3:
                    raise ValueError(
                        f"Nucleic-acid component has non-integral net charge {charge}"
                    )
                total += float(rounded)
            elif comp.kind == ComponentKind.LIGAND:
                molecule_charges = comp.metadata.get("molecule_charges")
                if isinstance(molecule_charges, dict):
                    counts: dict[str, int] = {}
                    seen: set[tuple[str, int, str]] = set()
                    for index in comp.atom_indices:
                        key = (
                            str(self.structure.resnames[index]).strip().upper(),
                            int(self.structure.resids[index]),
                            str(self.structure.chain_ids[index]),
                        )
                        if key not in seen:
                            seen.add(key)
                            counts[key[0]] = counts.get(key[0], 0) + 1
                    total += sum(
                        float(molecule_charges.get(name, 0)) * count
                        for name, count in counts.items()
                    )
                # Sum atom-level charges from topology if available and no
                # molecule-level charge contract was recorded.
                elif self.topology and comp.atom_indices:
                    for i in comp.atom_indices:
                        if i < len(self.topology.atom_types):
                            total += self.topology.atom_types[i].charge
        return total

    def list_component_kinds(self) -> list[ComponentKind]:
        return [c.kind for c in self.components]

    # ------------------------------------------------------------------
    # Checkpoint persistence — one authoritative coordinate state per step
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        """Persist the System to a checkpoint directory.

        Writes:
          - system.npz   — coordinates, box_vectors, per-atom metadata
          - system.json  — components, topology, misc metadata
        """
        import json

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        struct = self.structure

        # --- system.npz: all large numeric arrays ---
        npz_path = checkpoint_dir / "system.npz"
        arrays = {
            "coordinates": struct.coordinates,
            "box_vectors": struct.box_vectors,
        }
        # NumPy derives the Unicode width from the complete data.  Do not use a
        # hard-coded U4/U5 dtype here: checkpoints are authoritative and must
        # never silently alter atom, residue, chain, or segment identifiers.
        atom_names = _unicode_array(struct.atom_names)
        resnames = _unicode_array(struct.resnames)
        resids = np.array(struct.resids, dtype=np.int32)
        arrays["atom_names"] = atom_names
        arrays["resnames"] = resnames
        arrays["resids"] = resids
        if struct.chain_ids:
            arrays["chain_ids"] = _unicode_array(struct.chain_ids)
        if struct.elements:
            arrays["elements"] = _unicode_array(struct.elements)
        if struct.segids:
            arrays["segids"] = _unicode_array(struct.segids)
        if struct.occupancies:
            arrays["occupancies"] = np.array(struct.occupancies, dtype=np.float64)
        if struct.tempfactors:
            arrays["tempfactors"] = np.array(struct.tempfactors, dtype=np.float64)
        np.savez_compressed(npz_path, **arrays)

        # --- system.json: components + topology + metadata ---
        json_path = checkpoint_dir / "system.json"
        comps = []
        for c in self.components:
            comps.append({
                "name": c.name,
                "kind": c.kind.value,
                "atom_indices": c.atom_indices.tolist(),
                "metadata": c.metadata,
            })

        topo = None
        if self.topology:
            topo = {
                "atom_types": [
                    {"name": t.name, "mass": t.mass, "charge": t.charge,
                     "sigma": t.sigma, "epsilon": t.epsilon, "atom_class": t.atom_class}
                    for t in self.topology.atom_types
                ],
                "bonds": [
                    {"i": b.i, "j": b.j, "funct": b.funct, "r0": b.r0, "k_b": b.k_b}
                    for b in self.topology.bonds
                ],
                "angles": [
                    {"i": a.i, "j": a.j, "k": a.k, "funct": a.funct,
                     "theta0": a.theta0, "k_theta": a.k_theta}
                    for a in self.topology.angles
                ],
                "dihedrals": [
                    {"i": d.i, "j": d.j, "k": d.k, "l": d.l, "funct": d.funct,
                     "phi": d.phi, "k_psi": d.k_psi, "multiplicity": d.multiplicity}
                    for d in self.topology.dihedrals
                ],
                "impropers": [
                    {"i": d.i, "j": d.j, "k": d.k, "l": d.l, "funct": d.funct,
                     "phi0": d.phi0, "k_psi": d.k_psi}
                    for d in self.topology.impropers
                ],
                "molecule_blocks": [
                    {"atom_indices": list(mb.atom_indices), "nrexcl": mb.nrexcl,
                     "type_name": mb.type_name, "num_molecules": mb.num_molecules}
                    for mb in self.topology.molecule_blocks
                ],
                "force_field": self.topology.force_field,
            }

        data = {
            "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "num_atoms": struct.num_atoms,
            "components": comps,
            "metadata": self.metadata,
        }
        if topo is not None:
            data["topology"] = topo

        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: Path) -> System:
        """Load a System from a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory containing system.npz and system.json.

        Returns
        -------
        System

        Raises
        ------
        FileNotFoundError
            If the checkpoint directory or required files are missing.
        ValueError
            If required arrays or keys are missing from the checkpoint files.
        """
        import json
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)

        # --- Validate checkpoint directory ---
        npz_path = checkpoint_dir / "system.npz"
        json_path = checkpoint_dir / "system.json"

        missing = []
        if not npz_path.exists():
            missing.append(str(npz_path))
        if not json_path.exists():
            missing.append(str(json_path))
        if missing:
            raise FileNotFoundError(
                f"Checkpoint files not found: {', '.join(missing)}"
            )

        # --- Validate .npz required arrays ---
        arrays = np.load(npz_path, allow_pickle=False)
        _REQUIRED_NPZ_KEYS = {"coordinates", "box_vectors", "atom_names",
                              "resnames", "resids"}
        missing_keys = _REQUIRED_NPZ_KEYS - set(arrays.keys())
        if missing_keys:
            raise ValueError(
                f"Checkpoint {npz_path} is missing required arrays: "
                f"{', '.join(sorted(missing_keys))}"
            )

        # --- Validate .json required keys ---
        with open(json_path) as fh:
            data = json.load(fh)
        _REQUIRED_JSON_KEYS = {"num_atoms", "components"}
        missing_keys = _REQUIRED_JSON_KEYS - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"Checkpoint {json_path} is missing required keys: "
                f"{', '.join(sorted(missing_keys))}"
            )

        # v1 checkpoints used fixed-width U4 arrays.  A long selected lipid
        # whose four-character prefix is present cannot be reconstructed
        # safely: guessing would risk attaching the wrong topology.  Four-
        # character systems such as POPC remain backward compatible.
        schema_version = int(data.get("checkpoint_schema_version", 1))
        if schema_version < _CHECKPOINT_SCHEMA_VERSION:
            selected = data.get("metadata", {}).get("selected_lipid_names", [])
            observed = {
                str(name).strip().upper()
                for name in arrays["resnames"].tolist()
            }
            truncated = sorted({
                str(name).strip().upper()
                for name in selected
                if len(str(name).strip()) > 4
                and str(name).strip()[:4].upper() in observed
            })
            if truncated:
                raise ValueError(
                    "legacy checkpoint contains detectably truncated lipid "
                    f"name(s): {', '.join(truncated)}; re-run Membrane Check "
                    "from the previous valid step"
                )

        # --- Load numpy arrays (validated above) ---
        chain_arr = arrays.get("chain_ids")
        elem_arr = arrays.get("elements")
        segid_arr = arrays.get("segids")
        occupancy_arr = arrays.get("occupancies")
        tempfactor_arr = arrays.get("tempfactors")
        struct = Structure(
            coordinates=arrays["coordinates"],
            box_vectors=arrays["box_vectors"],
            atom_names=arrays["atom_names"].tolist(),
            resnames=arrays["resnames"].tolist(),
            resids=arrays["resids"].tolist(),
            chain_ids=chain_arr.tolist() if chain_arr is not None else None,
            elements=elem_arr.tolist() if elem_arr is not None else None,
            segids=segid_arr.tolist() if segid_arr is not None else None,
            occupancies=occupancy_arr.tolist() if occupancy_arr is not None else None,
            tempfactors=tempfactor_arr.tolist() if tempfactor_arr is not None else None,
        )
        # Ensure we don't hold numpy strings in the lists
        if struct.chain_ids is not None:
            struct.chain_ids = [str(c) for c in struct.chain_ids]
        if struct.elements is not None:
            struct.elements = [str(e) for e in struct.elements]
        if struct.segids is not None:
            struct.segids = [str(s) for s in struct.segids]
        struct.atom_names = [str(a) for a in struct.atom_names]
        struct.resnames = [str(r) for r in struct.resnames]
        struct.resids = [int(r) for r in struct.resids]

        # Components
        components = []
        for cd in data.get("components", []):
            components.append(Component(
                name=cd["name"],
                kind=ComponentKind(cd["kind"]),
                atom_indices=np.array(cd["atom_indices"], dtype=np.int64),
                metadata=cd.get("metadata", {}),
            ))

        # Topology
        topology = None
        topo_data = data.get("topology")
        if topo_data:
            from gmxbuilder.core.topology import (
                Topology, AtomType, Bond, Angle, Dihedral, Improper, MoleculeBlock,
            )
            topology = Topology(force_field=topo_data.get("force_field", ""))
            for td in topo_data.get("atom_types", []):
                topology.atom_types.append(AtomType(
                    name=td["name"], mass=td["mass"], charge=td["charge"],
                    sigma=td["sigma"], epsilon=td["epsilon"],
                    atom_class=td.get("atom_class", ""),
                ))
            for bd in topo_data.get("bonds", []):
                topology.bonds.append(Bond(
                    i=bd["i"], j=bd["j"], funct=bd.get("funct", 1),
                    r0=bd.get("r0"), k_b=bd.get("k_b"),
                ))
            for ad in topo_data.get("angles", []):
                topology.angles.append(Angle(
                    i=ad["i"], j=ad["j"], k=ad["k"], funct=ad.get("funct", 1),
                    theta0=ad.get("theta0"), k_theta=ad.get("k_theta"),
                ))
            for dd in topo_data.get("dihedrals", []):
                topology.dihedrals.append(Dihedral(
                    i=dd["i"], j=dd["j"], k=dd["k"], l=dd["l"],
                    funct=dd.get("funct", 9), phi=dd.get("phi"),
                    k_psi=dd.get("k_psi"), multiplicity=dd.get("multiplicity"),
                ))
            for id_ in topo_data.get("impropers", []):
                topology.impropers.append(Improper(
                    i=id_["i"], j=id_["j"], k=id_["k"], l=id_["l"],
                    funct=id_.get("funct", 2), phi0=id_.get("phi0"),
                    k_psi=id_.get("k_psi"),
                ))
            for mb in topo_data.get("molecule_blocks", []):
                topology.molecule_blocks.append(MoleculeBlock(
                    atom_indices=list(mb["atom_indices"]), nrexcl=mb.get("nrexcl", 3),
                    type_name=mb["type_name"], num_molecules=mb.get("num_molecules", 1),
                ))

        return cls(
            structure=struct,
            topology=topology,
            components=components,
            metadata=data.get("metadata", {}),
        )

    def write_viewer_pdb(self, path: Path) -> None:
        """Write a viewer-friendly PDB of the current system."""
        from gmxbuilder.io.pdb import PDBWriter
        PDBWriter.write(
            self.structure,
            path,
            title="GMXBUILDER Step Viewer",
            wrap_ids_for_viewer=True,
        )
