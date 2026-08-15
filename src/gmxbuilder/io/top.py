"""GROMACS topology (.top, .itp) writer — simulation-ready output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmxbuilder.core.topology import Topology
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.exceptions import TopologyError


class TopologyWriter:
    """Write GROMACS .top and .itp files with proper force field parameters."""

    def __init__(self, force_field: str = "amber14sb", ff_config: dict | None = None):
        self.force_field = force_field
        self.ff_config = ff_config or {}
        # Path to bundled force field data — try with and without .ff suffix
        base = Path(__file__).resolve().parent.parent / "data" / "forcefields"
        for cand in [base / force_field, base / f"{force_field}.ff"]:
            if cand.is_dir():
                self._ff_path = cand
                break
        else:
            self._ff_path = base / force_field  # fallback (may not exist)
        self._rtp_loaded = False

    def _bonded_function_types(self) -> tuple[int, int, int]:
        """Return RTP default (angle, proper, improper) function numbers."""
        if self.force_field in {"charmm36", "charmm36m"}:
            return 5, 9, 2
        if self.force_field in {"amber14sb", "amber99sb", "amber99sb-ildn"}:
            return 1, 9, 4
        if self.force_field == "oplsaa":
            return 1, 3, 1
        raise TopologyError(f"No bonded-function mapping for force field {self.force_field!r}")

    @staticmethod
    def _generate_graph_terms(
        bonds: list[tuple[int, int]],
    ) -> tuple[
        list[tuple[int, int, int]],
        list[tuple[int, int, int, int]],
        list[tuple[int, int]],
    ]:
        """Generate angles, proper dihedrals and 1-4 pairs from a bond graph."""
        canonical_bonds = sorted({tuple(sorted((i, j))) for i, j in bonds if i != j})
        adjacency: dict[int, set[int]] = {}
        for i, j in canonical_bonds:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)

        angles: set[tuple[int, int, int]] = set()
        for center, neighbours in adjacency.items():
            ordered = sorted(neighbours)
            for left_index in range(len(ordered)):
                for right_index in range(left_index + 1, len(ordered)):
                    angles.add((ordered[left_index], center, ordered[right_index]))

        dihedrals: set[tuple[int, int, int, int]] = set()
        for middle_left, middle_right in canonical_bonds:
            for outer_left in adjacency[middle_left] - {middle_right}:
                for outer_right in adjacency[middle_right] - {middle_left}:
                    if outer_left == outer_right:
                        continue
                    term = (outer_left, middle_left, middle_right, outer_right)
                    reverse = tuple(reversed(term))
                    dihedrals.add(min(term, reverse))

        bonded = set(canonical_bonds)
        angle_endpoints = {tuple(sorted((left, right))) for left, _center, right in angles}
        pairs = sorted(
            {
                tuple(sorted((outer_left, outer_right)))
                for outer_left, _middle_left, _middle_right, outer_right in dihedrals
                if tuple(sorted((outer_left, outer_right))) not in bonded
                and tuple(sorted((outer_left, outer_right))) not in angle_endpoints
            }
        )
        return sorted(angles), sorted(dihedrals), pairs

    @staticmethod
    def _ordered_residue_runs(
        structure: Structure, residue_names: set[str]
    ) -> list[tuple[str, int]]:
        """Return run-length encoded molecule types in coordinate order."""
        molecule_types: list[str] = []
        previous_key: tuple[str, int] | None = None
        for resname, resid in zip(structure.resnames, structure.resids):
            key = (str(resname), int(resid))
            if key[0] not in residue_names:
                previous_key = None
                continue
            if key != previous_key:
                molecule_types.append(key[0])
                previous_key = key

        runs: list[tuple[str, int]] = []
        for molecule_type in molecule_types:
            if runs and runs[-1][0] == molecule_type:
                runs[-1] = (molecule_type, runs[-1][1] + 1)
            else:
                runs.append((molecule_type, 1))
        return runs

    @staticmethod
    def _ordered_residue_run_records(
        structure: Structure,
        residue_names: set[str],
        *,
        excluded_indices: set[int] | None = None,
    ) -> list[tuple[int, str, int]]:
        """Return ``(first_atom, molecule_type, count)`` coordinate-order runs."""
        excluded = excluded_indices or set()
        runs: list[tuple[int, str, int]] = []
        previous_key: tuple[str, int, str] | None = None
        contiguous_block = False
        for index, (resname, resid) in enumerate(zip(structure.resnames, structure.resids)):
            if index in excluded or str(resname) not in residue_names:
                previous_key = None
                contiguous_block = False
                continue
            chain = str(structure.chain_ids[index]) if index < len(structure.chain_ids) else ""
            key = (str(resname), int(resid), chain)
            if key != previous_key:
                if contiguous_block and runs and runs[-1][1] == key[0]:
                    first, molecule_type, count = runs[-1]
                    runs[-1] = (first, molecule_type, count + 1)
                else:
                    runs.append((index, key[0], 1))
                previous_key = key
                contiguous_block = True
        return runs

    # ------------------------------------------------------------------
    # Master .top file
    # ------------------------------------------------------------------

    def write_top(
        self,
        structure: Structure,
        path: str | Path,
        system_name: str = "system",
        molecule_counts: dict[str, int] | None = None,
        topology: Topology | None = None,
    ) -> None:
        """Write a complete system topology (.top) with per-chain ITPs.

        Parameters
        ----------
        topology : Topology | None
            Optional pre-built Topology from force field module.
            When provided, its atom types/charges supplement the RTP data.
        """
        # Flat directory layout — all files in output root.  GROMACS
        # resolves #include relative to the working directory, and
        # forcefield.itp's own includes relative to its own directory.
        # Putting everything in one directory eliminates path issues.
        top_dir = path.parent
        self._copy_force_field(top_dir)
        water_model_name = str(self.ff_config.get("water_model", "tip3p")).lower()
        try:
            from gmxbuilder.modules.solvation.water_models import WaterRegistry

            water_model = WaterRegistry.get(water_model_name)
        except KeyError as exc:
            raise TopologyError(str(exc)) from exc
        water_itp = f"{water_model_name}.itp"
        if not (top_dir / water_itp).is_file():
            raise TopologyError(
                f"Water model {water_model_name!r} is unavailable for "
                f"force field {self.force_field!r}"
            )
        # New Amber ports bundle ion parameters per water model (for example
        # ions_tip3p.itp).  Falling through to the deprecated GROMACS-level
        # ions.itp produces a hard #error in GROMACS 2026.
        water_ion_itp = f"ions_{water_model_name}.itp"
        ion_itp = water_ion_itp if (top_dir / water_ion_itp).is_file() else "ions.itp"
        if not (top_dir / ion_itp).is_file():
            raise TopologyError(
                f"Ion parameters compatible with water model {water_model_name!r} "
                f"are unavailable for force field {self.force_field!r}"
            )

        # Collect unique residue names and their counts
        resnames = structure.resnames
        res_counts: dict[str, int] = {}
        for rn in resnames:
            res_counts[rn] = res_counts.get(rn, 0) + 1

        native_nucleic = self.ff_config.get("native_nucleic_topologies", []) or []
        if not isinstance(native_nucleic, list):
            raise TopologyError("Native nucleic-acid topology metadata must be a list")
        nucleic_indices: set[int] = set()
        for record in native_nucleic:
            if not isinstance(record, dict):
                raise TopologyError("Invalid native nucleic-acid topology record")
            indices = record.get("atom_indices", [])
            if not isinstance(indices, list) or not indices:
                raise TopologyError("Native nucleic-acid topology has no atom indices")
            parsed = {int(index) for index in indices}
            if min(parsed) < 0 or max(parsed) >= structure.num_atoms:
                raise TopologyError("Native nucleic-acid atom index is out of range")
            if nucleic_indices & parsed:
                raise TopologyError("Native nucleic-acid topology atom ranges overlap")
            if int(record.get("atom_count", -1)) != len(parsed):
                raise TopologyError("Native nucleic-acid atom count does not match coordinates")
            nucleic_indices.update(parsed)

        # Determine component groups
        from gmxbuilder.io.pdb import _PROTEIN_RESNAMES
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        _PROTEIN_SET = set(_PROTEIN_RESNAMES)

        protein_res = {rn for rn in res_counts if rn in _PROTEIN_SET}
        water_res = {rn for rn in res_counts if rn in ("SOL", "HOH", "TIP3", "WAT")}
        from gmxbuilder.modules.ions.catalog import KNOWN_IONS, molecule_types_in_itp

        ion_res = {rn for rn in res_counts if rn in KNOWN_IONS}
        defined_ions = molecule_types_in_itp(top_dir / ion_itp)
        unsupported_ions = sorted(ion_res - defined_ions)
        if unsupported_ions:
            raise TopologyError(
                f"Ion(s) {', '.join(unsupported_ions)} are not defined by "
                f"{ion_itp} for force field {self.force_field}"
            )
        registered_lipids = set(LipidRegistry.list())
        lipid_res = {rn for rn in res_counts if rn in registered_lipids}
        ligand_res = {
            rn
            for rn in res_counts
            if rn not in _PROTEIN_SET
            and rn not in water_res
            and rn not in ion_res
            and rn not in lipid_res
            and not any(
                index in nucleic_indices and structure.resnames[index] == rn
                for index in range(structure.num_atoms)
            )
        }
        molecule_res = lipid_res | ligand_res

        # ---- Discover protein chains ----
        protein_chains = self._get_protein_chains(structure, protein_res, topology)

        external_atomtype_includes: list[str] = []
        external_molecule_types: dict[str, str] = {}
        if lipid_res and self.force_field.lower().startswith("amber"):
            selected_lipid_ff = str(self.ff_config.get("lipid_ff", "gaff2")).lower()
            if selected_lipid_ff == "lipid21":
                from gmxbuilder.modules.forcefield.lipid21_backend import (
                    lipid21_atomtypes_path,
                )

                filename = "lipid21_atomtypes.itp"
                (top_dir / filename).write_text(lipid21_atomtypes_path().read_text())
                external_atomtype_includes.append(filename)
            elif selected_lipid_ff == "gaff2":
                from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
                from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp
                from gmxbuilder.modules.membrane.lipids import LipidRegistry

                rtp = load_force_field_rtp(self.force_field)
                for lipid_name in sorted(lipid_res):
                    if rtp.get_residue(lipid_name) is not None:
                        continue
                    lipid = LipidRegistry.get(lipid_name)
                    template = prepare_gaff_lipid(lipid_name, lipid.smiles, lipid.charge)
                    external_molecule_types[lipid_name] = template.name
                    filename = f"{lipid_name}_atomtypes.itp"
                    (top_dir / filename).write_text(template.atomtypes_path.read_text())
                    external_atomtype_includes.append(filename)
            else:
                raise TopologyError(f"Unsupported Amber lipid backend {selected_lipid_ff!r}")

        ligand_parameters = self.ff_config.get("ligand_parameters", {}) or {}
        for ligand_name in sorted(ligand_res):
            params = ligand_parameters.get(ligand_name, {})
            if params.get("source") not in {"gaff2", "cgenff"}:
                continue
            itp_path = Path(str(params.get("itp_path", "")))
            atomtypes_path = Path(str(params.get("atomtypes_path", "")))
            if not (itp_path.is_file() and atomtypes_path.is_file()):
                raise TopologyError(
                    f"External parameter files are missing for ligand {ligand_name}"
                )
            molecule_type = str(params.get("molecule_type") or ligand_name)
            external_molecule_types[ligand_name] = molecule_type
            atomtypes_name = f"{ligand_name}_atomtypes.itp"
            (top_dir / atomtypes_name).write_text(atomtypes_path.read_text())
            external_atomtype_includes.append(atomtypes_name)

        with open(path, "w") as fh:
            fh.write("; GMXBUILDER — simulation-ready topology\n;\n")

            # Force field includes (flat directory — no toppar/ prefix)
            fh.write('#include "forcefield.itp"\n')
            for filename in external_atomtype_includes:
                fh.write(f'#include "{filename}"\n')
            fh.write(f'#include "{water_itp}"\n')
            fh.write(f'#include "{ion_itp}"\n')

            # Per-chain protein ITPs
            for chain_id, chain_indices in protein_chains:
                moltype = f"Protein_chain_{chain_id}"
                chain_itp = top_dir / f"topol_{moltype}.itp"
                self._write_protein_itp_for_indices(
                    structure, chain_itp, chain_indices, moltype, topology=topology
                )
                fh.write(f'#include "topol_{moltype}.itp"\n')

            for record in native_nucleic:
                filename = str(record.get("itp_filename", ""))
                posre_filename = str(record.get("posre_filename", ""))
                if not filename or Path(filename).name != filename:
                    raise TopologyError("Invalid native nucleic-acid ITP filename")
                if not posre_filename or Path(posre_filename).name != posre_filename:
                    raise TopologyError("Invalid native nucleic-acid restraint filename")
                itp_text = str(record.get("itp_text", ""))
                posre_text = str(record.get("posre_text", ""))
                if "[ moleculetype ]" not in itp_text or "[ atoms ]" not in itp_text:
                    raise TopologyError("Native nucleic-acid ITP content is incomplete")
                if "[ position_restraints ]" not in posre_text:
                    raise TopologyError("Native nucleic-acid restraints are incomplete")
                (top_dir / filename).write_text(itp_text)
                (top_dir / posre_filename).write_text(posre_text)
                fh.write(f'#include "{filename}"\n')

            for lip in sorted(lipid_res):
                lip_itp = top_dir / f"{lip}.itp"
                self._write_lipid_itp(lip, structure, lip_itp)
                fh.write(f'#include "{lip}.itp"\n')

            for ligand in sorted(ligand_res):
                ligand_itp = top_dir / f"{ligand}.itp"
                params = ligand_parameters.get(ligand, {})
                if params.get("source") in {"gaff2", "cgenff"}:
                    source = Path(str(params.get("itp_path", "")))
                    ligand_itp.write_text(source.read_text())
                else:
                    self._write_lipid_itp(
                        ligand,
                        structure,
                        ligand_itp,
                        apply_lipid_restraints=False,
                    )
                fh.write(f'#include "{ligand}.itp"\n')

            # Water topology comes from force field (tip3p.itp / spc.itp etc.)
            # — no separate SOL.itp needed; GROMACS reads it from the FF.

            # System name
            fh.write(f"\n[ system ]\n{system_name}\n\n")

            # Molecules in exact coordinate order.  This supports PDB files in
            # which DNA/RNA precedes protein, rather than relying on a fixed
            # protein-first assumption.
            fh.write("[ molecules ]\n")
            molecule_records: list[tuple[int, str, int]] = []
            for chain_id, chain_indices in protein_chains:
                molecule_records.append((min(chain_indices), f"Protein_chain_{chain_id}", 1))

            for record in native_nucleic:
                indices = [int(index) for index in record["atom_indices"]]
                molecule_records.append((min(indices), str(record["molecule_type"]), 1))

            # GROMACS consumes coordinates in [molecules] order.  Mixed
            # leaflets are spatially shuffled, so grouping all POPC then all
            # CHOL would assign coordinates to the wrong molecule templates.
            for first, molecule_name, count in self._ordered_residue_run_records(
                structure, molecule_res, excluded_indices=nucleic_indices
            ):
                molecule_type = external_molecule_types.get(molecule_name, molecule_name)
                molecule_records.append((first, molecule_type, count))

            if water_res:
                water_atoms: dict[tuple[str, int, str], int] = {}
                for index, (name, resid) in enumerate(zip(structure.resnames, structure.resids)):
                    if name not in water_res:
                        continue
                    chain = structure.chain_ids[index] if index < len(structure.chain_ids) else ""
                    key = (str(name), int(resid), str(chain))
                    water_atoms[key] = water_atoms.get(key, 0) + 1
                invalid_sites = [
                    key for key, count in water_atoms.items() if count != water_model.n_atoms
                ]
                if invalid_sites:
                    name, resid, chain = invalid_sites[0]
                    raise TopologyError(
                        f"Water {chain or '?'}:{resid} {name} does not contain "
                        f"exactly {water_model.n_atoms} sites"
                    )
                n_water = sum(water_atoms.values())
                if n_water % water_model.n_atoms:
                    raise TopologyError(
                        f"SOL atom count {n_water} is not divisible by the "
                        f"{water_model_name} site count {water_model.n_atoms}"
                    )
                water_runs = self._ordered_residue_run_records(structure, water_res)
                molecule_records.extend(
                    (first, "SOL", count) for first, _source_name, count in water_runs
                )

            # Preserve coordinate order.  Different ion species are separate
            # one-atom molecule types, so alphabetically regrouping them makes
            # GROMACS assign (for example) CL parameters to an NA coordinate.
            molecule_records.extend(self._ordered_residue_run_records(structure, ion_res))
            for _first, molecule_type, count in sorted(molecule_records, key=lambda item: item[0]):
                fh.write(f"{molecule_type:<34s} {count}\n")

    # ------------------------------------------------------------------
    # Chain discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_protein_chains(
        structure: Structure,
        protein_res: set[str],
        topology: Topology | None = None,
    ) -> list[tuple[str, list[int]]]:
        """Group protein chains, merging chains joined by an explicit covalent bond."""
        n = min(structure.num_atoms, len(structure.resnames))
        chain_ids = getattr(structure, "chain_ids", None) or []
        chain_map: dict[str, list[int]] = {}
        for i in range(n):
            if structure.resnames[i] in protein_res:
                cid = chain_ids[i] if i < len(chain_ids) else "A"
                if not isinstance(cid, str) or cid.strip() == "":
                    cid = "A"
                chain_map.setdefault(cid.strip(), []).append(i)
        parent = {chain: chain for chain in chain_map}

        def find(chain: str) -> str:
            while parent[chain] != chain:
                parent[chain] = parent[parent[chain]]
                chain = parent[chain]
            return chain

        def union(first: str, second: str) -> None:
            left, right = find(first), find(second)
            if left != right:
                parent[max(left, right)] = min(left, right)

        if topology is not None:
            chain_by_atom = {
                index: chain for chain, indices in chain_map.items() for index in indices
            }
            for bond in topology.bonds:
                first = chain_by_atom.get(int(bond.i))
                second = chain_by_atom.get(int(bond.j))
                if first is not None and second is not None and first != second:
                    union(first, second)

        groups: dict[str, list[str]] = {}
        for chain in sorted(chain_map):
            groups.setdefault(find(chain), []).append(chain)
        result = []
        for chains in groups.values():
            identifier = "_".join(chains)
            indices = [index for chain in chains for index in chain_map[chain]]
            result.append((identifier, sorted(indices)))
        return result

    # ------------------------------------------------------------------
    # Protein ITP
    # ------------------------------------------------------------------

    def _write_protein_itp(
        self, structure: Structure, path: Path, protein_res: set[str] | None = None
    ) -> None:
        """Write a protein ITP (legacy — use _write_protein_itp_for_indices for per-chain)."""
        if protein_res:
            protein_indices = [
                i
                for i in range(min(structure.num_atoms, len(structure.resnames)))
                if structure.resnames[i] in protein_res
            ]
        else:
            protein_indices = list(range(min(structure.num_atoms, len(structure.resnames))))
        self._write_protein_itp_for_indices(structure, path, protein_indices, "Protein_chain")

    def _write_protein_itp_for_indices(
        self,
        structure: Structure,
        path: Path,
        atom_indices: list[int],
        moltype: str = "Protein_chain",
        topology: Topology | None = None,
    ) -> None:
        """Write a protein ITP for a specific set of atom indices (e.g. one chain).

        Includes [ atoms ], [ bonds ], [ angles ], [ dihedrals ], [ impropers ]
        sourced from the CHARMM36 .rtp residue templates.
        When *topology* is provided, pre-computed atom types/charges from
        the force field module are used in preference to inline RTP lookup.
        """
        from gmxbuilder.modules.forcefield.rtp_parser import (
            get_terminal_residue,
            load_force_field_rtp,
        )

        # Load RTP for the selected force field
        if not self._rtp_loaded:
            self._rtp = load_force_field_rtp(self.force_field)
            self._rtp_loaded = True

        n_resnames = len(structure.resnames)
        n_atom_names = len(structure.atom_names)
        n_resids = len(structure.resids)

        # Detect termini that StructureProcessor has reconciled with the
        # selected force field.  Residue labels stay PDB-compatible (ALA,
        # GLY...), while RTP lookup uses virtual NALA/CALA-style templates.
        raw_residue_order: list[tuple[str, str, int]] = []
        raw_residue_names: dict[tuple[str, str, int], set[str]] = {}
        for i in atom_indices:
            rn = structure.resnames[i] if i < n_resnames else "UNK"
            rid = structure.resids[i] if i < n_resids else i + 1
            chain = structure.chain_ids[i] if i < len(structure.chain_ids) else "A"
            key = (str(chain), rn, rid)
            if key not in raw_residue_names:
                raw_residue_order.append(key)
                raw_residue_names[key] = set()
            raw_residue_names[key].add(structure.atom_names[i].strip())

        terminal_templates: dict[tuple[str, str, int], str] = {}
        residues_by_chain: dict[str, list[tuple[str, str, int]]] = {}
        for key in raw_residue_order:
            residues_by_chain.setdefault(key[0], []).append(key)
        for chain_residues in residues_by_chain.values():
            if len(chain_residues) <= 1:
                continue
            for key, end in ((chain_residues[0], "N"), (chain_residues[-1], "C")):
                _chain, base_name, _rid = key
                # Explicit ACE/NME residues are already complete RTP residues.
                # Their adjacent amino acid must remain an internal residue so
                # its -C/+N references form the cap peptide bond.
                if (end == "N" and base_name == "ACE") or (end == "C" and base_name == "NME"):
                    continue
                variant_name, variant = get_terminal_residue(self.force_field, base_name, end)
                base = self._rtp.get_residue(base_name)
                if base is None:
                    raise TopologyError(f"RTP residue {base_name} not found in {self.force_field}")
                base_atoms = {atom[0] for atom in base["atoms"]}
                variant_atoms = {atom[0] for atom in variant["atoms"]}
                added = variant_atoms - base_atoms
                removed = base_atoms - variant_atoms
                current = raw_residue_names[key]
                if added and added.issubset(current) and not (removed & current):
                    terminal_templates[key] = variant_name

        # ---- Pass 1: write [ atoms ] and build per-residue name→seq map ----
        atoms_lines: list[str] = []
        # key=(chain, template-resname, resid) → atom records
        residue_atoms: dict[tuple[str, str, int], list[tuple[int, str, str, float]]] = {}

        for seq_idx, i in enumerate(atom_indices):
            rn = structure.resnames[i] if i < n_resnames else "UNK"
            an = structure.atom_names[i] if i < n_atom_names else "CA"
            rid = structure.resids[i] if i < n_resids else i + 1
            chain = structure.chain_ids[i] if i < len(structure.chain_ids) else "A"

            template_rn = terminal_templates.get((str(chain), rn, rid), rn)
            # RTP is authoritative here.  The pre-computed topology predates
            # protonation, PTM, and HDB/terminal atom changes.
            rtp_result = self._rtp.get_atom_type(template_rn, an)
            if rtp_result:
                atype, charge = rtp_result
            elif topology is not None and i < len(topology.atom_types):
                at = topology.atom_types[i]
                atype = at.name
                charge = at.charge
            else:
                raise TopologyError(f"No {self.force_field} RTP atom type for {template_rn}:{an}")

            seq1 = seq_idx + 1  # 1-based
            atoms_lines.append(
                f"{seq1:6d} {atype:>6s} {rid:6d} {rn:>6s} {an:>6s} {seq1:6d}  {charge:10.6f}\n"
            )

            key = (str(chain), template_rn, rid)
            if key not in residue_atoms:
                residue_atoms[key] = []
            residue_atoms[key].append((seq1, an, atype, charge))

        # ---- Pass 2: collect bonded parameters from RTP ----
        all_bonds: list[tuple[int, int]] = []
        all_angles: list[tuple[int, int, int]] = []
        all_dihedrals: list[tuple[int, int, int, int]] = []
        all_impropers: list[tuple] = []

        residue_order = list(residue_atoms)
        residue_name_maps = {
            key: {an.strip(): seq1 for seq1, an, _atype, _charge in atoms}
            for key, atoms in residue_atoms.items()
        }
        seq_to_structure_index = {
            seq_idx + 1: structure_index for seq_idx, structure_index in enumerate(atom_indices)
        }
        connected_boundaries: set[int] = set()
        for position in range(len(residue_order) - 1):
            if residue_order[position][0] != residue_order[position + 1][0]:
                continue
            left = residue_name_maps[residue_order[position]].get("C")
            right = residue_name_maps[residue_order[position + 1]].get("N")
            if left is None or right is None:
                continue
            left_coord = structure.coordinates[seq_to_structure_index[left]]
            right_coord = structure.coordinates[seq_to_structure_index[right]]
            # Normal peptide C-N is about 0.133 nm.  A generous 0.20 nm
            # threshold preserves valid geometries while respecting chain
            # breaks/missing segments that share the same PDB chain ID.
            if float(np.linalg.norm(left_coord - right_coord)) <= 0.20:
                connected_boundaries.add(position)

        def resolve_atom(token: str, residue_position: int) -> int | None:
            """Resolve RTP local, previous (-), or next (+) atom references."""
            name = token.strip()
            target_position = residue_position
            if name.startswith("-"):
                target_position -= 1
                name = name[1:]
                boundary = target_position
            elif name.startswith("+"):
                target_position += 1
                name = name[1:]
                boundary = residue_position
            else:
                boundary = None
            if not 0 <= target_position < len(residue_order):
                return None
            if boundary is not None and boundary not in connected_boundaries:
                return None
            return residue_name_maps[residue_order[target_position]].get(name)

        for residue_position, (_chain, rn, rid) in enumerate(residue_order):
            rtp_res = self._rtp.get_residue(rn)
            if rtp_res is None:
                continue

            # Map RTP bonds (by atom name) to sequential indices
            for a1, a2 in rtp_res.get("bonds", []):
                i1 = resolve_atom(a1, residue_position)
                i2 = resolve_atom(a2, residue_position)
                if i1 is not None and i2 is not None:
                    all_bonds.append((i1, i2))

            # Map RTP angles
            for a1, a2, a3 in rtp_res.get("angles", []):
                i1 = resolve_atom(a1, residue_position)
                i2 = resolve_atom(a2, residue_position)
                i3 = resolve_atom(a3, residue_position)
                if i1 is not None and i2 is not None and i3 is not None:
                    all_angles.append((i1, i2, i3))

            # Map RTP dihedrals
            for a1, a2, a3, a4 in rtp_res.get("dihedrals", []):
                i1 = resolve_atom(a1, residue_position)
                i2 = resolve_atom(a2, residue_position)
                i3 = resolve_atom(a3, residue_position)
                i4 = resolve_atom(a4, residue_position)
                if all(v is not None for v in (i1, i2, i3, i4)):
                    all_dihedrals.append((i1, i2, i3, i4))

            # Map RTP impropers
            for improper in rtp_res.get("impropers", []):
                a1, a2, a3, a4 = improper[:4]
                i1 = resolve_atom(a1, residue_position)
                i2 = resolve_atom(a2, residue_position)
                i3 = resolve_atom(a3, residue_position)
                i4 = resolve_atom(a4, residue_position)
                if all(v is not None for v in (i1, i2, i3, i4)):
                    all_impropers.append((i1, i2, i3, i4, *improper[4:]))

        # StructureProcessor records dedicated cross-residue chemistry (for
        # example SG-SG disulfides) in the authoritative topology.  Include
        # those bonds after RTP expansion; graph-derived angles, dihedrals and
        # exclusions below then cover the complete covalent graph.
        if topology is not None:
            structure_to_seq = {
                structure_index: seq_index
                for seq_index, structure_index in seq_to_structure_index.items()
            }
            for bond in topology.bonds:
                first = structure_to_seq.get(int(bond.i))
                second = structure_to_seq.get(int(bond.j))
                if first is not None and second is not None:
                    all_bonds.append((first, second))

        # pdb2gmx derives bonded terms from the complete bond graph.  RTP files
        # generally do not enumerate every angle/proper/pair, so explicit RTP
        # terms alone produce a syntactically valid but physically incomplete
        # topology.
        all_bonds = sorted({tuple(sorted(bond)) for bond in all_bonds})
        generated_angles, generated_dihedrals, all_pairs = self._generate_graph_terms(all_bonds)
        all_angles = sorted(set(all_angles) | set(generated_angles))
        all_dihedrals = sorted(set(all_dihedrals) | set(generated_dihedrals))
        all_impropers = sorted(set(all_impropers))
        angle_funct, proper_funct, improper_funct = self._bonded_function_types()

        # ---- Write file ----
        with open(path, "w") as fh:
            ff_name = self.force_field.upper()
            fh.write(f"; {moltype} topology — GMXBUILDER ({ff_name} .rtp)\n\n")
            fh.write(f"[ moleculetype ]\n{moltype}    3\n\n")
            fh.write("[ atoms ]\n")
            fh.write(";   nr  type  resnr residue  atom  cgnr  charge\n")
            fh.writelines(atoms_lines)

            # Write bonds
            if all_bonds:
                fh.write("\n[ bonds ]\n")
                fh.write(";   ai    aj  funct\n")
                for i1, i2 in all_bonds:
                    fh.write(f"{i1:6d} {i2:6d}    1\n")

            # Write angles
            if all_angles:
                fh.write("\n[ angles ]\n")
                fh.write(";   ai    aj    ak  funct\n")
                for i1, i2, i3 in all_angles:
                    fh.write(f"{i1:6d} {i2:6d} {i3:6d}    {angle_funct}\n")

            # Write dihedrals
            if all_dihedrals or all_impropers:
                fh.write("\n[ dihedrals ]\n")
                fh.write("; proper dihedrals generated from the bond graph\n")
                for i1, i2, i3, i4 in all_dihedrals:
                    fh.write(f"{i1:6d} {i2:6d} {i3:6d} {i4:6d}    {proper_funct}\n")
                if all_impropers:
                    fh.write("; impropers declared by RTP/TDB templates\n")
                    for improper in all_impropers:
                        i1, i2, i3, i4 = improper[:4]
                        parameters = " ".join(improper[4:])
                        suffix = f" {parameters}" if parameters else ""
                        fh.write(f"{i1:6d} {i2:6d} {i3:6d} {i4:6d}    {improper_funct}{suffix}\n")

            if all_pairs:
                fh.write("\n[ pairs ]\n")
                fh.write("; 1-4 pairs generated from proper dihedral paths\n")
                for i1, i2 in all_pairs:
                    fh.write(f"{i1:6d} {i2:6d}    1\n")

            # Staged equilibration restraints.  The force constants are
            # supplied by MDP preprocessor macros and decay across stages.
            # Restrain heavy atoms only; hydrogen geometry follows bonded
            # terms and constraints.
            fh.write("\n#ifdef POSRES\n")
            fh.write("[ position_restraints ]\n")
            fh.write("; atom  funct       fc_x             fc_y             fc_z\n")
            backbone_names = {"N", "CA", "C", "O", "OT1", "OT2", "OXT"}
            for local_index, structure_index in enumerate(atom_indices, start=1):
                atom_name = structure.atom_names[structure_index].strip()
                element = (
                    structure.elements[structure_index].strip().upper()
                    if structure_index < len(structure.elements)
                    else ""
                )
                if element == "H" or atom_name.upper().startswith("H"):
                    continue
                macro = "POSRES_FC_BB" if atom_name in backbone_names else "POSRES_FC_SC"
                fh.write(f"{local_index:6d}    1  {macro:>16s} {macro:>16s} {macro:>16s}\n")
            fh.write("#endif\n")

    # ------------------------------------------------------------------
    # Lipid ITP
    # ------------------------------------------------------------------

    @staticmethod
    def _append_exact_lipid_position_restraint(
        path: Path, coordinates: np.ndarray, atom_names: list[str] | tuple[str, ...]
    ) -> None:
        """Add one chemically selected headgroup Z restraint to an exact ITP."""
        from gmxbuilder.modules.membrane.lipid_orientation import (
            atom_element,
            infer_lipid_orientation,
        )

        phosphorus = [index for index, name in enumerate(atom_names) if atom_element(name) == "P"]
        if phosphorus:
            marker_index = phosphorus[0]
        else:
            orientation = infer_lipid_orientation(coordinates, atom_names)
            projections = (
                np.asarray(coordinates)[orientation.polar_indices] @ orientation.head_from_tail
            )
            marker_index = int(orientation.polar_indices[int(np.argmax(projections))])
        with path.open("a") as handle:
            handle.write("\n#ifdef POSRES\n")
            handle.write("[ position_restraints ]\n")
            handle.write("; atom  funct  fc_x  fc_y             fc_z\n")
            handle.write(f"{marker_index + 1:6d}    1   0.0   0.0  POSRES_FC_LIPID\n")
            handle.write("#endif\n")

    def _write_lipid_itp(
        self,
        lipid_name: str,
        structure: Structure,
        path: Path,
        *,
        apply_lipid_restraints: bool = True,
    ) -> None:
        from gmxbuilder.modules.forcefield.lipid_policy import (
            lipid_rtp_identity_issues,
            lipid_rtp_template,
        )

        first_index = next(
            (i for i, name in enumerate(structure.resnames) if name == lipid_name),
            None,
        )
        if first_index is None:
            raise TopologyError(f"No coordinates found for lipid {lipid_name}")
        first_resid = structure.resids[first_index]
        molecule_indices = []
        for index in range(first_index, structure.num_atoms):
            if structure.resnames[index] != lipid_name or structure.resids[index] != first_resid:
                break
            molecule_indices.append(index)

        rtp_name, rtp_residue = lipid_rtp_template(lipid_name, self.force_field)
        if rtp_residue is None:
            if not self.force_field.lower().startswith("amber"):
                raise TopologyError(
                    f"Lipid {lipid_name} has no {self.force_field} RTP parameters; "
                    f"use the Amber99SB-ILDN + GAFF2 lipid policy"
                )
            coordinate_order = tuple(
                structure.atom_names[index].strip() for index in molecule_indices
            )
            selected_lipid_ff = str(self.ff_config.get("lipid_ff", "gaff2")).lower()
            if selected_lipid_ff == "lipid21":
                from gmxbuilder.modules.forcefield.lipid21_backend import (
                    lipid21_itp_path,
                    load_lipid21_geometry,
                )

                _coordinates, atom_names = load_lipid21_geometry(lipid_name)
                if coordinate_order != tuple(atom_names):
                    raise TopologyError(
                        f"Lipid {lipid_name} coordinate order does not match its "
                        "exact Lipid21 topology"
                    )
                path.write_text(lipid21_itp_path(lipid_name).read_text())
                self._append_exact_lipid_position_restraint(path, _coordinates, atom_names)
            elif selected_lipid_ff == "gaff2":
                from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
                from gmxbuilder.modules.membrane.lipids import LipidRegistry

                lipid = LipidRegistry.get(lipid_name)
                template = prepare_gaff_lipid(lipid_name, lipid.smiles, lipid.charge)
                if coordinate_order != template.atom_names:
                    raise TopologyError(
                        f"Lipid {lipid_name} coordinate order does not match its "
                        f"cached GAFF2 topology"
                    )
                path.write_text(template.itp_path.read_text())
                self._append_exact_lipid_position_restraint(
                    path, template.coordinates, template.atom_names
                )
            else:
                raise TopologyError(f"Unsupported Amber lipid backend {selected_lipid_ff!r}")
            return

        identity_issues = lipid_rtp_identity_issues(lipid_name, self.force_field)
        if identity_issues:
            raise TopologyError(
                f"Lipid {lipid_name} is not chemically identical to CHARMM RTP "
                f"{rtp_name}: {'; '.join(identity_issues)}"
            )

        # Use the exact atom order of the first built molecule.  Lipid-library
        # coordinate order is authoritative for GRO; RTP is authoritative for
        # type, charge and connectivity.
        rtp_atoms = {
            atom_name: (atom_type, charge)
            for atom_name, atom_type, charge, _group in rtp_residue["atoms"]
        }
        coordinate_names = {structure.atom_names[index].strip() for index in molecule_indices}
        missing_atoms = sorted(set(rtp_atoms) - coordinate_names)
        extra_atoms = sorted(coordinate_names - set(rtp_atoms))
        if missing_atoms or extra_atoms:
            raise TopologyError(
                f"Lipid {lipid_name} does not match the {self.force_field} RTP "
                f"template (missing={missing_atoms}, extra={extra_atoms})"
            )
        lipid_atom_list = []
        for index in molecule_indices:
            atom_name = structure.atom_names[index].strip()
            if atom_name not in rtp_atoms:
                raise TopologyError(
                    f"Lipid {lipid_name} atom {atom_name!r} is absent from "
                    f"the {self.force_field} RTP template"
                )
            atom_type, charge = rtp_atoms[atom_name]
            lipid_atom_list.append((atom_name, atom_type, charge))

        # Build name → index map for bond generation
        name_to_idx: dict[str, int] = {}
        for i, (an, atype, charge) in enumerate(lipid_atom_list):
            name_to_idx[an.strip()] = i

        bonds = []
        for atom1, atom2 in rtp_residue.get("bonds", []):
            if atom1 in name_to_idx and atom2 in name_to_idx:
                bonds.append((name_to_idx[atom1], name_to_idx[atom2]))
        bonds = sorted({tuple(sorted(bond)) for bond in bonds})
        if not bonds:
            raise TopologyError(f"Lipid {lipid_name} RTP template contains no usable bonds")
        graph_bonds = [(left + 1, right + 1) for left, right in bonds]
        angles, dihedrals, pairs = self._generate_graph_terms(graph_bonds)
        angle_funct, proper_funct, improper_funct = self._bonded_function_types()
        impropers = []
        for improper in rtp_residue.get("impropers", []):
            atom_names = improper[:4]
            if all(name in name_to_idx for name in atom_names):
                impropers.append(
                    tuple(name_to_idx[name] + 1 for name in atom_names) + tuple(improper[4:])
                )

        # Protect native cis lipid double bonds during early equilibration.
        atom_type_by_name = {
            atom_name: atom_type for atom_name, atom_type, _charge in lipid_atom_list
        }
        adjacency: dict[str, set[str]] = {name: set() for name in name_to_idx}
        for left, right in rtp_residue.get("bonds", []):
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        cis_dihedrals: list[tuple[int, int, int, int]] = []
        for left, right in rtp_residue.get("bonds", []):
            if atom_type_by_name.get(left) != "CEL1" or atom_type_by_name.get(right) != "CEL1":
                continue
            left_outer = sorted(
                name
                for name in adjacency[left] - {right}
                if atom_type_by_name.get(name, "").startswith("C")
            )
            right_outer = sorted(
                name
                for name in adjacency[right] - {left}
                if atom_type_by_name.get(name, "").startswith("C")
            )
            if left_outer and right_outer:
                cis_dihedrals.append(
                    (
                        name_to_idx[left_outer[0]] + 1,
                        name_to_idx[left] + 1,
                        name_to_idx[right] + 1,
                        name_to_idx[right_outer[0]] + 1,
                    )
                )

        with open(path, "w") as fh:
            fh.write(f"; {lipid_name} topology — GMXBUILDER\n\n")
            fh.write(f"[ moleculetype ]\n{lipid_name}    3\n\n")
            fh.write("[ atoms ]\n")
            fh.write(";   nr  type  resnr residue  atom  cgnr  charge\n")

            for i, (an, atype, charge) in enumerate(lipid_atom_list):
                fh.write(
                    f"{i + 1:6d} {atype:>6s} {1:6d} {lipid_name:>6s} {an:>6s} {i + 1:6d}  {charge:10.6f}\n"
                )

            if bonds:
                fh.write("\n[ bonds ]\n")
                fh.write(";   ai    aj  funct\n")
                for bi, bj in sorted(bonds):
                    # Do not inject element-based approximate constants.  A
                    # supported CHARMM lipid must resolve every bond against
                    # the selected force field's exact [ bondtypes ] table;
                    # grompp then fails explicitly if a parameter is absent.
                    fh.write(f"{bi + 1:6d} {bj + 1:6d}    1\n")

            if angles:
                fh.write("\n[ angles ]\n")
                for atom1, atom2, atom3 in angles:
                    fh.write(f"{atom1:6d} {atom2:6d} {atom3:6d}    {angle_funct}\n")

            if dihedrals or impropers:
                fh.write("\n[ dihedrals ]\n")
                for atom1, atom2, atom3, atom4 in dihedrals:
                    fh.write(f"{atom1:6d} {atom2:6d} {atom3:6d} {atom4:6d}    {proper_funct}\n")
                for improper in impropers:
                    atom1, atom2, atom3, atom4 = improper[:4]
                    parameters = " ".join(improper[4:])
                    suffix = f" {parameters}" if parameters else ""
                    fh.write(
                        f"{atom1:6d} {atom2:6d} {atom3:6d} {atom4:6d}    {improper_funct}{suffix}\n"
                    )

            if pairs:
                fh.write("\n[ pairs ]\n")
                for atom1, atom2 in pairs:
                    fh.write(f"{atom1:6d} {atom2:6d}    1\n")

            # Planar Z restraints keep the headgroup planes near the initial
            # DHH while allowing lateral area relaxation.  Phosphorus is the
            # preferred phospholipid marker; O3 is used for sterols.
            if apply_lipid_restraints:
                marker_indices = [
                    name_to_idx[name] + 1 for name in ("P", "O3") if name in name_to_idx
                ]
                if not marker_indices:
                    marker_indices = [
                        index + 1
                        for index, (name, _atype, _charge) in enumerate(lipid_atom_list)
                        if name.upper().startswith(("O", "N", "P", "S"))
                    ]
                fh.write("\n#ifdef POSRES\n")
                fh.write("[ position_restraints ]\n")
                fh.write("; atom  funct  fc_x  fc_y             fc_z\n")
                for atom_index in marker_indices:
                    fh.write(f"{atom_index:6d}    1   0.0   0.0  POSRES_FC_LIPID\n")
                fh.write("#endif\n")

            if apply_lipid_restraints and cis_dihedrals:
                fh.write("\n#ifdef DIHRES\n")
                fh.write("[ dihedral_restraints ]\n")
                fh.write(";   ai    aj    ak    al  type  phi  dphi  kfac\n")
                for atom1, atom2, atom3, atom4 in cis_dihedrals:
                    fh.write(
                        f"{atom1:6d} {atom2:6d} {atom3:6d} {atom4:6d}    1  0.0  0.0  DIHRES_FC\n"
                    )
                fh.write("#endif\n")

    # ------------------------------------------------------------------
    # Force field bundling
    # ------------------------------------------------------------------

    def _copy_force_field(self, output_dir: Path) -> None:
        """Copy the bundled force field files to toppar/ (toppar convention).

        Also rewrites ``#include`` directives inside ``forcefield.itp`` so
        that GROMACS can resolve them from the working directory (the
        includes are written as ``toppar/ffnonbonded.itp`` etc.).
        """
        if (output_dir / "forcefield.itp").exists():
            return  # already copied

        src = self._ff_path
        if not src.is_dir():
            import warnings

            warnings.warn(f"Force field directory not found: {src} — topology may be incomplete")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        # Only copy files needed at runtime — skip docs, archives, and build artifacts
        _NEEDED_SUFFIXES = {".itp", ".rtp", ".atp", ".hdb", ".tdb", ".arn", ".r2b"}
        for f in src.iterdir():
            if f.is_file() and f.suffix in _NEEDED_SUFFIXES:
                dest = output_dir / f.name
                shutil.copy2(f, dest)
