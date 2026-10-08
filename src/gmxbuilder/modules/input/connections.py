"""Validate deposited inter-residue links before independent templates are used."""

from __future__ import annotations

import numpy as np

from gmxbuilder.core.chemistry import PROTEIN_RESNAMES
from gmxbuilder.io.residue_identity import clean_identifier
from gmxbuilder.modules.nucleic_acid.support import canonical_backbone_connections


def connection_issues(structure):
    info = structure.source_info
    source_atoms = info.get("atoms", {})
    retained = {uid: i for i, uid in enumerate(structure.source_ids) if uid}
    issues = []
    links = []
    if info.get("format") == "pdb":
        all_serials = {atom["serial"]: atom for atom in source_atoms.values()}
        serials = {}
        identities = {}
        for uid, atom in source_atoms.items():
            if uid not in retained:
                continue
            index = retained[uid]
            serials.setdefault(atom["serial"], []).append(index)
            key = (
                atom.get("author_chain", atom["chain"]),
                atom["resid"],
                clean_identifier(atom.get("icode", "")),
                atom["name"],
            )
            identities.setdefault(key, []).append(index)
        for a, b in info.get("connections", []):
            left, right = serials.get(a, []), serials.get(b, [])
            if len(left) > 1 or len(right) > 1:
                issues.append(
                    {
                        "code": "ambiguous_connection",
                        "message": f"CONECT {a}-{b} uses ambiguous atom serials; "
                        "provide unique identities",
                    }
                )
            elif left and right:
                links.append((left[0], right[0], "CONECT"))
            elif bool(left) != bool(right):
                removed = all_serials.get(b if left else a, {})
                if removed.get("element", "").upper() in {"H", "D"}:
                    continue
                issues.append(
                    {
                        "code": "dangling_connection",
                        "message": f"Selection removes an endpoint of CONECT {a}-{b}; "
                        "resolve the covalent fragment and its termini before building",
                    }
                )
        for line in info.get("link_records", []):
            try:
                if line.startswith("SSBOND"):
                    left = (line[15:16].strip(), int(line[17:21]), line[21:22].strip(), "SG")
                    right = (line[29:30].strip(), int(line[31:35]), line[35:36].strip(), "SG")
                else:
                    left = (
                        line[21:22].strip(),
                        int(line[22:26]),
                        line[26:27].strip(),
                        line[12:16].strip(),
                    )
                    right = (
                        line[51:52].strip(),
                        int(line[52:56]),
                        line[56:57].strip(),
                        line[42:46].strip(),
                    )
                a, b = identities.get(left, []), identities.get(right, [])
                if len(a) > 1 or len(b) > 1:
                    issues.append(
                        {
                            "code": "ambiguous_connection",
                            "message": "LINK/SSBOND uses repeated residue identities; "
                            "provide explicit endpoints",
                        }
                    )
                elif a and b:
                    if any(
                        line[start:end].strip() not in {"", "1555"}
                        for start, end in ((59, 65), (66, 72))
                    ):
                        issues.append(
                            {
                                "code": "unsupported_connection",
                                "message": "LINK/SSBOND references a symmetry mate; "
                                "construct the explicit assembly",
                            }
                        )
                    else:
                        links.append((a[0], b[0], line[:6].strip()))
                elif bool(a) != bool(b):
                    issues.append(
                        {
                            "code": "dangling_connection",
                            "message": "Selection removes a LINK/SSBOND endpoint; "
                            "resolve the connected fragment",
                        }
                    )
            except (ValueError, IndexError):
                issues.append(
                    {
                        "code": "invalid_connection",
                        "message": "Malformed LINK/SSBOND record; "
                        "repair the connection declaration",
                    }
                )
    elif info.get("format") == "mmcif":
        identities = {}
        mixed_names = any(
            clean_identifier(row.get(f"_struct_conn.{partner}_auth_seq_id"))
            and not clean_identifier(row.get(f"_struct_conn.{partner}_auth_atom_id"))
            for row in info.get("connections", [])
            for partner in ("ptnr1", "ptnr2")
        )
        for uid, atom in source_atoms.items():
            if uid not in retained:
                continue
            for namespace in ("auth", "label"):
                key = tuple(
                    atom.get(f"_atom_site.{namespace}_{field}", "")
                    for field in ("asym_id", "seq_id", "atom_id")
                )
                insertion = clean_identifier(atom.get("_atom_site.pdbx_PDB_ins_code", ""))
                if namespace == "auth":
                    key = (*key, insertion)
                identities.setdefault((namespace, *key), []).append(retained[uid])
            if mixed_names:
                # Join the two namespaces through the deposited atom row, not
                # by assuming that auth_atom_id equals label_atom_id.
                key = (
                    "auth_label",
                    atom.get("_atom_site.auth_asym_id", ""),
                    atom.get("_atom_site.auth_seq_id", ""),
                    atom.get("_atom_site.label_atom_id", ""),
                    insertion,
                )
                identities.setdefault(key, []).append(retained[uid])
        for row in info.get("connections", []):
            kind = row.get("_struct_conn.conn_type_id", "").lower()
            if kind in {"hydrog", "saltbr"}:
                continue
            ends = []
            for partner in ("ptnr1", "ptnr2"):
                index = None
                for namespace in ("auth", "label"):
                    key = tuple(
                        row.get(f"_struct_conn.{partner}_{namespace}_{field}", "")
                        for field in ("asym_id", "seq_id", "atom_id")
                    )
                    if namespace == "auth":
                        if not clean_identifier(key[2]):
                            namespace = "auth_label"
                            key = (*key[:2], row.get(f"_struct_conn.{partner}_label_atom_id", ""))
                        key = (
                            *key,
                            clean_identifier(
                                row.get(
                                    f"_struct_conn.pdbx_{partner}_PDB_ins_code",
                                    row.get(f"_struct_conn.pdbx_{partner}_pdb_ins_code", ""),
                                )
                            ),
                        )
                    matches = identities.get((namespace, *key), [])
                    if len(matches) == 1:
                        index = matches[0]
                        break
                    if len(matches) > 1:
                        issues.append(
                            {
                                "code": "ambiguous_connection",
                                "message": "Declared connection has ambiguous residue/atom "
                                "identities; "
                                "provide unique endpoints including insertion codes",
                            }
                        )
                        break
                ends.append(index)
            if all(index is not None for index in ends):
                links.append((*ends, kind or "struct_conn"))
            elif any(index is not None for index in ends) and kind.startswith("covale"):
                issues.append(
                    {
                        "code": "dangling_connection",
                        "message": "Selection removes a declared covalent connection endpoint; "
                        "resolve the fragment and its termini before building",
                    }
                )
    protein_order = list(
        dict.fromkeys(
            (structure.chain_ids[i], structure.resids[i], structure.resnames[i])
            for i in range(structure.num_atoms)
            if structure.resnames[i] in PROTEIN_RESNAMES
        )
    )
    peptide_neighbours = set(zip(protein_order, protein_order[1:]))
    nucleic_bonds = canonical_backbone_connections(structure)
    for left, right, kind in links:

        def residue(i):
            return (structure.chain_ids[i], structure.resids[i], structure.resnames[i])

        if residue(left) == residue(right):
            continue
        # Ordinary peptide bonds and ordinary disulfides have explicit support
        # in the later protein stage. Other declared cross-links cannot be
        # represented by independently parameterizing the two residues.
        names = {structure.atom_names[left], structure.atom_names[right]}
        distance = float(np.linalg.norm(structure.coordinates[left] - structure.coordinates[right]))
        carbon, nitrogen = (left, right) if structure.atom_names[left] == "C" else (right, left)
        peptide = (
            names == {"C", "N"}
            and (residue(carbon), residue(nitrogen)) in peptide_neighbours
            and structure.chain_ids[left] == structure.chain_ids[right]
            and 0.08 <= distance <= 0.20
        )
        disulfide = (
            names == {"SG"}
            and structure.resnames[left] == structure.resnames[right] == "CYS"
            and 0.15 <= distance <= 0.25
        )
        if not peptide and not disulfide and frozenset((left, right)) not in nucleic_bonds:
            issues.append(
                {
                    "code": "unsupported_connection",
                    "message": f"Declared {kind} link between {residue(left)} "
                    f"{structure.atom_names[left]} and "
                    f"{residue(right)} {structure.atom_names[right]} requires "
                    "an explicit supported "
                    "cross-link or coordination model; "
                    "independent residue templates are insufficient",
                }
            )
    return issues
