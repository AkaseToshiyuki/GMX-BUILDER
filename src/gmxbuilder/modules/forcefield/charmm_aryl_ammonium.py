"""Bounded research model for alkylbenzenes and primary alkylammonium ions.

Typing anchors and charges are read from the selected installed release. Charge
classes deliberately tie saturated carbon substitution states: this is an
explicit research assumption, not the official CGenFF charge model. Held-out
benzylammonium exposes the missing longer-range polarization; its error is
reported, never relabeled as physical validation or an official penalty.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from gmxbuilder.modules.forcefield.charmm_compat import (
    TEMPLATES,
    CharmmCompatError,
    ChemicalTemplate,
)
from gmxbuilder.modules.forcefield.charmm_research import reference_record

MODEL_ID = "aryl-primary-ammonium-tied-bci-v1"
TRAINING = ("MAMM", "EAMM", "BENZ", "EBEN", "CUME")
HOLDOUT = ("TOLU", "BZAM", "PRPA")
# Atom-type anchor only. Its zwitterionic charges are NEVER fitted or reused.
ALANINE = ChemicalTemplate(
    "ALAI",
    "C[C@H]([NH3+])C(=O)[O-]",
    ("C7", "C8", "N9", "C10", "O11", "O12"),
    "alanine zwitterion: local ammonium-adjacent CH typing only",
)


def check_domain(molecule):
    """One optional benzene ring, saturated acyclic substituents, zero/one NH3+."""
    from rdkit import Chem

    heavy = Chem.RemoveHs(molecule)
    rings = heavy.GetRingInfo().AtomRings()
    nitrogens = [a for a in heavy.GetAtoms() if a.GetAtomicNum() == 7]
    valid = (
        len(Chem.GetMolFrags(heavy)) == 1
        and 2 <= heavy.GetNumAtoms() <= 24
        and len(nitrogens) <= 1
        and Chem.GetFormalCharge(heavy) == len(nitrogens)
        and len(rings) <= 1
        and all(
            len(r) == 6 and all(heavy.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings
        )
    )
    for atom in heavy.GetAtoms():
        if atom.GetIsotope() or atom.GetNumRadicalElectrons():
            valid = False
        elif atom.GetAtomicNum() == 7:
            valid &= (
                atom.GetFormalCharge() == 1
                and atom.GetTotalNumHs() == 3
                and atom.GetDegree() == 1
                and all(
                    n.GetAtomicNum() == 6 and not n.GetIsAromatic() for n in atom.GetNeighbors()
                )
            )
        elif atom.GetAtomicNum() == 6:
            valid &= atom.GetFormalCharge() == 0 and (
                atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP3
            )
            # No quaternary carbon or tert-alkyl ammonium anchor in this model.
            valid &= atom.GetIsAromatic() or atom.GetTotalNumHs() >= 1
        else:
            valid = False
    valid &= all(
        b.GetIsAromatic() or b.GetBondType() == Chem.BondType.SINGLE for b in heavy.GetBonds()
    )
    if rings:
        # Multiple ring substituents need additional positional charge contexts.
        valid &= sum(heavy.GetAtomWithIdx(i).GetDegree() == 3 for i in rings[0]) <= 1
    if not valid:
        raise CharmmCompatError(
            "UNSUPPORTED_CHEMISTRY",
            "extended research domain: saturated acyclic C/H skeletons with at most one "
            "monosubstituted benzene and one primary alkyl [NH3+] group, 24 heavy atoms; "
            "no neutral/secondary amines, heteroaromatics, fused rings or other functional groups",
        )


def type_environment(atom):
    """First-shell element/charge/valence rule; not a force-field type-name guess."""
    from rdkit import Chem

    if atom.GetAtomicNum() == 1:
        parent = type_environment(atom.GetNeighbors()[0])
        return (1, parent) if parent is not None else None
    if atom.GetAtomicNum() not in {6, 7}:
        return None
    if not atom.GetIsAromatic() and atom.GetHybridization() != Chem.HybridizationType.SP3:
        return None
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors()),
        tuple(
            sorted(
                (n.GetAtomicNum(), n.GetFormalCharge())
                for n in atom.GetNeighbors()
                if n.GetAtomicNum() != 1
            )
        ),
    )


def charge_class(atom):
    """Versioned, explicitly tied charge classes; alpha C-H remains release-specific."""
    if atom.GetAtomicNum() == 1:
        return charge_class(atom.GetNeighbors()[0]) + "_H"
    if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
        return "NH3"
    if atom.GetAtomicNum() == 6:
        if atom.GetIsAromatic():
            return "AR"
        if any(n.GetAtomicNum() == 7 and n.GetFormalCharge() == 1 for n in atom.GetNeighbors()):
            return "ALPHA"
        return "ALK"
    raise CharmmCompatError("CHARGE_MODEL_MISSING", "no charge class for chemical environment")


def type_molecule(molecule, rules):
    types = []
    for atom in molecule.GetAtoms():
        rule = rules.get(type_environment(atom))
        if rule is None:
            raise CharmmCompatError(
                "ATOM_TYPE_UNASSIGNED",
                f"no reviewed local type environment for atom {atom.GetIdx() + 1}",
            )
        types.append(rule["type"])
    return types


def bond_classes(bond):
    atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
    classes = tuple(charge_class(a) for a in atoms)
    if all(a.GetAtomicNum() == 6 for a in atoms):
        # One declared saturated-carbon class for C-C transfers. Keep distinct
        # alpha-C/H and C/N classes for the release-dependent protonated group.
        classes = tuple("ALK" if c == "ALPHA" else c for c in classes)
    return classes


def transfer_matrix(molecule, features):
    matrix = np.zeros((molecule.GetNumAtoms(), len(features)))
    columns = {key: i for i, key in enumerate(features)}
    for bond in molecule.GetBonds():
        first, second = bond_classes(bond)
        if first == second:
            continue
        key = tuple(sorted((first, second)))
        if key not in columns:
            raise CharmmCompatError("CHARGE_MODEL_MISSING", f"untrained charge feature {key}")
        sign = 1 if first == key[0] else -1
        matrix[bond.GetBeginAtomIdx(), columns[key]] += sign
        matrix[bond.GetEndAtomIdx(), columns[key]] -= sign
    return matrix


def train_model(parser):
    references = {t.residue: t for t in (*TEMPLATES, ALANINE)}
    training = {n: reference_record(references[n], parser) for n in TRAINING}
    rules = {}
    for name, (mol, types, _, names) in {
        **training,
        "ALAI": reference_record(ALANINE, parser),
    }.items():
        for atom, atom_type, atom_name in zip(mol.GetAtoms(), types, names, strict=True):
            key = type_environment(atom)
            if key is None:
                continue
            if key in rules and rules[key]["type"] != atom_type:
                raise CharmmCompatError(
                    "ATOM_TYPE_AMBIGUOUS", "conflicting reference type environments"
                )
            rule = rules.setdefault(key, {"type": atom_type, "references": []})
            rule["references"].append({"residue": name, "atom": atom_name})
    features = sorted(
        {
            tuple(sorted(bond_classes(b)))
            for mol, _, _, _ in training.values()
            for b in mol.GetBonds()
            if bond_classes(b)[0] != bond_classes(b)[1]
        }
    )
    matrices, targets = [], []
    for mol, _, q, _ in training.values():
        baseline = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])
        if abs(float(q.sum() - baseline.sum())) > 1e-8:
            raise CharmmCompatError(
                "CHARGE_CONSERVATION_FAILED", "reference chemical state changed"
            )
        matrices.append(transfer_matrix(mol, features))
        targets.append(q - baseline)
    matrix, target = np.vstack(matrices), np.concatenate(targets)
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    train_error = float(np.max(np.abs(matrix @ coefficients - target)))
    if rank != len(features) or train_error > 1e-8 or not np.all(np.isfinite(coefficients)):
        raise CharmmCompatError(
            "CHARGE_MODEL_MISSING",
            "charge fit is underdetermined or does not reproduce its anchors",
        )
    heldout = []
    for name in HOLDOUT:
        mol, expected_types, expected_q, _ = reference_record(references[name], parser)
        if type_molecule(mol, rules) != expected_types:
            raise CharmmCompatError("ATOM_TYPE_AMBIGUOUS", f"held-out type disagreement: {name}")
        baseline = np.array([a.GetFormalCharge() for a in mol.GetAtoms()])
        prediction = baseline + transfer_matrix(mol, features) @ coefficients
        error = prediction - expected_q
        heldout.append(
            {
                "residue": name,
                "type_agreement": True,
                "charge_mae_e": float(np.mean(np.abs(error))),
                "charge_rmse_e": float(np.sqrt(np.mean(error**2))),
                "charge_max_error_e": float(np.max(np.abs(error))),
            }
        )
    report = {
        "model_id": MODEL_ID,
        "training_residues": list(TRAINING),
        "holdout_residues": list(HOLDOUT),
        "additional_type_only_references": ["ALAI"],
        "rank": int(rank),
        "feature_count": len(features),
        "singular_values": singular.tolist(),
        "features": [list(f) for f in features],
        "coefficients": coefficients.tolist(),
        "training_max_error_e": train_error,
        "holdout": heldout,
        "holdout_interpretation": "diagnostic, not an accuracy acceptance test; "
        "benzylammonium exposes missing longer-range polarization",
        "assumptions": [
            "bond transfers are shared across CH3/CH2/CH substitution within each charge class",
            "all saturated carbons share a class for C-C transfers, including aryl-alkyl bonds",
            "equal-class C-C bonds transfer zero; angle/dihedral charge increments are absent",
            "ALAI supplies the ammonium-adjacent CH type only; its charges are not transferred",
        ],
        "rules": [{"environment": repr(k), **v} for k, v in sorted(rules.items())],
        "physical_validation": "not_evaluated",
        "official_cgenff_equivalence": "not_established",
    }
    report["model_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    charge_environments = {
        type_environment(a) for m, _, _, _ in training.values() for a in m.GetAtoms()
    }
    return rules, features, coefficients, charge_environments, report


def assign_research(molecule, parser):
    from rdkit import Chem

    check_domain(molecule)
    molecule = Chem.AddHs(molecule)
    rules, features, coefficients, trained_environments, model = train_model(parser)
    types = type_molecule(molecule, rules)
    classes = [charge_class(a) for a in molecule.GetAtoms()]
    baseline = np.array([a.GetFormalCharge() for a in molecule.GetAtoms()])
    matrix = transfer_matrix(molecule, features)
    charges = baseline + matrix @ coefficients
    if not np.all(np.isfinite(charges)) or abs(float(charges.sum() - baseline.sum())) > 1e-10:
        raise CharmmCompatError(
            "CHARGE_CONSERVATION_FAILED", "charge transfers do not conserve the formal charge"
        )
    names = [
        f"{'H' if a.GetAtomicNum() == 1 else 'A'}{a.GetIdx() + 1}" for a in molecule.GetAtoms()
    ]
    template = ChemicalTemplate(
        "LCMP",
        Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=False),
        tuple(n for n in names if not n.startswith("H")),
        "experimental aryl/primary-ammonium assignment",
    )
    record = {
        "atoms": [(n, t, float(q), 1) for n, t, q in zip(names, types, charges, strict=True)],
        "bonds": [
            (names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]) for b in molecule.GetBonds()
        ],
    }
    report = {
        "charge_method": "formal_charge_plus_tied_bond_increments",
        "charge_model": model,
        "atom_assignments": [
            {
                "atom": n,
                "type": t,
                "charge_e": float(q),
                "formal_charge_e": int(baseline[i]),
                "charge_class": classes[i],
                "environment": repr(type_environment(a)),
                "source_references": rules[type_environment(a)]["references"],
                "charge_contributions_e": (matrix[i] * coefficients).tolist(),
                "charge_environment_extrapolated": type_environment(a) not in trained_environments,
            }
            for i, (n, t, q, a) in enumerate(
                zip(names, types, charges, molecule.GetAtoms(), strict=True)
            )
        ],
        "parameter_policy": "native exact/wildcard lookup only; missing parameters block export",
        "research_limitations": [
            "charge-class transfer is an experimental hypothesis, not an AMP-specific QM fit",
            "held-out benzylammonium charge errors are reported in charge_model.holdout",
            "finite static energy and complete topology do not establish physical accuracy",
        ],
    }
    return molecule, template, record, report
