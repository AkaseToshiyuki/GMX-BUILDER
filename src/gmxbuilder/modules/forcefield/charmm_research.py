"""Experimental, source-derived typing and conservative bond charge increments.

This is an independent small-domain model, not the official CGenFF algorithm.
No parameter values or fitted coefficients are distributed with the code.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from gmxbuilder.modules.forcefield.charmm_compat import TEMPLATES, CharmmCompatError

MODEL_ID = "linear-neutral-co-bci-v1"
TRAINING = ("MEOH", "ETOH", "PRPA", "DMEE", "DETE")
HOLDOUT = ("PROH", "METE")


def check_linear_domain(molecule):
    from rdkit import Chem

    if (
        Chem.GetFormalCharge(molecule)
        or molecule.GetNumAtoms() > 24
        or molecule.GetRingInfo().NumRings()
        or sum(a.GetAtomicNum() == 8 for a in molecule.GetAtoms()) > 1
        or any(
            a.GetAtomicNum() not in {6, 8}
            or a.GetFormalCharge()
            or a.GetDegree() > 2
            or a.GetIsotope()
            or a.GetNumRadicalElectrons()
            for a in molecule.GetAtoms()
        )
        or any(b.GetBondType() != Chem.BondType.SINGLE for b in molecule.GetBonds())
    ):
        raise CharmmCompatError(
            "UNSUPPORTED_CHEMISTRY",
            "research domain: neutral unbranched acyclic alkanes/alcohols/ethers, "
            "at most one oxygen and 24 heavy atoms; use CGenFF import for other chemistry",
        )


#: The research models, in the order they are tried: least-assuming first.
#:
#: Replaces a chain of ``try / except CharmmCompatError`` delegations between
#: modules. That chain caught the base class of *every* error code, so a real
#: CHARGE_CONSERVATION_FAILED raised inside one model would have been swallowed
#: and silently retried against another that was never asked about it. It also
#: reported whichever model happened to be last, rather than the one whose
#: domain the molecule actually belongs to.
#:
#: Order is by how much each model assumes. The general model assumes nothing
#: beyond what the installed release documents. The linear and tied-class
#: models each add an explicit hypothesis, and so are consulted only for
#: chemistry the release does not cover on its own -- an alpha-branched
#: alkylammonium such as amphetamine reaches the tied-class model this way.
def _models():
    from gmxbuilder.modules.forcefield import charmm_aryl_ammonium, charmm_general

    return (
        ("general", charmm_general.check_domain, charmm_general.assign_research, True),
        ("linear", lambda m, _ff: check_linear_domain(m), _assign_linear, False),
        (
            "tied-charge-class",
            lambda m, _ff: charmm_aryl_ammonium.check_domain(m),
            lambda m, p, _ff: charmm_aryl_ammonium.assign_research(m, p),
            False,
        ),
    )


def select_model(molecule, force_field: str):
    """The first model whose domain covers this molecule, and why the rest declined.

    Raises the *general* model's refusal when every model declines: it is the
    one whose domain is the installed release, so its error names what the
    release is missing rather than which hand-written predicate rejected the
    molecule last.
    """
    declined = []
    first_error = None
    for name, domain, assign, _needs_ff in _models():
        try:
            domain(molecule, force_field)
        except CharmmCompatError as error:
            if error.code not in {
                "UNSUPPORTED_CHEMISTRY",
                "ATOM_TYPE_UNASSIGNED",
                "ATOM_TYPE_AMBIGUOUS",
                "ATOM_TYPE_EXTRAPOLATED",
                "CHARGE_MODEL_MISSING",
            }:
                # Broken references, failed charge conservation and internal
                # calculations are failures, not permission to change models.
                raise
            declined.append((name, str(error)))
            if first_error is None:
                first_error = error
            continue
        return name, assign, declined
    raise first_error or CharmmCompatError(
        "UNSUPPORTED_CHEMISTRY", "no research model covers this chemistry"
    )


def check_domain(molecule, force_field: str) -> None:
    """Whether any research model covers this molecule."""
    select_model(molecule, force_field)


def environment(atom):
    """Versioned element/valence environment, independent of atom numbering."""
    if atom.GetAtomicNum() == 1:
        return (1, environment(atom.GetNeighbors()[0]))
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors()),
        tuple(sorted(n.GetAtomicNum() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1)),
    )


def reference_record(template, parser):
    """Bind reviewed explicit chemistry to the installed RTP; reject drift."""
    from rdkit import Chem

    molecule = Chem.AddHs(Chem.MolFromSmiles(template.smiles))
    record = parser.get_residue(template.residue)
    if not record:
        raise CharmmCompatError("FF_VERSION_MISMATCH", f"missing reference {template.residue}")
    atoms = {a[0]: a for a in record["atoms"]}
    bonds = {frozenset(b[:2]) for b in record["bonds"]}
    names = list(template.heavy_names)
    hydrogen_names = {}
    for index, name in enumerate(template.heavy_names):
        hydrogen_names[index] = sorted(
            n for b in bonds if name in b for n in b if n.startswith("H")
        )
    for atom in list(molecule.GetAtoms())[len(names) :]:
        parent = atom.GetNeighbors()[0].GetIdx()
        if not hydrogen_names[parent]:
            raise CharmmCompatError("FF_VERSION_MISMATCH", "reference hydrogen count changed")
        names.append(hydrogen_names[parent].pop(0))
    edges = {
        frozenset((names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]))
        for b in molecule.GetBonds()
    }
    if set(names) != set(atoms) or edges != bonds:
        raise CharmmCompatError("FF_VERSION_MISMATCH", "reference chemical graph changed")
    return molecule, [atoms[n][1] for n in names], np.array([atoms[n][2] for n in names]), names


def charge_matrix(molecule, types, features):
    """Each bond transfers charge antisymmetrically; equal-type bonds transfer zero."""
    matrix = np.zeros((molecule.GetNumAtoms(), len(features)))
    columns = {key: i for i, key in enumerate(features)}
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if types[a] == types[b]:
            continue
        key = tuple(sorted((types[a], types[b])))
        if key not in columns:
            raise CharmmCompatError("CHARGE_MODEL_MISSING", f"untrained charge feature {key}")
        sign = 1 if types[a] == key[0] else -1
        matrix[a, columns[key]] += sign
        matrix[b, columns[key]] -= sign
    return matrix


def train_model(parser):
    references = {t.residue: t for t in TEMPLATES}
    training = {n: reference_record(references[n], parser) for n in TRAINING}
    rules = {}
    for name, (mol, types, charges, names) in training.items():
        if abs(float(charges.sum())) > 1e-10:
            raise CharmmCompatError(
                "CHARGE_CONSERVATION_FAILED", "neutral training reference is charged"
            )
        for atom, atom_type, atom_name in zip(mol.GetAtoms(), types, names, strict=True):
            key = environment(atom)
            if key in rules and rules[key]["type"] != atom_type:
                raise CharmmCompatError("ATOM_TYPE_AMBIGUOUS", "conflicting reference environments")
            rule = rules.setdefault(key, {"type": atom_type, "references": []})
            rule["references"].append({"residue": name, "atom": atom_name})
    features = sorted(
        {
            tuple(sorted((types[b.GetBeginAtomIdx()], types[b.GetEndAtomIdx()])))
            for mol, types, _, _ in training.values()
            for b in mol.GetBonds()
            if types[b.GetBeginAtomIdx()] != types[b.GetEndAtomIdx()]
        }
    )
    matrix = np.vstack([charge_matrix(m, t, features) for m, t, _, _ in training.values()])
    target = np.concatenate([q for _, _, q, _ in training.values()])
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != len(features):
        raise CharmmCompatError("CHARGE_MODEL_MISSING", "charge model is rank deficient")
    train_error = float(np.max(np.abs(matrix @ coefficients - target)))
    heldout = []
    for name in HOLDOUT:
        mol, expected_types, q, _ = reference_record(references[name], parser)
        assigned = type_molecule(mol, rules)
        if assigned != expected_types:
            raise CharmmCompatError("ATOM_TYPE_AMBIGUOUS", f"held-out atom types differ for {name}")
        error = charge_matrix(mol, assigned, features) @ coefficients - q
        heldout.append(
            {
                "residue": name,
                "charge_mae_e": float(np.mean(np.abs(error))),
                "charge_rmse_e": float(np.sqrt(np.mean(error**2))),
                "charge_max_error_e": float(np.max(np.abs(error))),
            }
        )
    if train_error > 1e-8 or any(h["charge_max_error_e"] > 1e-8 for h in heldout):
        raise CharmmCompatError("CHARGE_MODEL_MISSING", "reference reproduction validation failed")
    report = {
        "model_id": MODEL_ID,
        "training_residues": list(TRAINING),
        "holdout_residues": list(HOLDOUT),
        "rank": int(rank),
        "feature_count": len(features),
        "singular_values": singular_values.tolist(),
        "training_max_error_e": train_error,
        "holdout": heldout,
        "split_limitation": "held-out homologues; no independent scaffold or physical benchmark",
        "features": [list(f) for f in features],
        "coefficients": coefficients.tolist(),
        "rules": [{"environment": repr(k), **v} for k, v in sorted(rules.items())],
        "physical_validation": "not_evaluated",
        "official_cgenff_equivalence": "not_established",
    }
    report["model_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return rules, features, coefficients, report


def type_molecule(molecule, rules):
    types = []
    for atom in molecule.GetAtoms():
        rule = rules.get(environment(atom))
        if rule is None:
            raise CharmmCompatError(
                "ATOM_TYPE_UNASSIGNED", f"no reviewed environment for atom {atom.GetIdx() + 1}"
            )
        types.append(rule["type"])
    return types


def assign_research(molecule, parser, force_field: str):
    """Assign with the first research model whose domain covers this molecule."""
    name, assign, declined = select_model(molecule, force_field)
    result = assign(molecule, parser, force_field)
    report = result[3]
    if isinstance(report, dict):
        report["research_model"] = name
        report["models_declined"] = [
            {"model": model, "reason": reason} for model, reason in declined
        ]
    return result


def _assign_linear(molecule, parser, _force_field=None):
    """Exact bond charge increments over neutral acyclic C/O chemistry."""
    from rdkit import Chem

    from gmxbuilder.modules.forcefield.charmm_compat import ChemicalTemplate

    molecule = Chem.AddHs(molecule)
    rules, features, coefficients, model_report = train_model(parser)
    types = type_molecule(molecule, rules)
    matrix = charge_matrix(molecule, types, features)
    charges = matrix @ coefficients
    if abs(float(charges.sum())) > 1e-10:
        raise CharmmCompatError(
            "CHARGE_CONSERVATION_FAILED", "charge increments do not conserve charge"
        )
    names = [
        f"{'H' if a.GetAtomicNum() == 1 else 'A'}{a.GetIdx() + 1}" for a in molecule.GetAtoms()
    ]
    template = ChemicalTemplate(
        "LCMP",
        Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=False),
        tuple(n for n in names if not n.startswith("H")),
        "experimental assignment",
    )
    record = {
        "atoms": [(n, t, float(q), 1) for n, t, q in zip(names, types, charges, strict=True)],
        "bonds": [
            (names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]) for b in molecule.GetBonds()
        ],
    }
    report = {
        "charge_method": "fitted_bond_increments",
        "charge_model": model_report,
        "atom_assignments": [
            {
                "atom": n,
                "type": t,
                "charge_e": float(q),
                "environment": repr(environment(a)),
                "source_references": rules[environment(a)]["references"],
                "charge_contributions_e": (matrix[i] * coefficients).tolist(),
            }
            for i, (n, t, q, a) in enumerate(
                zip(names, types, charges, molecule.GetAtoms(), strict=True)
            )
        ],
        "parameter_policy": "native exact/wildcard lookup only; uncalibrated analogy disabled",
    }
    return molecule, template, record, report
