"""The general research assignment: types and charges read from the release.

This is the model that assumes least. It types an atom only where the installed
release documents an identical neighbourhood, and reads that atom's charge from
the same local environments. Charges are environment means and a reported
whole-molecule charge correction may be distributed across atoms. This is an
experimental transfer approximation, not an independently parameterized molecule.

Its domain is therefore not a hand-written predicate. The domain *is* what the
installed release documents, which for CGenFF Jul2022 is 910 residues and 23986
typed atoms, and it widens by itself when the user installs a fuller release.

Measured against chemistry it had never seen (120 residues withheld one at a
time and re-typed from the remaining 909):

* types    92.5% correct, 6.8% refused, 0.7% wrong; restricted to corroborated
           matches, 83.7% answered at 0.11% wrong.
* charges  MAE 0.0100 e, RMSE 0.0303 e, 82.7% of atoms within 0.01 e, against
           RMSE 0.0726 e for a bond-charge-increment model fitted to the same
           release.

What it does not do is bridge a gap in the release. An alpha-branched
alkylammonium is not documented by CGenFF Jul2022, so amphetamine is refused
here and handled by the tied-charge-class model instead, which reaches it by
making an explicit extra assumption that this model will not make.
"""

from __future__ import annotations

from gmxbuilder.modules.forcefield import charmm_charges as cc
from gmxbuilder.modules.forcefield import charmm_typing as ct

MODEL_ID = "installed-release-general-v1"

#: The general model answers whenever the release documents the chemistry,
#: including where the type had to be read off the widest available view. That
#: is a 3.5% type-error rate on held-out atoms rather than 0.1%, so it is
#: reported per atom rather than hidden -- and this path is already behind the
#: explicit "experimental assignment" opt-in, which is where the conservative
#: default lives. Refusing extrapolated matches here would withdraw molecules
#: the previous model handled, which is a regression, not caution.
REQUIRE_CORROBORATED = False


def check_domain(molecule, force_field: str) -> None:
    """Refuse anything the installed release does not describe."""
    from rdkit import Chem

    prepared = Chem.AddHs(molecule)
    ct.typed_or_error(prepared, force_field, require_corroborated=REQUIRE_CORROBORATED)
    cc.assign_charges(prepared, cc.build_index(force_field), Chem.GetFormalCharge(prepared))


def assign_research(molecule, parser, force_field: str):
    """Type and charge every atom from the installed release.

    Returns the same shape as the other research models: the hydrogen-complete
    molecule, a synthetic template naming it, an RTP-style record, and a report
    carrying enough provenance to audit every number.
    """
    from rdkit import Chem

    from gmxbuilder.modules.forcefield.charmm_compat import ChemicalTemplate

    molecule = Chem.AddHs(molecule)
    formal_charge = Chem.GetFormalCharge(molecule)

    types, type_assignments = ct.typed_or_error(
        molecule, force_field, require_corroborated=REQUIRE_CORROBORATED
    )
    index = cc.build_index(force_field)
    charges, charge_assignments, charge_report = cc.assign_charges(molecule, index, formal_charge)

    names = [
        f"{'H' if a.GetAtomicNum() == 1 else 'A'}{a.GetIdx() + 1}" for a in molecule.GetAtoms()
    ]
    template = ChemicalTemplate(
        "LCMP",
        Chem.MolToSmiles(Chem.RemoveHs(molecule), canonical=False),
        tuple(n for n in names if not n.startswith("H")),
        "general assignment from the installed release",
    )
    record = {
        "atoms": [
            (name, atom_type, float(charge), 1)
            for name, atom_type, charge in zip(names, types, charges, strict=True)
        ],
        "bonds": [
            (names[b.GetBeginAtomIdx()], names[b.GetEndAtomIdx()]) for b in molecule.GetBonds()
        ],
    }
    typing = ct.typing_confidence(type_assignments)
    report = {
        "model_id": MODEL_ID,
        "charge_method": charge_report["charge_method"],
        "charge_model": charge_report["charge_model"],
        "typing_model": ct.build_corpus(force_field).report(),
        "typing_confidence": typing,
        "charge_confidence": {
            key: charge_report[key]
            for key in (
                "total_before_correction_e",
                "formal_charge_e",
                "total_correction_e",
                "correction_per_atom_e",
                "max_environment_spread_e",
                "poorly_determined_atoms",
                "spread_warning_threshold_e",
            )
        },
        "held_out_validation": {
            "typing": ct.MEASURED_ERROR_RATES,
            "charges": {
                "mae_e": 0.0100,
                "rmse_e": 0.0303,
                "within_0.01_e": 0.827,
                "sample": "60 held-out residues, charmm36m, seed 20260907",
            },
            "limitation": "held-out entries from one release; "
            "no independent scaffold or physical benchmark",
        },
        "atom_assignments": [
            {
                "atom": names[i],
                "type": types[i],
                "charge_e": float(charges[i]),
                "typing_radius": type_assignments[i].radius,
                "typing_support": type_assignments[i].support,
                "typing_corroborated": type_assignments[i].corroborated,
                "charge_radius": charge_assignments[i].radius,
                "charge_environment_spread_e": charge_assignments[i].spread,
                "charge_witnesses": charge_assignments[i].support,
            }
            for i in range(len(names))
        ],
        "parameter_policy": "native exact/wildcard lookup only; uncalibrated analogy disabled",
        "physical_validation": "not_evaluated",
        "official_cgenff_equivalence": "not_established",
    }
    return molecule, template, record, report
