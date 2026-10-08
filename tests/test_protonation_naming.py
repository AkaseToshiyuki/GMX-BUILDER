"""Protonation state names must match the force field that has to build them.

Amber and CHARMM disagree on the residue name of nearly every non-default
protonation state, and the table here used to be a single hard-coded mix of the
two -- CHARMM histidines with Amber acids. A CHARMM build therefore asked for
ASH and failed several steps later with a message blaming the input structure,
and an Amber build asking for a non-default histidine would have failed on HSD
the same way.

The force fields' own .rtp files are the authority, so that is what these
assert against rather than a second hard-coded list.
"""

from __future__ import annotations

import pytest

from gmxbuilder.modules.modifications.protonation import (
    assign_all_protonations,
    assign_protonation,
    get_titratable_residues,
    resolve_titratable_states,
)
from tests.prerequisites import requires_forcefield

FORCE_FIELDS = ("amber14sb", "charmm36m")


# --------------------------------------------------------------------------
# Against the force fields themselves


@pytest.mark.parametrize("force_field", FORCE_FIELDS)
@requires_forcefield("amber14sb")
@requires_forcefield("charmm36m")
def test_every_offered_state_has_a_template(force_field):
    """The defect: a state was offered that the force field cannot build."""
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    missing = [
        state.residue_name
        for states in get_titratable_residues(force_field).values()
        for state in states
        if rtp.get_residue(state.residue_name) is None
    ]
    assert missing == [], f"{force_field} has no template for {missing}"


@pytest.mark.parametrize("force_field", FORCE_FIELDS)
@requires_forcefield("amber14sb")
@requires_forcefield("charmm36m")
def test_every_assignment_across_the_ph_range_is_buildable(force_field):
    """Sweep the pH range a user can choose and check every name produced."""
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    rtp = load_force_field_rtp(force_field)
    parents = ["HIS", "ASP", "GLU", "LYS", "CYS", "TYR"]
    bad = []
    for pH in (1.0, 3.0, 5.0, 7.0, 7.4, 9.0, 11.0, 13.0):
        for tautomer in ("HSD", "HSE"):
            for entry in assign_all_protonations(
                parents, pH=pH, his_tautomer=tautomer, force_field=force_field
            ):
                if rtp.get_residue(entry["assigned_name"]) is None:
                    bad.append((pH, tautomer, entry["original"], entry["assigned_name"]))
    assert bad == [], f"{force_field} cannot build: {bad}"


# --------------------------------------------------------------------------
# The two families really do differ


def test_the_two_families_name_the_same_state_differently():
    """If these ever agree, the naming table has collapsed back to one set."""
    amber = {
        state.state_id: state.residue_name
        for states in resolve_titratable_states("amber14sb").values()
        for state in states
    }
    charmm = {
        state.state_id: state.residue_name
        for states in resolve_titratable_states("charmm36m").values()
        for state in states
    }
    shared = set(amber) & set(charmm)
    differing = {key for key in shared if amber[key] != charmm[key]}
    assert {"HIS_ND", "HIS_NE", "HIS_P", "ASP_H", "GLU_H", "LYS_N"} <= differing


@pytest.mark.parametrize(
    ("force_field", "residue", "pH", "expected"),
    [
        ("amber14sb", "ASP", 3.0, "ASH"),
        ("charmm36m", "ASP", 3.0, "ASPP"),
        ("amber14sb", "GLU", 3.0, "GLH"),
        ("charmm36m", "GLU", 3.0, "GLUP"),
        ("amber14sb", "LYS", 12.0, "LYN"),
        ("charmm36m", "LYS", 12.0, "LSN"),
        ("amber14sb", "HIS", 5.0, "HIP"),
        ("charmm36m", "HIS", 5.0, "HSP"),
        ("amber14sb", "HIS", 8.0, "HIE"),
        ("charmm36m", "HIS", 8.0, "HSE"),
    ],
)
def test_each_state_is_named_for_its_force_field(force_field, residue, pH, expected):
    assigned = assign_protonation(residue, pH=pH, force_field=force_field)
    assert assigned["assigned_name"] == expected


def test_the_histidine_tautomer_choice_is_family_independent():
    """ "HSD" is a selector token, not the name Amber will write."""
    for force_field, delta, epsilon in (("amber14sb", "HID", "HIE"), ("charmm36m", "HSD", "HSE")):
        assert (
            assign_protonation("HIS", pH=9.0, his_tautomer="HSD", force_field=force_field)[
                "assigned_name"
            ]
            == delta
        )
        assert (
            assign_protonation("HIS", pH=9.0, his_tautomer="HSE", force_field=force_field)[
                "assigned_name"
            ]
            == epsilon
        )


# --------------------------------------------------------------------------
# A state no force field provides


@pytest.mark.parametrize("force_field", FORCE_FIELDS)
def test_a_state_the_force_field_lacks_is_reported_not_emitted(force_field):
    """Neither shipped force field defines a deprotonated tyrosine.

    Emitting TYM anyway is what turns a chemistry limitation into a grompp
    failure about a missing template.
    """
    assigned = assign_protonation("TYR", pH=13.0, force_field=force_field)
    assert assigned["assigned_name"] == "TYR"
    assert assigned["force_field_lacks_state"] is True


def test_a_state_the_force_field_provides_is_not_flagged():
    assigned = assign_protonation("ASP", pH=3.0, force_field="charmm36m")
    assert assigned["force_field_lacks_state"] is False


# --------------------------------------------------------------------------
# The predicted-pKa path uses the same names


def test_the_propka_path_shares_the_model_pka_path():
    """It was a second copy of the selection, carrying hard-coded Amber names."""
    import inspect

    from gmxbuilder.modules.modifications import protonation

    source = inspect.getsource(protonation.assign_protonation_with_propka)
    assert "assign_protonation(" in source
    for amber_only in ('"ASH"', '"GLH"', '"LYN"', '"HSP"', '"CYM"', '"TYM"'):
        assert amber_only not in source, f"{amber_only} is hard-coded in the PROPKA path"


@pytest.mark.parametrize("force_field", FORCE_FIELDS)
def test_a_predicted_pka_names_the_state_for_the_force_field(force_field):
    from gmxbuilder.modules.modifications.protonation import assign_protonation_with_propka

    residues = [{"resname": "ASP", "chain": "B", "resid": 262, "index": 0}]
    predictions = [
        {"residue_name": "ASP", "chain": "B", "resid": 262, "predicted_pKa": 8.5, "shift": 4.7}
    ]
    assigned = assign_protonation_with_propka(
        residues, predictions, pH=7.0, force_field=force_field
    )[0]
    expected = "ASH" if force_field.startswith("amber") else "ASPP"
    assert assigned["assigned_name"] == expected
    assert assigned["predicted_pKa"] == 8.5
