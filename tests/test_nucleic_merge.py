"""The rules that make the nucleic-acid merge safe.

The upstream archive is a whole force field. Most of what it holds duplicates
what we already ship, and some of it quietly contradicts it. These assert the
two rules that decide what crosses over, and they use small synthetic force
fields so each rule can be shown failing on its own.

The merge against the real OL24 archive is checked in
``test_nucleic_forcefield.py``, which needs the installed force field.
"""

from __future__ import annotations

import pytest

from gmxbuilder.modules.forcefield.nucleic_merge import (
    base_reachable,
    merge_nucleic_parameters,
    nucleic_eligible,
    parse_sections,
    residue_atom_types,
)

# --------------------------------------------------------------------------
# Building a tiny force field pair


def _write_forcefield(root, *, atomtypes, bonded, residues, extra=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "ffnonbonded.itp").write_text("[ atomtypes ]\n" + atomtypes + "\n")
    (root / "ffbonded.itp").write_text(bonded + "\n")
    (root / "atomtypes.atp").write_text(
        "\n".join(f"{line.split()[0]:<12s} 12.01000" for line in atomtypes.strip().splitlines())
        + "\n"
    )
    (root / "aminoacids.rtp").write_text("[ bondedtypes ]\n1 1 9 4 1 3 1 0 0\n\n" + residues + "\n")
    for name, text in (extra or {}).items():
        (root / name).write_text(text)
    return root


_PROTEIN_ATOMTYPES = """\
  CT  6 12.01 0.0000 A 0.339966950842 0.457729600000
  N   7 14.01 0.0000 A 0.325000000000 0.711280000000
  Br 35 79.90 0.0000 A 0.395559030854 1.338880000000
"""

_NUCLEIC_ATOMTYPES = """\
  CT  6 12.01 0.0000 A 3.39967e-01 4.57730e-01
  N   7 14.01 0.0000 A 3.25000e-01 7.11280e-01
  Br 35 79.90 0.0000 A 4.64693e-01 2.45414e-01
  N*  7 14.01 0.0000 A 3.25000e-01 7.11280e-01
  CK  6 12.01 0.0000 A 3.39967e-01 3.59824e-01
  OS  8 16.00 0.0000 A 3.00001e-01 7.11280e-01
"""

_PROTEIN_RESIDUES = """\
[ ALA ]
 [ atoms ]
   N   N   -0.4157  1
   CA  CT   0.0337  2
[ RA ]
 [ atoms ]
   RA  Br   2.0000  1
"""

_NUCLEIC_RESIDUES = """\
[ bondedtypes ]
1 1 9 4 1 3 1 0

[ DA ]
 [ atoms ]
   C1'  CT   0.0431  1
   N9   N*  -0.0268  2
   C8   CK   0.1607  3
   O4'  OS  -0.3691  4
"""


@pytest.fixture
def pair(tmp_path):
    """A base we ship and an upstream archive that overlaps it."""
    base = _write_forcefield(
        tmp_path / "base",
        atomtypes=_PROTEIN_ATOMTYPES,
        bonded=(
            "[ dihedraltypes ]\n"
            "  OS  CT  N*  CK  9  0.0  10.46  1\n"
            "  CT  CT  N   CT  9  0.0   1.00  3\n"
        ),
        residues=_PROTEIN_RESIDUES,
    )
    upstream = _write_forcefield(
        tmp_path / "upstream",
        atomtypes=_NUCLEIC_ATOMTYPES,
        bonded=(
            "[ dihedraltypes ]\n"
            "  OS  CT  N*  CK  9  74.8  2.95  1\n"
            "  OS  CT  N*  CK  9   6.2  4.46  2\n"
            "  CT  CT  N   CT  9   0.0  9.99  3\n"
            "  N*  CK  OS  CT  9   0.0  1.11  2\n"
        ),
        residues=_PROTEIN_RESIDUES,
        extra={
            "dna.rtp": _NUCLEIC_RESIDUES,
            "rna.rtp": _NUCLEIC_RESIDUES.replace("[ DA ]", "[ RA ]"),
            "dna.hdb": "",
            "rna.hdb": "",
            "dna.r2b": "",
            "rna.r2b": "",
            "dna.arn": "",
            "rna.arn": "",
        },
    )
    return base, upstream


# --------------------------------------------------------------------------
# Rule 1: only what the nucleic acids need


def test_a_parameter_the_nucleic_acids_do_not_need_is_left_upstream():
    """The archive is a whole force field; most of it is not ours to take."""
    nucleic = {"CT", "N*", "CK", "OS"}
    only = {"N*", "CK", "OS"}
    assert nucleic_eligible(("OS", "CT", "N*", "CK", "9"), nucleic, only)
    # Histidine/tryptophan impropers: upstream has them, we do not, and taking
    # them would add improper terms to every protein built with this field.
    assert not nucleic_eligible(("NA", "CW", "CC", "CT", "4"), nucleic, only)
    # Its own water and ion types, which would collide with ours.
    assert not nucleic_eligible(("HW", "OW", "HW"), nucleic, only)
    assert not nucleic_eligible(("Na",), nucleic, only)


def test_a_wildcard_needs_a_nucleic_only_type_to_be_taken():
    """A wildcard is a claim about every molecule, not only this one."""
    nucleic = {"CT", "N*", "CK", "OS", "C1"}
    only = {"N*", "CK", "OS", "C1"}
    assert nucleic_eligible(("X", "C1", "CT", "X", "9"), nucleic, only)
    # Every named type is shared, so this would silently match protein atoms.
    assert not nucleic_eligible(("X", "CT", "CT", "X", "9"), nucleic, only)


# --------------------------------------------------------------------------
# Rule 2: never move something a supported residue can reach


def test_reachability_is_per_residue_not_a_union_over_the_force_field():
    """``CT`` is in proteins and nucleic acids alike.

    A union would call every nucleic dihedral containing ``CT`` reachable and
    forbid the whole merge. What matters is whether the term can form, and it
    forms within one residue's reach.
    """
    residues = {"ALA": {"N", "CT"}, "SER": {"N", "CT", "OH"}}
    assert base_reachable(("CT", "CT", "N", "CT", "9"), residues)
    assert not base_reachable(("OS", "CT", "N*", "CK", "9"), residues)


def test_a_wildcard_position_does_not_make_a_key_reachable_by_itself():
    """A wildcard matches any atom; the *named* positions must still be there.

    Treating a wildcard as universally reachable rejected sixteen dihedrals
    the nucleic acids need -- ``X-C1-CT-X`` and its siblings, whose ``C1`` no
    protein residue has. What keeps an all-shared wildcard out is
    :func:`nucleic_eligible`, not this.
    """
    residues = {"ALA": {"N", "CT"}}
    assert base_reachable(("X", "CT", "CT", "X", "9"), residues)
    assert not base_reachable(("X", "C1", "CT", "X", "9"), residues)


def test_a_parameter_a_supported_residue_can_reach_is_not_even_added(tmp_path, pair):
    """Adding is not automatically safe either.

    GROMACS builds bonded terms from connectivity, so a *new* type whose key a
    protein residue can supply starts applying to proteins that previously had
    no such term. Against the real archive this refuses three constraint types
    over ``C``/``CA``/``CT``/``HO``, which are inert today only because no
    ``.rtp`` declares a constraints section -- luck, not a guarantee.
    """
    base, upstream = pair
    (upstream / "ffbonded.itp").write_text(
        (upstream / "ffbonded.itp").read_text() + "\n[ constrainttypes ]\n  CT  N  2  0.195\n"
    )
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts

    # ALA supplies both CT and N, so the constraint is refused rather than added.
    assert "CT-N" in report.rejected.get("constrainttypes", [])
    assert "CT-N" not in report.added.get("constrainttypes", [])
    merged = parse_sections((tmp_path / "merged" / "ffbonded.itp").read_text())
    assert ("CT", "N") not in merged.get("constrainttypes", {})


def test_every_rtp_the_base_ships_is_protected_not_only_the_amino_acids(tmp_path, pair):
    """The phosphorylated residues introduce ``P``, which nucleic acids use too.

    Reading only ``aminoacids.rtp`` would leave them unprotected, so the merge
    reads every ``.rtp`` in the base.
    """
    base, upstream = pair
    (base / "phosaa14sb.rtp").write_text(
        "[ bondedtypes ]\n1 1 9 4 1 3 1 0 0\n\n"
        "[ SEP ]\n [ atoms ]\n   OG  OS  -0.5  1\n   CB  CT   0.1  2\n"
        "   N9  N*   0.0  3\n   C8  CK   0.0  4\n"
    )
    # SEP now supplies OS, CT, N* and CK, so the glycosidic torsion becomes
    # reachable and must not be silently replaced.
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")

    assert not report.ok
    assert any("OS-CT-N*-CK" in conflict for conflict in report.conflicts)


# --------------------------------------------------------------------------
# The merge as a whole


def test_the_glycosidic_torsion_is_replaced_and_the_protein_one_is_not(tmp_path, pair):
    """The override the OL series exists for, and the one it must not make."""
    base, upstream = pair
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts

    assert "OS-CT-N*-CK-9" in report.overridden["dihedraltypes"]
    # CT-CT-N-CT is reachable from ALA, and upstream disagrees about it. It is
    # neither adopted nor overridden: the base's value stands.
    merged = parse_sections((tmp_path / "merged" / "ffbonded.itp").read_text())
    protein_rows = merged["dihedraltypes"][("CT", "CT", "N", "CT", "9")]
    assert [row[-2] for row in protein_rows] == ["1.00"], (
        "a parameter a protein residue can reach was changed"
    )
    nucleic_rows = merged["dihedraltypes"][("OS", "CT", "N*", "CK", "9")]
    assert sorted(row[-2] for row in nucleic_rows) == ["2.95", "4.46"]


def test_a_halogen_the_nucleic_acids_never_use_is_left_alone(tmp_path, pair):
    """Upstream really does disagree about ``Br``; it is none of our business.

    Ours is 0.3956/1.3389, upstream's 0.4647/0.2454 -- a different bromide
    parameter set. Adopting it would change every halogenated ligand, and
    nothing about nucleic acids requires it.
    """
    base, upstream = pair
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts

    merged = parse_sections((tmp_path / "merged" / "ffnonbonded.itp").read_text())
    assert merged["atomtypes"][("Br",)][0][-2:] == ["0.395559030854", "1.338880000000"]
    assert "Br" in report.rejected.get("atomtypes", [])


def test_truncated_precision_is_not_mistaken_for_a_different_parameter(tmp_path, pair):
    """Upstream writes ``3.39967e-01`` where the base writes ``0.339966950842``."""
    base, upstream = pair
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts
    assert "CT" not in report.overridden.get("atomtypes", [])
    assert "N" not in report.overridden.get("atomtypes", [])


def test_a_residue_name_that_means_two_things_is_resolved_towards_the_nucleic_acid(tmp_path, pair):
    """``RA`` is radium in ff14SB and adenosine in ``rna.rtp``.

    ``pdb2gmx`` refuses a force field that defines a residue twice. The
    nucleic residue wins against a monatomic ion, because this force field is
    for nucleic-acid systems and nobody is simulating radium.
    """
    base, upstream = pair
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts
    assert "aminoacids.rtp:RA" in report.residues_dropped

    residues = residue_atom_types((tmp_path / "merged" / "aminoacids.rtp").read_text())
    assert "RA" not in residues
    assert "ALA" in residues, "the collision resolver removed more than it should"


def test_a_collision_with_a_real_molecule_stops_the_merge(tmp_path, pair):
    """Only a monatomic ion loses by preference. Anything else is a decision."""
    base, upstream = pair
    (base / "aminoacids.rtp").write_text(
        (base / "aminoacids.rtp")
        .read_text()
        .replace(
            "[ RA ]\n [ atoms ]\n   RA  Br   2.0000  1\n",
            "[ RA ]\n [ atoms ]\n   N   N  -0.4  1\n   CA  CT  0.1  2\n",
        )
    )
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")

    assert not report.ok
    assert any("RA" in conflict for conflict in report.conflicts)


def test_an_adopted_atom_type_reaches_the_database_pdb2gmx_reads(tmp_path, pair):
    """``pdb2gmx`` resolves ``.rtp`` types against ``atomtypes.atp``, not the .itp.

    Adding a type to one and not the other builds a force field ``grompp``
    accepts and ``pdb2gmx`` rejects -- which is what the first version of this
    merge produced.
    """
    base, upstream = pair
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert report.ok, report.conflicts

    declared = {
        line.split()[0]
        for line in (tmp_path / "merged" / "atomtypes.atp").read_text().splitlines()
        if line.split(";")[0].strip()
    }
    for name in report.added.get("atomtypes", []):
        assert name in declared, f"{name} would be invisible to pdb2gmx"


def test_the_base_is_copied_before_anything_is_changed(tmp_path, pair):
    """Anything the merge does not name must survive byte-identical."""
    base, upstream = pair
    (base / "opc.itp").write_text("; a file the merge knows nothing about\n")
    merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert (tmp_path / "merged" / "opc.itp").read_text() == (base / "opc.itp").read_text()


def test_an_upstream_without_nucleic_files_is_refused(tmp_path, pair):
    base, upstream = pair
    (upstream / "dna.rtp").unlink()
    report = merge_nucleic_parameters(base, upstream, tmp_path / "merged")
    assert not report.ok
    assert any("dna.rtp" in conflict for conflict in report.conflicts)
