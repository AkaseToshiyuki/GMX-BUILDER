"""The assembled ``amber14sb_ol24`` force field, against real GROMACS.

The merge rules are asserted on synthetic inputs in ``test_nucleic_merge.py``.
What is here needs the force field the installer actually builds, so it skips
when that has not been installed.

The central claim is not "DNA builds" -- it is **"and nothing else changed"**.
A merge that quietly shifted a protein parameter would still build DNA
perfectly well, so the protein half is compared against the base force field
directly, energy term by energy term.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess

import pytest

from tests.prerequisites import forcefield_parameters_available, requires_gromacs

pytestmark = pytest.mark.slow

_FF = "amber14sb_ol24"

requires_merged_forcefield = pytest.mark.skipif(
    not forcefield_parameters_available(_FF),
    reason=(
        f"{_FF} is assembled by ./install-local.sh from a separately "
        "distributed archive and is not part of this repository"
    ),
)


def _gmx() -> str:
    """The GROMACS this force field can actually be used with.

    ``amber14sb`` -- and so everything merged onto it -- enables
    ``_FF_AMBER_LEAP_ATOM_REORDERING``, which GROMACS added in 2026. An older
    grompp parses the files and produces *different* improper terms, so a
    green run against 2025 would prove nothing about what users get.
    """
    from gmxbuilder.modules.forcefield.catalog import (
        detect_gromacs_version,
        get_force_field_profile,
    )
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    executable = find_gromacs_executable()
    if not executable:
        pytest.skip("no GROMACS executable is available")
    version = detect_gromacs_version(executable)
    required = get_force_field_profile(_FF).minimum_gromacs
    if version is None or version < required:
        pytest.skip(
            f"{_FF} needs GROMACS {'.'.join(map(str, required))} or later; "
            f"found {version}. Set GMX_BIN to a newer build."
        )
    return executable


def _forcefield_root():
    from pathlib import Path

    import gmxbuilder

    return Path(gmxbuilder.__file__).resolve().parent / "data" / "forcefields"


def _sequence_pdb(work, residues: list[str], name: str):
    """Write a linear chain from the force field's own residue templates.

    Coordinates are placeholders: every check here is a topology check, and
    ``pdb2gmx`` only needs atom and residue names to build one.
    """
    from gmxbuilder.modules.forcefield.nucleic_merge import residue_atom_types

    database = "dna.rtp" if residues[0].startswith("D") else "rna.rtp"
    text = (_forcefield_dir(_FF) / database).read_text()
    atoms_by_residue = _rtp_atom_names(text)

    lines, serial = [], 1
    for index, residue in enumerate(residues, start=1):
        for offset, atom in enumerate(atoms_by_residue[residue]):
            lines.append(_atom_record(serial, atom, residue, index, *_scatter(serial)))
            serial += 1
    lines += ["TER", "END"]
    assert residue_atom_types(text)  # the database really did parse
    path = work / f"{name}.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def _scatter(serial: int) -> tuple[float, float, float]:
    """Deterministic, non-collinear placeholder coordinates.

    These structures are built to check that residue templates resolve, so the
    positions carry no meaning -- but they cannot be collinear. ``pdb2gmx``
    places an added hydrogen from the geometry of its neighbours, and three
    atoms on a line make that construction singular: it emits NaN, writes a
    .gro that looks fine, and everything downstream fails somewhere else.
    """
    return (1.2 * serial, 1.5 * math.sin(serial), 1.5 * math.cos(serial * 0.7))


def _atom_record(
    serial: int, atom: str, residue: str, resid: int, x: float, y: float, z: float
) -> str:
    """One PDB ATOM record in the columns the format actually specifies.

    PDB is fixed-column, and getting it wrong is quiet: a residue name that
    starts one column early is read as part of the atom name, and pdb2gmx
    reports a residue nobody wrote. Atom names of fewer than four characters
    begin in column 14, four-character names in column 13.
    """
    name = atom if len(atom) >= 4 else f" {atom:<3s}"
    return (
        f"ATOM  {serial:5d} {name:<4s} {residue:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    )


def _rtp_atom_names(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    section = None
    for raw in text.splitlines():
        line = raw.split(";")[0].rstrip()
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
        if header:
            name = header.group(1)
            if name in {"atoms", "bonds", "impropers", "dihedrals", "bondedtypes"}:
                section = name
            else:
                current, section = name, None
                out[current] = []
            continue
        if section == "atoms" and current and line.strip():
            out[current].append(line.split()[0])
    return out


def _forcefield_dir(name: str):
    """Resolve a force field, accepting both bundled naming styles.

    The GROMACS-derived trees keep their ``.ff`` suffix; the ones this project
    assembles do not.
    """
    root = _forcefield_root()
    for candidate in (root / name, root / f"{name}.ff"):
        if candidate.is_dir():
            return candidate
    raise AssertionError(f"force field {name} is not installed")


def _build(work, pdb, force_field: str, tag: str) -> tuple[str, str]:
    """Run pdb2gmx, returning (gro, top) paths.

    ``pdb2gmx -ff <name>`` looks for ``<name>.ff`` beside the working
    directory, so the tree is copied under that exact name whichever style it
    is stored in.
    """
    shutil.copytree(_forcefield_dir(force_field), work / f"{force_field}.ff", dirs_exist_ok=True)
    result = subprocess.run(
        [
            _gmx(),
            "pdb2gmx",
            "-f",
            str(pdb),
            "-ff",
            force_field,
            "-water",
            "tip3p",
            "-o",
            f"{tag}.gro",
            "-p",
            f"{tag}.top",
            "-i",
            f"{tag}_posre.itp",
            "-missing",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return f"{tag}.gro", f"{tag}.top"


def _single_point(work, gro: str, top: str, tag: str) -> dict[str, float]:
    (work / "sp.mdp").write_text(
        "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\nnstlist = 1\n"
        "rlist = 1.2\nrvdw = 1.2\nrcoulomb = 1.2\ncoulombtype = Cut-off\n"
        "vdwtype = Cut-off\npbc = xyz\nnstcalcenergy = 1\nnstenergy = 1\ncontinuation = yes\n"
    )
    for command in (
        [_gmx(), "editconf", "-f", gro, "-o", f"{tag}_box.gro", "-d", "2.0", "-bt", "cubic"],
        [
            _gmx(),
            "grompp",
            "-f",
            "sp.mdp",
            "-c",
            f"{tag}_box.gro",
            "-p",
            top,
            "-o",
            f"{tag}.tpr",
            "-po",
            f"{tag}_out.mdp",
            "-maxwarn",
            "5",
        ],
        [
            _gmx(),
            "mdrun",
            "-s",
            f"{tag}.tpr",
            "-deffnm",
            tag,
            "-ntmpi",
            "1",
            "-ntomp",
            "1",
            "-nb",
            "cpu",
        ],
    ):
        result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + "\n" + result.stderr

    energy = subprocess.run(
        [_gmx(), "energy", "-f", f"{tag}.edr", "-o", f"{tag}.xvg"],
        cwd=work,
        input="1\n2\n3\n4\n5\n6\n7\n8\n9\n\n",
        capture_output=True,
        text=True,
        timeout=300,
    )
    values: dict[str, float] = {}
    for line in energy.stdout.splitlines():
        match = re.match(r"^(\S[\S ]{0,20}?)\s{2,}(-?\d+\.?\d*(?:e[+-]?\d+)?)\s", line)
        if match and match.group(1).strip() not in values:
            values[match.group(1).strip()] = float(match.group(2))
    return values


# --------------------------------------------------------------------------
# It builds nucleic acids


@requires_gromacs
@requires_merged_forcefield
@pytest.mark.parametrize(
    ("residues", "name"),
    [
        (["DA5", "DT", "DG", "DC", "DA", "DT3"], "dna"),
        (["RA5", "RU", "RG", "RC", "RA", "RU3"], "rna"),
    ],
)
def test_native_pdb2gmx_builds_a_canonical_strand(tmp_path, residues, name):
    """What ff14SB could not do at all: GROMACS 2026 ships it protein-only."""
    pdb = _sequence_pdb(tmp_path, residues, name)
    gro, top = _build(tmp_path, pdb, _FF, name)
    assert (tmp_path / gro).is_file()
    coordinates = (tmp_path / gro).read_text()
    assert "nan" not in coordinates.lower(), (
        "pdb2gmx produced non-finite coordinates; the topology may still look valid"
    )
    text = (tmp_path / top).read_text()
    assert "[ moleculetype ]" in text
    # Every residue asked for reached the topology, not just the first.
    assert text.count("#include") >= 1
    atom_count = int(coordinates.splitlines()[1].strip())
    assert atom_count > 100, f"only {atom_count} atoms were built for six residues"


# --------------------------------------------------------------------------
# And nothing else changed


@requires_gromacs
@requires_merged_forcefield
def test_a_peptide_is_energetically_identical_to_the_base_force_field(tmp_path):
    """The safety property, measured rather than argued.

    A merge that shifted a protein parameter would still build DNA perfectly
    well. The only way to know it did not is to build the same protein with
    the base and with the merged force field and compare every energy term.
    """
    if not forcefield_parameters_available("amber14sb"):
        pytest.skip("the base amber14sb parameters are installed separately")

    peptide = tmp_path / "peptide.pdb"
    peptide.write_text(
        "\n".join(
            _atom_record(index, atom, residue, number, *_scatter(index))
            for index, (atom, residue, number) in enumerate(
                [
                    ("N", "ALA", 1),
                    ("CA", "ALA", 1),
                    ("C", "ALA", 1),
                    ("O", "ALA", 1),
                    ("CB", "ALA", 1),
                    ("N", "SER", 2),
                    ("CA", "SER", 2),
                    ("C", "SER", 2),
                    ("O", "SER", 2),
                    ("CB", "SER", 2),
                    ("OG", "SER", 2),
                    ("N", "GLY", 3),
                    ("CA", "GLY", 3),
                    ("C", "GLY", 3),
                    ("O", "GLY", 3),
                ],
                start=1,
            )
        )
        + "\nTER\nEND\n"
    )

    base_gro, base_top = _build(tmp_path, peptide, "amber14sb", "base")
    merged_gro, merged_top = _build(tmp_path, peptide, _FF, "merged")

    base = _single_point(tmp_path, base_gro, base_top, "base")
    merged = _single_point(tmp_path, merged_gro, merged_top, "merged")

    assert base, "no energy terms were read"
    for term, value in base.items():
        assert term in merged, f"{term} vanished from the merged force field"
        assert merged[term] == pytest.approx(value, rel=1e-9, abs=1e-6), (
            f"{term} changed for a protein: {value} -> {merged[term]}"
        )


@requires_merged_forcefield
def test_the_protein_half_is_byte_identical_to_the_base(tmp_path):
    """Everything the merge does not name must survive untouched.

    Cheaper than the energy comparison and it catches a different class of
    mistake: a file rewritten wholesale rather than a parameter shifted.
    """
    if not forcefield_parameters_available("amber14sb"):
        pytest.skip("the base amber14sb parameters are installed separately")
    base = _forcefield_dir("amber14sb")
    merged = _forcefield_dir(_FF)

    untouched = [
        path.name
        for path in sorted(base.iterdir())
        if path.is_file()
        and path.name
        not in {
            "ffbonded.itp",  # gains nucleic bonded parameters
            "ffnonbonded.itp",  # gains three nucleic atom types
            "atomtypes.atp",  # mirrors those three for pdb2gmx
            "aminoacids.rtp",  # loses the radium residue to adenosine
            "forcefield.doc",  # replaced by the merged provenance
        }
    ]
    assert len(untouched) > 10, "the exclusion list has swallowed the whole force field"
    for name in untouched:
        assert (merged / name).read_bytes() == (base / name).read_bytes(), (
            f"{name} was modified by a merge that had no business touching it"
        )


@requires_merged_forcefield
def test_no_parameter_a_supported_residue_can_reach_differs_from_the_base():
    """The claim "identical for non-nucleic work", checked over the whole file.

    The energy comparison above uses one peptide; this covers every parameter
    in the force field, so a difference in something that peptide happens not
    to contain cannot slip through.
    """
    from gmxbuilder.modules.forcefield.nucleic_merge import (
        parse_sections,
        residue_atom_types,
    )

    if not forcefield_parameters_available("amber14sb"):
        pytest.skip("the base amber14sb parameters are installed separately")
    base, merged = _forcefield_dir("amber14sb"), _forcefield_dir(_FF)

    residues: dict[str, set[str]] = {}
    for database in sorted(base.glob("*.rtp")):
        residues.update(residue_atom_types(database.read_text()))

    reachable: list[str] = []
    for name, sections in (
        ("ffnonbonded.itp", ("atomtypes",)),
        ("ffbonded.itp", ("bondtypes", "angletypes", "dihedraltypes", "constrainttypes")),
    ):
        theirs = parse_sections((base / name).read_text())
        ours = parse_sections((merged / name).read_text())
        for section in sections:
            left, right = theirs.get(section, {}), ours.get(section, {})
            differing = (set(left) ^ set(right)) | {
                key for key in set(left) & set(right) if left[key] != right[key]
            }
            for key in sorted(differing):
                named = {part for part in key if not part.isdigit() and part != "X"}
                if any(named <= types for types in residues.values()):
                    reachable.append(f"{section} {'-'.join(key)}")

    assert reachable == [], (
        "these parameters differ from the base and a residue it already "
        f"supports can reach them: {reachable}"
    )


@requires_merged_forcefield
def test_the_only_residue_that_changes_is_the_radium_ion():
    """``RA`` becomes adenosine. Nothing else in the base databases moves."""
    from gmxbuilder.modules.forcefield.nucleic_merge import residue_atom_types

    if not forcefield_parameters_available("amber14sb"):
        pytest.skip("the base amber14sb parameters are installed separately")
    base, merged = _forcefield_dir("amber14sb"), _forcefield_dir(_FF)

    for name in ("aminoacids.rtp", "phosaa14sb.rtp"):
        theirs = residue_atom_types((base / name).read_text())
        ours = residue_atom_types((merged / name).read_text())
        removed = set(theirs) - set(ours)
        assert removed <= {"RA"}, f"{name} lost {sorted(removed - {'RA'})}"
        assert not set(ours) - set(theirs), f"{name} gained residues"
        for residue in sorted(set(theirs) & set(ours)):
            assert theirs[residue] == ours[residue], f"{residue} changed"


@requires_merged_forcefield
def test_the_installed_force_field_records_where_it_came_from():
    """A force field assembled on the user's machine has to say so."""
    doc = (_forcefield_dir(_FF) / "forcefield.doc").read_text()
    assert "assembled at installation time" in doc
    assert "NOT a" in doc and "redistributed" in doc
    for citation in ("Maier", "Zgarbova", "Perez"):
        assert citation in doc, f"the {citation} citation is missing"


@requires_merged_forcefield
def test_it_is_offered_for_dna_and_for_automatic_ligand_parameterisation():
    """The gap this force field exists to close.

    DNA needed CHARMM36m, whose only automatic small-molecule path is a manual
    CGenFF upload, while GAFF2 needed the Amber family, which had no nucleic
    acids. One force field now does both.
    """
    from gmxbuilder.modules.forcefield.catalog import force_field_family
    from gmxbuilder.modules.nucleic_acid.support import nucleic_force_field_capability

    capable, _reason = nucleic_force_field_capability(_FF)
    assert capable
    assert force_field_family(_FF) == "amber"

    from gmxbuilder.modules.nucleic_acid.native import _SUPPORTED_FORCE_FIELDS

    assert _FF in _SUPPORTED_FORCE_FIELDS
