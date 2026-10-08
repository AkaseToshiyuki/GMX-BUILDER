#!/usr/bin/env python3
"""Check the assembled nucleic-acid force field against the published one.

The suite proves the merge did not disturb the protein half, using only files
this repository ships. It cannot prove the other half -- that the nucleic
acids came across *faithfully* -- because that needs the upstream archive,
which is deliberately not redistributed.

So this is a release step rather than a test. It downloads the pinned archive,
merges it exactly as the installer does, builds the same nucleic acid with
both force fields, and compares every energy term.

    python scripts/verify_nucleic_forcefield.py

Two differences are expected and are reported as such rather than hidden:

* **Impropers.** This project's ff14SB enables
  ``_FF_AMBER_LEAP_ATOM_REORDERING``, matching how Amber orders improper
  dihedrals. The upstream port is from 2019 and predates it. Removing the
  define reproduces the upstream value exactly, which is how the difference
  was attributed rather than assumed.
* **1-4 terms**, by around 1e-5 relative. The bundled files carry more
  significant figures than the upstream port (``fudgeQQ`` 0.83333333333333333
  against 0.8333), so ours is the more precise of the two.

Everything else must agree exactly. Exit status is 0 when it does.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: Terms that must match to the last digit GROMACS prints.
EXACT_TERMS = ("Bond", "Angle", "Proper Dih.", "Coulomb (SR)")

#: Terms whose difference is explained above; the bound is generous enough to
#: pass and tight enough that a real parameter error would break it.
TOLERATED = {"LJ-14": 1e-4, "Coulomb-14": 1e-4, "Potential": 1e-4, "LJ (SR)": 1e-4}

#: The improper difference scales with how distorted the structure is, so it
#: is bounded relatively. Measured on a tleap-built hexanucleotide: 0.004
#: kJ/mol out of 0.217, or 2%. The bound is set an order of magnitude above
#: that -- large enough not to be a tripwire for geometry, small enough that a
#: wrong improper *parameter* (which would move the term by tens of percent)
#: cannot pass.
IMPROPER_RELATIVE_BOUND = 0.05

SEQUENCE = ["DA5", "DT", "DG", "DC", "DA", "DT3"]


def _run(command: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{command[1] if len(command) > 1 else command[0]} failed:\n{result.stderr}"
        )
    return result


def _energies(gmx: str, work: Path, tag: str) -> dict[str, float]:
    out = _run(
        [gmx, "energy", "-f", f"{tag}.edr", "-o", f"{tag}.xvg"],
        work,
        "1\n2\n3\n4\n5\n6\n7\n8\n9\n\n",
    )
    values: dict[str, float] = {}
    for line in out.stdout.splitlines():
        match = re.match(r"^(\S[\S ]{0,20}?)\s{2,}(-?\d+\.?\d*(?:e[+-]?\d+)?)\s", line)
        if match and match.group(1).strip() not in values:
            values[match.group(1).strip()] = float(match.group(2))
    return values


def _build_and_measure(
    gmx: str, work: Path, forcefield: Path, tag: str, pdb: Path
) -> dict[str, float]:
    shutil.copytree(forcefield, work / f"{tag}.ff", dirs_exist_ok=True)
    _run(
        [
            gmx,
            "pdb2gmx",
            "-f",
            str(pdb),
            "-ff",
            tag,
            "-water",
            "tip3p",
            "-o",
            f"{tag}.gro",
            "-p",
            f"{tag}.top",
            "-i",
            f"{tag}_p.itp",
            "-missing",
        ],
        work,
    )
    (work / "sp.mdp").write_text(
        "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\nnstlist = 1\n"
        "rlist = 1.2\nrvdw = 1.2\nrcoulomb = 1.2\ncoulombtype = Cut-off\n"
        "vdwtype = Cut-off\npbc = xyz\nnstcalcenergy = 1\nnstenergy = 1\ncontinuation = yes\n"
    )
    _run(
        [gmx, "editconf", "-f", f"{tag}.gro", "-o", f"{tag}_b.gro", "-d", "2.0", "-bt", "cubic"],
        work,
    )
    _run(
        [
            gmx,
            "grompp",
            "-f",
            "sp.mdp",
            "-c",
            f"{tag}_b.gro",
            "-p",
            f"{tag}.top",
            "-o",
            f"{tag}.tpr",
            "-po",
            f"{tag}_o.mdp",
            "-maxwarn",
            "5",
        ],
        work,
    )
    _run(
        [
            gmx,
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
        work,
    )
    return _energies(gmx, work, tag)


def missing_energy_terms(theirs: dict, ours: dict) -> list[str]:
    """Missing values are a failed comparison, never an empty successful intersection."""
    required = set(EXACT_TERMS) | set(TOLERATED)
    failures = [
        f"{side}: missing {term}"
        for side, values in (("upstream", theirs), ("merged", ours))
        for term in sorted(required - values.keys())
    ]
    improper_theirs = {term for term in theirs if term.startswith(("Per.", "Improper"))}
    improper_ours = {term for term in ours if term.startswith(("Per.", "Improper"))}
    if not improper_theirs or improper_theirs != improper_ours:
        failures.append("missing or unmatched improper energy terms")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmx", default=None, help="GROMACS 2026+ executable")
    arguments = parser.parse_args()

    from install_external_assets import _archive_mode, download, sha256

    from gmxbuilder.modules.forcefield.catalog import detect_gromacs_version
    from gmxbuilder.modules.forcefield.nucleic_merge import merge_nucleic_parameters
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    gmx = arguments.gmx or find_gromacs_executable()
    if not gmx:
        raise SystemExit("no GROMACS executable found; pass --gmx")
    version = detect_gromacs_version(gmx)
    if version is None or version < (2026, 0):
        raise SystemExit(f"this force field needs GROMACS 2026 or later; {gmx} is {version}")

    manifest = json.loads((ROOT / "scripts" / "external_assets.json").read_text())
    spec = next(a for a in manifest["assets"] if a.get("merge_base"))
    base = ROOT / "src" / "gmxbuilder" / "data" / "forcefields" / spec["merge_base"]

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-ffverify-") as temporary:
        work = Path(temporary)
        archive = work / "asset.tar"
        print(f"downloading {spec['name']}")
        download(spec["url"], archive)
        if sha256(archive) != spec["sha256"]:
            raise SystemExit("SHA-256 verification failed; the pinned source has changed")

        import tarfile

        extract = work / "extract"
        extract.mkdir()
        with tarfile.open(archive, mode=_archive_mode(archive)) as handle:
            handle.extractall(extract, filter="data")
        upstream = extract / spec["archive_root"]

        report = merge_nucleic_parameters(base, upstream, work / "merged")
        print(report.summary())
        if not report.ok:
            raise SystemExit("the merge reported conflicts; nothing was compared")

        # The upstream port cannot resolve the base's ninth bondedtypes column,
        # so it is compared exactly as published, against our merged result.
        pdb = work / "nucleic.pdb"
        if not _write_sequence_with_tleap(pdb, work):
            _write_sequence(pdb, upstream)
            print(
                "\nNOTE: AmberTools was not found, so the structure is a synthetic\n"
                "      placeholder. Term-by-term agreement still means what it says,\n"
                "      but the absolute energies are not physical."
            )

        print(f"\nbuilding {'-'.join(SEQUENCE)} with both force fields")
        theirs = _build_and_measure(gmx, work, upstream, "upstream", pdb)
        ours = _build_and_measure(gmx, work, work / "merged", "merged", pdb)

    print(f"\n{'term':16s} {'upstream':>15s} {'merged':>15s} {'difference':>14s}  verdict")
    failures = missing_energy_terms(theirs, ours)
    for term in sorted(set(theirs) & set(ours)):
        difference = ours[term] - theirs[term]
        scale = max(1.0, abs(theirs[term]))
        if term in EXACT_TERMS:
            verdict = "exact" if difference == 0.0 else "DIFFERS"
        elif term.startswith(("Per.", "Improper")):
            verdict = (
                "improper convention"
                if abs(difference) / scale <= IMPROPER_RELATIVE_BOUND
                else "DIFFERS"
            )
        elif term in TOLERATED:
            verdict = "precision" if abs(difference) / scale <= TOLERATED[term] else "DIFFERS"
        else:
            continue
        if verdict == "DIFFERS":
            failures.append(term)
        print(f"{term:16s} {theirs[term]:15.5f} {ours[term]:15.5f} {difference:14.6f}  {verdict}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)} differ beyond what the two conventions explain")
        return 1
    print("\nPASS: the nucleic-acid parameters came across faithfully")
    return 0


def _write_sequence_with_tleap(path: Path, work: Path) -> bool:
    """Build the test strand with the AmberTools the installer already provides.

    A real geometry matters here. On a scattered placeholder the bond term
    reaches 1e7 kJ/mol and the improper term 463; both force fields agree on
    those numbers, so the comparison is still valid, but nothing about the
    magnitudes is interpretable and the tolerances have nothing to anchor to.
    """
    from gmxbuilder.modules.forcefield.gaff_backend import gaff_environment_path

    tleap = gaff_environment_path() / "bin" / "tleap"
    if not tleap.is_file():
        return False
    script = work / "build.leap"
    script.write_text(
        "source leaprc.DNA.OL15\n"
        f"dna = sequence {{ {' '.join(SEQUENCE)} }}\n"
        f"savepdb dna {path}\n"
        "quit\n"
    )
    result = subprocess.run(
        [str(tleap), "-f", str(script)], cwd=work, capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0 and path.is_file()


def _write_sequence(path: Path, forcefield: Path) -> None:
    import math

    from gmxbuilder.modules.forcefield.nucleic_merge import residue_atom_types

    text = (forcefield / "dna.rtp").read_text()
    assert residue_atom_types(text)
    names: dict[str, list[str]] = {}
    current, section = None, None
    for raw in text.splitlines():
        line = raw.split(";")[0].rstrip()
        header = re.match(r"^\s*\[\s*(\S+)\s*\]", line)
        if header:
            key = header.group(1)
            if key in {"atoms", "bonds", "impropers", "dihedrals", "bondedtypes"}:
                section = key
            else:
                current, section = key, None
                names[current] = []
            continue
        if section == "atoms" and current and line.strip():
            names[current].append(line.split()[0])

    lines, serial = [], 1
    for index, residue in enumerate(SEQUENCE, start=1):
        for atom in names[residue]:
            name = atom if len(atom) >= 4 else f" {atom:<3s}"
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} {residue:>3s} A{index:4d}    "
                f"{1.2 * serial:8.3f}{1.5 * math.sin(serial):8.3f}"
                f"{1.5 * math.cos(serial * 0.7):8.3f}  1.00  0.00"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nTER\nEND\n")


if __name__ == "__main__":
    raise SystemExit(main())
