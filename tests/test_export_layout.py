"""The sorted package: where files land, and what the root is allowed to hold.

The package used to be flat -- a real membrane build put about forty files in
one directory. Sorting it moved every path that GROMACS is handed, so these
assert the two things that can now silently break: that the root stays small,
and that ``#include`` still resolves through the nesting.

The GROMACS half of that is checked for real in
``test_gromacs_smoke.py::test_the_sorted_package_still_builds_a_tpr``. What is
here is the part that needs no external tool.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules.export.exporter import ExportModule
from gmxbuilder.modules.export.layout import (
    FORCEFIELD_DIR,
    MDP_DIR,
    ROOT_FILES,
    STRUCTURE_DIR,
    TOPOLOGY_DIR,
)
from gmxbuilder.modules.forcefield.assign import ForceFieldAssigner
from gmxbuilder.pipeline.provenance import record_step
from tests.prerequisites import requires_forcefield


def _tiny_system() -> System:
    names = ["N", "CA", "C", "O", "CB"]
    structure = Structure(
        coordinates=np.array(
            [
                [2.000, 2.000, 2.000],
                [2.145, 2.000, 2.000],
                [2.245, 2.100, 2.000],
                [2.225, 2.220, 2.000],
                [2.160, 1.850, 2.000],
            ]
        ),
        box_vectors=np.eye(3) * 5.0,
        atom_names=names,
        resnames=["ALA"] * 5,
        resids=[1] * 5,
        chain_ids=["A"] * 5,
        elements=["N", "C", "C", "O", "C"],
    )
    return System(
        structure=structure,
        components=[Component("PROTEIN_A", ComponentKind.PROTEIN, np.arange(len(names)))],
        metadata={"force_field": "amber14sb", "water_model": "tip3p", "seed": 7},
    )


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One real export, reused: it copies a force field and writes 12 MDPs."""
    from tests.prerequisites import forcefield_parameters_available

    if not forcefield_parameters_available("amber14sb"):
        pytest.skip("amber14sb parameters are installed separately")

    system = _tiny_system()
    record_step(
        system,
        "input",
        {"pdb": str(tmp_path_factory.mktemp("upload") / "absent.pdb"), "_task_dir": "/host/task"},
        {"num_atoms": 5, "box_dimensions_nm": [5.0, 5.0, 5.0]},
        1.25,
    )
    record_step(
        system,
        "forcefield",
        {"name": "amber14sb", "water_model": "tip3p"},
        {"num_atoms": 5, "box_dimensions_nm": [5.0, 5.0, 5.0]},
        2.5,
    )
    assigned = ForceFieldAssigner().execute(system, {})
    assert assigned.success, assigned.log
    root = tmp_path_factory.mktemp("package")
    result = ExportModule().execute(
        assigned.system, {"output_dir": str(root), "system_name": "layout"}
    )
    assert result.success, result.log
    return root


@requires_forcefield("amber14sb")
def test_the_package_root_holds_only_what_a_person_opens_first(exported):
    """Citations, the record of the build, and the command that runs it.

    Dotfiles are excluded: ``.authoritative-archive`` is the exporter's own
    bookkeeping, it is not in the archive the user downloads, and it is not
    something anyone opens.
    """
    root_files = {
        path.name for path in exported.iterdir() if path.is_file() and not path.name.startswith(".")
    }
    assert root_files == set(ROOT_FILES) | {"layout.zip"}


@requires_forcefield("amber14sb")
@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        (STRUCTURE_DIR, "input.gro"),
        (STRUCTURE_DIR, "index.ndx"),
        (TOPOLOGY_DIR, "topol.top"),
        (f"{TOPOLOGY_DIR}/{FORCEFIELD_DIR}", "forcefield.itp"),
    ],
)
def test_each_kind_of_file_lands_in_its_own_directory(exported, directory, expected):
    assert (exported / directory / expected).is_file()


@requires_forcefield("amber14sb")
def test_no_include_climbs_out_of_the_directory_it_sits_in(exported):
    """A ``..`` in an include is what would break an extracted archive.

    The package is written on one machine and unpacked on another. Every
    include must therefore point downward from the file that contains it, and
    must resolve to a file that is actually in the package.
    """
    root = exported.resolve()
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".top", ".itp"}:
            continue
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped.startswith("#include"):
                continue
            target = stripped.split('"')[1]
            assert not target.startswith(("/", "..")), f"{path.name} includes {target!r}"
            resolved = (path.parent / target).resolve()
            assert root in resolved.parents, f"{path.name} includes outside the package: {target}"
            assert resolved.is_file(), f"{path.name} includes a missing file: {target}"


@requires_forcefield("amber14sb")
def test_the_archive_carries_the_sorted_layout_not_a_flat_copy(exported):
    with zipfile.ZipFile(exported / "layout.zip") as archive:
        members = set(archive.namelist())
    assert {name for name in members if "/" not in name} == set(ROOT_FILES)
    assert f"{STRUCTURE_DIR}/input.gro" in members
    assert f"{TOPOLOGY_DIR}/topol.top" in members
    assert any(name.startswith(f"{MDP_DIR}/") for name in members)


@requires_forcefield("amber14sb")
def test_a_rerun_does_not_leave_a_stale_flat_copy_behind(exported, tmp_path):
    """An export directory is reused, and two topologies is worse than none.

    Before the layout was sorted, ``topol.top`` sat in the root. A rerun that
    left it there would put a stale topology beside the new one with nothing
    on disk to say which ``grompp`` had read.
    """
    import shutil

    from gmxbuilder.modules.export.layout import LEGACY_ROOT_ARTIFACTS

    # On a copy: this deletes the sorted directories, and the export fixture
    # is shared with every other test in this module.
    package = tmp_path / "rerun"
    shutil.copytree(exported, package)
    for name in LEGACY_ROOT_ARTIFACTS:
        (package / name).write_text("stale copy from an older release")
    (package / "forcefield.itp").write_text("stale force field")
    keep = package / "notes"
    keep.mkdir()
    (keep / "mine.txt").write_text("a directory the user added")

    ExportModule._clear_previous_export(package)

    for name in LEGACY_ROOT_ARTIFACTS:
        assert not (package / name).exists(), f"{name} survived the rerun"
    assert not (package / "forcefield.itp").exists()
    assert (keep / "mine.txt").is_file(), "a rerun deleted something the user added"


@requires_forcefield("amber14sb")
def test_the_readme_and_the_manifest_agree_about_the_build(exported):
    manifest = json.loads((exported / "manifest.json").read_text())
    readme = (exported / "README.txt").read_text()
    duration = manifest["provenance"]["build_duration_s"]
    assert duration == pytest.approx(3.75)
    assert f"{duration:.1f} s" in readme
    assert "gmxbuilder build -c manifest.json" in readme
