"""Fresh-installer names must preserve independently audited parameter identity."""

import hashlib
import json
from pathlib import Path

import pytest

from gmxbuilder.modules.membrane.parameter_provenance import (
    canonical_forcefield_files,
    installed_forcefield_matches,
)

DATA = Path(__file__).parents[1] / "src/gmxbuilder/data"


def reviewed_layout(ff):
    # The independent, earlier CHARMM reuse audit pins the original file set.
    prior = json.loads((DATA / "v4_charmm_reuse.json").read_text())["parameter_files"][ff]
    installed = dict(prior)
    installed.pop("charmm36.tgz", None)
    for path in (DATA / "forcefield_overlays" / ff).glob("*.itp"):
        if path.name.startswith("gmxbuilder-"):
            del installed[path.name.removeprefix("gmxbuilder-")]
        installed[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return prior, installed


@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
def test_installer_inputs_preserve_the_independent_audit(ff):
    prior, installed = reviewed_layout(ff)
    prior.pop("charmm36.tgz", None)
    assert canonical_forcefield_files(ff, installed) == prior


@pytest.mark.parametrize("ff", ["charmm36", "charmm36m"])
@pytest.mark.parametrize("damage", ["entrypoint", "supplement", "missing", "duplicate", "extra"])
def test_changed_or_ambiguous_installer_inputs_never_inherit_old_identity(ff, damage):
    prior, installed = reviewed_layout(ff)
    prior.pop("charmm36.tgz", None)
    supplement = "gmxbuilder-plasmalogen-bonded.itp"
    if damage == "entrypoint":
        installed["forcefield.itp"] = "0" * 64
    elif damage == "supplement":
        installed[supplement] = "0" * 64
    elif damage == "missing":
        del installed[supplement]
    elif damage == "duplicate":
        installed["plasmalogen-bonded.itp"] = installed[supplement]
    else:
        installed["unreviewed.itp"] = "0" * 64
    assert canonical_forcefield_files(ff, installed) != prior


def test_layout_compatibility_does_not_apply_to_other_force_fields():
    _, installed = reviewed_layout("charmm36")
    assert canonical_forcefield_files("amber14sb", installed) == installed


def test_reuse_still_checks_every_unmapped_file(tmp_path):
    (tmp_path / "parameter.itp").write_text("reviewed coefficients")
    expected = {"parameter.itp": hashlib.sha256(b"reviewed coefficients").hexdigest()}
    assert installed_forcefield_matches("charmm36", tmp_path, expected)
    (tmp_path / "parameter.itp").write_text("changed coefficients")
    assert not installed_forcefield_matches("charmm36", tmp_path, expected)
    (tmp_path / "parameter.itp").write_text("reviewed coefficients")
    (tmp_path / "unknown.itp").write_text("unexpected")
    assert not installed_forcefield_matches("charmm36", tmp_path, expected)


def test_only_the_pinned_unused_download_can_be_absent(tmp_path):
    prior, _ = reviewed_layout("charmm36")
    expected = {"charmm36.tgz": prior["charmm36.tgz"]}
    assert installed_forcefield_matches("charmm36", tmp_path, expected)
    assert not installed_forcefield_matches("charmm36", tmp_path, {"charmm36.tgz": "0" * 64})
    (tmp_path / "charmm36.tgz").write_bytes(b"changed download")
    assert not installed_forcefield_matches("charmm36", tmp_path, expected)
