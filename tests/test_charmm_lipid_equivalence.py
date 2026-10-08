"""CHARMM36 may reuse a CHARMM36m lipid entry only where they really agree."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gmxbuilder.modules.forcefield.charmm_lipid_equivalence import (
    equivalent_lipid_definition,
)
from tests.dry_initial_fixture import dry_initial_fixture
from tests.prerequisites import requires_forcefield, requires_gromacs

pytestmark = pytest.mark.slow


def _needs_gromacs():
    """The comparison asks grompp what it resolved, so it needs GROMACS."""
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    if find_gromacs_executable() is None:
        pytest.skip("GROMACS is not installed")


def _mirror_force_field(source: Path, target: Path, replacements: dict[str, str]) -> Path:
    """A force field identical to another but for the named files.

    Symlinked rather than copied: charmm36m is hundreds of megabytes, and an
    include inside forcefield.itp resolves against the directory holding it, so
    a directory of links behaves exactly like the original.
    """
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name in replacements:
            (target / entry.name).write_text(replacements[entry.name])
        else:
            os.symlink(entry, target / entry.name)
    return target


def _use_tampered_charmm36m(monkeypatch, directory: Path) -> None:
    from gmxbuilder.modules.forcefield import catalog

    real = catalog.force_field_directory

    def redirected(force_field: str):
        if str(force_field).lower() == "charmm36m":
            return directory
        return real(force_field)

    monkeypatch.setattr(catalog, "force_field_directory", redirected)
    equivalent_lipid_definition.cache_clear()


def test_the_common_lipids_are_the_same_under_both_releases():
    """CHARMM36m refined the protein force field, not the lipids.

    If this stops holding, the library must go back to simulating both, so it
    is asserted rather than assumed -- around sixty entries at two GPU-hours
    each rest on it.
    """
    _needs_gromacs()
    for lipid in ("POPC", "DPPC", "DOPC", "POPE", "POPG", "DMPC"):
        equivalent, reason = equivalent_lipid_definition(lipid)
        assert equivalent, f"{lipid}: {reason}"


@pytest.mark.parametrize("lipid", ["SOPS", "SAPE", "PAPI"])
def test_lipids_the_two_releases_disagree_about_are_not_reused(lipid):
    """These differ in atoms, charges or impropers and must be built twice."""
    equivalent, reason = equivalent_lipid_definition(lipid)
    assert not equivalent
    assert reason


@pytest.mark.parametrize("lipid", ["CHOL", "CAMP", "SITO"])
def test_a_one_four_interaction_resolved_differently_is_not_reused(lipid):
    """The sterols' 1-4 interactions are not the same under the two releases.

    charmm36m gives the HAL1-HGA1 pair an explicit ``[ pairtypes ]`` entry, so
    its 1-4 term carries the NBFIX-corrected value; charmm36 has no such entry
    and the combination rule fills one in from the plain Lennard-Jones
    parameters instead. Comparing the bonded tables alone never saw this --
    1-4 interactions are not in them.
    """
    _needs_gromacs()
    equivalent, reason = equivalent_lipid_definition(lipid)
    assert not equivalent
    assert "LJ-14" in reason, reason


@pytest.mark.parametrize("lipid", ["POP2", "POP3"])
def test_an_ion_lipid_nbfix_only_one_release_carries_is_not_reused(lipid):
    """Sodium against the phosphoinositide phosphate oxygen.

    charmm36m carries an NBFIX for SOD against OC2DP and charmm36 does not, so
    with 0.15 M NaCl in the box the two releases do not simulate the same
    system -- and the interaction they differ over is the one that governs how
    these headgroups behave. It lives in ``nbfix.itp``, which the comparison
    that read parameter files never opened.
    """
    _needs_gromacs()
    equivalent, reason = equivalent_lipid_definition(lipid)
    assert not equivalent
    assert "SOD" in reason and "OC2DP" in reason, reason


def test_an_accepted_pairing_says_what_it_compared():
    _needs_gromacs()
    equivalent, reason = equivalent_lipid_definition("POPC")
    assert equivalent
    assert "interactions" in reason and "Lennard-Jones" in reason, reason


def test_a_later_periodicity_term_cannot_hide(tmp_path, monkeypatch):
    """A dihedral's second term is as much a parameter as its first.

    The comparison that read the files kept only the first line for each set of
    atom types, so 13% of the terms these lipids need were never compared at
    all. Tampering with the *second* periodicity term of a torsion POPC's own
    topology uses is the exact defect, and it must not survive.
    """
    _needs_gromacs()
    from gmxbuilder.modules.forcefield.catalog import force_field_directory

    root = force_field_directory("charmm36m")
    quadruple = ("OSLP", "CTL2", "CTL1", "OSL")
    lines = (root / "ffbonded.itp").read_text().splitlines()
    seen = 0
    for position, line in enumerate(lines):
        fields = line.partition(";")[0].split()
        if len(fields) >= 8 and tuple(fields[:4]) == quadruple:
            seen += 1
            if seen == 2:  # the term the old comparison threw away
                fields[6] = f"{float(fields[6]) * 2.0:.6f}"
                lines[position] = " ".join(fields)
                break
    assert seen == 2, "expected this torsion to have more than one periodicity term"

    tampered = _mirror_force_field(
        root, tmp_path / "charmm36m", {"ffbonded.itp": "\n".join(lines) + "\n"}
    )
    _use_tampered_charmm36m(monkeypatch, tampered)
    try:
        equivalent, reason = equivalent_lipid_definition("POPC")
        assert not equivalent, "doubling a later periodicity term must be seen"
        assert "Proper Dih." in reason, reason
    finally:
        equivalent_lipid_definition.cache_clear()


def test_a_changed_nbfix_cannot_hide(tmp_path, monkeypatch):
    """The same, for the off-diagonal Lennard-Jones corrections."""
    _needs_gromacs()
    from gmxbuilder.modules.forcefield.catalog import force_field_directory

    root = force_field_directory("charmm36m")
    lines = (root / "nbfix.itp").read_text().splitlines()
    changed = False
    for position, line in enumerate(lines):
        fields = line.partition(";")[0].split()
        if len(fields) >= 5 and set(fields[:2]) == {"SOD", "OCL"}:
            fields[4] = f"{float(fields[4]) * 1.5:.9f}"
            lines[position] = " ".join(fields)
            changed = True
            break
    assert changed, "expected a sodium NBFIX to tamper with"

    tampered = _mirror_force_field(
        root, tmp_path / "charmm36m", {"nbfix.itp": "\n".join(lines) + "\n"}
    )
    _use_tampered_charmm36m(monkeypatch, tampered)
    try:
        equivalent, reason = equivalent_lipid_definition("POPS")
        assert not equivalent, "an ion-lipid NBFIX is part of what makes two systems the same"
        assert "SOD" in reason, reason
    finally:
        equivalent_lipid_definition.cache_clear()


def test_an_unresolvable_lipid_is_never_reused():
    """The oxysterols have no CHARMM36 template at all."""
    equivalent, reason = equivalent_lipid_definition("22RHC")
    assert not equivalent
    assert "charmm36" in reason


@requires_forcefield("charmm36")
@requires_forcefield("charmm36m")
def test_converter_rounding_is_not_mistaken_for_a_reparameterisation():
    """The two installations print five and six decimals of the same numbers.

    Comparing the text called all 43 atom types this project's lipids use
    different, on 0.23430 against 0.2343040. Numerically the worst disagreement
    among them is 4e-5 -- the fifth decimal place, not a reparameterisation.

    The tolerance is not vacuous: across every atom type the two releases share,
    including the protein ones CHARMM36m actually retuned, the worst
    disagreement is larger than a third of the value.
    """
    from gmxbuilder.modules.forcefield.charmm_lipid_equivalence import (
        CONVERTER_ROUNDING,
        _nonbonded,
        _residues,
    )
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_name
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    old, new = _nonbonded("charmm36"), _nonbonded("charmm36m")
    residues = _residues("charmm36m")
    lipid_types = set()
    for name in LipidRegistry.list_builtin():
        record = residues.get(lipid_rtp_name(name, "charmm36m").upper())
        if record:
            lipid_types |= {str(a[1]) for a in record["atoms"]}
    lipid_types &= set(old) & set(new)
    assert len(lipid_types) > 30, "expected the lipid atom types to be found"

    def worst_over(types):
        return max(
            abs(a - b) / max(abs(b), 1e-12)
            for atom_type in types
            for a, b in zip(old[atom_type], new[atom_type], strict=True)
            if b
        )

    assert worst_over(lipid_types) < CONVERTER_ROUNDING, "lipid parameters should agree"
    assert worst_over(set(old) & set(new)) > CONVERTER_ROUNDING, (
        "the tolerance must be tight enough to still see a real difference"
    )


def _write_conformers(directory: Path, count: int = 2) -> None:
    """Real conformer files: a named placeholder is no longer a conformer."""
    for number in range(count):
        np.savez(
            directory / f"conf_{number:04d}.npz",
            coords=np.arange(30, dtype=float).reshape(10, 3),
            atom_names=np.asarray([f"C{index}" for index in range(10)]),
        )


@requires_forcefield("charmm36")
@requires_forcefield("charmm36m")
@requires_gromacs
def test_the_queue_reuses_rather_than_repeats(tmp_path):
    """A CHARMM36 entry is published from the CHARMM36m one, and says so."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = {
        "family": "charmm36-lipid",
        "lipid": "POPC",
        "force_field": "charmm36",
        "lipid_ff": "charmm36",
    }
    source = module.reuse_source(entry)
    assert source is not None and source["family"] == "charmm36m-lipid"

    origin = tmp_path / "replica-1" / "charmm36m-lipid" / "POPC"
    origin.mkdir(parents=True)
    from gmxbuilder.modules.membrane.area_observable import AreaMeasurement
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature

    finished = {
        "status": "ready",
        "test_mode": False,
        "quality": {"passed": True, "initial_water_exclusion": dry_initial_fixture()},
        "lipid_name": "POPC",
        "force_field": "charmm36m",
        "lipid_ff": "charmm36m",
        "parameter_family": "charmm36m-lipid",
        "canonical_smiles": "CCO",
        "topology_sha256": topology_signature(
            [f"C{i}" for i in range(10)], "charmm36m", "charmm36m"
        ),
        "temperature_K": 310,
        "npt_ps": 50000,
        "atom_names": [f"C{i}" for i in range(10)],
        "observables": AreaMeasurement(
            mean_nm2=0.643,
            standard_error_nm2=0.002,
            n_frames=10000,
            n_blocks=30,
            analysis_window_ps=(10000, 50000),
            equilibration_ps=10000,
            retained_fraction=0.8,
            relative_half_drift=0.002,
            drift_sigma=0.5,
            autocorrelation_ps=1000,
            effective_samples=30,
            converged=True,
            rejection=None,
        ).as_metadata(),
    }
    for r in (1, 2):
        directory = tmp_path / f"replica-{r}" / "charmm36m-lipid" / "POPC"
        directory.mkdir(parents=True, exist_ok=True)
        from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

        finished["parameter_fingerprint"] = fingerprint_from_metadata(finished)
        (directory / "metadata.json").write_text(json.dumps(finished))
        _write_conformers(directory, 1)
    assert module.publish_reuse(tmp_path, entry, source, replicas=2)
    published = json.loads(
        (tmp_path / "replica-1" / "charmm36-lipid" / "POPC" / "metadata.json").read_text()
    )
    assert published["force_field"] == "charmm36"
    assert published["reused_from"]["family"] == "charmm36m-lipid"
    assert (tmp_path / "replica-1" / "charmm36-lipid" / "POPC" / "conf_0000.npz").is_file()


def test_a_lipid_the_releases_disagree_about_is_not_reused_by_the_queue():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.reuse_source(
            {
                "family": "charmm36-lipid",
                "lipid": "SOPS",
                "force_field": "charmm36",
                "lipid_ff": "charmm36",
            }
        )
        is None
    )
    assert (
        module.reuse_source(
            {
                "family": "amber-gaff2",
                "lipid": "POPC",
                "force_field": "amber14sb",
                "lipid_ff": "gaff2",
            }
        )
        is None
    )


def test_reuse_refuses_a_source_that_is_not_a_finished_entry(tmp_path):
    """A metadata file is not an entry.

    A failed build, an interrupted write, or a V3 entry unpacked from the
    prebuilt archive all leave one behind, and copying any of those would mark
    the pairing done without a thing having been simulated.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = {
        "family": "charmm36-lipid",
        "lipid": "POPC",
        "force_field": "charmm36",
        "lipid_ff": "charmm36",
    }
    source = {**entry, "family": "charmm36m-lipid", "force_field": "charmm36m"}
    origin = tmp_path / "replica-1" / "charmm36m-lipid" / "POPC"
    origin.mkdir(parents=True)
    target = tmp_path / "replica-1" / "charmm36-lipid" / "POPC"

    for description, metadata, conformers in (
        ("a failed build", {"status": "failed", "observables": {}}, "real"),
        ("no observables", {"status": "ready"}, "real"),
        (
            "a non-finite area",
            {
                "status": "ready",
                "observables": {
                    "area_per_lipid_nm2": {"mean": float("nan"), "effective_samples": 12.0}
                },
            },
            "real",
        ),
        (
            "no conformers",
            {
                "status": "ready",
                "observables": {"area_per_lipid_nm2": {"mean": 0.64, "effective_samples": 12.0}},
            },
            "none",
        ),
        # An area is a number with evidence under it or it is nothing: an entry
        # that reached the end of the length ladder with no independent samples
        # would otherwise be marked done and never run again.
        (
            "no independent samples",
            {
                "status": "ready",
                "observables": {"area_per_lipid_nm2": {"mean": 0.64, "effective_samples": 0.0}},
            },
            "real",
        ),
        # A file called conf_0000.npz is not a conformer. This one fails at
        # np.load, which without the check happens in a user's build instead.
        (
            "an unreadable conformer",
            {
                "status": "ready",
                "observables": {"area_per_lipid_nm2": {"mean": 0.64, "effective_samples": 12.0}},
            },
            "broken",
        ),
    ):
        (origin / "metadata.json").write_text(json.dumps(metadata))
        for stale in origin.glob("conf_*.npz"):
            stale.unlink()
        if conformers == "real":
            _write_conformers(origin, 1)
        elif conformers == "broken":
            (origin / "conf_0000.npz").write_bytes(b"placeholder")
        assert not module.publish_reuse(tmp_path, entry, source, replicas=1), description
        assert not target.exists(), f"{description} must leave no target behind"


def test_a_failed_replica_is_not_counted_as_built(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = {
        "family": "amber-gaff2",
        "lipid": "DPPC",
        "force_field": "amber14sb",
        "lipid_ff": "gaff2",
    }
    directory = tmp_path / "replica-1" / "amber-gaff2" / "DPPC"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"status": "failed", "observables": {}}))
    assert not module.entry_done(tmp_path, entry, replicas=1), (
        "a failed entry counted as built would be skipped forever by a resumed queue"
    )


@pytest.fixture(autouse=True)
def _storage_parameter_sources(parameter_sources):
    return parameter_sources
