"""Construction failure recovery and the two-replica production barrier."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.modules.membrane import lipid_equilibration as equil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_v4_library as queue  # noqa: E402


def bare_builder(monkeypatch):
    builder = object.__new__(equil.LipidEquilibrationBuilder)
    builder.v4_protocol = {"test": "protocol"}
    builder.gmx = "not-executed"
    builder.threads = 1
    monkeypatch.setattr(builder, "_entry_lock", lambda *args: nullcontext())
    return builder


@pytest.mark.parametrize("failures", [1, 2])
def test_bounded_recovery_preserves_evidence_and_uses_gentler_geometry(
    tmp_path, monkeypatch, failures
):
    builder = bare_builder(monkeypatch)
    work = tmp_path / "replica-2"
    attempts = []

    def build(*args, **kwargs):
        work.mkdir()
        attempts.append(builder._closure_step_limit)
        (work / "bad.gro").write_text(f"attempt {len(attempts)}")
        if len(attempts) <= failures:
            raise equil.PreparationError("stereochemistry mismatch")
        return work

    monkeypatch.setattr(builder, "_build_once", build)
    if failures == 2:
        with pytest.raises(equil.PreparationError):
            builder.build("POPS", "charmm36m", retain_work=work)
    else:
        assert builder.build("POPS", "charmm36m", retain_work=work) == work
    assert attempts == [0.2, 0.1]
    (archive,) = tmp_path.glob("replica-2.failed-preparation-*")
    assert (archive / "bad.gro").read_text() == "attempt 1"
    assert json.loads((archive / "preparation-failure.json").read_text())["attempt"] == 1
    assert (work / "bad.gro").read_text() == "attempt 2"


def test_production_failure_is_never_rebuilt(tmp_path, monkeypatch):
    builder = bare_builder(monkeypatch)
    work = tmp_path / "replica-1"
    calls = []

    def build(*args, **kwargs):
        work.mkdir()
        (work / "nvt.log").write_text("started")
        calls.append(True)
        raise equil.PreparationError("must not retry past the phase boundary")

    monkeypatch.setattr(builder, "_build_once", build)
    with pytest.raises(RuntimeError, match="after dynamics"):
        builder.build("POPS", "charmm36m", retain_work=work)
    assert len(calls) == 1
    assert (work / "nvt.log").read_text() == "started"
    assert not list(tmp_path.glob("*.failed*"))


def test_identity_changed_closure_is_rolled_back_before_retry(tmp_path, monkeypatch):
    builder = bare_builder(monkeypatch)
    system = System(
        Structure(
            coordinates=np.array([[1.0, 1.0, 3.0], [1.0, -1e-20, 1.0]]),
            box_vectors=np.eye(3) * 5,
            atom_names=["C", "C"],
            resnames=["POPS", "POPS"],
            resids=[1, 2],
            elements=["C", "C"],
        ),
        metadata={"force_field": "charmm36m"},
    )
    monkeypatch.setattr(
        equil,
        "construction_core_gap",
        lambda structure, offsets: float(
            structure.coordinates[0, 2] - structure.coordinates[1, 2] - 1
        ),
    )
    monkeypatch.setattr(builder, "_reimage_bilayer_z", lambda system: None)
    monkeypatch.setattr(builder, "_mdp", lambda *args, **kwargs: "integrator = steep\n")
    seen = []

    def validate(system):
        if system.coordinates[0, 0] == 99:
            raise ValueError("stereochemistry mismatch")

    def run(args, work, **kwargs):
        if args[1] != "mdrun":
            return
        prefix = args[args.index("-deffnm") + 1]
        cycle = int(prefix.rsplit("_", 1)[1])
        structure = GROReader().read(work / f"closed_{cycle:02d}.gro")
        seen.append(structure.coordinates.copy())
        if len(seen) == 1:
            structure.coordinates[0, 0] = 99
        GROWriter.write(structure, work / f"{prefix}.gro")
        (work / f"{prefix}.log").write_text("Potential Energy = -100\nMaximum force = 10\n")

    monkeypatch.setattr(builder, "_validate_membrane_identity", validate)
    monkeypatch.setattr(builder, "_run", run)
    builder._close_leaflet_gap(system, tmp_path, np.array([0, 1, 2]), 1)
    assert len(seen) >= 2
    assert all(xyz[0, 0] == 1 for xyz in seen)
    np.testing.assert_allclose(seen[1][:, 2], [2.95, 1.05])
    assert system.coordinates[0, 0] == 1
    records = [
        json.loads(line) for line in (tmp_path / "closure-attempts.jsonl").read_text().splitlines()
    ]
    assert "chemical identity" in records[0]["rejected"]
    assert any(record.get("accepted") for record in records[1:])


@pytest.mark.parametrize("failure", ["preparation", "fingerprint", None])
def test_pair_never_starts_production_until_both_preparations_pass(tmp_path, monkeypatch, failure):
    entry = dict(
        lipid="POPS", family="charmm36m-lipid", force_field="charmm36m", lipid_ff="charmm36m"
    )
    events = []

    class Child:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

        def wait(self):
            return self.code

    def launch(entry, replica, gpu, args, ns, *, phase):
        events.append((phase, replica))
        return Child(1 if failure == "preparation" and replica == 2 else 0)

    def validate(work, protocol, *, allow_started=False):
        replica = int(work.name[-1])
        events.append(("validated", replica))
        if failure == "fingerprint" and replica == 2:
            raise ValueError("Prepared input changed")
        return dict(
            lipid_name="POPS",
            force_field="charmm36m",
            lipid_ff="charmm36m",
            replica_seed=queue.entry_seed(entry["family"], "POPS", replica),
            npt_steps=15000000,
            test_mode=False,
        )

    monkeypatch.setattr(queue, "launch", launch)
    monkeypatch.setattr(equil, "validate_prepared_work", validate)
    args = SimpleNamespace(v4_root=tmp_path)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            queue.build_pair(entry, [(1, "build", 30), (2, "build", 30)], args, [0, 1])
        assert not any(phase == "production" for phase, _ in events)
    else:
        assert queue.build_pair(entry, [(1, "build", 30), (2, "build", 30)], args, [0, 1]) == {
            1: 0,
            2: 0,
        }
        assert events == [
            ("prepare", 1),
            ("prepare", 2),
            ("validated", 1),
            ("validated", 2),
            ("production", 1),
            ("production", 2),
        ]


def test_failure_is_reported_while_other_replica_is_still_running(monkeypatch, capsys):
    class Child:
        def __init__(self, code, polls):
            self.code, self.polls = code, polls

        def poll(self):
            self.polls -= 1
            return self.code if self.polls <= 0 else None

        def wait(self):
            return self.code

    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    assert queue.wait_for_replicas([(1, lambda: Child(0, 3)), (2, lambda: Child(1, 1))]) == {
        1: 0,
        2: 1,
    }
    output = capsys.readouterr().out
    assert output.index("Replica 2") < output.index("Replica 1")


def test_resume_first_npt_uses_checkpoint_and_never_repeats_nvt(tmp_path, monkeypatch):
    builder = bare_builder(monkeypatch)
    record = dict(
        lipid_name="POPS",
        force_field="charmm36m",
        lipid_ff="charmm36m",
        replica_seed=6402,
        npt_steps=15000000,
    )
    for name in ("production-started.json", "npt.tpr", "npt.cpt", "npt.edr", "nvt.gro", "nvt.cpt"):
        (tmp_path / name).write_text("checkpoint fixture")
    events = []

    def validate(work, protocol, *, allow_started=False):
        assert allow_started is True
        events.append("validated")
        return record

    monkeypatch.setattr(equil, "validate_prepared_work", validate)
    monkeypatch.setattr(builder, "_entry_context", lambda *args, **kwargs: "context")
    monkeypatch.setattr(builder, "_mdrun", lambda *args, **kwargs: pytest.fail("NVT restarted"))
    monkeypatch.setattr(
        builder,
        "_mdrun_continue",
        lambda stage, work, **kwargs: events.append((stage, kwargs["until_ps"])),
    )
    monkeypatch.setattr(builder, "_publish_prepared", lambda *args, **kwargs: tmp_path)
    assert builder.resume_prepared(tmp_path) == tmp_path
    assert events == ["validated", ("npt", 30000)]
    (tmp_path / "npt.cpt").unlink()
    with pytest.raises(RuntimeError, match="checkpoint"):
        builder.resume_prepared(tmp_path)


@pytest.mark.parametrize("case,expected", [("retry", 68), ("last", 128), ("exhausted", 128)])
def test_precompression_uses_full_128_budget_and_stops_on_success(
    tmp_path, monkeypatch, case, expected
):
    from gmxbuilder.core.component import Component
    from gmxbuilder.core.enums import ComponentKind

    builder = bare_builder(monkeypatch)
    builder._construction_contraction = 0.98
    initial = 24.97548
    target = {
        "retry": np.sqrt(64 * 0.63),
        "last": initial * 0.98**128 / 1.005,
        "exhausted": initial * 0.98**150,
    }[case]
    structure = Structure(
        coordinates=np.array([[1.0, 1.0, 3.0], [2.0, 2.0, 1.0]]),
        box_vectors=np.diag([initial, initial, 7.5]),
        atom_names=["C", "C"],
        resnames=["POPS", "POPS"],
        resids=[1, 2],
    )
    system = System(
        structure,
        components=[
            Component(
                "membrane",
                ComponentKind.MEMBRANE,
                np.arange(2),
                metadata={"lipid_sizes": [1, 1], "n_lipids_upper": 1},
            )
        ],
    )
    monkeypatch.setattr(equil.TopologyWriter, "write_top", lambda *a, **kw: None)
    monkeypatch.setattr(builder, "_mdp", lambda *a, **kw: "integrator = steep")
    monkeypatch.setattr(builder, "_reimage_bilayer_z", lambda *a: None)
    monkeypatch.setattr(builder, "_check_preparation_identity", lambda *a: None)
    monkeypatch.setattr(builder, "_close_leaflet_gap", lambda *a: None)
    inputs, cycles = {}, []

    def run(args, work, **kwargs):
        if args[1] == "grompp":
            inputs[args[args.index("-o") + 1]] = args[args.index("-c") + 1]
        else:
            prefix = args[args.index("-deffnm") + 1]
            tpr = args[args.index("-s") + 1]
            gro = GROReader().read(work / inputs[tpr])
            GROWriter.write(gro, work / f"{prefix}.gro")
            (work / f"{prefix}.log").write_text("Potential Energy = -100\nMaximum force = 10\n")
            if prefix.startswith("compress_em_"):
                cycles.append(prefix)

    monkeypatch.setattr(builder, "_run", run)
    if case == "exhausted":
        with pytest.raises(RuntimeError, match="128 cycles"):
            builder._precompress_bilayer(
                system, tmp_path, "charmm36m", "tip3p", target**2, test_mode=False
            )
        assert not (tmp_path / "compress_final.gro").exists()
    else:
        builder._precompress_bilayer(
            system, tmp_path, "charmm36m", "tip3p", target**2, test_mode=False
        )
        assert system.structure.dimensions()[0] <= target * 1.01
        assert (tmp_path / "compress_final.gro").exists()
    assert len(cycles) == expected


def test_rejected_closure_stops_at_coordinate_precision(tmp_path, monkeypatch):
    builder = bare_builder(monkeypatch)
    system = System(
        Structure(
            coordinates=np.array([[1.0, 1.0, 3.0], [3.0, 3.0, 1.0]]),
            box_vectors=np.eye(3) * 5,
            atom_names=["C", "C"],
            resnames=["POPS", "POPS"],
            resids=[1, 2],
            elements=["C", "C"],
        ),
        metadata={"force_field": "charmm36m"},
    )
    original = system.coordinates.copy()
    monkeypatch.setattr(equil, "construction_core_gap", lambda *a: 1.0)
    monkeypatch.setattr(builder, "_mdp", lambda *a, **k: "integrator = steep\n")
    calls = []

    def fail(*a, **k):
        calls.append(a)
        raise RuntimeError("rejected minimization")

    monkeypatch.setattr(builder, "_run", fail)
    builder._close_leaflet_gap(system, tmp_path, np.array([0, 1, 2]), 1)
    assert len(calls) == 7
    records = [
        json.loads(line) for line in (tmp_path / "closure-attempts.jsonl").read_text().splitlines()
    ]
    assert min(r["step_nm"] for r in records) >= 0.002
    np.testing.assert_array_equal(system.coordinates, original)


def test_head_heavy_bilayer_midplane_uses_hydrophobic_slab():
    coordinates, names, residues = [], [], []
    for resid, sign in [(1, 1), (2, -1)]:
        for name, z in [("P", 2.5), ("C1", 1.5), ("C2", 1.0), ("C3", 0.5)] + [("H", 2.5)] * 20:
            coordinates.append([float(resid), 1.0, sign * z])
            names.append(name)
            residues.append(resid)
    structure = Structure(
        coordinates=np.array(coordinates),
        box_vectors=np.eye(3) * 7.14,
        atom_names=names,
        resnames=["POPC"] * len(names),
        resids=residues,
        elements=[name[0] for name in names],
    )
    assert equil.construction_core_gap(structure) == pytest.approx(
        1.02
    )  # 1st/99th percentiles of three tail carbons
    structure.coordinates[:, 2] += 3.7
    structure.coordinates[24:, 2] += 7.14
    assert equil.construction_core_gap(structure) == pytest.approx(
        1.02
    )  # 1st/99th percentiles of three tail carbons


@pytest.mark.parametrize("always_reject", [False, True])
def test_compression_retries_from_valid_coordinates_and_bounds_rejections(
    tmp_path, monkeypatch, always_reject
):
    from gmxbuilder.core.component import Component
    from gmxbuilder.core.enums import ComponentKind

    builder = bare_builder(monkeypatch)
    system = System(
        Structure(
            coordinates=np.array([[1.0, 1.0, 3.0], [2.0, 2.0, 1.0]]),
            box_vectors=np.diag([10.0, 10.0, 8.0]),
            atom_names=["C", "C"],
            resnames=["POPG", "POPG"],
            resids=[1, 2],
        ),
        components=[
            Component(
                "membrane",
                ComponentKind.MEMBRANE,
                np.arange(2),
                metadata={"lipid_sizes": [1, 1], "n_lipids_upper": 1},
            )
        ],
    )
    monkeypatch.setattr(equil.TopologyWriter, "write_top", lambda *a, **kw: None)
    monkeypatch.setattr(builder, "_reimage_bilayer_z", lambda *a: None)
    monkeypatch.setattr(builder, "_close_leaflet_gap", lambda *a: None)
    inputs, widths = {}, []

    def identity(system):
        if system.coordinates[0, 2] == 99:
            raise equil.PreparationError("E/Z configuration differs")

    def run(args, work, **kwargs):
        if args[1] == "grompp":
            inputs[args[args.index("-o") + 1]] = args[args.index("-c") + 1]
            assert "emtol = 500\n" in (work / args[args.index("-f") + 1]).read_text()
            return
        prefix = args[args.index("-deffnm") + 1]
        gro = GROReader().read(work / inputs[args[args.index("-s") + 1]])
        assert gro.coordinates[0, 2] == 3  # Never retry from a rejected molecule.
        if prefix.startswith("compress_em_"):
            widths.append(gro.dimensions()[0])
            if len(widths) == 2 or (always_reject and len(widths) > 1):
                gro.coordinates[0, 2] = 99
        GROWriter.write(gro, work / f"{prefix}.gro")
        (work / f"{prefix}.log").write_text("Potential Energy = -100\nMaximum force = 10\n")

    monkeypatch.setattr(builder, "_check_preparation_identity", identity)
    monkeypatch.setattr(builder, "_run", run)
    if always_reject:
        with pytest.raises(equil.PreparationError, match="bounded backoff"):
            builder._precompress_bilayer(
                system, tmp_path, "charmm36m", "tip3p", 64, test_mode=False
            )
        assert len(widths) < 10
        assert system.structure.dimensions()[0] == 10
        assert not (tmp_path / "compress_final.gro").exists()
    else:
        builder._precompress_bilayer(system, tmp_path, "charmm36m", "tip3p", 64, test_mode=False)
        assert system.structure.dimensions()[0] <= 8.08
    assert widths[:3] == pytest.approx([10, 9.6, 9.8])
    assert system.coordinates[0, 2] == 3
    evidence = json.loads((tmp_path / "compression-rejections.json").read_text())
    assert evidence[0]["cycle"] == 2
    assert "E/Z" in evidence[0]["reason"]


def test_prepared_geometry_checks_whole_molecules_and_host_identity(tmp_path, monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from gmxbuilder.modules.membrane import lipid_graph
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    smiles = "C[C@H](O)C(=O)O"
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=721) == 0
    elements = tuple(a.GetSymbol() for a in molecule.GetAtoms())
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds())
    xyz = molecule.GetConformer().GetPositions() / 10
    names = tuple(f"{e}{i}" for i, e in enumerate(elements))
    monkeypatch.setattr(LipidRegistry, "get", lambda name: SimpleNamespace(smiles=smiles))
    monkeypatch.setattr(lipid_graph, "molecular_graph", lambda *a: (names, elements, bonds))
    n = len(names)
    # Each molecule spans three residue ids, as native Lipid21 templates can.
    residues = [1 + min(i * 3 // n, 2) for i in range(n)]
    structure = Structure(
        coordinates=np.vstack([xyz + [1, 1, 1], xyz + [3, 3, 3]]),
        box_vectors=np.eye(3) * 8,
        atom_names=list(names) * 2,
        resnames=["TARGET"] * n + ["HOST"] * n,
        resids=residues + [r + 3 for r in residues],
    )
    protocol = {"composition": {"TARGET": 50, "HOST": 50}, "lipids_per_leaflet": 1}
    # Use two lipids per leaflet for an integral one-per-species count in each leaflet.
    protocol["lipids_per_leaflet"] = 2
    structure = Structure(
        coordinates=np.vstack([structure.coordinates, structure.coordinates + [0, 0, 1]]),
        box_vectors=structure.box_vectors,
        atom_names=list(names) * 4,
        resnames=["TARGE"] * n + ["HOST"] * n + ["TARGE"] * n + ["HOST"] * n,
        resids=list(structure.resids) + [int(r) + 6 for r in structure.resids],
    )
    record = {"force_field": "charmm36m", "lipid_ff": "charmm36m"}
    GROWriter.write(structure, tmp_path / "em.gro")
    equil.validate_prepared_geometry(tmp_path, record, protocol)
    structure.coordinates[n : 2 * n, 0] *= -1
    GROWriter.write(structure, tmp_path / "em.gro")
    with pytest.raises(ValueError, match="stereochemistry mismatch"):
        equil.validate_prepared_geometry(tmp_path, record, protocol)
