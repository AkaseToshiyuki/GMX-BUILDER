"""Warning admission tests independent of optional installed force fields."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gmxbuilder.modules.membrane import grompp_policy as policy


def warning(charge="-128.000000", body=policy.EWALD_WARNING):
    return (
        f"System has non-zero total charge: {charge}\n\nWARNING 1 [file topol.top]:\n"
        f"  {body}\n\nFatal error:\nToo many warnings (1).\n"
    )


def topology(name="POPG", count=128, charge="-1"):
    return (
        f"[ moleculetype ]\n{name} 3\n[ atoms ]\n1 C 1 {name} C 1 {charge} 12\n"
        f"[ system ]\nTest\n[ molecules ]\n{name} {count}\n"
    )


@pytest.fixture
def work(tmp_path):
    (tmp_path / "topol.top").write_text(topology())
    (tmp_path / "em.mdp").write_text("integrator=steep\n")
    (tmp_path / "input.gro").write_text("coordinates")
    return tmp_path


def args():
    return [
        "gmx",
        "grompp",
        "-f",
        "em.mdp",
        "-c",
        "input.gro",
        "-p",
        "topol.top",
        "-o",
        "em.tpr",
        "-maxwarn",
        "1",
    ]


def runner(monkeypatch, work, logs, *, mutate=False):
    calls = []

    def run(command, **kwargs):
        assert command[1] == "grompp", "No simulation may run in a policy test"
        calls.append(command.copy())
        index = len(calls) - 1
        code, log = logs[index]
        expanded = Path(command[command.index("-pp") + 1])
        assert not expanded.is_relative_to(work)
        expanded.write_text(topology())
        if not code:
            Path(command[command.index("-o") + 1]).write_bytes(b"new-compiled-input")
        if mutate:
            (work / "topol.top").write_text("changed during preprocessing")
        return SimpleNamespace(returncode=code, stdout=log, stderr="")

    monkeypatch.setattr(policy.subprocess, "run", run)
    return calls


def test_strict_success_ignores_requested_warning_budget(monkeypatch, work):
    calls = runner(monkeypatch, work, [(0, "")])
    receipt = json.loads(policy.run_grompp(args(), work).read_text())
    assert receipt["strict_passed"] and receipt["maxwarn"] == 0
    assert calls[0][calls[0].index("-maxwarn") + 1] == "0"
    assert (work / "em.tpr").read_bytes() == b"new-compiled-input"
    assert not list((work / ".grompp-policy").rglob("*.top"))


def test_formal_charge_is_separate_verified_preionization_exception(monkeypatch, work):
    calls = runner(monkeypatch, work, [(1, warning()), (0, warning())])
    result = json.loads(policy.run_grompp(args(), work, stage="before_ions").read_text())
    assert result["exception"] == "verified-pre-ionization-formal-charge"
    assert result["formal_charge"] == -128
    assert not result["strict_passed"] and result["maxwarn"] == 1
    assert [c[c.index("-maxwarn") + 1] for c in calls] == ["0", "1"]


@pytest.mark.parametrize(
    "log,stage",
    [
        (warning(), "neutralized"),
        (warning("-127.000000"), "before_ions"),
        (warning(body="Unrelated dangerous warning"), "before_ions"),
        (warning() + "\nWARNING 2 [file x]:\n  Different warning\n\n", "before_ions"),
        ("Other fatal error", "before_ions"),
        ("WARNING changed format\n", "before_ions"),
    ],
)
def test_rejects_other_warnings_and_unexplained_charge(monkeypatch, work, log, stage):
    (work / "em.tpr").write_bytes(b"old-tpr")
    calls = runner(monkeypatch, work, [(1, log)])
    with pytest.raises(RuntimeError, match="policy rejected"):
        policy.run_grompp(args(), work, stage=stage)
    assert len(calls) == 1
    assert (work / "em.tpr").read_bytes() == b"old-tpr"
    receipt = json.loads(next((work / ".grompp-policy").glob("*/receipt.json")).read_text())
    assert not receipt["accepted"]


@pytest.mark.parametrize("code,log", [(0, warning()), (1, warning())])
def test_inputs_cannot_change_during_preprocessing(monkeypatch, work, code, log):
    runner(monkeypatch, work, [(code, log)], mutate=True)
    with pytest.raises(RuntimeError):
        policy.run_grompp(args(), work, stage="before_ions")
    assert not (work / "em.tpr").exists()


@pytest.mark.parametrize("retry_log", ["", warning("-127.000000"), warning(body="Other")])
def test_retry_must_reproduce_the_exact_accepted_warning(monkeypatch, work, retry_log):
    runner(monkeypatch, work, [(1, warning()), (0, retry_log)])
    with pytest.raises(RuntimeError):
        policy.run_grompp(args(), work, stage="before_ions")
    assert not (work / "em.tpr").exists()


@pytest.mark.parametrize(
    "text",
    [topology(charge="NaN"), topology(charge="-0.8"), topology(name="SSM", charge="0.000003")],
)
def test_formal_charge_does_not_admit_unknown_residuals(tmp_path, text):
    path = tmp_path / "processed.txt"
    path.write_text(text)
    with pytest.raises((ValueError, KeyError)):
        policy.charge_exception(path, warning(), tmp_path, tmp_path / "topol.top", "dry")


@pytest.mark.parametrize("resume", [False, True])
def test_policy_failure_prevents_both_dynamics_launches(monkeypatch, tmp_path, resume):
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.gmx = "gmx"
    builder._run = lambda *a, **kw: pytest.fail("Dynamics dispatched before policy admission")

    def reject(*a):
        raise RuntimeError("Unreviewed retained TPR")

    monkeypatch.setattr(policy, "ensure_dynamics_policy", reject)
    with pytest.raises(RuntimeError, match="Unreviewed retained TPR"):
        if resume:
            builder._mdrun_continue("npt", tmp_path, until_ps=100, timeout=1)
        else:
            builder._mdrun("nvt", tmp_path, timeout=1)


# Independent frozen charge vector from AmberTools 24.8 PA + SPM + SA.
PSM_CHARGES = """
-0.12544700 0.02904700 0.02904700 0.02904700 0.01397500 0.00929200 0.00929200 -0.02008600
0.01592900 0.01592900 -0.03309600 0.01462100 0.01462100 -0.02763300 0.01136800 0.01136800
-0.02520600 0.01433400 0.01433400 -0.02883100 0.01469100 0.01469100 -0.03047200 0.01389700
0.01389700 -0.01579300 0.00906700 0.00906700 -0.01963000 0.01104100 0.01104100 -0.02442700
0.01333400 0.01333400 -0.02938700 0.01676300 0.01676300 -0.03009900 0.02087700 0.02087700
-0.00488200 0.01900700 0.01900700 -0.14925900 0.04734500 0.04734500 0.01273900 0.08745500
0.17662500 0.05162200 0.05162200 -0.53335600 1.32476200 -0.45930800 0.28244700 0.04988600
0.04988600 -0.49366500 0.23384000 0.23384000 0.24527500 -0.38661100 0.19156900 0.19156900
0.19156900 -0.38661100 0.19156900 0.19156900 0.19156900 -0.38661100 0.19156900 0.19156900
0.19156900 -0.88478900 -0.88478900 -0.38841000 0.27359200 0.61723100 -0.64530300
0.32309200 0.06920800 -0.72303300 0.43496400 -0.19999900 0.04947100 0.04947100 0.04947100
-0.04678100 0.02648400 0.02648400 0.04464800 0.01097000 0.01097000 -0.11922500 0.04033000
0.04033000 0.00104800 0.01693500 0.01693500 -0.07941700 0.02350200 0.02350200 -0.06637200
0.02172300 0.02172300 0.05335200 0.02020000 0.02020000 -0.18788900 0.04690900 0.04690900
0.04928800 0.00604500 0.00604500 -0.01299700 0.00322300 0.00322300 0.09317100 -0.01169000
-0.01169000 -0.25813800 0.07684300 0.07684300 0.02829000 0.11628100 -0.34188900 0.14555000
""".split()


@pytest.fixture
def psm_topology(tmp_path, monkeypatch):
    text = "[ moleculetype ]\nPSM 3\n[ atoms ]\n"
    text += "".join(f"{i} C 1 PSM C{i} 1 {q} 12\n" for i, q in enumerate(PSM_CHARGES, 1))
    text += (
        "[ moleculetype ]\nSOL 2\n[ atoms ]\n1 OW 1 SOL O 1 -0.834 16\n"
        "2 HW 1 SOL H1 1 0.417 1\n3 HW 1 SOL H2 1 0.417 1\n"
        "[ moleculetype ]\nNA 1\n[ atoms ]\n1 Na 1 NA NA 1 1 23\n"
        "[ moleculetype ]\nCL 1\n[ atoms ]\n1 Cl 1 CL CL 1 -1 35\n"
        "[ molecules ]\nPSM 128\nSOL 2698\nNA 22\nCL 22\n"
    )
    path = tmp_path / "processed.txt"
    path.write_text(text)
    # Isolate numerical/composition admission from the separate source gate.
    monkeypatch.setattr(policy, "_psm_sources", lambda *a: {"parameter_fingerprint": "reviewed"})
    return path


def test_reviewed_psm_residual_is_explicit_and_does_not_modify_charges(psm_topology, tmp_path):
    before = psm_topology.read_bytes()
    result = policy.charge_exception(
        psm_topology, warning("0.000379"), tmp_path, tmp_path / "topol.top", "neutralized"
    )
    assert result["exception"] == "reviewed-psm128-source-residual"
    assert result["decimal_charge"] == "0.00038400"
    assert result["predicted_reported_charge"] == "0.000379"
    assert psm_topology.read_bytes() == before


@pytest.mark.parametrize(
    "old,new",
    [
        ("PSM 128", "PSM 256"),
        ("CL 22", "CL 21"),
        ("SOL 2698\n", ""),
        ("0.00000000", "0.00010000"),
        ("-0.834", "-0.835"),
        ("NA 22", "NA 22\nOther 1"),
        ("PSM 128", "PSM 128\nPSM 1"),
    ],
)
def test_psm_composition_and_charge_negative_controls(psm_topology, tmp_path, old, new):
    text = psm_topology.read_text()
    # This atom perturbation uses an actual charge in the frozen vector.
    if old == "0.00000000":
        old = PSM_CHARGES[0]
    assert old in text
    psm_topology.write_text(text.replace(old, new, 1))
    with pytest.raises((ValueError, KeyError)):
        policy.charge_exception(
            psm_topology, warning("0.000379"), tmp_path, tmp_path / "topol.top", "neutralized"
        )


def test_psm_source_gate_failure_is_not_bypassed(psm_topology, tmp_path, monkeypatch):
    def reject(*a):
        raise ValueError("Unreviewed source fingerprint")

    monkeypatch.setattr(policy, "_psm_sources", reject)
    with pytest.raises(ValueError, match="Unreviewed source"):
        policy.charge_exception(
            psm_topology, warning("0.000379"), tmp_path, tmp_path / "topol.top", "neutralized"
        )


@pytest.mark.parametrize(
    "version,precision,accepted",
    [
        ("2026.3", "mixed", True),
        ("2026.2", "mixed", False),
        ("2026.3", "double", False),
    ],
)
def test_psm_binary_version_and_precision_are_bound(
    monkeypatch, work, psm_topology, version, precision, accepted
):
    compiled = psm_topology.read_text()
    calls = []

    def run(command, **kwargs):
        if command[1] == "--version":
            return SimpleNamespace(
                returncode=0,
                stdout=f"GROMACS version: {version}\nPrecision: {precision}\n",
                stderr="",
            )
        calls.append(command)
        Path(command[command.index("-pp") + 1]).write_text(compiled)
        code = 1 if len(calls) == 1 else 0
        if not code:
            Path(command[command.index("-o") + 1]).write_bytes(b"reviewed-psm")
        return SimpleNamespace(returncode=code, stdout=warning("0.000379"), stderr="")

    monkeypatch.setattr(policy.subprocess, "run", run)
    if accepted:
        result = json.loads(policy.run_grompp(args(), work).read_text())
        assert result["maxwarn"] == 1 and not result["strict_passed"]
        assert len(calls) == 2
    else:
        with pytest.raises(RuntimeError, match="GROMACS 2026.3 mixed"):
            policy.run_grompp(args(), work)
        assert len(calls) == 1 and not (work / "em.tpr").exists()


def test_legacy_audit_requires_same_compiled_topology_and_preserves_original(monkeypatch, work):
    (work / "npt.tpr").write_bytes(b"old-checkpoint-compatible-tpr")
    (work / "npt.mdp").write_text("integrator=md")
    (work / "nvt.gro").write_text("coordinates")
    (work / "nvt.cpt").write_bytes(b"checkpoint")
    calls = runner(monkeypatch, work, [(0, "")])
    monkeypatch.setattr(
        policy, "_compiled_identity", lambda gmx, p: ("ld-seed = 7\n", p.read_bytes())
    )
    with pytest.raises(RuntimeError, match="differs from newly audited"):
        policy.ensure_dynamics_policy("gmx", work, "npt")
    assert len(calls) == 1
    assert (work / "npt.tpr").read_bytes() == b"old-checkpoint-compatible-tpr"
    assert (work / "nvt.cpt").read_bytes() == b"checkpoint"


def test_legacy_receipt_rechecks_inputs_and_tpr_hash(monkeypatch, work):
    # A derived audit TPR may have identical bytes; its receipt cannot stand
    # in for the original input binding merely because the hashes match.
    (work / "npt.tpr").write_bytes(b"new-compiled-input")
    (work / "npt.mdp").write_text("integrator=md")
    (work / "nvt.gro").write_text("coordinates")
    (work / "nvt.cpt").write_bytes(b"checkpoint")
    calls = runner(monkeypatch, work, [(0, ""), (0, ""), (0, "")])
    monkeypatch.setattr(
        policy, "_compiled_identity", lambda *a: ("ld-seed = 7\n", "same-parameters")
    )
    receipt = policy.ensure_dynamics_policy("gmx", work, "npt")
    assert json.loads(receipt.read_text())["legacy_audit"]
    assert policy.ensure_dynamics_policy("gmx", work, "npt") == receipt
    assert len(calls) == 1
    (work / "npt.tpr").write_bytes(b"extended-tpr")
    assert policy.ensure_dynamics_policy("gmx", work, "npt") != receipt
    assert len(calls) == 2
    (work / "npt.mdp").write_text("changed-input")
    policy.ensure_dynamics_policy("gmx", work, "npt")
    assert len(calls) == 3


def test_interleaved_molecule_blocks_are_accumulated(tmp_path):
    path = tmp_path / "processed.txt"
    path.write_text(topology(count=64) + "POPG 64\n")
    result = policy.charge_exception(path, warning(), tmp_path, tmp_path / "topol.top", "dry")
    assert result["molecules"] == {"POPG": 128}
    assert result["formal_charge"] == -128


def test_cpu_fallback_npt_uses_the_inputs_actually_preprocessed(monkeypatch, work):
    (work / "npt.mdp").write_text("integrator=md")
    (work / "nvt_cpu.gro").write_text("CPU coordinates")
    (work / "nvt_cpu.cpt").write_bytes(b"CPU checkpoint")
    calls = runner(monkeypatch, work, [(0, ""), (0, "")])
    command = [
        "gmx",
        "grompp",
        "-f",
        "npt.mdp",
        "-c",
        "nvt_cpu.gro",
        "-t",
        "nvt_cpu.cpt",
        "-p",
        "topol.top",
        "-o",
        "npt.tpr",
    ]
    receipt = policy.run_grompp(command, work)
    assert policy.ensure_dynamics_policy("gmx", work, "npt") == receipt
    assert len(calls) == 1
    (work / "nvt_cpu.gro").write_text("changed CPU coordinates")
    monkeypatch.setattr(
        policy, "_compiled_identity", lambda *a: ("ld-seed = 7\n", "same-parameters")
    )
    assert policy.ensure_dynamics_policy("gmx", work, "npt") != receipt
    assert len(calls) == 2


def test_failed_tpr_publication_never_records_acceptance(monkeypatch, work):
    runner(monkeypatch, work, [(0, "")])
    (work / "em.tpr").mkdir()  # Destination cannot be replaced by a file.
    with pytest.raises(RuntimeError, match="policy rejected"):
        policy.run_grompp(args(), work)
    receipt = json.loads(next((work / ".grompp-policy").glob("*/receipt.json")).read_text())
    assert not receipt["accepted"]


def test_policy_rejection_does_not_mark_preparation_as_started(monkeypatch, tmp_path):
    from gmxbuilder.modules.membrane import lipid_equilibration as equil

    builder = equil.LipidEquilibrationBuilder.__new__(equil.LipidEquilibrationBuilder)
    builder.gmx = "gmx"
    record = {
        "lipid_name": "PSM",
        "force_field": "amber14sb",
        "lipid_ff": "lipid21",
        "replica_seed": 1,
    }
    monkeypatch.setattr(equil, "validate_prepared_work", lambda *a: record)
    monkeypatch.setattr(builder, "_entry_context", lambda *a, **kw: None)

    def reject(*a):
        raise RuntimeError("Unreviewed preparation")

    monkeypatch.setattr(policy, "ensure_dynamics_policy", reject)
    with pytest.raises(RuntimeError, match="Unreviewed preparation"):
        builder._run_prepared_once(tmp_path, retain_work=tmp_path)
    assert not (tmp_path / "production-started.json").exists()


@pytest.mark.parametrize(
    "mdp",
    ["integrator=md\n", "integrator=md\nld-seed=-1\n", "integrator=md\nld_seed=71 ; retained\n"],
)
def test_legacy_audit_reproduces_original_stochastic_seed(mdp):
    result, seed = policy._retained_seed_mdp(mdp, "   ld-seed = 71\n")
    assert seed == 71 and result.count("ld-seed = 71") == 1
    assert "integrator=md" in result
    assert "ld_seed" not in result and "ld-seed=-1" not in result


@pytest.mark.parametrize(
    "mdp,inputrec",
    [
        ("ld-seed=72\n", "ld-seed = 71\n"),
        ("ld-seed=71\nld_seed=71\n", "ld-seed = 71\n"),
        ("integrator=md\n", "integrator = md\n"),
    ],
)
def test_legacy_audit_rejects_inconsistent_stochastic_seed(mdp, inputrec):
    with pytest.raises(ValueError, match="seed"):
        policy._retained_seed_mdp(mdp, inputrec)
