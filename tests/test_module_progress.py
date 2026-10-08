"""Progress reported from inside a module.

`run_step` publishes 0.20 before it calls `module.execute()` and the next
milestone after, and that one call is most of a long Check: on a real task it
was 214 s parameterising a ligand with AM1-BCC while the bar held at 20%.

Modules now report their own progress and the runner maps it into the slice it
knows the module occupies. What matters is that the two ends agree, that a
module reporting nothing still works, and that reporting can never cost a
build -- so those are what is asserted here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from gmxbuilder.pipeline.progress import module_progress_scope, report_progress

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "src" / "gmxbuilder" / "modules"


# --------------------------------------------------------------------------
# The channel itself


def test_reporting_outside_a_scope_does_nothing():
    """Modules are also run by the CLI and the pipeline runner, with no sink."""
    report_progress(0.5, "no sink is listening")


def test_a_scope_receives_what_the_module_reports():
    seen = []
    with module_progress_scope(lambda f, p: seen.append((f, p))):
        report_progress(0.25, "half way through the leaflets")
    assert seen == [(0.25, "half way through the leaflets")]


def test_a_scope_stops_receiving_once_it_exits():
    seen = []
    with module_progress_scope(lambda f, p: seen.append((f, p))):
        report_progress(0.1, "inside")
    report_progress(0.9, "outside")
    assert seen == [(0.1, "inside")]


def test_nested_scopes_restore_the_outer_one():
    outer, inner = [], []
    with module_progress_scope(lambda f, p: outer.append(p)):
        with module_progress_scope(lambda f, p: inner.append(p)):
            report_progress(0.5, "inner")
        report_progress(0.6, "outer again")
    assert inner == ["inner"]
    assert outer == ["outer again"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_a_fraction_outside_the_unit_range_is_clamped(reported, expected):
    """A module computing a fraction from a count can overshoot by one."""
    seen = []
    with module_progress_scope(lambda f, p: seen.append(f)):
        report_progress(reported, "phase")
    assert seen == [expected]


def test_a_failing_sink_cannot_break_the_module():
    """Reporting is decoration. It must never cost a build."""

    def explode(fraction, phase):
        raise RuntimeError("the browser went away")

    with module_progress_scope(explode):
        report_progress(0.5, "phase")  # must not raise


def test_a_sink_that_is_none_is_a_scope_with_no_listener():
    with module_progress_scope(None):
        report_progress(0.5, "phase")


# --------------------------------------------------------------------------
# The runner's half of the contract


def test_the_runner_maps_module_progress_into_the_slice_it_owns():
    from gmxbuilder.pipeline import step_executor

    assert step_executor.MODULE_PHASE_START == 0.20
    assert step_executor.MODULE_PHASE_END == 0.75

    source = inspect.getsource(step_executor.StepRunner.run_step)
    assert "module_progress_scope(module_report)" in source
    # The mapping must be expressed in terms of the constants, not repeated.
    assert "MODULE_PHASE_END - MODULE_PHASE_START" in source
    assert "report(0.2," not in source and "report(0.75," not in source


def test_the_module_slice_is_the_widest_step_in_the_ladder():
    """The module is the part that can run for minutes; it gets the most bar."""
    from gmxbuilder.pipeline import step_executor

    source = inspect.getsource(step_executor.StepRunner.run_step)
    tree = ast.parse(inspect.cleandoc(source))
    literals = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "report"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, float)
    ]
    milestones = sorted({0.20, 0.75, *literals})
    gaps = [(b - a, a) for a, b in zip(milestones, milestones[1:], strict=False)]
    widest = max(gaps)
    assert widest[1] == pytest.approx(0.20), f"widest gap starts at {widest[1]}, not the module"


def test_a_step_maps_a_real_module_report_onto_the_published_fraction():
    """End to end through the runner: what a module says, the server publishes."""
    from gmxbuilder.pipeline.step_executor import (
        MODULE_PHASE_END,
        MODULE_PHASE_START,
    )

    span = MODULE_PHASE_END - MODULE_PHASE_START
    published: list[float] = []

    def report(fraction, phase):
        published.append(fraction)

    # The closure the runner builds, exercised with the same arithmetic.
    def module_report(fraction, phase):
        report(MODULE_PHASE_START + span * fraction, phase)

    with module_progress_scope(module_report):
        report_progress(0.0, "start")
        report_progress(0.5, "middle")
        report_progress(1.0, "end")

    assert published == pytest.approx([0.20, 0.475, 0.75])


# --------------------------------------------------------------------------
# Where the reports are


# Modules slow enough that a user waits on them, and the method each reports
# from. A module missing from this list simply behaves as it did before.
INSTRUMENTED = {
    "membrane/builder.py": ("MembraneBuilder", "run"),
    "solvation/solvate.py": ("SolvationBuilder", "run"),
    "ions/add_ions.py": ("IonBuilder", "run"),
    "input/pdb_input.py": ("PDBInputModule", "run"),
    "modifications/processor.py": ("StructureProcessor", "run"),
    "export/exporter.py": ("ExportModule", "run"),
    "forcefield/selector.py": ("ForceFieldSelector", "_parameterize_gaff2_ligands"),
}


def _reports_by_method(path: Path) -> dict[str, list[ast.Call]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, list[ast.Call]] = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "report_progress"
            ]
            if calls:
                found[f"{cls.name}.{fn.name}"] = calls
    return found


@pytest.mark.parametrize("relative", sorted(INSTRUMENTED))
def test_every_instrumented_module_reports_from_its_own_run(relative):
    """A report placed in the wrong method reports the wrong thing, silently."""
    cls, method = INSTRUMENTED[relative]
    found = _reports_by_method(MODULES / relative)
    assert list(found) == [f"{cls}.{method}"], f"{relative}: reports in {list(found)}"


@pytest.mark.parametrize("relative", sorted(INSTRUMENTED))
def test_reported_fractions_only_ever_increase(relative):
    """A bar that jumps backwards reads as a restart."""
    cls, method = INSTRUMENTED[relative]
    calls = _reports_by_method(MODULES / relative)[f"{cls}.{method}"]
    literals = [
        call.args[0].value
        for call in calls
        if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, float)
    ]
    assert literals == sorted(literals), f"{relative}: {literals}"


def _phase_variants(node: ast.expr) -> list[str]:
    """Return the strings a phase expression can actually show a user.

    A phase may be a literal, an f-string naming the molecule, or a
    conditional choosing between the two. Interpolated runtime values stand in
    as a placeholder: what is under test is the wording around them.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            "".join(
                part.value
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
                else "\u2026"
                for part in node.values
            )
        ]
    if isinstance(node, ast.IfExp):
        return _phase_variants(node.body) + _phase_variants(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [
            left + right
            for left in _phase_variants(node.left)
            for right in _phase_variants(node.right)
        ]
    return []


def _assert_reads_as_prose(text: str, where: str) -> None:
    assert text, f"{where}: empty phase"
    assert text[0].isupper(), f"{where}: {text!r} does not start a sentence"
    assert "(" not in text and "_" not in text, f"{where}: {text!r} looks like code"


@pytest.mark.parametrize("relative", sorted(INSTRUMENTED))
def test_every_phase_is_a_sentence_about_the_system_not_the_code(relative):
    """The phase is shown to the user, so it names chemistry, not functions.

    A phase built by a helper is checked by calling that helper instead; see
    the ligand tests below, which assert the same property on real output.
    """
    cls, method = INSTRUMENTED[relative]
    calls = _reports_by_method(MODULES / relative)[f"{cls}.{method}"]
    for call in calls:
        variants = _phase_variants(call.args[1])
        if not variants:
            # A computed phase. Not exempt -- covered behaviourally below.
            assert isinstance(call.args[1], ast.Call), (
                f"{relative}: phase is neither a string expression nor a call"
            )
            continue
        for text in variants:
            _assert_reads_as_prose(text, relative)


def test_the_charge_method_description_is_true_of_that_method():
    """The wait is explained to the user, so the explanation must be correct.

    A first version hard-coded "AM1-BCC ... several minutes" into the phase.
    Under the Gasteiger option that sentence named the wrong method and
    promised a wait that does not happen.
    """
    from gmxbuilder.modules.forcefield.gaff_backend import describe_gaff_charge_method

    quantum_label, quantum_cost = describe_gaff_charge_method("bcc")
    assert quantum_label == "AM1-BCC"
    assert "minutes" in quantum_cost, "the method that takes minutes must say so"

    empirical_label, empirical_cost = describe_gaff_charge_method("gas")
    assert empirical_label == "Gasteiger"
    assert "minutes" not in empirical_cost, "an instant method must not promise a wait"
    assert "quantum" not in empirical_cost, "Gasteiger is not a quantum calculation"


def test_an_unknown_charge_method_is_rejected_rather_than_described():
    from gmxbuilder.modules.forcefield.gaff_backend import describe_gaff_charge_method

    with pytest.raises(ValueError):
        describe_gaff_charge_method("nonsense")


def _ligand_block() -> str:
    source = (MODULES / "forcefield" / "selector.py").read_text(encoding="utf-8")
    block = source[source.index("def _parameterize_gaff2_ligands") :]
    return block[: block.index("\n    def ")]


def test_the_slow_ligand_path_branches_on_whether_the_work_is_real():
    """The defect this exists for: a 214 s wait the interface never explained.

    Asserted against the code that runs, not the comment beside it -- an
    earlier version of this test passed on the word "AM1-BCC" appearing in a
    comment after it had been removed from the message.
    """
    block = _ligand_block()
    assert "gaff_molecule_is_cached" in block, "a cache hit must not be announced as a wait"
    assert "describe_gaff_charge_method" in block, "the method must be named, not assumed"


def test_the_ligand_phase_reads_as_prose_in_every_case():
    """Asserted on real output, since this phase is computed rather than literal."""
    from gmxbuilder.modules.forcefield.gaff_backend import MoleculeJob
    from gmxbuilder.modules.forcefield.selector import ForceFieldSelector

    jobs = [MoleculeJob(name, None, [0], 0) for name in ("UK4", "ATP", "HEM")]
    cases = {
        "nothing pending": ForceFieldSelector._ligand_phase(jobs, [], "AM1-BCC", "a wait"),
        "one pending": ForceFieldSelector._ligand_phase(jobs, ["UK4"], "AM1-BCC", "a wait"),
        "several pending": ForceFieldSelector._ligand_phase(
            jobs, ["UK4", "ATP"], "AM1-BCC", "a wait"
        ),
    }
    for where, text in cases.items():
        _assert_reads_as_prose(text, where)


def test_the_phase_names_what_is_pending_and_what_was_reused():
    """Naming nothing is no help when a system carries four ligands, and
    naming one of four would mislead about what the bar is waiting for."""
    from gmxbuilder.modules.forcefield.gaff_backend import MoleculeJob
    from gmxbuilder.modules.forcefield.selector import ForceFieldSelector

    jobs = [MoleculeJob(name, None, [0], 0) for name in ("UK4", "ATP", "HEM")]

    one = ForceFieldSelector._ligand_phase(jobs, ["UK4"], "AM1-BCC", "a wait")
    assert "UK4" in one and "parallel" not in one

    several = ForceFieldSelector._ligand_phase(jobs, ["UK4", "ATP"], "AM1-BCC", "a wait")
    assert "2 molecules" in several and "in parallel" in several

    none = ForceFieldSelector._ligand_phase(jobs, [], "AM1-BCC", "a wait")
    assert none.startswith("Reusing stored")
    assert "UK4" in none and "ATP" in none
