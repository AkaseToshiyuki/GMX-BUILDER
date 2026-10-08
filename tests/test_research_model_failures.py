"""Model selection must distinguish an unsupported domain from a failed computation."""

import pytest

from gmxbuilder.modules.forcefield import charmm_research as research
from gmxbuilder.modules.forcefield.charmm_compat import CharmmCompatError


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("calculation failed"),
        KeyError("missing internal state"),
        CharmmCompatError("CHARGE_CONSERVATION_FAILED", "inconsistent charge"),
        CharmmCompatError("FF_VERSION_MISMATCH", "reference changed"),
        CharmmCompatError("DEPENDENCY_UNAVAILABLE", "missing force field"),
    ],
)
def test_failed_computation_never_selects_another_model(monkeypatch, failure):
    tried = []

    def fail(*args):
        raise failure

    def later(*args):
        tried.append(True)

    monkeypatch.setattr(
        research,
        "_models",
        lambda: (
            ("first", fail, None, True),
            ("later", later, None, True),
        ),
    )
    with pytest.raises(type(failure)) as caught:
        research.select_model(None, "charmm36m")
    assert caught.value is failure
    assert not tried


def test_explicit_domain_refusal_can_try_the_next_model(monkeypatch):
    def unsupported(*args):
        raise CharmmCompatError("UNSUPPORTED_CHEMISTRY", "outside this domain")

    monkeypatch.setattr(
        research,
        "_models",
        lambda: (
            ("first", unsupported, None, True),
            ("later", lambda *a: None, "assignment", True),
        ),
    )
    name, assignment, declined = research.select_model(None, "charmm36m")
    assert (name, assignment) == ("later", "assignment")
    assert declined[0][0] == "first"


def test_native_guard_failure_cannot_admit_an_upload(monkeypatch):
    from gmxbuilder.modules.forcefield import native_lipids
    from gmxbuilder.web.custom_lipids import _refuse_if_the_force_field_already_provides_it

    def unreadable(*args):
        raise OSError("unreadable database")

    monkeypatch.setattr(native_lipids, "find_native_lipid", unreadable)
    with pytest.raises(ValueError, match="could not be checked"):
        _refuse_if_the_force_field_already_provides_it("CCO", "charmm36m")
