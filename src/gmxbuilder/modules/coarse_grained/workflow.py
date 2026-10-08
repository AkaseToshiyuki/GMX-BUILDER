"""Declarative admission rules that distinguish the Martini 3 workflows.

The bilayer and solution builders run the same construction code; what differs
is which tasks each will accept and which settings it pins. Expressing that as
class attributes lets one implementation serve both, instead of two forks that
drift apart -- as they had, to the point where a fix applied to one copy was
invisible in the other.

A class that sets no attributes here admits everything, which is the behaviour
the shared modules had before the workflows were expressed this way.
"""

from __future__ import annotations

from gmxbuilder.core.exceptions import ModuleConfigError


class CGWorkflowAdmission:
    """Mixin applying one workflow's admission rules to a shared module."""

    #: Required value of ``system.metadata["cg_environment"]``; None accepts any.
    cg_environment: str | None = None
    #: Whether this step checks that metadata. The earliest steps run before
    #: ``cg_environment`` has been recorded on the system, so they must not.
    checks_task_environment: bool = False
    #: Name used in refusal messages, so the user is told which builder refused.
    workflow_label: str = "Martini 3 Builder"
    #: Settings this workflow has no meaning for, refused rather than ignored.
    forbidden_config_keys: frozenset[str] = frozenset()
    #: What the forbidden keys belong to, for the message.
    forbidden_config_subject: str = "setting(s)"
    #: Keys pinned after admission, e.g. ``{"environment": "solution"}``.
    pinned_config: dict = {}  # noqa: RUF012 - overridden per workflow, never mutated
    #: Refusal text per pinned key. Each key states its own reason, because
    #: "cannot switch to bilayer mode" and "requires an uploaded protein" are
    #: different things to tell a user.
    pin_refusals: dict = {}  # noqa: RUF012 - overridden per workflow, never mutated

    def admit(self, system, config: dict) -> dict:
        """Refuse a task this workflow does not build, and pin its settings.

        Returns the config to use, copied when anything is pinned so the
        caller's dictionary is never mutated.
        """
        expected = self.cg_environment
        if expected is None:
            return config

        if self.checks_task_environment and system.metadata.get("cg_environment") != expected:
            raise ModuleConfigError(
                f"{self.workflow_label} accepts only "
                f"{'bilayer' if expected == 'bilayer' else 'solution'} tasks"
            )

        supplied = sorted(self.forbidden_config_keys & set(config))
        if supplied:
            raise ModuleConfigError(
                f"{self.workflow_label} does not accept "
                f"{self.forbidden_config_subject}: " + ", ".join(supplied)
            )

        for key, value in self.pinned_config.items():
            if config.get(key, value) is not value and config.get(key, value) != value:
                raise ModuleConfigError(
                    self.pin_refusals.get(key, f"{self.workflow_label} requires {key}={value!r}")
                )
        if self.pinned_config:
            config = {**config, **self.pinned_config}
        return config
