"""Progress reporting from inside a running module.

`StepRunner.run_step` publishes a milestone before it calls `module.execute()`
and the next one after, so a browser watching a Check sees the bar hold at 20%
for the whole of that call. On a real task that was 214 seconds: the module was
parameterising a ligand with GAFF2/AM1-BCC and had no way to say so.

Modules report progress through their own work as a fraction of that work, and
the runner maps it into the slice of the step the module occupies. A module
that reports nothing is not a special case; its step simply behaves as before.

Why a scope and not a parameter: module instances are cached in
`step_executor._MODULE_CACHE` and shared between tasks that run concurrently in
the step thread pool, so a callback stored on the instance would be a data
race between two tasks running the same step. Threading a parameter instead
would change all 26 `run()` signatures for the handful of modules that are
slow enough to need it. A `ContextVar` is per-thread here -- the scope is
entered inside the pool worker that runs the step -- and follows the idiom the
codebase already uses for per-task ambient state in `LipidRegistry.task_scope`,
`task_gaff_cache` and `task_equilibrated_library`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Called as sink(fraction, phase) where fraction is 0..1 of the module's work.
ModuleProgressSink = Callable[[float, str], None]

_sink: ContextVar[ModuleProgressSink | None] = ContextVar(
    "gmxbuilder_module_progress", default=None
)


@contextmanager
def module_progress_scope(sink: ModuleProgressSink | None) -> Iterator[None]:
    """Route `report_progress` calls made in this thread to *sink*."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def report_progress(fraction: float, phase: str) -> None:
    """Report how far this module has got through its own work.

    *fraction* is 0..1 of this module's work, not of the step: a module does
    not know what share of the step it is. *phase* is shown to the user, so it
    names what is happening in the terms they chose their inputs in -- the
    molecule being parameterised, the leaflet being filled -- rather than the
    function running.

    Outside a scope this does nothing, which is what makes it safe to call from
    a module used by the CLI, a test, or the pipeline runner. Failures are
    swallowed: reporting is decoration and must never cost a build.
    """
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(max(0.0, min(1.0, float(fraction))), str(phase))
    except Exception:  # noqa: BLE001 - progress must never break a build
        logger.debug("Module progress sink failed", exc_info=True)
