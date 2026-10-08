"""Parameterize a task's ligands before the user asks for them.

GAFF2 parameterization is the slowest thing a Check does and the least
expected: on a real task the Force Field step took 214 s, of which 99.98% was
one `acpype` call running `sqm` -- a semi-empirical charge calculation that is
single-threaded, so no amount of hardware shortens it. Every other part of that
step was milliseconds.

Nothing about that calculation depends on the Force Field panel. The molecule,
its coordinates, its atom names and the charge the panel will suggest are all
determined once the input Check has saved its checkpoint, and
`_coordinate_identity_signature` is translation-invariant, so the cache entry
computed from the input checkpoint is the one the forcefield step will look
for. Starting it when the input Check finishes therefore overlaps it with the
time the user spends choosing force fields, and the step becomes a cache
lookup.

What this deliberately does not do is guess. It runs `estimate_gaff_net_charge`
at the same default pH the Force Field panel prefills, and a molecule whose
charge is ambiguous is left alone rather than parameterized at a charge the
user has not seen. A user who then changes the charge or the pH gets a correct
result from a second calculation; the speculative one is wasted, never wrong.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Executor, Future
from pathlib import Path

logger = logging.getLogger(__name__)

# What the Force Field panel prefills before the user touches it. Matching it
# is the whole point: a different value computes a different cache entry.
PREWARM_PH = 7.0

_status: dict[str, dict] = {}
_futures: dict[str, Future] = {}
_cancelled: set[str] = set()
_guard = threading.Lock()


def status(task_id: str) -> dict:
    """Return a snapshot of what the background parameterization has done.

    The nested per-molecule mapping is copied too, not just the outer dict: the
    worker mutates it from another thread, so handing out a reference lets a
    caller observe a state that never existed, and lets the JSON encoder walk a
    dict while it changes size.
    """
    with _guard:
        entry = _status.get(task_id)
        if not entry:
            return {"state": "idle", "molecules": {}}
        snapshot = dict(entry)
        snapshot["molecules"] = dict(entry.get("molecules") or {})
        return snapshot


def cancel(task_id: str) -> None:
    """Stop before the next molecule, and forget this task.

    A parameterization already in flight is a subprocess with its own timeout
    and is left to finish: killing it would leave a half-written cache entry,
    and its result stays valid for anyone who asks for the same molecule.
    """
    with _guard:
        _cancelled.add(task_id)
        future = _futures.pop(task_id, None)
        _status.pop(task_id, None)
    if future is not None:
        future.cancel()


def _set(task_id: str, **fields) -> None:
    with _guard:
        entry = _status.setdefault(task_id, {"state": "running", "molecules": {}})
        entry.update(fields)


def _set_molecule(task_id: str, name: str, state: str) -> None:
    with _guard:
        entry = _status.setdefault(task_id, {"state": "running", "molecules": {}})
        entry["molecules"][name] = state


def start(task_id: str, task_dir: str | Path, executor: Executor) -> bool:
    """Begin parameterizing this task's ligands. Returns whether one started.

    The executor must not be the build queue: this is speculative work and must
    never delay a build the user actually asked for.
    """
    with _guard:
        _cancelled.discard(task_id)
        if task_id in _futures and not _futures[task_id].done():
            return False
        _status[task_id] = {
            "state": "running",
            "molecules": {},
            "started_at": time.time(),
            "finished_at": None,
        }
    try:
        future = executor.submit(_prewarm, task_id, Path(task_dir))
    except RuntimeError:
        # The executor is shutting down; the forcefield step will do the work.
        with _guard:
            _status.pop(task_id, None)
        return False
    with _guard:
        _futures[task_id] = future
    return True


def _prewarm(task_id: str, task_dir: Path) -> None:
    from gmxbuilder.core.system import System
    from gmxbuilder.modules.forcefield.compatibility import molecule_groups
    from gmxbuilder.modules.forcefield.gaff_backend import (
        estimate_gaff_net_charge,
        gaff_available,
        gaff_molecule_is_cached,
        prepare_gaff_molecule,
    )
    from gmxbuilder.web.custom_lipids import task_custom_lipid_scope

    try:
        if not gaff_available():
            _set(task_id, state="unavailable", finished_at=time.time())
            return
        checkpoint = task_dir / "steps" / "input"
        if not (checkpoint / "system.npz").is_file():
            _set(task_id, state="idle", finished_at=time.time())
            return

        # The same scope the step runs in, so a task-private custom lipid is
        # recognised as a lipid here too and is not parameterized as a ligand.
        with task_custom_lipid_scope(task_dir):
            system = System.load_checkpoint(checkpoint)
            groups = molecule_groups(system)
            if not groups:
                _set(task_id, state="done", finished_at=time.time())
                return

            for name, instances in groups.items():
                with _guard:
                    if task_id in _cancelled:
                        return
                estimates = [
                    estimate_gaff_net_charge(name, system.structure, indices, PREWARM_PH)
                    for indices in instances
                ]
                charges = {estimate.net_charge for estimate in estimates}
                if len(charges) != 1:
                    # The panel will ask the user; parameterizing at a charge
                    # they have not chosen would be a guess, not a head start.
                    _set_molecule(task_id, name, "ambiguous")
                    continue
                net_charge = charges.pop()
                if gaff_molecule_is_cached(
                    name,
                    system.structure,
                    instances[0],
                    net_charge,
                    target_pH=PREWARM_PH,
                ):
                    _set_molecule(task_id, name, "ready")
                    continue
                _set_molecule(task_id, name, "computing")
                prepare_gaff_molecule(
                    name,
                    system.structure,
                    instances[0],
                    net_charge,
                    target_pH=PREWARM_PH,
                )
                _set_molecule(task_id, name, "ready")
        _set(task_id, state="done", finished_at=time.time())
    except Exception:  # noqa: BLE001 - speculative work must not surface
        # The forcefield step runs the same calculation and reports its own
        # failure with the context the user needs. Nothing is lost here.
        logger.info("Background ligand parameterization failed", exc_info=True)
        _set(task_id, state="error", finished_at=time.time())
    finally:
        with _guard:
            _futures.pop(task_id, None)
