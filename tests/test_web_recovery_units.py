"""The online recovery timer must not create a default-target boot cycle."""

import configparser
from graphlib import TopologicalSorter

import pytest

from gmxbuilder.web.recovery import recovery_units


def boot_order(units):
    # Model systemd's relevant implicit timer/service ordering explicitly.
    graph = {"basic.target": {"timers.target"}, "gmxbuilder-storage.service": {"basic.target"}}
    for name, text in units.items():
        if "/" in name:
            continue
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(text)
        graph.setdefault(name, set()).update(config["Unit"].get("After", "").split())
        if name.endswith(".timer") and config["Unit"].getboolean("DefaultDependencies", True):
            graph.setdefault("timers.target", set()).add(name)
    return tuple(TopologicalSorter(graph).static_order())


def test_recovery_timer_preserves_online_lifecycle_without_boot_cycle(tmp_path):
    units = recovery_units(tmp_path)
    order = boot_order(units)
    assert order.index("basic.target") < order.index("gmxbuilder-storage.service")
    assert order.index("gmxbuilder-storage.service") < order.index("gmxbuilder-online.target")
    assert order.index("gmxbuilder-online.target") < order.index("gmxbuilder-recovery.timer")
    timer = configparser.ConfigParser(interpolation=None)
    timer.read_string(units["gmxbuilder-recovery.timer"])
    assert timer["Unit"]["PartOf"] == timer["Unit"]["Requisite"] == "gmxbuilder-online.target"
    assert timer["Timer"]["Unit"] == "gmxbuilder-recovery.service"
    # The historical default dependency closes the independently modelled cycle.
    from graphlib import CycleError

    units["gmxbuilder-recovery.timer"] = units["gmxbuilder-recovery.timer"].replace(
        "DefaultDependencies=no\n", ""
    )
    with pytest.raises(CycleError):
        boot_order(units)
