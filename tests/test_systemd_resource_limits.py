"""Small real cgroup probes; these never invoke molecular simulation tools."""

import asyncio
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from gmxbuilder.web.resource_policy import ResourcePolicy
from gmxbuilder.web.resource_workers import SystemdWorkers

pytestmark = pytest.mark.slow


@pytest.fixture
def systemd_units():
    if not shutil.which("systemd-run"):
        pytest.skip("systemd user manager unavailable")
    result = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True)
    if result.returncode:
        pytest.skip("systemd user manager unavailable")
    units = []
    yield units
    for unit in units:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)


def start(units, program, *properties):
    unit = "gmxbuilder-op-" + uuid.uuid4().hex + ".service"
    units.append(unit)
    args = [
        "systemd-run",
        "--user",
        "--quiet",
        "--unit=" + unit,
        "--property=RuntimeMaxSec=20",
        "--property=TimeoutStopSec=2",
        "--property=StandardOutput=null",
        "--property=StandardError=null",
    ]
    args += ["--property=" + value for value in properties]
    result = subprocess.run(
        args + ["/usr/bin/python3", "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return unit


def show(unit, property_name):
    return subprocess.check_output(
        ["systemctl", "--user", "show", unit, "--property=" + property_name, "--value"], text=True
    ).strip()


def quota(unit):
    group = show(unit, "ControlGroup")
    maximum, period = (Path("/sys/fs/cgroup") / group.lstrip("/") / "cpu.max").read_text().split()
    return int(maximum) / int(period)


def test_dynamic_cpu_grants_are_enforced_by_kernel(systemd_units, tmp_path, monkeypatch):
    monkeypatch.setenv("GMXBUILDER_CPU_CORES", "2")
    policy = ResourcePolicy.from_environment()
    workers = SystemdWorkers(tmp_path, policy)
    first = start(systemd_units, "import time; time.sleep(15)", "CPUQuota=200%")
    assert quota(first) == 2
    second = start(systemd_units, "import time; time.sleep(15)", "CPUQuota=100%")
    asyncio.run(workers.rebalance([{"unit": first}, {"unit": second}]))
    assert quota(first) + quota(second) == 2
    assert quota(first) == quota(second) == 1
    asyncio.run(workers.stop(second))
    asyncio.run(workers.rebalance([{"unit": first}]))
    assert quota(first) == 2


def test_memory_limit_covers_sum_of_child_processes(systemd_units):
    program = """
import os,time
for _ in range(2):
    if os.fork() == 0:
        memory = bytearray(70*1024*1024)
        time.sleep(15)
        os._exit(0)
time.sleep(15)
"""
    unit = start(systemd_units, program, "MemoryMax=96M", "MemorySwapMax=0", "OOMPolicy=kill")
    for _ in range(100):
        result = show(unit, "Result")
        if result == "oom-kill":
            break
        time.sleep(0.05)
    assert result == "oom-kill"
    assert show(unit, "ActiveState") in {"failed", "inactive", "deactivating"}
