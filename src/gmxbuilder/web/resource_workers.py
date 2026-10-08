"""Systemd process-tree limits for Web operations, separate from offline V4."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

from gmxbuilder.web.resource_policy import ResourcePolicy

UNIT_PATTERN = re.compile(r"gmxbuilder-op-[0-9a-f]{32}\.service\Z")


class SystemdWorkers:
    def __init__(self, root: Path, policy: ResourcePolicy):
        self.root = root
        self.policy = policy
        self._quotas = {}

    async def command(self, *arguments):
        process = await asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def launch(self, operation, threads):
        unit = f"gmxbuilder-op-{operation['id']}.service"
        ttl = max(1, int(operation["expires"] - time.time()))
        arguments = [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            f"--property=MemoryMax={self.policy.task_memory_bytes}",
            "--property=MemorySwapMax=0",
            "--property=OOMPolicy=kill",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=5",
            f"--property=RuntimeMaxSec={ttl}",
            f"--property=CPUQuota={threads * 100}%",
            "--property=CPUAffinity="
            + " ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))),
            "--property=StandardOutput=null",
            "--property=StandardError=null",
            "--setenv=PYTHONDONTWRITEBYTECODE=1",
        ]
        parent = os.environ.get("GMXBUILDER_SERVICE_UNIT", "")
        if parent:
            if not re.fullmatch(r"[a-zA-Z0-9_.@-]+\.service", parent):
                raise ValueError("Invalid GMXBUILDER_SERVICE_UNIT")
            arguments += [f"--property=PartOf={parent}", f"--property=After={parent}"]
        arguments += [
            sys.executable,
            "-m",
            "gmxbuilder.web.resource_worker",
            str(self.root),
            operation["id"],
            str(threads),
        ]
        code, _, _ = await self.command(*arguments)
        if code:
            raise RuntimeError("Cannot start the isolated worker; check the resource service")
        self._quotas[unit] = threads * 100
        return unit

    async def state(self, unit):
        if not UNIT_PATTERN.fullmatch(unit or ""):
            return "missing"
        _, output, _ = await self.command(
            "systemctl", "--user", "show", unit, "--property=ActiveState", "--value"
        )
        return output.strip()

    async def stop(self, unit):
        if not UNIT_PATTERN.fullmatch(unit or ""):
            raise ValueError("Refusing to stop a unit outside the Web operation namespace")
        await self.command("systemctl", "--user", "stop", unit)
        if await self.state(unit) in {"active", "activating", "deactivating"}:
            raise RuntimeError("Worker has not stopped; its files must remain protected")

    async def memory_current(self, unit):
        if not UNIT_PATTERN.fullmatch(unit or ""):
            return 0
        _, output, _ = await self.command(
            "systemctl", "--user", "show", unit, "--property=MemoryCurrent", "--value"
        )
        return int(output.strip()) if output.strip().isdigit() else 0

    async def rebalance(self, operations):
        units = {op.get("unit") for op in operations}
        self._quotas = {unit: value for unit, value in self._quotas.items() if unit in units}
        if not operations:
            return
        share = self.policy.cpu_cores / len(operations)
        # Reduce every oversized grant before expanding another one. systemd
        # controls allowed CPU time; existing external thread counts stay put.
        grants = [(op, share) for op in operations]
        grants.sort(
            key=lambda pair: (
                -self._quotas.get(
                    pair[0].get("unit"), (pair[0].get("threads") or self.policy.cpu_cores) * 100
                )
            )
        )
        for op, cores in grants:
            unit = op.get("unit")
            if not UNIT_PATTERN.fullmatch(unit or ""):
                continue
            quota = int(cores * 100)
            if self._quotas.get(unit) == quota:
                continue
            code, _, _ = await self.command(
                "systemctl",
                "--user",
                "set-property",
                "--runtime",
                unit,
                f"CPUQuota={quota}%",
            )
            if not code:
                self._quotas[unit] = quota
            if code and await self.state(unit) == "active":
                raise RuntimeError("Cannot adjust worker CPU quota")


def available_memory() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        values[key] = int(value.split()[0]) * 1024
    return values["MemAvailable"]
