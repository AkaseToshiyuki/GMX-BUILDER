"""Linux Landlock write confinement, inherited by every scientific subprocess.

Reads remain available for installed force fields and executables. Every file
creation, deletion, truncation and write is confined to the managed filesystem.
No root privilege, mount namespace or changes to offline builders are required.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def restrict_writes(root: Path) -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Managed workers require Linux Landlock ABI 3 or newer")
    libc = ctypes.CDLL(None, use_errno=True)
    create, add, restrict = 444, 445, 446
    abi = libc.syscall(create, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Managed workers require Landlock ABI 3 (including truncate)")
    # Linux uapi: WRITE_FILE, REMOVE_*, MAKE_*, REFER and TRUNCATE.
    writes = (1 << 1) | sum(1 << bit for bit in range(4, 15))

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    rules = Ruleset(writes)
    rules_fd = libc.syscall(create, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if rules_fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot create filesystem write sandbox")
    try:
        for path, access in [(root, writes), (Path("/dev/null"), 1 << 1)]:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                if libc.syscall(add, rules_fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "Cannot configure filesystem write sandbox")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(restrict, rules_fd, 0):
            raise OSError(ctypes.get_errno(), "Cannot enforce filesystem write sandbox")
    finally:
        os.close(rules_fd)
