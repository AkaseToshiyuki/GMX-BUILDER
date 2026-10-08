"""Private FUSE write volume with atomic aggregate and per-task quotas.

The backing directory is exclusively owned by this daemon. Workers see only
the mount and are write-confined there with Landlock. Charges include rounded
logical file lengths and one 4 KiB metadata charge per inode, so sparse files,
empty-file floods and unlinked-but-open files cannot evade the limits. No
writeback cache is enabled: writes fail with EDQUOT before exceeding a quota.
"""

from __future__ import annotations

import argparse
import errno
import functools
import json
import logging
import os
import stat
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyfuse3
import trio

BLOCK = 4096
logger = logging.getLogger(__name__)
_MUTATIONS = {"create", "mkdir", "write", "setattr", "unlink", "rmdir", "rename", "symlink"}


class QuotaStateError(RuntimeError):
    """An invariant or data-integrity failure requiring storage recovery."""


def charged(size: int, mode: int) -> int:
    return BLOCK + (((size + BLOCK - 1) // BLOCK) * BLOCK if stat.S_ISREG(mode) else 0)


def translate_errors(function):
    @functools.wraps(function)
    async def wrapper(*args, **kwargs):
        try:
            if args[0].write_fault and function.__name__ in _MUTATIONS:
                raise pyfuse3.FUSEError(errno.EIO)
            return await function(*args, **kwargs)
        except pyfuse3.FUSEError:
            raise
        except OSError as exc:
            raise pyfuse3.FUSEError(exc.errno or errno.EIO) from exc
        except Exception as exc:
            # Keep the session readable, but do not write through a potentially
            # inconsistent quota/index state after an internal handler failure.
            args[0].write_fault = True
            logger.exception("Quota filesystem handler %s failed", function.__name__)
            raise pyfuse3.FUSEError(errno.EIO) from exc

    return wrapper


@dataclass
class Node:
    path: Path | None
    owner: str
    charge: int
    handles: int = 0
    control: bool = False
    lookups: int = 0
    detached_fd: int | None = None


class QuotaFS(pyfuse3.Operations):
    """Single-threaded low-level FUSE operations; no await in quota mutations."""

    def __init__(self, backing: Path, total_bytes: int, task_bytes: int):
        super().__init__()
        self.backing = backing.resolve()
        self.total_limit = total_bytes
        self.task_limit = task_bytes
        self.used = 0
        self.write_fault = False
        self._cleanup_log_times: dict[str, float] = {}
        self.owners: dict[str, int] = defaultdict(int)
        self.nodes: dict[int, Node] = {}
        self.paths: dict[Path, int] = {}
        self.fds: dict[int, int] = {}
        self.dir_handles: dict[int, tuple[int, int]] = {}
        self.next_inode = pyfuse3.ROOT_INODE
        self.next_dir = 1
        self._register(self.backing)
        for directory, dirs, files in os.walk(self.backing, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.lstat().st_nlink > 1 and path.is_file() and not path.is_symlink():
                    raise ValueError(f"Backing volume must not contain hard links: {path}")
                self._register(path)
        if self.used > total_bytes or any(v > task_bytes for k, v in self.owners.items() if k):
            raise ValueError("Existing backing data exceeds the configured storage quota")

    def _owner(self, path: Path) -> str:
        parts = path.relative_to(self.backing).parts
        return parts[1] if len(parts) >= 2 and parts[0] == "tasks" else ""

    def _register(self, path: Path) -> int:
        if path in self.paths:
            return self.paths[path]
        st = path.lstat()
        inode = self.next_inode
        self.next_inode += 1
        node = Node(
            path, self._owner(path), charged(st.st_size, st.st_mode), control=self._control(path)
        )
        self.nodes[inode] = node
        self.paths[path] = inode
        self._account(node, node.charge)
        return inode

    def _account(self, node: Node, difference: int) -> None:
        total = self.used + difference
        owner_total = self.owners.get(node.owner, 0) + difference
        if total < 0 or owner_total < 0:
            raise QuotaStateError("Quota accounting would become negative")
        if owner_total:
            self.owners[node.owner] = owner_total
        else:
            self.owners.pop(node.owner, None)
        self.used = total

    def _control(self, path: Path) -> bool:
        parts = path.relative_to(self.backing).parts
        return bool(parts and parts[0] == ".control")

    def _check(self, owner: str, difference: int, *, control=False) -> None:
        if difference <= 0:
            return
        # Keep a small part of the same total quota for failure records and
        # cancellation transactions when an unexpected write fills the volume.
        reserve = 0 if control else min(1024**2, self.total_limit // 100)
        if self.used + difference > self.total_limit - reserve:
            raise pyfuse3.FUSEError(errno.EDQUOT)
        if owner and self.owners.get(owner, 0) + difference > self.task_limit:
            raise pyfuse3.FUSEError(errno.EDQUOT)

    def _path(self, inode: int) -> Path:
        node = self.nodes.get(inode)
        if node is None or node.path is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return node.path

    def _child(self, parent: int, name: bytes) -> Path:
        if name in {b"", b".", b".."} or b"/" in name or b"\0" in name:
            raise pyfuse3.FUSEError(errno.EINVAL)
        return self._path(parent) / os.fsdecode(name)

    def _attrs(self, inode: int) -> pyfuse3.EntryAttributes:
        node = self.nodes.get(inode)
        if node is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        if node.path is not None:
            st = node.path.lstat()
        elif node.detached_fd is not None:
            st = os.fstat(node.detached_fd)
        else:
            fd = next((fd for fd, ino in self.fds.items() if ino == inode), None)
            if fd is None:
                raise pyfuse3.FUSEError(errno.ENOENT)
            st = os.fstat(fd)
        result = pyfuse3.EntryAttributes()
        for key in (
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_rdev",
            "st_size",
            "st_atime_ns",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_blocks",
            "st_blksize",
        ):
            setattr(result, key, getattr(st, key))
        result.st_ino = inode
        result.generation = 0
        result.entry_timeout = 0
        result.attr_timeout = 0
        return result

    def _refresh(self, inode: int, st) -> None:
        node = self.nodes[inode]
        new = charged(st.st_size, st.st_mode)
        self._account(node, new - node.charge)
        node.charge = new

    def _lookup_attrs(self, inode: int) -> pyfuse3.EntryAttributes:
        attributes = self._attrs(inode)
        self.nodes[inode].lookups += 1
        return attributes

    def _retain_detached(self, inode: int) -> int | None:
        node = self.nodes[inode]
        if node.lookups or node.handles:
            # Open before unlink/overwrite. O_PATH also preserves symlink and
            # directory metadata without following their targets or opening IO.
            return os.open(self._path(inode), os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        return None

    def _cleanup_warning(self, kind: str, error_number: int | None = None) -> None:
        # Keys are a fixed set of cleanup kinds, never per-inode identifiers.
        now = time.monotonic()
        if now - self._cleanup_log_times.get(kind, float("-inf")) >= 60:
            self._cleanup_log_times[kind] = now
            logger.warning("Quota cleanup anomaly: %s (errno=%s)", kind, error_number)

    def _close_once(self, fd: int, *, metadata_only: bool = False) -> None:
        # Linux releases the descriptor even on close errors (EBADF means it
        # was already invalid). Never retry: its integer may have been reused.
        try:
            os.close(fd)
        except OSError as exc:
            if metadata_only and exc.errno in {errno.EBADF, errno.EINTR}:
                self._cleanup_warning("metadata-close", exc.errno)
                return
            # In particular, EIO/ENOSPC/EDQUOT can report failed data writes.
            raise QuotaStateError(f"Backing descriptor close failed (errno={exc.errno})") from exc

    def _collect(self, inode: int) -> None:
        node = self.nodes[inode]
        if node.handles < 0 or node.lookups < 0:
            raise QuotaStateError("Negative inode reference count")
        if node.path is None and not node.handles and not node.lookups:
            if (
                inode in self.fds.values()
                or any(open_inode == inode for open_inode, _fd in self.dir_handles.values())
                or node.charge < BLOCK
                or self.used < node.charge
                or self.owners.get(node.owner, 0) < node.charge
            ):
                raise QuotaStateError("Unreferenced inode still has handles or invalid quota")
            if node.detached_fd is not None:
                fd, node.detached_fd = node.detached_fd, None
                self._close_once(fd, metadata_only=True)
            self._account(node, -node.charge)
            self.nodes.pop(inode)

    def _detach(self, inode: int, retained_fd: int | None) -> None:
        node = self.nodes[inode]
        self.paths.pop(node.path, None)
        node.path = None
        node.detached_fd = retained_fd
        self._collect(inode)

    async def forget(self, inode_list):
        # FUSE forget has no reply and must not raise, even on shutdown races.
        for inode, count in inode_list:
            try:
                node = self.nodes.get(inode)
                if node is None:
                    continue
                if count < 0 or count > node.lookups:
                    self.write_fault = True
                    logger.error("Invalid quota inode lookup reference count")
                    continue
                node.lookups -= count
                self._collect(inode)
            except Exception:
                self.write_fault = True
                logger.exception("Quota filesystem could not forget an inode")

    @translate_errors
    async def lookup(self, parent_inode, name, ctx=None):
        if name == b".":
            return self._lookup_attrs(parent_inode)
        if name == b"..":
            parent = self._path(parent_inode).parent
            return self._lookup_attrs(self.paths.get(parent, pyfuse3.ROOT_INODE))
        return self._lookup_attrs(self._register(self._child(parent_inode, name)))

    @translate_errors
    async def getattr(self, inode, ctx=None):
        return self._attrs(inode)

    @translate_errors
    async def opendir(self, inode, ctx):
        handle = self.next_dir
        self.next_dir += 1
        fd = os.open(self._path(inode), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.dir_handles[handle] = (inode, fd)
        self.nodes[inode].handles += 1
        return handle

    @translate_errors
    async def readdir(self, fh, start_id, token):
        directory_inode, fd = self.dir_handles[fh]
        path = self.nodes[directory_inode].path
        if path is None:
            # A successfully removed directory was empty; its open handle
            # remains valid until releasedir even after its name disappears.
            return
        # Inode cookies remain stable if another request creates/removes files.
        entries = sorted((self._register(path / name), name) for name in os.listdir(fd))
        for inode, name in entries:
            if inode > start_id:
                if not pyfuse3.readdir_reply(token, os.fsencode(name), self._attrs(inode), inode):
                    break
                self.nodes[inode].lookups += 1

    @translate_errors
    async def releasedir(self, fh):
        entry = self.dir_handles.get(fh)
        if entry is None:
            self._cleanup_warning("unknown-directory-handle")
            return
        inode, fd = entry
        node = self.nodes[inode]
        if node.handles <= 0:
            raise QuotaStateError("Directory handle has no inode reference")
        self.dir_handles.pop(fh)
        node.handles -= 1
        self._close_once(fd, metadata_only=True)
        self._collect(inode)

    @translate_errors
    async def mkdir(self, parent_inode, name, mode, ctx):
        path = self._child(parent_inode, name)
        self._check(self._owner(path), BLOCK, control=self._control(path))
        path.mkdir(mode=mode)
        return self._lookup_attrs(self._register(path))

    def _opened(self, inode, fd):
        self.fds[fd] = inode
        self.nodes[inode].handles += 1
        info = pyfuse3.FileInfo(fh=fd)
        # SQLite WAL coordinates readers through a small shared mmap. Only
        # its -shm file needs mmap support; data writes retain direct I/O.
        path = self.nodes[inode].path
        info.direct_io = not (
            path
            and path.name.endswith("-shm")
            and ".control" in path.relative_to(self.backing).parts
        )
        info.keep_cache = False
        return info

    @translate_errors
    async def create(self, parent_inode, name, mode, flags, ctx):
        path = self._child(parent_inode, name)
        self._check(self._owner(path), BLOCK, control=self._control(path))
        fd = os.open(path, (flags & ~os.O_APPEND) | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        inode = self._register(path)
        return self._opened(inode, fd), self._lookup_attrs(inode)

    @translate_errors
    async def open(self, inode, flags, ctx):
        if self.write_fault and flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_CREAT):
            raise pyfuse3.FUSEError(errno.EIO)
        node = self.nodes.get(inode)
        if node is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        if node.path is None and node.detached_fd is not None:
            # Only reopen a retained regular inode, never an unlinked symlink.
            if not stat.S_ISREG(os.fstat(node.detached_fd).st_mode):
                raise pyfuse3.FUSEError(errno.ELOOP)
            fd = os.open(f"/proc/self/fd/{node.detached_fd}", flags & ~os.O_APPEND)
        else:
            fd = os.open(self._path(inode), (flags & ~os.O_APPEND) | os.O_NOFOLLOW)
        self._refresh(inode, os.fstat(fd))
        return self._opened(inode, fd)

    @translate_errors
    async def read(self, fh, off, size):
        return os.pread(fh, size, off)

    @translate_errors
    async def write(self, fh, off, buf):
        inode = self.fds[fh]
        node = self.nodes[inode]
        st = os.fstat(fh)
        wanted = charged(max(st.st_size, off + len(buf)), st.st_mode)
        self._check(node.owner, wanted - node.charge, control=node.control)
        try:
            return os.pwrite(fh, buf, off)
        finally:
            self._refresh(inode, os.fstat(fh))

    @translate_errors
    async def setattr(self, inode, attr, fields, fh, ctx):
        node = self.nodes.get(inode)
        if node is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        current = self._attrs(inode)
        symlink = stat.S_ISLNK(current.st_mode)
        if symlink and (fields.update_size or fields.update_mode):
            raise pyfuse3.FUSEError(errno.EOPNOTSUPP)
        if fh is not None:
            target = fh
        elif node.path is None and node.detached_fd is not None and not symlink:
            target = f"/proc/self/fd/{node.detached_fd}"
        else:
            target = self._path(inode)
        # Symlink metadata operations must not modify the target outside the
        # backing volume. Regular retained inodes use our own stable O_PATH fd.
        options = {"follow_symlinks": False} if symlink else {}
        if fields.update_size:
            wanted = charged(attr.st_size, current.st_mode)
            self._check(node.owner, wanted - node.charge, control=node.control)
            os.truncate(target, attr.st_size)
            st = os.fstat(fh) if fh is not None else os.stat(target)
            self._refresh(inode, st)
        if fields.update_mode:
            os.chmod(target, attr.st_mode)
        if fields.update_uid or fields.update_gid:
            os.chown(
                target,
                attr.st_uid if fields.update_uid else -1,
                attr.st_gid if fields.update_gid else -1,
                **options,
            )
        if fields.update_atime or fields.update_mtime:
            current = self._attrs(inode)
            os.utime(
                target,
                ns=(
                    attr.st_atime_ns if fields.update_atime else current.st_atime_ns,
                    attr.st_mtime_ns if fields.update_mtime else current.st_mtime_ns,
                ),
                **options,
            )
        return self._attrs(inode)

    @translate_errors
    async def flush(self, fh):
        os.fsync(fh)

    @translate_errors
    async def fsync(self, fh, datasync):
        os.fdatasync(fh) if datasync else os.fsync(fh)

    @translate_errors
    async def release(self, fh):
        inode = self.fds.get(fh)
        if inode is None:
            self._cleanup_warning("unknown-file-handle")
            return
        node = self.nodes[inode]
        if node.handles <= 0:
            raise QuotaStateError("File handle has no inode reference")
        self.fds.pop(fh)
        node.handles -= 1
        self._close_once(fh)
        self._collect(inode)

    @translate_errors
    async def unlink(self, parent_inode, name, ctx):
        path = self._child(parent_inode, name)
        inode = self._register(path)
        retained = self._retain_detached(inode)
        try:
            path.unlink()
        except BaseException:
            if retained is not None:
                os.close(retained)
            raise
        self._detach(inode, retained)

    @translate_errors
    async def rmdir(self, parent_inode, name, ctx):
        path = self._child(parent_inode, name)
        inode = self._register(path)
        retained = self._retain_detached(inode)
        try:
            path.rmdir()
        except BaseException:
            if retained is not None:
                os.close(retained)
            raise
        self._detach(inode, retained)

    @translate_errors
    async def rename(self, parent_inode_old, name_old, parent_inode_new, name_new, flags, ctx):
        old = self._child(parent_inode_old, name_old)
        new = self._child(parent_inode_new, name_new)
        if (
            flags
            or self._owner(old) != self._owner(new)
            or self._control(old) != self._control(new)
        ):
            raise pyfuse3.FUSEError(errno.EXDEV)
        if old == new:
            return
        replaced = self.paths.get(new)
        retained = self._retain_detached(replaced) if replaced is not None else None
        try:
            os.rename(old, new)
        except BaseException:
            if retained is not None:
                os.close(retained)
            raise
        if replaced is not None:
            self._detach(replaced, retained)
        for path, inode in list(self.paths.items()):
            if path == old or old in path.parents:
                moved = new / path.relative_to(old)
                self.nodes[inode].path = moved
                self.paths.pop(path)
                self.paths[moved] = inode

    @translate_errors
    async def symlink(self, parent_inode, name, target, ctx):
        path = self._child(parent_inode, name)
        self._check(self._owner(path), BLOCK, control=self._control(path))
        path.symlink_to(os.fsdecode(target))
        return self._lookup_attrs(self._register(path))

    @translate_errors
    async def readlink(self, inode, ctx):
        node = self.nodes.get(inode)
        if node is not None and node.path is None and node.detached_fd is not None:
            return os.fsencode(os.readlink("", dir_fd=node.detached_fd))
        return os.fsencode(os.readlink(self._path(inode)))

    @translate_errors
    async def statfs(self, ctx):
        result = pyfuse3.StatvfsData()
        underlying = os.statvfs(self.backing)
        remaining = min(self.total_limit - self.used, underlying.f_bavail * underlying.f_frsize)
        result.f_bsize = result.f_frsize = BLOCK
        result.f_blocks = self.total_limit // BLOCK
        result.f_bfree = result.f_bavail = max(0, remaining // BLOCK)
        result.f_files = self.total_limit // BLOCK
        result.f_ffree = result.f_favail = max(0, remaining // BLOCK)
        result.f_namemax = underlying.f_namemax
        return result

    async def getxattr(self, inode, name, ctx):
        if inode == pyfuse3.ROOT_INODE and name == b"user.gmxbuilder.health":
            return json.dumps({"write_fault": self.write_fault}).encode()
        if inode == pyfuse3.ROOT_INODE and name == b"user.gmxbuilder.quota":
            return json.dumps(
                {"total_bytes": self.total_limit, "task_bytes": self.task_limit}
            ).encode()
        raise pyfuse3.FUSEError(errno.ENODATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backing", type=Path, required=True)
    parser.add_argument("--mount", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--task-bytes", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.task_bytes < args.total_bytes:
        parser.error("Require 0 < task bytes < total bytes")
    if args.backing.resolve() == args.mount.resolve():
        parser.error("Mount and backing paths must be distinct")
    args.backing.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.mount.mkdir(parents=True, exist_ok=True, mode=0o700)
    fs = QuotaFS(args.backing, args.total_bytes, args.task_bytes)
    options = set(pyfuse3.default_options) | {"fsname=gmxbuilder-quota", "default_permissions"}
    pyfuse3.init(fs, str(args.mount), options)
    try:
        trio.run(pyfuse3.main)
    finally:
        pyfuse3.close(unmount=True)


if __name__ == "__main__":
    main()
