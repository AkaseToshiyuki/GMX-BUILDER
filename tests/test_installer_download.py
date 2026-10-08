"""Bound untrusted transfer bytes before hashing or publishing installer assets."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


def helper(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("header", [None, "nonsense", "-1", "1", "1000"])
def test_actual_stream_budget_and_cleanup_independent_of_header(tmp_path, header):
    module = helper("installer_download")
    target = tmp_path / "asset"
    target.write_bytes(b"previous")
    response = io.BytesIO(b"x" * 50)
    response.headers = {} if header is None else {"Content-Length": header}
    with pytest.raises(RuntimeError, match="size budget"):
        module.download_verified(
            "https://example.invalid/asset",
            target,
            "0" * 64,
            max_bytes=10,
            opener=lambda *a, **k: response,
        )
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.download-*"))


def test_productive_stream_still_obeys_total_deadline(tmp_path, monkeypatch):
    module = helper("installer_download")
    clock = [0.0]

    class Trickle(io.BytesIO):
        def read1(self, size):
            clock[0] += 500
            return b"x"

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    with pytest.raises(RuntimeError, match="time budget"):
        module.download_verified(
            "https://example.invalid/asset",
            tmp_path / "asset",
            "0" * 64,
            opener=lambda *a, **k: Trickle(),
        )
    assert not list(tmp_path.iterdir())


def test_verified_download_is_atomic_and_cached(tmp_path):
    module = helper("installer_download")
    body = b"valid asset"
    digest = hashlib.sha256(body).hexdigest()
    target = tmp_path / "asset"
    module.download_verified(
        "https://example.invalid/asset",
        target,
        digest,
        opener=lambda *a, **k: io.BytesIO(body),
    )
    assert target.read_bytes() == body
    module.download_verified(
        "https://example.invalid/asset",
        target,
        digest,
        opener=lambda *a, **k: pytest.fail("Verified cache must not redownload"),
    )
    assert not list(tmp_path.glob("*.download-*"))


@pytest.mark.parametrize("installer_name", ["install_gromacs", "install_gaff_runtime"])
def test_installer_entry_points_use_bounded_transfer(tmp_path, monkeypatch, installer_name):
    installer = helper(installer_name)
    original = installer._download.download_verified
    monkeypatch.setattr(
        installer._download,
        "download_verified",
        lambda *a, **k: original(*a, **{**k, "max_bytes": 10}),
    )
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 50))
    with pytest.raises(RuntimeError, match="size budget"):
        if installer_name == "install_gromacs":
            installer._fetch_verified("https://example.invalid/asset", tmp_path / "asset", "0" * 64)
        else:
            installer._fetch_archive(
                {"url": "https://example.invalid/asset", "sha256": "0" * 64}, tmp_path
            )
    assert not list(tmp_path.iterdir())


def test_micromamba_transfer_is_bounded_before_execution(tmp_path, monkeypatch):
    installer = helper("install_gaff_runtime")
    original = installer._download.download_verified
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        installer._download,
        "download_verified",
        lambda *a, **k: original(*a, **{**k, "max_bytes": 10}),
    )
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 50))
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: pytest.fail("Unverified tool executed")
    )
    with pytest.raises(RuntimeError, match="size budget"):
        installer._install_locked(tmp_path / "prefix", tmp_path / "runtime")
    assert not list((tmp_path / "runtime/tools").iterdir())


def test_real_http_trailers_cannot_extend_absolute_deadline(tmp_path, monkeypatch):
    import socket
    import threading
    import time

    helper_module = helper("installer_download")
    monkeypatch.setattr(helper_module, "MAX_TRANSFER_SECONDS", 0.3)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    done = threading.Event()

    def serve():
        try:
            with sock.accept()[0] as conn:
                conn.recv(4096)
                conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n")
                while not done.wait(0.01):
                    conn.sendall(b"X-Trace: alive\r\n")
        except OSError:
            pass

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="time budget"):
            helper_module.download_verified(
                f"http://127.0.0.1:{sock.getsockname()[1]}/",
                tmp_path / "asset",
                "0" * 64,
                max_bytes=1,
            )
        assert time.monotonic() - started < 2
        assert not list(tmp_path.iterdir())
    finally:
        done.set()
        sock.close()
        worker.join(timeout=1)
