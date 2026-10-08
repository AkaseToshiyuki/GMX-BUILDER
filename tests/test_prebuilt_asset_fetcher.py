"""Tests for the Git-LFS-independent prebuilt-asset bootstrap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_prebuilt_assets.py"
SPEC = importlib.util.spec_from_file_location("gmxbuilder_asset_fetcher", SCRIPT)
assert SPEC and SPEC.loader
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


def _manifest(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "archive": "assets.tar.xz",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "download_url": "https://example.invalid/assets.tar.xz",
            }
        )
    )
    return path


def test_fetch_replaces_lfs_pointer_with_verified_payload(tmp_path, monkeypatch):
    payload = b"verified release payload"
    manifest = _manifest(tmp_path, payload)
    archive = tmp_path / "assets.tar.xz"
    archive.write_text("version https://git-lfs.github.com/spec/v1\n")

    class Response:
        status = 200

        def __init__(self):
            self.sent = False

        def __enter__(self):
            self.offset = 0
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            result = payload[self.offset : self.offset + size]
            self.offset += len(result)
            return result

    monkeypatch.setattr(fetcher, "urlopen", lambda *_args, **_kwargs: Response())
    assert fetcher.fetch(manifest).startswith("downloaded:")
    assert archive.read_bytes() == payload
    assert fetcher.fetch(manifest).startswith("present:")


def test_fetch_rejects_unverified_payload(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, b"expected")

    class Response:
        status = 200

        def __init__(self):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if self.sent:
                return b""
            self.sent = True
            return b"wrong"

    monkeypatch.setattr(fetcher, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match="size or SHA-256"):
        fetcher.fetch(manifest)


def test_fetch_requires_https(tmp_path):
    manifest = _manifest(tmp_path, b"payload")
    data = json.loads(manifest.read_text())
    data["download_url"] = "http://example.invalid/assets.tar.xz"
    manifest.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="HTTPS"):
        fetcher.fetch(manifest)


def test_checkout_fetches_only_its_verified_asset_from_origin_lfs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    payload = b"pre-equilibrated release from private origin LFS"
    manifest = _manifest(tmp_path, payload)
    archive = tmp_path / "assets.tar.xz"
    archive.write_text("version https://git-lfs.github.com/spec/v1\n")
    checksum = hashlib.sha256(payload).hexdigest()
    media = tmp_path / "git-lfs-objects"
    obj = media / checksum[:2] / checksum[2:4] / checksum
    obj.parent.mkdir(parents=True)
    obj.write_bytes(payload)
    commands = []
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")

    def run(command, **kwargs):
        commands.append(command)
        if "rev-parse" in command:
            output = str(tmp_path)
        elif "get-url" in command:
            output = "https://example.invalid/private/project.git"
        elif "smudge" in command:
            assert f"oid sha256:{checksum}".encode() in kwargs["input"]
            assert "GIT_LFS_SKIP_SMUDGE" not in kwargs["env"]
            kwargs["stdout"].write(payload)
            output = ""
        else:
            output = ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(fetcher.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(fetcher.subprocess, "run", run)
    monkeypatch.setattr(fetcher, "urlopen", lambda *a, **k: pytest.fail("public fallback used"))
    assert fetcher.fetch(manifest).startswith("downloaded from origin LFS:")
    assert archive.read_bytes() == payload
    fetched = next(command for command in commands if "smudge" in command)
    assert "lfs.fetchinclude=" in fetched
    assert "lfs.fetchexclude=" in fetched
    assert "lfs.url=https://example.invalid/private/project.git/info/lfs" in fetched
    assert fetched[-2:] == ["smudge", "assets.tar.xz"]
