#!/usr/bin/env python3
"""Verify release distributions contain hydrated assets and no private payloads."""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

from check_public_export import classify


def verify(directory: Path) -> None:
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/gmxbuilder/data/prebuilt_assets/manifest.json"
        ).read_text()
    )
    archives = [*directory.glob("*.whl"), *directory.glob("*.tar.gz")]
    assert len(archives) == 2, "expected one wheel and one source distribution"
    for path in archives:
        source = path.name.endswith(".tar.gz")
        archive = tarfile.open(path) if source else zipfile.ZipFile(path)
        names = archive.getnames() if source else archive.namelist()
        root = path.name.removesuffix(".tar.gz") + "/" if source else ""
        payloads = 0
        for name in names:
            relative = name.removeprefix(root)
            canonical = relative if source else "src/" + relative
            assert classify(canonical) is None, f"private package member: {name}"
            assert not any(
                part in {"AGENTS.md", "CLAUDE.md", "Private"} for part in Path(name).parts
            )
            if relative.endswith(".tar.xz") and "/prebuilt_assets/" in relative:
                handle = archive.extractfile(name) if source else archive.open(name)
                with handle:
                    data = handle.read()
                assert hashlib.sha256(data).hexdigest() == manifest["archive_sha256"]
                assert len(data) == manifest["archive_bytes"]
                payloads += 1
        assert payloads == 1, f"missing bundled lipid archive in {path}"
        archive.close()
        print(f"{path.name}: {len(names)} members; hydrated lipid asset; private boundary passed")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
