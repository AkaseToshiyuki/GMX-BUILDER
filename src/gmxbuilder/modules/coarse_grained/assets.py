"""Versioned Martini 3 assets owned by the coarse-grained module."""

from __future__ import annotations

import hashlib
import json
import shutil
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    root = resources.files("gmxbuilder") / "data" / "martini3"
    with resources.as_file(root / "manifest.json") as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _asset_root() -> Path:
    root = resources.files("gmxbuilder") / "data" / "martini3" / "v3.0.0"
    with resources.as_file(root) as path:
        return Path(path)


def verify_assets() -> list[str]:
    """Verify every bundled force-field file and return its name."""
    manifest = load_manifest()
    root = _asset_root()
    verified: list[str] = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ModuleConfigError(f"Required Martini 3 asset is missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ModuleConfigError(
                f"Martini 3 asset checksum mismatch for {name}; reinstall GMXBUILDER"
            )
        verified.append(name)
    return sorted(verified)


def materialize_assets(destination: Path) -> list[Path]:
    """Copy verified force-field inputs into a task-private ``toppar``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = _asset_root()
    paths: list[Path] = []
    for name in verify_assets():
        target = destination / name
        shutil.copy2(root / name, target)
        paths.append(target)
    return paths


def tool_versions() -> dict[str, str | None]:
    manifest = load_manifest()
    result: dict[str, str | None] = {}
    for distribution in ("vermouth", "COBY", "mdtraj"):
        try:
            result[distribution.lower()] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution.lower()] = None
    result["expected"] = dict(manifest["tools"])
    return result


def validate_toolchain() -> dict[str, str]:
    versions = tool_versions()
    expected = versions.pop("expected")
    missing = [name for name, value in versions.items() if value is None]
    if missing:
        raise ModuleConfigError(
            "Martini 3 support is not installed; missing: " + ", ".join(missing)
        )
    mismatched = [
        f"{name}={versions[name]} (expected {expected[name]})"
        for name in expected
        if versions.get(name) != expected[name]
    ]
    if mismatched:
        raise ModuleConfigError(
            "Unsupported Martini 3 tool versions: " + "; ".join(mismatched)
        )
    verify_assets()
    return {name: str(value) for name, value in versions.items()}


def public_capabilities() -> dict:
    manifest = load_manifest()
    error = None
    try:
        tools = validate_toolchain()
        ready = True
    except ModuleConfigError as exc:
        tools = tool_versions()
        ready = False
        error = str(exc)
    return {
        "ready": ready,
        "error": error,
        "bundle_id": manifest["bundle_id"],
        "force_field": manifest["force_field"],
        "tools": tools,
        "lipids": [
            {"name": name, **values}
            for name, values in manifest["lipids"].items()
        ],
        "environments": ["solution", "bilayer"],
        "water_model": "Martini regular water (W)",
        "ions": ["NA", "CL"],
        "citations": list(manifest["citations"]),
        "boundaries": {
            "standard_protein_residues": True,
            "elastic_network": True,
            "custom_molecules": False,
            "post_translational_modifications": False,
            "curved_membranes": False,
            "backmapping": False
        }
    }
