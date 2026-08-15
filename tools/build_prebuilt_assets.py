#!/usr/bin/env python3
"""Build a deterministic release archive from validated runtime lipid caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from gmxbuilder import __version__
from gmxbuilder.modules.forcefield.gaff_backend import (
    _cache_key,
    _load_cached,
    _safe_name,
)
from gmxbuilder.modules.forcefield.lipid_policy import gaff_lipid_capability
from gmxbuilder.modules.membrane.equilibrated_library import (
    SCHEMA_VERSION as LIBRARY_SCHEMA_VERSION,
    EquilibratedLipidLibrary,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry


ASSET_VERSION = 3
FORCE_FIELDS = ("amber14sb", "charmm36m", "charmm36")
PUBLIC_ASSET_URL = (
    "https://media.githubusercontent.com/media/AkaseToshiyuki/GMX-BUILDER/"
    "main/src/gmxbuilder/data/prebuilt_assets/"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_deterministic(bundle: tarfile.TarFile, path: Path, root: Path) -> None:
    info = bundle.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if path.is_file():
        with path.open("rb") as handle:
            bundle.addfile(info, handle)
    else:
        bundle.addfile(info)


def build_archive(
    lipid_source: Path,
    gaff_source: Path,
    destination: Path,
) -> dict:
    library = EquilibratedLipidLibrary([lipid_source])
    strict_entries: dict[tuple[str, str], Path] = {}
    coverage = library.coverage(list(FORCE_FIELDS))
    for job in coverage:
        if not job["ready"]:
            continue
        entry = library.inspect(
            job["lipid_name"],
            job["force_field"],
            job["lipid_ff"],
        )
        if entry is None:
            raise RuntimeError(f"Validated entry disappeared: {job}")
        strict_entries[(job["parameter_family"], job["lipid_name"])] = entry.path

    if not strict_entries:
        raise RuntimeError("No validated lipid conformer libraries are available")

    gaff_cache_dirs: dict[str, Path] = {}
    for family, name in sorted(strict_entries):
        if family != "amber-gaff2":
            continue
        capable, reason = gaff_lipid_capability(name)
        if not capable:
            raise RuntimeError(
                f"Validated Amber/GAFF2 entry has no usable GAFF capability for {name}: {reason}"
            )
        lipid = LipidRegistry.get(name)
        safe_name = _safe_name(name)
        key = _cache_key(safe_name, lipid.smiles, int(lipid.charge), "bcc")
        cache_dir = gaff_source / f"{safe_name}-{key}"
        if _load_cached(cache_dir) is None:
            raise RuntimeError(f"Validated GAFF2 cache is missing for {name}")
        gaff_cache_dirs[name] = cache_dir

    with tempfile.TemporaryDirectory(prefix="gmxbuilder-prebuilt-build-") as temp:
        staging = Path(temp)
        strict_root = staging / "lipid_equilibrated"
        gaff_root = staging / "gaff2"
        for (family, name), source in sorted(strict_entries.items()):
            shutil.copytree(source, strict_root / family / name)
        for name, source in sorted(gaff_cache_dirs.items()):
            shutil.copytree(source, gaff_root / source.name)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, mode="w:xz", preset=6) as bundle:
            for path in sorted(staging.rglob("*")):
                _add_deterministic(bundle, path, staging)

        conformers = len(list(strict_root.rglob("conf_*.npz")))
        strict_files = len([path for path in strict_root.rglob("*") if path.is_file()])
        gaff_files = len([path for path in gaff_root.rglob("*") if path.is_file()])
        contents = {
            "compatible_force_field_jobs": len(coverage),
            "validated_force_field_jobs": sum(bool(job["ready"]) for job in coverage),
            "unavailable_force_field_jobs": sum(not bool(job["ready"]) for job in coverage),
            "strict_library_entries": len(strict_entries),
            "strict_library_files": strict_files,
            "conformations": conformers,
            "gaff2_cache_entries": len(gaff_cache_dirs),
            "gaff2_cache_files": gaff_files,
            "force_fields": list(FORCE_FIELDS),
        }

    manifest = {
        "schema_version": 1,
        "asset_version": ASSET_VERSION,
        "library_schema_version": LIBRARY_SCHEMA_VERSION,
        "software_version": __version__,
        "archive": destination.name,
        "download_url": PUBLIC_ASSET_URL + destination.name,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": _sha256(destination),
        "contents": contents,
        "scientific_contract": {
            "strict_method": "explicit_solvent_semiisotropic_npt",
            "minimum_npt_ps": 1000,
            "gaff_charge_method": "am1-bcc",
            "quarantined_or_unavailable_entries_included": False,
        },
    }
    (destination.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lipid-source",
        type=Path,
        default=Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated",
    )
    parser.add_argument(
        "--gaff-source",
        type=Path,
        default=Path.home() / ".cache" / "gmxbuilder" / "gaff2",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "src"
            / "gmxbuilder"
            / "data"
            / "prebuilt_assets"
            / f"gmxbuilder-lipid-assets-v{ASSET_VERSION}.tar.xz"
        ),
    )
    args = parser.parse_args()
    manifest = build_archive(
        args.lipid_source.resolve(),
        args.gaff_source.resolve(),
        args.destination.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
