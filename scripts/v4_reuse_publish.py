"""Stage and validate all V4 reuse artifacts before exposing any destination."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from gmxbuilder.modules.membrane.equilibrated_library import (
    EquilibratedLipidLibrary,
    conformer_files,
)
from gmxbuilder.modules.membrane.v4_reuse import (
    file_digest,
    reuse_evidence,
    source_entry,
    target_metadata,
)


def publish(root: Path, entry: dict, source: dict, replicas: int) -> bool:
    """The caller owns the queue lock. Existing targets are never overwritten.

    A failure before publication exposes no target. If a rename fails, remove
    only directories this transaction just installed; a process crash leaves
    identifiable partial work for explicit recovery, never a completed pair.
    """
    from assemble_v4_library import assemble_entry
    from build_v4_library import entry_done, replica_root

    from gmxbuilder.modules.membrane.v4_tpr import verify_tpr
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    destinations = [
        replica_root(root, r) / entry["family"] / entry["lipid"] for r in range(1, replicas + 1)
    ]
    library_target = root / "library" / entry["family"] / entry["lipid"]
    if any(
        path.exists() or path.is_symlink()
        for path in [
            *destinations,
            library_target,
            root / "work" / entry["family"] / entry["lipid"],
        ]
    ):
        return False
    staged = None
    installed = []
    try:
        evidence = reuse_evidence(entry)
        if source_entry(entry) != source or not entry_done(root, source, replicas):
            return False
        original_reader = EquilibratedLipidLibrary(roots=[root / "library"])
        if original_reader.inspect(entry["lipid"], "charmm36m", "charmm36m") is None:
            return False
        gmx = find_gromacs_executable()
        if not gmx or file_digest(gmx) != evidence["gmx_sha256"]:
            return False
        staged = Path(tempfile.mkdtemp(prefix=f".reuse-{entry['lipid']}-", dir=root))
        source_snapshots = []
        for replica in range(1, replicas + 1):
            origin = replica_root(root, replica) / source["family"] / source["lipid"]
            work = root / "work" / source["family"] / source["lipid"] / f"replica-{replica}"
            target = replica_root(staged, replica) / entry["family"] / entry["lipid"]
            target.mkdir(parents=True)
            metadata = json.loads((origin / "metadata.json").read_text())
            from gmxbuilder.modules.membrane.parameter_provenance import validate_work_parameters

            validate_work_parameters(work, metadata)
            checksums = {path.name: file_digest(path) for path in conformer_files(origin)}
            source_snapshots.append(
                (origin, {**checksums, "metadata.json": file_digest(origin / "metadata.json")})
            )
            if {name: file_digest(work / name) for name in evidence["itp_sha256"]} != evidence[
                "itp_sha256"
            ]:
                raise ValueError("Source simulation topology differs from audited parameters")
            original = json.loads((work / "v4-protocol.json").read_text())
            from gmxbuilder.modules.membrane.v4_reanalysis import analysis_upgrade_allowed

            if not analysis_upgrade_allowed(original, evidence["source_protocol"]):
                raise ValueError("Original simulation conditions differ from reuse conditions")
            # Verify a staged TPR: the verifier writes a sidecar, so it must not
            # receive the immutable original simulation path.
            tpr = staged / f"source-{replica}.tpr"
            shutil.copy2(work / "npt.tpr", tpr)
            verification = verify_tpr(gmx, tpr, original, "charmm36m", "npt")
            simulation = {
                "tpr_sha256": verification["tpr_sha256"],
                "original_protocol_sha256": original["sha256"],
                "gmx_sha256": evidence["gmx_sha256"],
                "itp_sha256": evidence["itp_sha256"],
                "effective_inputrec": verification["effective"],
            }
            for name, checksum in checksums.items():
                shutil.copy2(origin / name, target / name)
                if file_digest(target / name) != checksum:
                    raise ValueError("Copied conformer bytes changed")
            metadata = target_metadata(metadata, evidence, checksums, simulation)
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        if not entry_done(staged, entry, replicas):
            raise ValueError("Staged target replicas do not meet V4 admission")
        result = assemble_entry(
            staged,
            [replica_root(staged, r) for r in range(1, replicas + 1)],
            entry["family"],
            entry["lipid"],
            staged / "library",
        )
        if (
            "skipped" in result
            or EquilibratedLipidLibrary(roots=[staged / "library"]).inspect(
                entry["lipid"], "charmm36", "charmm36"
            )
            is None
        ):
            raise ValueError("Staged target failed the production library reader")
        for origin, checksums in source_snapshots:
            if any(file_digest(origin / name) != checksum for name, checksum in checksums.items()):
                raise ValueError("Source changed during reuse publication")
        pairs = [
            (replica_root(staged, r) / entry["family"] / entry["lipid"], destination)
            for r, destination in enumerate(destinations, 1)
        ]
        pairs.append((staged / "library" / entry["family"] / entry["lipid"], library_target))
        for candidate, destination in pairs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ValueError("Target appeared during publication; preserve existing work")
            candidate.rename(destination)
            installed.append(destination)
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        for path in reversed(installed):
            shutil.rmtree(path)
        return False
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
