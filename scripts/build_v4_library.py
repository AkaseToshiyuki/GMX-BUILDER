#!/usr/bin/env python3
"""Build the V4 pre-equilibrated lipid library, one entry at a time.

V3 published ``box_x * box_y / 64`` of the final frame of a 1 ns NPT run.
Measured against two 100 ns replicas of the same system, that number moves
2.26% between velocity seeds and sits 8% below the equilibrium area -- it is a
construction artefact, not a measurement. V4 runs long enough for the evidence
to exist and runs each entry twice so the answer can be checked against itself.

Two replicas rather than one longer run because the cross-seed comparison is
the whole difference between V4 and V3. It does *not* reduce the cost: two
50 ns runs and one 100 ns run are the same simulation.

**V3 is not touched.** Each replica writes to its own root under the V4
directory, so the shipped library keeps working throughout and the switch is a
separate, later decision.

    python scripts/build_v4_library.py                # start or resume
    python scripts/build_v4_library.py --dry-run      # show the queue
    python scripts/build_v4_library.py --only POPC    # one lipid, all families

Resume preserves physically valid existing replicas. Each replica needs at least
ten effective local-conformation samples for reviewed V4 queue completion.
Area convergence remains an
independent diagnostic; neither statistic limits how many frames are analysed.
Total runtime depends on selected families, device throughput and extensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_V3_ROOT = Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated"
DEFAULT_V4_ROOT = Path.home() / ".cache" / "gmxbuilder" / "lipid_equilibrated_v4"

#: Seeds are derived from the entry so a resumed queue reproduces exactly what
#: an uninterrupted one would have built.
SEED_BASE = 20260905

# First assessment for a newly initialized replica, not a minimum duration
# that establishes equilibrium. Existing checkpoints retain their elapsed time.
# Prefix replay of BSM/CAMP/CER18 at 30 and 40 ns did not admit any of them;
# this limited check supports earlier assessment, not a universal error bound.
INITIAL_NS = 30.0

# Continue from the same checkpoint in 10 ns increments until construction
# acceptance or the resource cap. More frequent analysis does not guarantee
# new independent samples, and failed physical/identity evidence is not waived.
EXTENSION_NS = 10.0

#: A resource limit, not evidence of a phase or convergence.
MAXIMUM_NS = 200.0

# Maintainer policy: ten independent samples per replica is a minimum; V4
# additionally requires physical gates and late-window practical equivalence.
# Area-model convergence is a separate diagnostic and never silently extends
# the queue's resource budget. Keep both verdicts in progress records.
MINIMUM_EFFECTIVE_SAMPLES = 10.0


def entry_seed(family: str, lipid: str, replica: int) -> int:
    """A stable seed per entry and replica.

    hashlib, not ``hash()``: the built-in is salted per process, so a queue
    restarted tomorrow would give every remaining entry different velocities
    from the ones an uninterrupted run would have used -- which would quietly
    make "resume" mean something other than "continue".
    """
    key = f"{family}/{lipid}/replica-{replica}/{SEED_BASE}".encode()
    digest = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return SEED_BASE + (digest % 900_000_000)


def unresolved_stereochemistry(lipid: str, *, compare_native: bool = True) -> str | None:
    """Why this lipid cannot be built yet, or None.

    A registry structure that does not say which stereoisomer it is cannot
    produce a library entry that does. RDKit embeds an arbitrary configuration
    at every unspecified centre, independently per conformer, which is how the
    CHARMM conformer sets came to hold four different diastereomers of POPG.
    An entry built that way looks perfectly healthy: bulk area and thickness
    are almost unchanged, so no quality gate notices.

    So this is checked before anything is simulated rather than after.
    """
    from rdkit import Chem

    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    smiles = LipidRegistry.get(lipid).smiles
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "its structure cannot be parsed"
    from gmxbuilder.geometry.stereochemistry import stereo_reference

    molecule = stereo_reference(molecule)
    centres = sum(
        1
        for index, code in Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        if code == "?" and molecule.GetAtomWithIdx(index).GetAtomicNum() == 6
    )
    bonds = sum(
        1
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() == Chem.BondStereo.STEREONONE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        and not bond.GetBeginAtom().GetIsAromatic()
        and not (bond.GetBeginAtom().IsInRing() or bond.GetEndAtom().IsInRing())
    )
    if centres or bonds:
        return (
            f"its structure leaves {centres} stereocentre(s) and {bonds} double bond(s) "
            "unspecified, so any entry built from it would be an arbitrary isomer"
        )
    return _unmappable_structure(lipid) if compare_native else None


def _unmappable_structure(lipid: str) -> str | None:
    """Check connectivity without embedding or spending simulation resources.

    A geometry embedding failure is not evidence of a connectivity mismatch.
    Compare the complete element/bond/hydrogen graph directly here; coordinate
    stereochemistry is independently checked when preparation produces atoms.
    """
    from gmxbuilder.geometry.molecular_identity import reference_mappings
    from gmxbuilder.geometry.rdkit_lipid import _molecule_from_rtp
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    smiles = LipidRegistry.get(lipid).smiles
    if not smiles or "@" not in smiles:
        return None
    for force_field in ("charmm36m", "charmm36"):
        try:
            _name, rtp = lipid_rtp_template(lipid, force_field)
        except (KeyError, ValueError, RuntimeError):
            continue
        if rtp is None:
            continue
        try:
            molecule, _names = _molecule_from_rtp(rtp)
            reference_mappings(
                smiles,
                tuple(atom.GetSymbol() for atom in molecule.GetAtoms()),
                tuple(
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
                ),
            )
            return None
        except (KeyError, ValueError, RuntimeError):
            pass
        return (
            "its declared structure cannot be mapped onto the force field's own "
            f"{force_field} residue, so the two disagree about its connectivity"
        )
    return None


def policy_refusal(family: str, lipid: str, force_field: str) -> str | None:
    """Why a pairing cannot enter V4 preparation.

    Historical failures remain unavailable in ordinary user builds, but are
    explicitly retried here with corrected inputs. Admission to this queue is
    not acceptance: exact identity, parameter and physical gates still run.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import (
        charmm_lipid_capability,
        gaff_lipid_capability,
        rebuilding_library_entry,
    )

    # Asked inside the rebuild scope. Outside it, a lipid whose entry was built
    # from a since-corrected structure reports as unavailable -- which is right
    # for a user and exactly wrong here, because this queue exists to rebuild
    # those entries. Without the scope it skipped 41 of the entries it was
    # started to produce, and reported them as "policy forbids the pairing".
    with rebuilding_library_entry(retry_failed_validation=True):
        if family == "amber-gaff2":
            allowed, reason = gaff_lipid_capability(lipid)
        elif family.startswith("charmm36"):
            allowed, reason = charmm_lipid_capability(lipid, force_field)
        else:
            return None
    return None if allowed else reason


def capability_entries(*, include_gaff_overlap: bool = False) -> list[dict]:
    """Enumerate supported lipid/force-field pairs from installed capabilities.

    Include pairs absent from the historical V3 cache. Coverage is determined by
    current identity and parameter support; queue policy checks eligibility later."""
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability
    from gmxbuilder.modules.forcefield.lipid_policy import (
        charmm_lipid_capability,
        gaff_lipid_capability,
        rebuilding_library_entry,
    )
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    families = (
        (
            "charmm36m-lipid",
            "charmm36m",
            "charmm36m",
            lambda name: charmm_lipid_capability(name, "charmm36m")[0],
        ),
        (
            "charmm36-lipid",
            "charmm36",
            "charmm36",
            lambda name: charmm_lipid_capability(name, "charmm36")[0],
        ),
        ("amber-lipid21", "amber14sb", "lipid21", lambda name: lipid21_capability(name)[0]),
        ("amber-gaff2", "amber14sb", "gaff2", lambda name: gaff_lipid_capability(name)[0]),
    )
    entries: list[dict] = []
    with rebuilding_library_entry(retry_failed_validation=True):
        for name in sorted(LipidRegistry.list_builtin()):
            for family, force_field, lipid_ff, capable in families:
                try:
                    supported = bool(capable(name))
                except (KeyError, ValueError, RuntimeError):
                    supported = False
                if family == "amber-gaff2" and not include_gaff_overlap:
                    supported = supported and not lipid21_capability(name)[0]
                    lipid_ff = "amber-mixed"
                if supported:
                    entries.append(
                        {
                            "family": family,
                            "lipid": name,
                            "force_field": force_field,
                            "lipid_ff": lipid_ff,
                        }
                    )
    return entries


def discover_entries(v3_root: Path, *, include_gaff_overlap: bool = False) -> list[dict]:
    """Current parameters define coverage; old cache metadata cannot add models."""
    return capability_entries(include_gaff_overlap=include_gaff_overlap)


def replica_root(v4_root: Path, replica: int) -> Path:
    return v4_root / f"replica-{replica}"


def effective_samples(v4_root: Path, entry: dict, replicas: int) -> float | None:
    """The weakest replica's independent-sample count, or None if unreadable."""
    counts = []
    for replica in range(1, replicas + 1):
        path = replica_root(v4_root, replica) / entry["family"] / entry["lipid"] / "metadata.json"
        try:
            metadata = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        observables = metadata.get("observables")
        if not isinstance(observables, dict):
            return None
        area = observables.get("area_per_lipid_nm2")
        if not isinstance(area, dict) or "effective_samples" not in area:
            return None
        if metadata.get("v4_protocol"):
            from gmxbuilder.modules.membrane.local_relaxation import assess_local

            local = metadata.get("trajectory_analysis", {}).get("local_conformations")
            if not local:
                return None
            counts.append(assess_local(local)["minimum_effective_samples"])
        else:
            counts.append(float(area["effective_samples"]))
    return min(counts) if counts else None


def sampling_summary(v4_root, entry, replicas):
    """Report independent times separately from correlated molecular conformers."""
    result = []
    for replica in range(1, replicas + 1):
        path = replica_root(v4_root, replica) / entry["family"] / entry["lipid"] / "metadata.json"
        metadata = json.loads(path.read_text())
        analysis = metadata.get("trajectory_analysis")
        if not analysis:
            continue
        from gmxbuilder.modules.membrane.local_relaxation import assess_local

        local = assess_local(analysis["local_conformations"])
        fields = {
            key: value
            for key, value in local["statistics"].items()
            if key.split(":", 1)[0] in local["policy"]["hard_interval_observables"]
        }
        limiting = min(fields, key=lambda name: fields[name]["effective_samples"])
        result.append(
            {
                "replica": replica,
                "npt_ns": metadata["npt_ps"] / 1000,
                "analysis_window_ns": [time / 1000 for time in local["analysis_window_ps"]],
                "limiting_observable": limiting,
                "minimum_effective_samples": fields[limiting]["effective_samples"],
                "retained_sampling_times": len(
                    {item["time_ps"] for item in metadata["conformer_provenance"]}
                ),
                "stored_conformers": metadata["n_conformations"],
            }
        )
    return result


def reuse_source(entry: dict) -> dict | None:
    """Reuse only audited CHARMM pairs; V4 also binds conditions and provenance."""
    if entry.get("v4_protocol"):
        from gmxbuilder.modules.membrane.v4_reuse import source_entry

        return source_entry(entry)
    if entry["family"] != "charmm36-lipid":
        return None
    from gmxbuilder.modules.forcefield.charmm_lipid_equivalence import (
        equivalent_lipid_definition,
    )

    equivalent, _reason = equivalent_lipid_definition(entry["lipid"])
    if not equivalent:
        return None
    return {**entry, "family": "charmm36m-lipid", "force_field": "charmm36m"}


def publish_reuse(v4_root: Path, entry: dict, source: dict, replicas: int) -> bool:
    """Publish a CHARMM36 entry from the CHARMM36m one, recording that it was.

    Every replica of the source must be a finished entry before anything is
    copied -- a source that merely has a metadata file could be a failed build,
    an interrupted write, or a V3 entry that arrived when the prebuilt archive
    was unpacked, and copying one of those would mark the pairing done without
    anything having been simulated.

    Reuse only fills an entirely absent entry. Existing replicas or retained
    simulation work are preserved for normal resume or explicit recovery.
    Each replica is staged beside its destination and published by rename;
    an interrupted multi-replica publication can leave a valid partial pair.
    """
    if entry.get("v4_protocol"):
        from v4_reuse_publish import publish

        return publish(v4_root, entry, source, replicas)
    sources = [
        replica_root(v4_root, replica) / source["family"] / source["lipid"]
        for replica in range(1, replicas + 1)
    ]
    if not entry_done(v4_root, source, replicas):
        return False

    targets = [origin.parent.parent / entry["family"] / entry["lipid"] for origin in sources]
    if (
        any(target.exists() for target in targets)
        or (v4_root / "work" / entry["family"] / entry["lipid"]).exists()
    ):
        return False

    staged: list[tuple[Path, Path]] = []
    try:
        for origin in sources:
            target = origin.parent.parent / entry["family"] / entry["lipid"]
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=target.name + ".staging-", dir=target.parent))
            staged.append((staging, target))
            shutil.copytree(origin, staging, dirs_exist_ok=True)
            metadata_path = staging / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            from gmxbuilder.modules.membrane.equilibrated_library import topology_signature

            source_signature = metadata["topology_sha256"]
            metadata["force_field"] = entry["force_field"]
            metadata["lipid_ff"] = entry["lipid_ff"]
            metadata["parameter_family"] = entry["family"]
            # The reader's identity signature includes the force-field namespace.
            # Preserve the source signature as provenance, then sign the new namespace.
            metadata["topology_sha256"] = topology_signature(
                metadata["atom_names"], entry["force_field"], entry["lipid_ff"]
            )
            from gmxbuilder.modules.membrane.parameter_provenance import fingerprint_from_metadata

            metadata["parameter_fingerprint"] = fingerprint_from_metadata(metadata)
            metadata["reused_from"] = {
                "source_topology_sha256": source_signature,
                "family": source["family"],
                "force_field": source["force_field"],
                "reason": _reuse_reason(entry["lipid"]),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            if not _replica_complete(staging):
                raise RuntimeError("the staged copy is not a finished entry")
    except (OSError, ValueError, RuntimeError):
        for staging, _target in staged:
            shutil.rmtree(staging, ignore_errors=True)
        return False

    try:
        for staging, target in staged:
            staging.rename(target)
    finally:
        for staging, _target in staged:
            shutil.rmtree(staging, ignore_errors=True)
    return entry_done(v4_root, entry, replicas)


def _reuse_reason(lipid: str) -> str:
    from gmxbuilder.modules.forcefield.charmm_lipid_equivalence import (
        equivalent_lipid_definition,
    )

    _equivalent, detail = equivalent_lipid_definition(lipid)
    return (
        "CHARMM36 and CHARMM36m resolve this lipid to the same template: "
        f"{detail}. Simulating both would be the same simulation twice."
    )


def entry_done(v4_root: Path, entry: dict, replicas: int) -> bool:
    """Physical artifacts and ten independent samples per replica complete a job."""
    return entry_assessment(v4_root, entry, replicas)["queue_complete"]


def entry_assessment(v4_root: Path, entry: dict, replicas: int) -> dict:
    from gmxbuilder.modules.membrane.area_observable import AreaMeasurement, pool_replicas

    result = {"built": False, "queue_complete": False, "area_converged": False, "reason": None}
    metadatas = []
    for replica in range(1, replicas + 1):
        directory = replica_root(v4_root, replica) / entry["family"] / entry["lipid"]
        if not _replica_complete(directory):
            return {**result, "reason": f"replica {replica} is missing or physically invalid"}
        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata.get("lipid_name") != entry["lipid"]
            or metadata.get("parameter_family") != entry["family"]
        ):
            return {
                **result,
                "reason": f"replica {replica} identity does not match the queue entry",
            }
        if entry.get("v4_protocol") and metadata.get("v4_protocol") != entry["v4_protocol"]:
            return {**result, "reason": "Entry predates or differs from the reviewed V4 protocol"}
        metadatas.append(metadata)
    if any(m.get("v4_protocol") for m in metadatas):
        return v4_entry_assessment(entry, metadatas)
    for field in (
        "canonical_smiles",
        "topology_sha256",
        "atom_names",
        "temperature_K",
        "equilibration_host",
    ):
        values = [m.get(field) for m in metadatas]
        if any(value != values[0] for value in values[1:]) or (
            field != "equilibration_host" and not values[0]
        ):
            return {**result, "reason": f"replicas have absent or inconsistent {field}"}
    try:
        measurements = [AreaMeasurement.from_metadata(m["observables"]) for m in metadatas]
        pooled = pool_replicas(measurements)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return {**result, "reason": f"invalid area measurement: {exc}"}
    return {
        "built": True,
        "queue_complete": all(
            m.effective_samples >= MINIMUM_EFFECTIVE_SAMPLES for m in measurements
        ),
        "area_converged": pooled.converged,
        "area_rejection": pooled.rejection,
        "reason": None
        if all(m.effective_samples >= MINIMUM_EFFECTIVE_SAMPLES for m in measurements)
        else "fewer than ten effective samples in at least one replica",
        "spread_sigma": pooled.spread_sigma if math.isfinite(pooled.spread_sigma) else None,
        "ns_per_replica": [float(m["npt_ps"]) / 1000.0 for m in metadatas],
    }


def v4_entry_assessment(entry, metadatas):
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
    from gmxbuilder.modules.membrane.v4_sampling import (
        quantitative_area,
    )

    result = {"built": False, "queue_complete": False, "area_converged": False, "reason": None}
    try:
        protocol = resolve_protocol(entry["lipid"], entry["family"])
        if any(m.get("v4_protocol") != protocol for m in metadatas):
            raise ValueError("Replica protocol identity differs from reviewed conditions")
        for field in (
            "canonical_smiles",
            "topology_sha256",
            "atom_names",
            "temperature_K",
            "equilibration_host",
        ):
            if any(m.get(field) != metadatas[0].get(field) for m in metadatas):
                raise ValueError(f"Replicas disagree on {field}")
        from gmxbuilder.modules.membrane.v4_construction import qualify_construction

        qualification = qualify_construction(metadatas, protocol)
        if qualification["status"] == "failed":
            return {
                **result,
                "construction": qualification,
                "reason": "; ".join(qualification["failures"]),
            }
        analyses = [m["trajectory_analysis"] for m in metadatas]
        failures = qualification["failures"]
        agreement = qualification["replica_agreement"]
        area = quantitative_area(analyses)
        return {
            "built": True,
            "queue_complete": qualification["accepted"],
            "area_converged": area["converged"],
            "area_rejection": area["rejection"],
            "reason": None if qualification["accepted"] else "; ".join(failures),
            "spread_sigma": agreement["spread_sigma"].get("area_per_lipid_nm2"),
            "all_failures": failures,
            "ns_per_replica": [float(m["npt_ps"]) / 1000 for m in metadatas],
            "construction": qualification,
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {**result, "reason": f"Invalid V4 analysis: {exc}"}


def replica_action(directory: Path, work: Path, initial_ns: float) -> tuple[str, float]:
    """Preserve existing data: continue valid runs, refuse ambiguous partial builds."""
    if _replica_complete(directory):
        metadata = json.loads((directory / "metadata.json").read_text())
        ns = float(metadata.get("npt_ps", 0)) / 1000.0
        if not math.isfinite(ns) or ns <= 0:
            raise RuntimeError(f"Invalid production duration in {directory}")
        return "present", ns
    if (work / "prepared.json").is_file() and not directory.exists():
        if not (work / "production-started.json").exists() and not any(work.glob("nvt*.log")):
            return "prepared", initial_ns
        if (work / "production-started.json").is_file() and continuation_available(work):
            return "resume", initial_ns
    if directory.exists() or work.exists():
        raise RuntimeError(
            "Existing incomplete artifacts need inspection; refusing to overwrite "
            f"{directory} / {work}"
        )
    return "build", initial_ns


def assemble_completed(v4_root: Path, entry: dict, replicas: int) -> dict:
    """Publish accepted sources idempotently, including recovery after a crash.

    This writes only the V4 output library, never the installed runtime library.
    Failure to publish is a storage/integrity problem, not a reason for more MD.
    """
    from assemble_v4_library import assemble_entry

    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.modules.membrane.v4_evidence import assembled_evidence_valid

    library = EquilibratedLipidLibrary(roots=[v4_root / "library"])
    library.require_v4 = True
    roots = [replica_root(v4_root, r) for r in range(1, replicas + 1)]
    signatures = [
        hashlib.sha256(
            (root / entry["family"] / entry["lipid"] / "metadata.json").read_bytes()
        ).hexdigest()
        for root in roots
    ]
    target = v4_root / "library" / entry["family"] / entry["lipid"]
    try:
        metadata = json.loads((target / "metadata.json").read_text())
        if (
            metadata.get("assembled_from", {}).get("source_metadata_sha256") == signatures
            and assembled_evidence_valid(metadata)
            and _conformers_readable(target, metadata)
            and library.inspect(
                entry["lipid"], metadata.get("force_field", ""), metadata.get("lipid_ff")
            )
            is not None
        ):
            return {"status": "present", "path": str(target)}
    except (OSError, ValueError, TypeError):
        pass
    result = assemble_entry(v4_root, roots, entry["family"], entry["lipid"], v4_root / "library")
    if "skipped" in result:
        raise RuntimeError(f"Accepted entry could not be assembled: {result['skipped']}")
    return {"status": "assembled", "path": str(target), **result}


def continuation_available(work: Path) -> bool:
    return all(
        (work / name).is_file() and (work / name).stat().st_size > 0
        for name in ("npt.tpr", "npt.cpt", "npt.edr", "topol.top")
    )


def _replica_complete(directory: Path) -> bool:
    """Whether one replica directory holds a finished V4 entry.

    Every clause here is a way a directory has been, or could be, finished in
    name only. A metadata file saying ``{"status": "failed", "observables": {}}``
    satisfied the shape and was counted as built. An area measured from zero
    independent samples is a number with no evidence under it, and the queue
    would have marked the entry done and never run it again. And a conformer
    file is not a conformer: ``conf_1.npz`` of zero bytes, or holding no arrays,
    or holding something that is not an npz at all, passed a check that only
    asked whether a file of that name existed -- and would have failed later, in
    a user's build, at ``np.load``.
    """
    path = directory / "metadata.json"
    if not path.is_file():
        return False
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(metadata, dict) or str(metadata.get("status")) != "ready":
        return False
    from gmxbuilder.modules.membrane.parameter_provenance import parameters_current

    if not parameters_current(metadata):
        return False
    quality = metadata.get("quality")
    from gmxbuilder.modules.membrane.initial_water import initial_water_evidence_valid

    if (
        metadata.get("test_mode")
        or not isinstance(quality, dict)
        or quality.get("passed") is not True
        or not initial_water_evidence_valid(quality.get("initial_water_exclusion"))
    ):
        return False
    observables = metadata.get("observables")
    if not isinstance(observables, dict):
        return False
    area = observables.get("area_per_lipid_nm2")
    if not isinstance(area, dict):
        return False
    try:
        mean = float(area["mean"])
        samples = float(area["effective_samples"])
        duration = float(metadata["npt_ps"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (math.isfinite(mean) and math.isfinite(samples) and math.isfinite(duration)):
        return False
    if mean <= 0.0 or samples <= 0.0 or duration <= 0.0:
        return False
    from gmxbuilder.modules.membrane.equilibrated_library import topology_signature

    names = metadata.get("atom_names")
    force_field = metadata.get("force_field")
    lipid_ff = metadata.get("lipid_ff")
    if not names or not force_field or not lipid_ff:
        return False
    if metadata.get("topology_sha256") != topology_signature(names, force_field, lipid_ff):
        return False
    return _conformers_readable(directory, metadata)


def _conformers_readable(directory: Path, metadata: dict) -> bool:
    """Whether the entry's conformers are on disk and can actually be read."""
    import numpy as np

    from gmxbuilder.modules.membrane.v4_reuse import reused_coordinates_valid

    if not reused_coordinates_valid(directory, metadata):
        return False

    files = sorted(directory.glob("conf_*.npz"))
    if not files:
        return False
    declared = metadata.get("n_conformations")
    if isinstance(declared, int) and declared != len(files):
        return False
    for conformer in files:
        try:
            with np.load(conformer) as stored:
                coordinates = np.asarray(stored["coords"], dtype=float)
                names = [str(name) for name in stored["atom_names"]]
        except (OSError, ValueError, KeyError, EOFError):
            return False
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            return False
        if metadata.get("atom_names") and names != metadata["atom_names"]:
            return False
        if len(names) != len(coordinates) or not names:
            return False
        if not np.isfinite(coordinates).all():
            return False
    return True


def launch(
    entry: dict, replica: int, gpu: int, arguments, ns: float, *, phase: str = "build"
) -> subprocess.Popen:
    """Start one replica as its own process.

    A subprocess per replica rather than threads: a 50 ns build is hours long,
    GROMACS is called through the shell anyway, and a segfault or an OOM in one
    entry must not take the other GPU's work or the queue with it.
    """
    verify_launch_source(arguments)
    if phase not in {"build", "prepare", "production", "resume"}:
        raise ValueError(f"Unknown replica phase: {phase}")
    record_invocation(arguments, entry, replica, gpu, phase, ns)
    work = arguments.v4_root / "work" / entry["family"] / entry["lipid"] / f"replica-{replica}"
    log = arguments.v4_root / "logs" / entry["family"]
    log.mkdir(parents=True, exist_ok=True)
    handle = (log / f"{entry['lipid']}-replica-{replica}.log").open("a")

    environment = dict(os.environ)
    # Deliberately *not* pointing GMXBUILDER_LIPID_LIBRARY at the V4 root. That
    # variable decides where the shipped V3 archive is installed, so aiming it
    # here unpacks 228 V3 entries into the V4 tree. Where output is written is
    # decided solely by the explicit roots= below.
    environment.pop("GMXBUILDER_LIPID_LIBRARY", None)
    environment["PYTHONPATH"] = str(ROOT / "src")

    code = f"""
import sys
from pathlib import Path
from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.modules.membrane.lipid_equilibration import (
    LipidEquilibrationBuilder, lipid_gpu_device,
)
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
root = Path({str(replica_root(arguments.v4_root, replica))!r})
builder = LipidEquilibrationBuilder(
    library=EquilibratedLipidLibrary(roots=[root, root]),
    gmx={getattr(arguments, "gmx_bin", None)!r},
    require_gpu_update=True,
    v4_protocol=resolve_protocol({entry["lipid"]!r}, {entry["family"]!r}),
)
builder.threads = {arguments.threads}
with lipid_gpu_device({gpu}):
    if {phase == "resume"!r}:
        builder.resume_prepared(Path({str(work)!r}))
    elif {phase == "production"!r}:
        builder.run_prepared(Path({str(work)!r}))
    else:
        builder.build(
        {entry["lipid"]!r}, {entry["force_field"]!r}, {entry["lipid_ff"]!r},
        npt_ps={ns * 1000.0},
        replica_seed={entry_seed(entry["family"], entry["lipid"], replica)},
        retain_work=Path({str(work)!r}),
        force=True,
        prepare_only={phase == "prepare"!r},
    )
"""
    try:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            pass_fds=getattr(arguments, "lock_fds", ()),
        )
    finally:
        handle.close()


def launch_extension(entry: dict, replica: int, gpu: int, arguments, to_ns: float):
    """Carry one replica's finished NPT run further, in its own process.

    The same shape as ``launch`` and for the same reason, but it continues the
    run in ``work`` from its checkpoint instead of building the entry again.
    What comes out is republished over the entry it extends, measured by the
    same code that measured it the first time.
    """
    verify_launch_source(arguments)
    record_invocation(arguments, entry, replica, gpu, "extend", to_ns)
    work = arguments.v4_root / "work" / entry["family"] / entry["lipid"] / f"replica-{replica}"
    log = arguments.v4_root / "logs" / entry["family"]
    log.mkdir(parents=True, exist_ok=True)
    handle = (log / f"{entry['lipid']}-replica-{replica}.log").open("a")

    environment = dict(os.environ)
    environment.pop("GMXBUILDER_LIPID_LIBRARY", None)
    environment["PYTHONPATH"] = str(ROOT / "src")

    code = f"""
import sys
from pathlib import Path
from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
from gmxbuilder.modules.membrane.lipid_equilibration import (
    LipidEquilibrationBuilder, lipid_gpu_device,
)
from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol
root = Path({str(replica_root(arguments.v4_root, replica))!r})
builder = LipidEquilibrationBuilder(
    library=EquilibratedLipidLibrary(roots=[root, root]),
    gmx={getattr(arguments, "gmx_bin", None)!r},
    require_gpu_update=True,
    v4_protocol=resolve_protocol({entry["lipid"]!r}, {entry["family"]!r}),
)
builder.threads = {arguments.threads}
with lipid_gpu_device({gpu}):
    builder.extend(
        {entry["lipid"]!r}, {entry["force_field"]!r}, {entry["lipid_ff"]!r},
        to_ps={to_ns * 1000.0},
        work_dir=Path({str(work)!r}),
        replica_seed={entry_seed(entry["family"], entry["lipid"], replica)},
    )
"""
    try:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            pass_fds=getattr(arguments, "lock_fds", ()),
        )
    finally:
        handle.close()


def verify_launch_source(arguments):
    if getattr(arguments, "source_manifest", None) is not None:
        from v4_queue_state import verify_source

        verify_source(ROOT, arguments.source_manifest)
        if hashlib.sha256(Path(arguments.gmx_bin).read_bytes()).hexdigest() != arguments.gmx_sha256:
            raise RuntimeError("GROMACS executable changed; refusing another replica")


def record_invocation(arguments, entry, replica, gpu, action, target_ns):
    """Keep operation provenance separate from unknown legacy trajectory origins."""
    manifest = getattr(arguments, "manifest_path", None)
    if manifest is None:
        return
    previous = (
        replica_root(arguments.v4_root, replica)
        / entry["family"]
        / entry["lipid"]
        / "metadata.json"
    )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entry,
        "replica": replica,
        "gpu": gpu,
        "action": action,
        "target_ns": target_ns,
        "seed": entry_seed(entry["family"], entry["lipid"], replica),
        "previous_metadata_sha256": hashlib.sha256(previous.read_bytes()).hexdigest()
        if previous.is_file()
        else None,
    }
    with Path(manifest).with_suffix(".operations.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def wait_for_replicas(requests):
    """Report exits as they happen and preserve/reap every launched replica."""
    processes = []
    try:
        for replica, start in requests:
            processes.append((replica, start()))
        pending = dict(processes)
        while pending:
            for replica, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    print(f"    Replica {replica} exited with code {code}", flush=True)
                    del pending[replica]
            if pending:
                time.sleep(0.5)
    finally:
        codes = {replica: process.wait() for replica, process in processes}
    return codes


def prepare_pair(entry, actions, arguments, gpus):
    """Complete and validate every missing replica before launching any MD."""
    from gmxbuilder.modules.membrane.lipid_equilibration import validate_prepared_work
    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    codes = wait_for_replicas(
        [
            (
                r,
                lambda r=r, ns=ns: launch(
                    entry, r, gpus[(r - 1) % len(gpus)], arguments, ns, phase="prepare"
                ),
            )
            for r, action, ns in actions
            if action == "build"
        ]
    )
    if any(code != 0 for code in codes.values()):
        raise RuntimeError(
            "Replica preparation failed; pair production was not started. "
            "Completed preparations and failed construction evidence were preserved."
        )
    protocol = resolve_protocol(entry["lipid"], entry["family"])
    for replica, action, ns in actions:
        if action not in {"build", "prepared", "resume"}:
            continue
        work = arguments.v4_root / "work" / entry["family"] / entry["lipid"] / f"replica-{replica}"
        record = validate_prepared_work(work, protocol, allow_started=action == "resume")
        if (
            record["lipid_name"] != entry["lipid"]
            or record["force_field"] != entry["force_field"]
            or record["lipid_ff"] != entry["lipid_ff"]
            or record["replica_seed"] != entry_seed(entry["family"], entry["lipid"], replica)
            or record["npt_steps"] != max(50000, int(ns * 1000 * 500))
            or record["test_mode"]
        ):
            raise ValueError(f"Prepared replica {replica} does not match the queued request")
    return codes


def build_pair(entry, actions, arguments, gpus):
    prepare_pair(entry, actions, arguments, gpus)
    return wait_for_replicas(
        [
            (
                r,
                lambda r=r, ns=ns, action=action: launch(
                    entry,
                    r,
                    gpus[(r - 1) % len(gpus)],
                    arguments,
                    ns,
                    phase="resume" if action == "resume" else "production",
                ),
            )
            for r, action, ns in actions
            if action in {"build", "prepared", "resume"}
        ]
    )


def order_pending(entries, priority_lipids=""):
    """Prioritize named lipids across families without reversing reuse dependencies."""
    names = list(dict.fromkeys(n.strip().upper() for n in priority_lipids.split(",") if n.strip()))
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    unknown = set(names) - set(LipidRegistry.list_builtin()) - {e["lipid"] for e in entries}
    if unknown:
        raise ValueError(f"Unknown priority lipids: {', '.join(sorted(unknown))}")
    priority = {name: index for index, name in enumerate(names)}
    # Every source precedes its CHARMM36 target, including in the priority tier.
    families = {"amber-lipid21": 0, "charmm36m-lipid": 1, "charmm36-lipid": 2, "amber-gaff2": 3}
    return sorted(
        entries,
        key=lambda e: (
            priority.get(e["lipid"], len(priority)),
            e["lipid"],
            families.get(e["family"], 9),
            e["family"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=DEFAULT_V3_ROOT,
        help="Deprecated compatibility option; selection uses current capabilities",
    )
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument(
        "--ns", type=float, default=INITIAL_NS, help="NPT length per replica before any extension"
    )
    parser.add_argument(
        "--maximum-ns",
        type=float,
        default=MAXIMUM_NS,
        help="NPT budget per replica, at most 200 ns; unresolved entries remain needs_review",
    )
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--only", default=None, help="restrict to one lipid name")
    parser.add_argument(
        "--priority-lipids",
        default="POPC,POPS,POPG,POPE,POP2,POP3,CHOL,POPI,PSM,DOPC,DOPE,DOPS,DPPC,DPPE",
        help="ordered lipids to finish across families before the alphabetical queue",
    )
    parser.add_argument(
        "--families",
        default=None,
        help=(
            "comma-separated parameter families to build, e.g. "
            "charmm36m-lipid,amber-lipid21. Everything else is left for a later run"
        ),
    )
    parser.add_argument(
        "--include-gaff-overlap",
        action="store_true",
        help="also simulate GAFF2 for lipids already covered by Lipid21",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-entry", action="store_true", help="finish one entry and release the queue"
    )
    arguments = parser.parse_args()

    if arguments.replicas < 2 or arguments.threads < 1:
        parser.error("V4 needs at least two replicas and positive thread count")
    if not (
        math.isfinite(arguments.ns)
        and math.isfinite(arguments.maximum_ns)
        and 0 < arguments.ns <= arguments.maximum_ns <= MAXIMUM_NS
    ):
        parser.error("require 0 < ns <= maximum-ns <= 200 with finite durations")
    if not arguments.gpus.strip():
        parser.error("at least one GPU identifier is required")
    if arguments.dry_run:
        return run_queue(arguments)
    from v4_queue_state import queue_ownership

    from gmxbuilder.modules.membrane.lipid_equilibration import find_gromacs

    arguments.gmx_bin = str(Path(find_gromacs()).resolve())
    arguments.gmx_sha256 = hashlib.sha256(Path(arguments.gmx_bin).read_bytes()).hexdigest()
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }
    configuration["gmx_version"] = subprocess.check_output(
        [arguments.gmx_bin, "--version"],
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    with queue_ownership(arguments.v4_root, ROOT, configuration=configuration) as (
        manifest,
        path,
        fd,
    ):
        arguments.source_manifest = manifest
        arguments.manifest_path = str(path)
        arguments.lock_fds = (fd,)
        return run_queue(arguments)


def run_queue(arguments) -> int:
    gpus = [int(value) for value in arguments.gpus.split(",") if value.strip() != ""]
    if len(gpus) < arguments.replicas:
        print(
            f"note: {arguments.replicas} replicas over {len(gpus)} GPU(s); "
            "replicas will share devices"
        )

    entries = discover_entries(
        arguments.v3_root, include_gaff_overlap=getattr(arguments, "include_gaff_overlap", False)
    )
    if arguments.only:
        entries = [e for e in entries if e["lipid"].upper() == arguments.only.upper()]
    if arguments.families:
        wanted = {name.strip() for name in arguments.families.split(",") if name.strip()}
        unknown = wanted - {e["family"] for e in entries}
        if unknown:
            raise SystemExit(f"no such parameter family: {', '.join(sorted(unknown))}")
        entries = [e for e in entries if e["family"] in wanted]
    if not entries:
        raise SystemExit("No supported entries match the current capability selection")

    from gmxbuilder.modules.membrane.v4_protocol import protocol_refusal

    refused = [
        (
            e,
            unresolved_stereochemistry(e["lipid"], compare_native=e["family"] != "amber-gaff2")
            or policy_refusal(e["family"], e["lipid"], e["force_field"])
            or protocol_refusal(e),
        )
        for e in entries
    ]
    quarantined = [(e, why) for e, why in refused if why]
    entries = [e for e, why in refused if not why]
    if quarantined:
        print(f"{len(quarantined)} V3 entries skipped: current policy forbids the pairing")
        for entry, why in quarantined:
            print(f"    {entry['family']:18s} {entry['lipid']:6s} {why[:66]}")

    from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

    entries = [{**e, "v4_protocol": resolve_protocol(e["lipid"], e["family"])} for e in entries]
    # Validate against all selected entries: completed priorities are valid on resume.
    try:
        entries = order_pending(entries, getattr(arguments, "priority_lipids", ""))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    pending = [e for e in entries if not entry_done(arguments.v4_root, e, arguments.replicas)]
    total_ns = len(pending) * arguments.replicas * arguments.ns
    print(f"{len(entries)} entries, {len(pending)} still to build")
    print(
        f"Nominal initial budget: {total_ns:.0f} ns before existing work, reuse and extensions; "
        "completion time is not yet estimated"
    )
    if arguments.dry_run:
        for entry in pending:
            print(f"  {entry['family']:18s} {entry['lipid']}")
        return 0

    arguments.v4_root.mkdir(parents=True, exist_ok=True)
    selection = {
        "priority_lipids": getattr(arguments, "priority_lipids", ""),
        "candidates": entries,
        "held": [{**entry, "reason": reason} for entry, reason in quarantined],
        "run_manifest": getattr(arguments, "manifest_path", None),
    }
    (arguments.v4_root / "protocol-selection.json").write_text(json.dumps(selection, indent=2))
    progress = arguments.v4_root / "queue-progress.jsonl"
    # Acceptance and assembly are separate durable writes. Resume completes
    # publication after an interrupted scheduler without rerunning its MD.
    for entry in entries:
        if entry not in pending and entry.get("v4_protocol"):
            verify_launch_source(arguments)
            assemble_completed(arguments.v4_root, entry, arguments.replicas)

    started_at = time.time()
    for index, entry in enumerate(pending, start=1):
        verify_launch_source(arguments)
        label = f"{entry['family']}/{entry['lipid']}"
        began = time.time()
        print(f"[{index}/{len(pending)}] {label} ...", flush=True)

        source = reuse_source(entry)
        if source is not None and publish_reuse(
            arguments.v4_root, entry, source, arguments.replicas
        ):
            print("    reused approved CHARMM36m conformers under audited conditions", flush=True)
            with progress.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "family": entry["family"],
                            "lipid": entry["lipid"],
                            "ok": True,
                            "reused_from": source["family"],
                            "seconds": 0.0,
                        }
                    )
                    + "\n"
                )
            if arguments.stop_after_entry or (arguments.v4_root / "STOP_AFTER_ENTRY").exists():
                break
            continue

        codes = {}
        failure = None
        try:
            actions = []
            for replica in range(1, arguments.replicas + 1):
                directory = (
                    replica_root(arguments.v4_root, replica) / entry["family"] / entry["lipid"]
                )
                work = (
                    arguments.v4_root
                    / "work"
                    / entry["family"]
                    / entry["lipid"]
                    / f"replica-{replica}"
                )
                action, ns = replica_action(directory, work, arguments.ns)
                actions.append((replica, action, ns))
            codes = build_pair(entry, actions, arguments, gpus)
            assessment = entry_assessment(arguments.v4_root, entry, arguments.replicas)
            while (
                all(c == 0 for c in codes.values())
                and assessment["built"]
                and not assessment["queue_complete"]
            ):
                lengths = assessment["ns_per_replica"]
                if min(lengths) >= arguments.maximum_ns:
                    break
                target = min(max(lengths) + EXTENSION_NS, arguments.maximum_ns)
                to_extend = [r for r, ns in enumerate(lengths, 1) if ns < target]
                for replica in to_extend:
                    work = (
                        arguments.v4_root
                        / "work"
                        / entry["family"]
                        / entry["lipid"]
                        / f"replica-{replica}"
                    )
                    if not continuation_available(work):
                        raise RuntimeError(
                            f"Missing continuation artifacts: {work}; existing results preserved"
                        )
                print(
                    f"    {assessment['reason']}; continuing eligible replicas to {target:g} ns",
                    flush=True,
                )
                codes = wait_for_replicas(
                    [
                        (
                            r,
                            lambda r=r: launch_extension(
                                entry, r, gpus[(r - 1) % len(gpus)], arguments, target
                            ),
                        )
                        for r in to_extend
                    ]
                )
                assessment = entry_assessment(arguments.v4_root, entry, arguments.replicas)
        except (OSError, ValueError, RuntimeError) as exc:
            failure = str(exc)
        verify_launch_source(arguments)
        assessment = entry_assessment(arguments.v4_root, entry, arguments.replicas)
        ok = (
            failure is None and all(c == 0 for c in codes.values()) and assessment["queue_complete"]
        )
        publication = None
        samples = None
        sampling = []
        postprocessing_failure = None
        try:
            if assessment["built"]:
                samples = effective_samples(arguments.v4_root, entry, arguments.replicas)
                sampling = sampling_summary(arguments.v4_root, entry, arguments.replicas)
            if ok and entry.get("v4_protocol"):
                publication = assemble_completed(arguments.v4_root, entry, arguments.replicas)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            ok = False
            postprocessing_failure = f"Completed evidence needs review: {exc}"
        elapsed = time.time() - began
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "family": entry["family"],
            "lipid": entry["lipid"],
            "ok": ok,
            "returncodes": codes,
            "effective_samples": samples,
            "sampling": sampling,
            "ns_per_replica": assessment.get("ns_per_replica"),
            "status": "complete"
            if ok
            else ("failed" if failure or not assessment["built"] else "needs_review"),
            "reason": postprocessing_failure or failure or assessment["reason"],
            "spread_sigma": assessment.get("spread_sigma"),
            "area_converged": assessment["area_converged"],
            "area_rejection": assessment.get("area_rejection"),
            "construction_status": assessment.get("construction", {}).get("status"),
            "construction_scope": assessment.get("construction", {}).get("policy", {}).get("scope"),
            "diagnostic_warnings": assessment.get("construction", {}).get("warnings", []),
            "publication": publication,
            "run_manifest": getattr(arguments, "manifest_path", None),
            "samples_target_met": bool(
                samples is not None and samples >= MINIMUM_EFFECTIVE_SAMPLES
            ),
            "seconds": round(elapsed, 1),
        }
        with progress.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        if entry["lipid"] == "POPC" and entry["family"] == "amber-lipid21" and not ok:
            print("POPC/Lipid21 did not pass; later entries remain on hold", flush=True)
            break

        done = index
        rate = (time.time() - started_at) / done
        remaining = rate * (len(pending) - done) / 3600.0
        print(
            f"    {record['status']} in {elapsed / 60:.0f} min; ~{remaining:.0f} h left", flush=True
        )
        if record["reason"]:
            print(f"    {record['reason']}", flush=True)
        if arguments.stop_after_entry or (arguments.v4_root / "STOP_AFTER_ENTRY").exists():
            print("queue stopped at entry boundary", flush=True)
            break

    unresolved = [e for e in entries if not entry_done(arguments.v4_root, e, arguments.replicas)]
    print(f"queue finished: {len(unresolved)} entries remain incomplete under the queue policy")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
