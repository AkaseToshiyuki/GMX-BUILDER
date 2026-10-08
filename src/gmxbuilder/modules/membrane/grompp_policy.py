"""Strict preprocessing with auditable, narrowly scoped charge exceptions.

The PSM policy was approved on 2026-10-06. It preserves upstream charges;
acceptance is neither a charge correction nor an equilibrium certificate.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

POLICY = "lipid-grompp-2-psm128-20261006"
PSM_FINGERPRINT = "1647fc3e2eb761d2b269c718587a3eeed8c49915a56d5e23267eec4e53d060d3"
EWALD_WARNING = (
    "You are using Ewald electrostatics in a system with net charge. This can "
    "lead to severe artifacts, such as ions moving into regions with low "
    "dielectric, due to the uniform background charge. We suggest to "
    "neutralize your system with counter ions, possibly in combination with a "
    "physiological salt concentration."
)
PSM_RESTRAINT = (
    "\n#ifdef POSRES\n[ position_restraints ]\n"
    "; atom  funct  fc_x  fc_y             fc_z\n"
    "    53    1   0.0   0.0  POSRES_FC_LIPID\n#endif\n"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def warning_bodies(log: str) -> list[str]:
    headers = re.findall(r"^\s*WARNING\b[^\n]*", log, re.M)
    bodies = re.findall(r"^WARNING \d+[^\n]*\n(.*?)(?=\n\s*\n)", log, re.M | re.S)
    if len(headers) != len(bodies):
        raise ValueError("Unrecognized preprocessing warning format")
    return [" ".join(body.split()) for body in bodies]


def topology_charge(text: str) -> tuple[dict, dict]:
    """Read ordered A-state charges and molecule counts from expanded topology."""
    charges, molecules = {}, {}
    section = name = None
    for line in text.splitlines():
        raw = line.split(";", 1)[0].strip()
        if not raw:
            continue
        if raw.startswith("#"):
            raise ValueError("Unresolved topology directive")
        if raw.startswith("["):
            section = raw.strip("[] ").lower()
            continue
        fields = raw.split()
        if section == "moleculetype":
            name = fields[0]
            if name in charges:
                raise ValueError("Duplicate molecule definition")
            charges[name] = []
        elif section == "atoms":
            if (
                name is None
                or len(fields) not in {7, 8}
                or int(fields[0]) != len(charges[name]) + 1
            ):
                raise ValueError("Unsupported atom definition or B-state charges")
            charge = Decimal(fields[6])
            if not charge.is_finite():
                raise ValueError("Non-finite atom charge")
            charges[name].append(charge)
        elif section == "molecules":
            if len(fields) != 2 or int(fields[1]) <= 0:
                raise ValueError("Invalid molecule count")
            # Host/guest bilayers interleave molecule blocks in coordinate order.
            molecules[fields[0]] = molecules.get(fields[0], 0) + int(fields[1])
    if not molecules or any(not charges.get(name) for name in molecules):
        raise ValueError("Incomplete expanded topology")
    return charges, molecules


def _psm_sources(work: Path, top: Path) -> dict:
    from gmxbuilder.modules.forcefield.catalog import force_field_directory
    from gmxbuilder.modules.membrane.parameter_provenance import parameter_fingerprint

    if parameter_fingerprint("PSM", "amber14sb", "lipid21") != PSM_FINGERPRINT:
        raise ValueError("PSM parameter fingerprint has not been reviewed")
    data = Path(__file__).resolve().parents[2] / "data"
    base = (data / "lipid21/itp/PSM.itp").read_text()
    if (work / "PSM.itp").read_text() not in (base, base + PSM_RESTRAINT):
        raise ValueError("PSM topology differs from the reviewed source")
    includes = []
    section = None
    for line in top.read_text().splitlines():
        raw = line.split(";", 1)[0].strip()
        if not raw:
            continue
        if raw.startswith("#"):
            match = re.fullmatch(r'#include "([^"\n]+)"', raw)
            if not match or section is not None:
                raise ValueError("Unreviewed PSM topology directive")
            includes.append(match[1])
        elif raw.startswith("["):
            section = raw.strip("[] ").lower()
            if section not in {"system", "molecules"}:
                raise ValueError("Unreviewed PSM topology section")
        elif section is None:
            raise ValueError("Unreviewed PSM topology content")
    if includes != [
        "forcefield/forcefield.itp",
        "lipid21_atomtypes.itp",
        "forcefield/tip3p.itp",
        "forcefield/ions_tip3p.itp",
        "PSM.itp",
    ]:
        raise ValueError("Unreviewed PSM topology includes")
    source_ff = force_field_directory("amber14sb")
    expected = {p.relative_to(source_ff): p for p in source_ff.rglob("*.itp")}
    actual = {p.relative_to(work / "forcefield"): p for p in (work / "forcefield").rglob("*.itp")}
    if set(expected) != set(actual) or any(sha(actual[p]) != sha(expected[p]) for p in actual):
        raise ValueError("Working force field differs from reviewed PSM source")
    atomtypes = work / "lipid21_atomtypes.itp"
    if sha(atomtypes) != sha(data / "lipid21/lipid21_atomtypes.itp"):
        raise ValueError("PSM atom types differ from reviewed source")
    return {"parameter_fingerprint": PSM_FINGERPRINT, "psm_itp_sha256": sha(work / "PSM.itp")}


def charge_exception(processed: Path, log: str, work: Path, top: Path, stage: str) -> dict:
    if warning_bodies(log) != [EWALD_WARNING]:
        raise ValueError("Only the single reviewed Ewald charge warning is eligible")
    charges, counts = topology_charge(processed.read_text())
    decimal_total = sum(sum(charges[n]) * count for n, count in counts.items())
    if counts.get("PSM") == 128 and set(counts) <= {"PSM", "SOL", "NA", "CL"}:
        if stage != "dry" and not counts.get("SOL"):
            raise ValueError("Solvated PSM policy requires TIP3P water")
        provenance = _psm_sources(work, top)
        if counts.get("NA", 0) != counts.get("CL", 0):
            raise ValueError("PSM requires equal sodium and chloride counts")
        if len(charges["PSM"]) != 127 or sum(charges["PSM"]) != Decimal("0.000003"):
            raise ValueError("PSM source charge differs from reviewed residual")
        for name, expected in {"SOL": 0, "NA": 1, "CL": -1}.items():
            if name in counts and sum(charges[name]) != expected:
                raise ValueError("Unexpected solvent or ion charge")
        if decimal_total != Decimal("0.000384"):
            raise ValueError("Unexplained PSM system charge")
        # Version/precision are separately verified against the running binary.
        predicted = sum(
            sum(struct.unpack("f", struct.pack("f", float(q)))[0] for q in charges[n]) * count
            for n, count in counts.items()
        )
        kind = "reviewed-psm128-source-residual"
    elif stage in {"dry", "before_ions"}:
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        formal = 0
        for name, count in counts.items():
            expected = 0 if name == "SOL" else LipidRegistry.get(name).charge
            if name in {"NA", "CL"} or abs(sum(charges[name]) - expected) > Decimal("0.000001"):
                raise ValueError("Pre-ionization charge does not match declared formal charges")
            formal += expected * count
        if not formal:
            raise ValueError("No verified pre-ionization formal charge")
        predicted = formal
        provenance = {"formal_charge": formal}
        kind = "verified-pre-ionization-formal-charge"
    else:
        raise ValueError("Charge warning is outside the approved composition and stage")
    observed = re.findall(r"System has non-zero total charge: ([+-]?[0-9.]+)", log)
    if observed != [f"{predicted:.6f}"]:
        raise ValueError("Reported charge differs from the independently predicted charge")
    return {
        "exception": kind,
        "stage": stage,
        "molecules": counts,
        "decimal_charge": str(decimal_total),
        "predicted_reported_charge": f"{predicted:.6f}",
        **provenance,
    }


def _option(args: list[str], key: str) -> str:
    if args.count(key) != 1:
        raise ValueError(f"Preprocessing requires exactly one {key}")
    return args[args.index(key) + 1]


def _replace(args: list[str], key: str, value: str) -> None:
    if key in args:
        if args.count(key) != 1:
            raise ValueError(f"Repeated option {key}")
        args[args.index(key) + 1] = value
    else:
        args.extend([key, value])


def _inputs(args: list[str], work: Path) -> dict:
    paths = set(work.rglob("*.itp")) | set(work.glob("*.top"))
    for key in ("-f", "-c", "-r", "-t", "-n", "-p"):
        if key in args:
            paths.add((work / _option(args, key)).resolve())
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def _psm_version(gmx: str) -> str:
    version = subprocess.run(
        [gmx, "--version"], text=True, capture_output=True, check=True, timeout=30
    ).stdout
    if not re.search(r"GROMACS version:\s+2026\.3\s*$", version, re.M) or not re.search(
        r"Precision:\s+mixed\s*$", version, re.M
    ):
        raise ValueError("PSM residual policy requires reviewed GROMACS 2026.3 mixed")
    return version


def run_grompp(args: list[str], work: Path, *, stage="neutralized", timeout=3600) -> Path:
    """Publish a TPR only after strict success or a fully rechecked exception."""
    if stage not in {"dry", "before_ions", "neutralized"}:
        raise ValueError("Unknown preprocessing charge stage")
    work = work.resolve()
    target = work / _option(args, "-o")
    top = work / _option(args, "-p")
    audit_root = work / ".grompp-policy"
    audit_root.mkdir(exist_ok=True)
    audit = Path(tempfile.mkdtemp(prefix=target.stem + "-", dir=audit_root))
    before = _inputs(args, work)
    # GROMACS insists on a .top suffix. Keep that transient output outside
    # the retained workspace so concurrent provenance reads (or a killed
    # compiler) cannot mistake an expanded audit copy for a new parameter.
    scratch = tempfile.TemporaryDirectory(prefix="gmxbuilder-grompp-")
    expanded = Path(scratch.name) / "processed.top"
    command = list(args)
    for key, value in {
        "-o": str(audit / "accepted.tpr"),
        "-pp": str(expanded),
        "-po": str(audit / "effective.mdp"),
        "-maxwarn": "0",
    }.items():
        _replace(command, key, value)
    receipt = {
        "policy": POLICY,
        "stage": stage,
        "inputs": before,
        "command": list(args),
        "accepted": False,
    }
    try:
        result = subprocess.run(command, cwd=work, text=True, capture_output=True, timeout=timeout)
        if expanded.exists():
            (audit / "processed.txt").write_bytes(expanded.read_bytes())
            expanded.unlink()
        log = result.stdout + "\n" + result.stderr
        (audit / "strict.log").write_text(log)
        warnings = warning_bodies(log)
        if result.returncode:
            if "Too many warnings (1)." not in log or warnings != [EWALD_WARNING]:
                raise ValueError("Strict preprocessing failed without an eligible charge warning")
            evidence = charge_exception(audit / "processed.txt", log, work, top, stage)
            if evidence["exception"] == "reviewed-psm128-source-residual":
                version = _psm_version(args[0])
                (audit / "gromacs-version.txt").write_text(version)
            if _inputs(args, work) != before:
                raise ValueError("Preprocessing inputs changed during strict attempt")
            first_top = sha(audit / "processed.txt")
            (audit / "strict-processed.txt").write_bytes((audit / "processed.txt").read_bytes())
            receipt["processed_sha256"] = first_top
            _replace(command, "-maxwarn", "1")
            result = subprocess.run(
                command, cwd=work, text=True, capture_output=True, timeout=timeout
            )
            if expanded.exists():
                (audit / "processed.txt").write_bytes(expanded.read_bytes())
                expanded.unlink()
            log = result.stdout + "\n" + result.stderr
            (audit / "exception.log").write_text(log)
            if (
                result.returncode
                or charge_exception(audit / "processed.txt", log, work, top, stage) != evidence
                or sha(audit / "processed.txt") != first_top
            ):
                raise ValueError("Conditional preprocessing changed or failed")
            receipt.update(evidence, maxwarn=1, strict_passed=False)
        else:
            if warnings:
                raise ValueError("Strict preprocessing returned success with warnings")
            receipt.update(maxwarn=0, strict_passed=True, exception=None)
        if _inputs(args, work) != before:
            raise ValueError("Preprocessing inputs changed before acceptance")
        receipt["tpr_sha256"] = sha(audit / "accepted.tpr")
        (audit / "accepted.tpr").replace(target)
        receipt.update(accepted=True, target=str(target))
        return audit / "receipt.json"
    except (ValueError, KeyError, OSError) as exc:
        receipt["error"] = str(exc)
        raise RuntimeError(
            f"Preprocessing policy rejected {target.name}: {exc}; evidence: {audit}"
        ) from exc
    finally:
        if expanded.exists():
            (audit / "processed.txt").write_bytes(expanded.read_bytes())
        scratch.cleanup()
        (audit / "receipt.json").write_text(json.dumps(receipt, indent=2))


def _compiled_identity(gmx: str, tpr: Path) -> tuple[str, str]:
    """Compare compiled parameters, not coordinates or regenerated velocities.

    Only continuation length may differ. The entire compiled topology
    (including charges, exclusions, interactions and groups) must match exactly.
    """
    with tempfile.TemporaryFile(mode="w+") as output:
        subprocess.run(
            [gmx, "dump", "-s", str(tpr)],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600,
        )
        output.seek(0)
        text = output.read()
    try:
        inputrec = text.split("\ninputrec:\n", 1)[1].split("\nheader:\n", 1)[0]
        topology = text.split("\ntopology:\n", 1)[1].split("\nbox (3x3):", 1)[0]
    except IndexError as exc:
        raise ValueError("Unrecognized compiled topology dump") from exc
    if "moltype" not in topology or "ffparams:" not in topology:
        raise ValueError("Incomplete compiled topology dump")
    inputrec = re.sub(r"^\s*nsteps\s*=.*$", "", inputrec, flags=re.M)
    return inputrec, topology


def _retained_seed_mdp(text: str, inputrec: str) -> tuple[str, int]:
    """Reproduce the retained stochastic seed in the scratch compilation.

    GROMACS replaces the default -1 with a random integer at preprocessing.
    This seed is also used by V-rescale/C-rescale, so compare it explicitly.
    An explicitly configured seed must already agree with the retained TPR.
    """
    matches = re.findall(r"^\s*ld-seed\s*=\s*(-?\d+)\s*$", inputrec, re.M)
    if len(matches) != 1:
        raise ValueError("Retained TPR lacks an unambiguous stochastic seed")
    seed = int(matches[0])
    lines = []
    explicit = []
    for line in text.splitlines():
        raw = line.split(";", 1)[0]
        key, separator, value = raw.partition("=")
        if separator and key.strip().lower().replace("_", "-") == "ld-seed":
            explicit.append(int(value.strip()))
        else:
            lines.append(line)
    if len(explicit) > 1 or (explicit and explicit[0] not in {-1, seed}):
        raise ValueError("Configured stochastic seed differs from retained TPR")
    return "\n".join([*lines, f"ld-seed = {seed}", ""]), seed


def ensure_dynamics_policy(gmx: str, work: Path, stage: str) -> Path:
    """Audit legacy/extended TPRs without rewriting checkpoints or old evidence."""
    if stage not in {"nvt", "npt"}:
        raise ValueError("Unsupported dynamics stage")
    work = work.resolve()
    tpr = work / f"{stage}.tpr"
    digest = sha(tpr)
    audit_root = work / ".grompp-policy"
    if audit_root.exists():
        for path in sorted(audit_root.glob("*/receipt.json"), reverse=True):
            try:
                record = json.loads(path.read_text())
                if (
                    record.get("policy") == POLICY
                    and record.get("accepted") is True
                    and record.get("tpr_sha256") == digest
                    and record.get("target") == str(tpr)
                    and record.get("stage") == "neutralized"
                    and record.get("inputs") == _inputs(record["command"], work)
                ):
                    if record.get("exception") == "reviewed-psm128-source-residual":
                        _psm_version(gmx)
                    return path
            except (OSError, KeyError, ValueError):
                # Incomplete/relocated receipts never confer acceptance.
                continue
    nvt_output = "nvt"
    if stage == "npt" and not (work / "nvt.gro").exists():
        nvt_output = "nvt_cpu"
    args = [
        gmx,
        "grompp",
        "-f",
        f"{stage}.mdp",
        "-c",
        "em.gro" if stage == "nvt" else f"{nvt_output}.gro",
        "-p",
        "topol.top",
        "-o",
        str(tpr),
    ]
    if stage == "npt":
        args.extend(["-t", f"{nvt_output}.cpt"])
    current = _inputs(args, work)
    audit_root.mkdir(exist_ok=True)
    migration = Path(tempfile.mkdtemp(prefix=f"legacy-{stage}-", dir=audit_root))
    original_args = list(args)
    original_identity = _compiled_identity(gmx, tpr)
    seeded_mdp, seed = _retained_seed_mdp((work / f"{stage}.mdp").read_text(), original_identity[0])
    (migration / "retained-seed.mdp").write_text(seeded_mdp)
    _replace(args, "-f", str(migration / "retained-seed.mdp"))
    _replace(args, "-o", str(migration / "review.tpr"))
    receipt_path = run_grompp(args, work)
    if original_identity != _compiled_identity(gmx, migration / "review.tpr"):
        raise RuntimeError(f"Retained {stage}.tpr differs from newly audited compiled inputs")
    if sha(tpr) != digest or _inputs(original_args, work) != current:
        raise RuntimeError("Retained dynamics inputs changed during policy audit")
    receipt = json.loads(receipt_path.read_text())
    receipt.update(
        tpr_sha256=digest,
        target=str(tpr),
        legacy_audit=True,
        retained_stochastic_seed=seed,
        command=original_args,
        inputs=current,
        preprocessing_receipt=str(receipt_path),
    )
    result = migration / "receipt.json"
    result.write_text(json.dumps(receipt, indent=2))
    return result
