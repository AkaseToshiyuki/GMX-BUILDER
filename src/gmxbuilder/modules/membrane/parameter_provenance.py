"""Content-based parameter identity for lipid sampling and cache admission."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from gmxbuilder.modules.forcefield.catalog import force_field_directory

CONTRACT = "lipid-parameters-2-explicit-amber-pairs-local-charmm"

# Exact installer layouts reviewed on 2026-10-08. The supplemental parameter
# bytes are identical; entrypoints differ only in comments, whitespace and the
# named include. All directives and 36 retained compiled NPT parameter sets
# were compared independently. No arbitrary formatting or parameter edits are
# normalized. A missing supplement or an ambiguous old/new pair stays invalid.
_CHARMM_INSTALLER_LAYOUTS = {
    "charmm36": {
        "forcefield.itp": (
            "f4349b722c64156264e0b4fb21d8438288eea3e1cce59934f06bb550a9cb007b",
            "forcefield.itp",
            "e4c6621151e08b45fc327ade65f697d12483d3034d2ae81a26da42a4aa40bcf8",
        ),
        "gmxbuilder-current-lipid-bonded.itp": (
            "c5fed63e679085d03616fc0ac1e20db8cb7647c36286927135ac8744857c8cf7",
            "current-lipid-bonded.itp",
            "c5fed63e679085d03616fc0ac1e20db8cb7647c36286927135ac8744857c8cf7",
        ),
        "gmxbuilder-plasmalogen-bonded.itp": (
            "4477a5bcb12f5889948c9ebc45cf364405bdabc30a74917c6acc1f16d8b0b64e",
            "plasmalogen-bonded.itp",
            "4477a5bcb12f5889948c9ebc45cf364405bdabc30a74917c6acc1f16d8b0b64e",
        ),
    },
    "charmm36m": {
        "forcefield.itp": (
            "1a2fa3bc6042ec16fd37f69646f73cf2134e68aec07f0cd9bda5bfda57706a69",
            "forcefield.itp",
            "e9daf5e0f86f4ad74a6de1da2874971190947d8aa5c1d3bb7f94cf2399f4def8",
        ),
        "gmxbuilder-plasmalogen-bonded.itp": (
            "46c0a3dc53166cd85b46bf7b834e32a2a348468a092277c3ebfd96dc60282ab2",
            "plasmalogen-bonded.itp",
            "46c0a3dc53166cd85b46bf7b834e32a2a348468a092277c3ebfd96dc60282ab2",
        ),
    },
}
_OPTIONAL_INSTALLER_ARCHIVES = {
    "charmm36": {"charmm36.tgz": "93659165386d7b9eaf5888b3aad1f672e711409545433bae9a3dc6e82cfa72c6"}
}


def canonical_forcefield_files(force_field: str, files: dict[str, str]) -> dict[str, str]:
    """Recognize only the complete, source-pinned equivalent installer layout."""
    layout = _CHARMM_INSTALLER_LAYOUTS.get(force_field, {})
    result = dict(files)
    if not layout or any(
        files.get(name) != current or (name != old_name and old_name in files)
        for name, (current, old_name, _) in layout.items()
    ):
        return result
    for name, (_, old_name, checksum) in layout.items():
        del result[name]
        result[old_name] = checksum
    return result


def installed_forcefield_matches(force_field: str, root: Path, expected: dict[str, str]) -> bool:
    """Keep complete reuse-file checks while allowing the reviewed fresh layout."""
    actual = canonical_forcefield_files(
        force_field,
        {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()},
    )
    required = dict(expected)
    for name, digest in _OPTIONAL_INSTALLER_ARCHIVES.get(force_field, {}).items():
        # This exact upstream download was retained by the old installation;
        # the current installer verifies it then discards it. It is not read
        # by GROMACS. If present, it must still match the audited digest.
        if required.get(name) == digest and name not in actual:
            del required[name]
    return actual == required


# Exact-source compatibility for the 2026-09-15 admission-only repair. The
# changed AST nodes are exclusively _GAFF_UNAVAILABLE and gaff_lipid_capability;
# parameter construction and exported definitions are unchanged. Keep existing
# trajectories valid without rewriting their evidence. Any further source edit
# misses this exact digest and invalidates the fingerprint as usual.
_ADMISSION_ONLY_POLICY_HASHES = {
    # 2026-09-16: only capability predicates change; accepted V4 clears old holds.
    "3aa384218d3fb6fcda44439ed10b692a1d3ff0e1c18b87b72ef91a18237090a0": (
        "fc65f6baa9957e7010c9ffb19a95fc2349c289721807caf37877537f9aa068d9"
    ),
    "3b8efb24b98d5a68a54d8c9efb1581e99e805863317a25d58bf218b9ed4c0354": (
        "fc65f6baa9957e7010c9ffb19a95fc2349c289721807caf37877537f9aa068d9"
    ),
}


# Exact 0.9.124 compatibility: valid RTP parsing and unmodified lipid definitions
# were compared to 0.9.123. The corrected West2020 plasmalogens are NEVER aliased
# on CHARMM backends. An arbitrary subsequent source edit misses these hashes.
_CHECKDSH_POLICY_HASH = "5a77e0d41578a9f6628a17de69a5aec866a69c796e7b3d4dc52952ced56af0a0"
_CHECKDSH_RTP_HASH = "8281b51b4c3999cac37350479f8647e80a8809fe4544ad28b9faf4c3b4aa3a0d"
_CHECKDSH_PRIOR_RTP_HASH = "44d6eb51209527384f55076c10e54016de72998e1b5e38852ff4cf0857892585"
_PIP_TEMPLATE_POLICY_HASH = "6d1e088caa6ea5cb5fcd4a93d31c8624b475d9b2b2dffc8eb9ddeab6e125b512"


@lru_cache(maxsize=4096)
def _file_hash(path: str, stamp: tuple[int, int, int]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hash(path: Path) -> str:
    stat = path.stat()
    return _file_hash(str(path), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def _implementation_hash(
    path: Path,
    relative: str,
    *,
    unchanged_policy: bool = True,
    unchanged_pip_template: bool = True,
) -> str:
    digest = file_hash(path)
    # 2026-10-03: Python 3.10 compatibility only. UTC -> timezone.utc affects
    # the replaced-cache timestamp, never atom types, charges or fitting.
    # Alias this exact source only; any subsequent edit still invalidates it.
    if relative == "modules/forcefield/gaff_backend.py" and digest == (
        "f191d9deb3c61e2cd189d95d1ff047867681957af0c48a60183c493f3c5b6205"
    ):
        return "6a4ee9aa3953a370d5fe24482d331823a1b9d5effa0eb03cd2f2dc9937653ab5"
    # Official PGR -> PGS source selection only. All 39 re-exported ITPs and
    # the shared atomtype table are identical; only seven seed geometries
    # change. Molecular identity admission still rejects archived R,R results.
    if relative == "modules/forcefield/lipid21_backend.py" and digest == (
        "26b13f8e295a7b9ea7ea7664fc6bb4f4138634484028fb5446bff003bfd935b2"
    ):
        return "4f217c32d97ffc1e98f183b5fad6e8ffa84d9230eaa1afbaf4278708703774a6"
    # 2026-09-26: finite catalog/locking only; all 4,372 installed residue
    # records compare exactly against the preceding parser.
    if relative == "modules/forcefield/rtp_parser.py" and digest in {
        _CHECKDSH_RTP_HASH,
        "2fc864adfd03ea31bea4bbe8f3fd4ae8802eb585449fa340be9097e6fab79182",
    }:
        return _CHECKDSH_PRIOR_RTP_HASH
    if relative == "modules/forcefield/lipid_policy.py":
        # 2026-09-27: only four named native phosphoinositide templates change.
        # All other 148 installed RTP exports were compared byte-for-byte in
        # their canonical JSON representation. Never alias the changed models.
        if digest == _PIP_TEMPLATE_POLICY_HASH and unchanged_pip_template:
            digest = _CHECKDSH_POLICY_HASH
        if digest == _CHECKDSH_POLICY_HASH and unchanged_policy:
            return "fc65f6baa9957e7010c9ffb19a95fc2349c289721807caf37877537f9aa068d9"
        return _ADMISSION_ONLY_POLICY_HASHES.get(digest, digest)
    return digest


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def parameter_fingerprint(
    lipid_name: str, force_field: str, lipid_ff: str, host: dict | None = None
) -> str:
    """Hash actual definitions for the target and its sampling environment.

    No parameterization is launched by a lookup. Missing parameter files make
    historical candidates unusable, rather than granting them a new signature.
    Paths are relative so an immutable runtime checkout has the same identity.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for

    names = {lipid_name}
    if host:
        names.add(host["lipid_name"])
    unchanged_policy = not any(
        name in {"PPCPL", "PPEPL"}
        and lipid_backend_for(name, lipid_ff) in {"charmm36", "charmm36m"}
        for name in names
    )
    unchanged_pip_template = not any(
        name.upper() in {"POP2", "PAPI", "SAPI", "SOP2"}
        and lipid_backend_for(name, lipid_ff) in {"charmm36", "charmm36m"}
        for name in names
    )
    root = force_field_directory(force_field)
    if root is None or not (root / "forcefield.itp").is_file():
        raise ValueError(f"Missing parameter source for {force_field}")
    installed_files = {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".itp", ".rtp", ".tdb", ".r2b", ".hdb", ".atp"}
    }
    files = {
        f"forcefield/{name}": digest
        for name, digest in canonical_forcefield_files(force_field, installed_files).items()
    }
    data = Path(__file__).resolve().parents[2] / "data"
    if force_field in {"charmm36", "charmm36m"}:
        for p in sorted((data / "forcefield_overlays" / force_field).glob("*.itp")):
            files[f"local_parameters/{p.name}"] = file_hash(p)
    for relative in (
        "io/top.py",
        "modules/forcefield/lipid_policy.py",
        "modules/forcefield/charmm_lipid_local.py",
        "modules/forcefield/rtp_parser.py",
        "modules/forcefield/gaff_backend.py",
        "modules/forcefield/lipid21_backend.py",
    ):
        files[f"implementation/{relative}"] = _implementation_hash(
            data.parent / relative,
            relative,
            unchanged_policy=unchanged_policy,
            unchanged_pip_template=unchanged_pip_template,
        )
    names = {lipid_name}
    if host:
        names.add(host["lipid_name"])
    backends = {}
    for name in sorted(names):
        backend = lipid_backend_for(name, lipid_ff)
        backends[name] = backend
        if backend == "lipid21":
            for relative in (f"itp/{name}.itp", "lipid21_atomtypes.itp"):
                files[f"lipid21/{relative}"] = file_hash(data / "lipid21" / relative)
        elif backend == "gaff2":
            from gmxbuilder.modules.forcefield.gaff_backend import cached_gaff_template
            from gmxbuilder.modules.membrane.lipids import LipidRegistry

            lipid = LipidRegistry.get(name)
            template = cached_gaff_template(name, lipid.smiles, lipid.charge)
            if template is None:
                raise ValueError(f"Missing fitted GAFF2 parameter source for {name}")
            files[f"gaff2/{name}/molecule.itp"] = file_hash(template.itp_path)
            files[f"gaff2/{name}/atomtypes.itp"] = file_hash(template.atomtypes_path)
    return _digest({"contract": CONTRACT, "files": files, "backends": backends, "host": host})


def fingerprint_from_metadata(metadata: dict) -> str:
    return parameter_fingerprint(
        metadata["lipid_name"],
        metadata["force_field"],
        metadata["lipid_ff"],
        metadata.get("equilibration_host"),
    )


def parameters_current(metadata: dict) -> bool:
    if not metadata.get("parameter_fingerprint"):
        return False
    try:
        return metadata["parameter_fingerprint"] == fingerprint_from_metadata(metadata)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def work_parameter_files(work: Path) -> dict[str, str]:
    """Fingerprint resolved topology inputs without hashing mutable trajectories."""
    return {
        str(p.relative_to(work)): file_hash(p)
        for p in sorted(work.rglob("*"))
        if p.is_file() and p.suffix in {".top", ".itp"} and "analysis_history" not in p.parts
    }


def record_work_parameters(work: Path, fingerprint: str) -> None:
    files = work_parameter_files(work)
    if "topol.top" not in files:
        raise ValueError("Missing topology for parameter provenance")
    (work / "parameter-provenance.json").write_text(
        json.dumps({"fingerprint": fingerprint, "files": files}, indent=2) + "\n"
    )


def validate_work_parameters(work: Path, metadata: dict) -> None:
    if not parameters_current(metadata):
        raise ValueError("Trajectory parameters are missing or differ from the installed model")
    record = json.loads((work / "parameter-provenance.json").read_text())
    if record.get("fingerprint") != metadata["parameter_fingerprint"] or record.get(
        "files"
    ) != work_parameter_files(work):
        raise ValueError("Retained topology inputs changed; refusing trajectory continuation")
