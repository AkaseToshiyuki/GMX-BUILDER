"""Versioned Martini 3 assets owned by the coarse-grained module."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from functools import lru_cache
from importlib import metadata, resources
from pathlib import Path

from gmxbuilder.core.exceptions import ModuleConfigError


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    root = resources.files("gmxbuilder") / "data" / "martini3"
    with resources.as_file(root / "manifest.json") as path:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["lipids"] = _expand_bundled_lipidome(manifest)
    return manifest


def _asset_root() -> Path:
    root = resources.files("gmxbuilder") / "data" / "martini3" / "v3.0.0"
    with resources.as_file(root) as path:
        return Path(path)


_LIPID_FAMILIES = {
    "martini_v3.0.0_phospholipids_PC_v2.itp": ("PC", ("NC3", "PO4"), 0.64),
    "martini_v3.0.0_phospholipids_PE_v2.itp": ("PE", ("NH3", "PO4"), 0.60),
    "martini_v3.0.0_phospholipids_PG_v2.itp": ("PG", ("GL0", "PO4"), 0.65),
    "martini_v3.0.0_phospholipids_PS_v2.itp": ("PS", ("CNO", "PO4"), 0.64),
    "martini_v3.0.0_phospholipids_SM_v2.itp": ("SM", ("NC3", "PO4"), 0.63),
}


def _itp_molecule_atoms(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Read molecule atom names/charges from one trusted bundled ITP."""
    molecules: dict[str, list[tuple[str, float]]] = {}
    section = ""
    molecule: str | None = None
    waiting_for_name = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            waiting_for_name = section == "moleculetype"
            continue
        fields = line.split()
        if waiting_for_name:
            molecule = fields[0].upper()
            molecules.setdefault(molecule, [])
            waiting_for_name = False
            continue
        if section == "atoms" and molecule and len(fields) >= 7 and fields[0].isdigit():
            try:
                charge = float(fields[6])
            except ValueError:
                continue
            molecules[molecule].append((fields[4].upper(), charge))
    return molecules


@lru_cache(maxsize=1)
def lipid_viewer_topologies() -> dict[str, dict[str, tuple]]:
    """Return trusted bead order and topology edges for bundled lipids.

    Browser PDB viewers cannot infer Martini bonds reliably from distance.
    These edges are therefore read from the same bundled ITP files used by
    GROMACS.  Constraints and virtual-site construction links are included so
    rigid sterols are displayed as connected molecules as well.
    """
    root = _asset_root()
    manifest = load_manifest()
    wanted = set(manifest["lipids"])
    topologies: dict[str, dict[str, tuple]] = {}

    for filename in manifest.get("files", {}):
        if not filename.endswith(".itp"):
            continue
        path = root / filename
        section = ""
        molecule: str | None = None
        waiting_for_name = False
        atom_names: list[str] = []
        edges: set[tuple[int, int]] = set()

        def save_current() -> None:
            if molecule not in wanted or not atom_names:
                return
            topologies[molecule] = {
                "atom_names": tuple(atom_names),
                "edges": tuple(sorted(edges)),
            }

        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                next_section = line[1:-1].strip().lower()
                if next_section == "moleculetype":
                    save_current()
                    molecule = None
                    atom_names = []
                    edges = set()
                    waiting_for_name = True
                section = next_section
                continue
            fields = line.split()
            if waiting_for_name:
                molecule = fields[0].upper()
                waiting_for_name = False
                continue
            if molecule not in wanted:
                continue
            if section == "atoms" and len(fields) >= 5 and fields[0].isdigit():
                atom_names.append(fields[4].upper())
                continue
            if section in {"bonds", "constraints"} and len(fields) >= 2:
                if fields[0].isdigit() and fields[1].isdigit():
                    a, b = int(fields[0]), int(fields[1])
                    if a != b:
                        edges.add(tuple(sorted((a, b))))
                continue
            if section in {"virtual_sites2", "virtual_sites3", "virtual_sites4"}:
                constructor_count = int(section[-1])
                if len(fields) >= constructor_count + 1 and all(
                    value.isdigit() for value in fields[: constructor_count + 1]
                ):
                    site = int(fields[0])
                    for value in fields[1 : constructor_count + 1]:
                        constructor = int(value)
                        if site != constructor:
                            edges.add(tuple(sorted((site, constructor))))
        save_current()

    return topologies


def _coby_ltf_lipids() -> set[str]:
    """Return exact lipid names constructible by the pinned COBY LTF library."""
    try:
        from COBY.molecule_definitions import lipid_scaffolds
    except (ImportError, ModuleNotFoundError):
        return set()
    names: set[str] = set()
    for (_kind, parameter_set), scaffold in lipid_scaffolds.items():
        if parameter_set != "LTF":
            continue
        for key in scaffold.get("lipids", {}):
            name = key[0] if isinstance(key, tuple) else key
            names.add(str(name).upper())
    return names


def _expand_bundled_lipidome(manifest: dict) -> dict:
    """Expose every constructible lipid in the bundled official ITP classes.

    The JSON file retains individually curated overrides.  Additional entries
    are derived only when an exact molecule topology and a compatible COBY/LTF
    scaffold are both already shipped.  ``apl_nm2`` is a conservative initial
    construction-area default, not an equilibrium APL claim; NPT equilibration
    is expected to relax the periodic area.
    """
    lipids = {str(name).upper(): dict(values) for name, values in manifest["lipids"].items()}
    root = _asset_root()
    coby_ltf_lipids = _coby_ltf_lipids()
    for filename, (family, expected_heads, construction_area) in _LIPID_FAMILIES.items():
        if filename not in manifest.get("files", {}):
            continue
        for name, atoms in _itp_molecule_atoms(root / filename).items():
            if name in lipids or name not in coby_ltf_lipids:
                continue
            atom_names = {atom_name for atom_name, _charge in atoms}
            heads = [head for head in expected_heads if head in atom_names]
            tails = sorted(
                (atom_name for atom_name in atom_names if re.fullmatch(r"[CDT]\d+[AB]", atom_name)),
                key=lambda value: (value[-1], int(value[1:-1]), value[0]),
            )
            by_chain = {
                suffix: [name for name in tails if name.endswith(suffix)] for suffix in ("A", "B")
            }
            if not heads or any(not values for values in by_chain.values()):
                continue
            tail_markers = [name for values in by_chain.values() for name in values[-2:]]
            midplane_markers = [values[min(1, len(values) - 1)] for values in by_chain.values()]
            lipids[name] = {
                "family": family,
                "charge": int(round(sum(charge for _atom, charge in atoms))),
                "apl_nm2": construction_area,
                "builder_params": "LTF",
                "head_beads": heads,
                "midplane_beads": midplane_markers,
                "tail_beads": tail_markers,
                "construction_area_source": "conservative family default",
                "parameter_source": filename,
            }
    return dict(sorted(lipids.items()))


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
    available_beads: dict[str, set[str]] = {}
    for name in verified:
        if not name.endswith(".itp"):
            continue
        section = ""
        for raw_line in (root / name).read_text(encoding="utf-8").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if section != "atoms" or not line:
                continue
            fields = line.split()
            if len(fields) >= 5 and fields[0].isdigit():
                available_beads.setdefault(fields[3].upper(), set()).add(fields[4].upper())
    for lipid, definition in manifest["lipids"].items():
        required = {
            str(bead).upper()
            for field in ("head_beads", "midplane_beads", "tail_beads")
            for bead in definition.get(field, [])
        }
        missing = required - available_beads.get(lipid.upper(), set())
        if missing:
            raise ModuleConfigError(
                f"Martini 3 manifest references unknown {lipid} beads: "
                + ", ".join(sorted(missing))
            )
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
        raise ModuleConfigError("Unsupported Martini 3 tool versions: " + "; ".join(mismatched))
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
        "lipids": [{"name": name, **values} for name, values in manifest["lipids"].items()],
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
            "backmapping": False,
        },
    }
