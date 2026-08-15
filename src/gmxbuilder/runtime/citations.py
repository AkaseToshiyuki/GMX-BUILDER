"""Select reproducibility citations for one atomistic export."""

from __future__ import annotations

from importlib import resources
import json


def _registry() -> dict[str, dict]:
    root = resources.files("gmxbuilder") / "data" / "citations.json"
    with resources.as_file(root) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported citation registry schema")
    return dict(payload["references"])


def atomistic_citations(metadata: dict) -> dict:
    """Return a stable, method-aware citation manifest for an exported system."""
    force_field = str(metadata.get("force_field", "")).lower()
    lipid_ff = str(metadata.get("lipid_ff", "")).lower()
    ligand_ff = str(metadata.get("ligand_ff", "")).lower()
    water = str(metadata.get("water_model", "")).lower()
    selected = ["gmxbuilder", "gromacs", "v-rescale", "c-rescale"]
    if force_field == "amber14sb":
        selected.append("ff14sb")
    elif force_field in {"amber99sb", "amber99sb-ildn"}:
        selected.append("amber99sb")
    elif force_field == "charmm36m":
        selected.extend(["charmm36m", "charmm-gromacs"])
    elif force_field == "charmm36":
        selected.extend(["charmm36-protein", "charmm-gromacs"])
    elif force_field.startswith("opls"):
        selected.append("oplsaa")
    if lipid_ff == "lipid21":
        selected.append("lipid21")
    elif lipid_ff == "gaff2":
        selected.extend(["gaff", "acpype"])
    elif lipid_ff.startswith("charmm"):
        selected.append("charmm36-lipid")
    if ligand_ff == "gaff2":
        selected.extend(["gaff", "acpype", "rdkit"])
    elif ligand_ff == "cgenff":
        selected.append("cgenff")
    if water == "tip3p":
        selected.append("tip3p")
    if str(metadata.get("_orientation_method", "")).lower() in {"ppm", "auto"}:
        selected.extend(["ppm", "wimley-white"])
    ion_method = str(metadata.get("ions", {}).get("placement_method", "")).lower()
    if ion_method == "mc":
        selected.append("metropolis")
    registry = _registry()
    unique = list(dict.fromkeys(selected))
    return {
        "schema_version": 1,
        "generated_by": "GMXBUILDER",
        "notice": "Cite the methods and parameter sets actually used; review this list before publication.",
        "references": [{"id": key, **registry[key]} for key in unique],
    }
