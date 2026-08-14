"""pH-dependent amino-acid protonation state assignment.

Uses PROPKA 3.x for environment-sensitive pKa prediction when a PDB
structure is available.  Falls back to standard model-pKa values for
sequence-only assignment.

Supports residue renaming for CHARMM/AMBER force-field conventions.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Standard model-pKa values (solvent-exposed residues in unfolded state)
# ---------------------------------------------------------------------------
_MODEL_PKA: dict[str, dict[str, float]] = {
    # residue → { protonated_form: pKa }
    # pKa is the pH at which half the residues are protonated.
    # For acidic residues: pKa of sidechain COOH → COO⁻ + H⁺
    # For basic residues: pKa of sidechain NH₃⁺ → NH₂ + H⁺
    "HIS": {"neutral": 6.0},     # imidazole H⁺ dissociation
    "ASP": {"neutral": 3.9},     # β-COOH → β-COO⁻
    "GLU": {"neutral": 4.3},     # γ-COOH → γ-COO⁻
    "CYS": {"thiolate": 8.3},    # -SH → -S⁻
    "LYS": {"neutral": 10.5},    # ε-NH₃⁺ → ε-NH₂
    "TYR": {"phenolate": 10.1},  # -OH → -O⁻
    # N-terminal NH₃⁺: pKa ~8.0 (model compound)
    # C-terminal COOH: pKa ~3.5 (model compound)
    "NTER": {"neutral": 8.0},
    "CTER": {"neutral": 3.5},
}


# ---------------------------------------------------------------------------
# Protonation states and residue names (CHARMM / AMBER conventions)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProtonationState:
    """A possible protonation state of a residue at a given pH."""

    residue_name: str   # new residue name, e.g. "HSD"
    charge: int         # net sidechain charge
    description: str    # human-readable

_TITRATABLE_STATES: dict[str, list[ProtonationState]] = {
    "HIS": [
        ProtonationState("HSD", 0, "Neutral (proton on Nδ)"),
        ProtonationState("HSE", 0, "Neutral (proton on Nε)"),
        ProtonationState("HSP", 1, "Doubly protonated (+1)"),
    ],
    "ASP": [
        ProtonationState("ASP", -1, "Deprotonated (-1) — aspartate"),
        ProtonationState("ASH", 0, "Neutral (0) — aspartic acid"),
    ],
    "GLU": [
        ProtonationState("GLU", -1, "Deprotonated (-1) — glutamate"),
        ProtonationState("GLH", 0, "Neutral (0) — glutamic acid"),
    ],
    "LYS": [
        ProtonationState("LYS", 1, "Protonated (+1) — lysine"),
        ProtonationState("LYN", 0, "Neutral (0) — deprotonated"),
    ],
    "CYS": [
        ProtonationState("CYS", 0, "Protonated (0) — free cysteine"),
        ProtonationState("CYM", -1, "Deprotonated (-1) — thiolate"),
    ],
    "TYR": [
        ProtonationState("TYR", 0, "Protonated (0) — tyrosine"),
        ProtonationState("TYM", -1, "Deprotonated (-1) — tyrosinate"),
    ],
}


# ---------------------------------------------------------------------------
# Protonation assignment
# ---------------------------------------------------------------------------

def get_titratable_residues() -> dict[str, list[ProtonationState]]:
    """Return a copy of the titratable residues dictionary."""
    return dict(_TITRATABLE_STATES)


def assign_protonation(
    residue_name: str,
    pH: float,
    his_tautomer: str = "HSE",
) -> dict:
    """Determine the protonation state of a single residue at a given pH.

    Parameters
    ----------
    residue_name : str
        The original residue name (3-letter code, e.g. "HIS").
    pH : float
        Target pH.
    his_tautomer : str
        Preferred HIS tautomer when neutral: "HSD" (Nδ) or "HSE" (Nε).

    Returns
    -------
    dict with keys:
        original, assigned_name, charge, state_label, pKa, is_titratable
    """
    rn = residue_name.strip().upper()
    if rn not in _TITRATABLE_STATES:
        return {
            "original": rn,
            "assigned_name": rn,
            "charge": 0,
            "state_label": "non-titratable",
            "pKa": None,
            "is_titratable": False,
        }

    pka = list(_MODEL_PKA.get(rn, {}).values())
    pka_val = pka[0] if pka else 7.0
    states = _TITRATABLE_STATES[rn]

    # Assign based on pH vs pKa
    if rn in ("ASP", "GLU"):
        # Acidic: protonated (neutral) below pKa, deprotonated (-1) above
        if pH < pka_val:
            state = next(s for s in states if s.charge == 0)  # ASH/GLH
        else:
            state = next(s for s in states if s.charge == -1)  # ASP/GLU

    elif rn in ("LYS", "TYR"):
        # Basic: protonated above pKa? No — actually:
        # LYS: +1 below pKa, neutral above
        # TYR: neutral below pKa, -1 above
        if rn == "LYS":
            if pH < pka_val:
                state = next(s for s in states if s.charge == 1)  # LYS
            else:
                state = next(s for s in states if s.charge == 0)  # LYN
        else:  # TYR
            if pH < pka_val:
                state = next(s for s in states if s.charge == 0)  # TYR
            else:
                state = next(s for s in states if s.charge == -1)  # TYM

    elif rn == "CYS":
        # Neutral below pKa, thiolate above
        if pH < pka_val:
            state = next(s for s in states if s.charge == 0)  # CYS
        else:
            state = next(s for s in states if s.charge == -1)  # CYM

    elif rn == "HIS":
        # +1 below pKa, neutral above
        if pH < pka_val:
            state = next(s for s in states if s.charge == 1)  # HSP
        else:
            # Pick preferred tautomer
            if his_tautomer == "HSD":
                state = next(s for s in states if s.residue_name == "HSD")
            else:
                state = next(s for s in states if s.residue_name == "HSE")
    else:
        state = states[0]

    return {
        "original": rn,
        "assigned_name": state.residue_name,
        "charge": state.charge,
        "state_label": state.description,
        "pKa": round(pka_val, 1),
        "is_titratable": True,
        "ambiguous_at_pka": abs(float(pH) - pka_val) < 1e-9,
        "alternatives": [
            {"name": s.residue_name, "charge": s.charge, "label": s.description}
            for s in states
        ],
    }


def assign_all_protonations(
    residue_list: list[str],
    pH: float = 7.0,
    his_tautomer: str = "HSE",
) -> list[dict]:
    """Assign protonation states to a list of residues.

    Parameters
    ----------
    residue_list : list[str]
        Ordered list of 3-letter residue names.
    pH : float
    his_tautomer : str

    Returns
    -------
    list of dicts, one per residue, with additional 'index' and 'chain' fields
    derived from input context.
    """
    results = []
    for i, rn in enumerate(residue_list):
        result = assign_protonation(rn, pH=pH, his_tautomer=his_tautomer)
        result["index"] = i
        results.append(result)
    return results


def compute_net_charge_from_protonation(
    assignments: list[dict],
) -> int:
    """Sum the sidechain charges from protonation assignments."""
    return sum(a.get("charge", 0) for a in assignments)


def get_charge_adjustment(
    original_residues: list[str],
    pH: float = 7.0,
) -> dict:
    """Compute the net change in protein charge after protonation at given pH.

    Returns dict with original_charge, new_charge, delta, and per-residue details.
    """
    assignments = assign_all_protonations(original_residues, pH=pH)
    reference_assignments = assign_all_protonations(original_residues, pH=7.0)
    original_charge = compute_net_charge_from_protonation(reference_assignments)
    new_charge = compute_net_charge_from_protonation(assignments)
    return {
        "assignments": assignments,
        "reference_pH": 7.0,
        "original_charge": original_charge,
        "new_charge": new_charge,
        "delta": new_charge - original_charge,
    }


# =============================================================================
# PROPKA integration — environment-sensitive pKa prediction
# =============================================================================

def predict_pka_from_pdb(pdb_path: str | Path) -> list[dict]:
    """Run PROPKA on a PDB file and return per-residue pKa predictions.

    Uses the `propka3` command-line tool via subprocess, which is the
    most robust way to invoke PROPKA across versions.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file.

    Returns
    -------
    list of dicts, each with keys:
        residue_name, chain, resid, model_pKa, predicted_pKa, shift
    """
    import subprocess

    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdb = Path(tmpdir) / pdb_path.name
        # Normalize atom-name alignment before passing the file to PROPKA.
        # Older checkpoints and some third-party writers left-align one-letter
        # element names, which causes PROPKA to infer an incomplete bond graph.
        from gmxbuilder.io.pdb import format_pdb_atom_name

        normalized_lines = []
        for line in pdb_path.read_text(errors="replace").splitlines(keepends=True):
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 16:
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) >= 78 else ""
                line = line[:12] + format_pdb_atom_name(atom_name, element) + line[16:]
            normalized_lines.append(line)
        tmp_pdb.write_text("".join(normalized_lines))

        # Try 'propka3' first, then 'propka'.  A command that starts but exits
        # non-zero is a calculation failure; never parse its possibly partial
        # .pka file as if it were complete.
        failures: list[str] = []
        executable_found = False
        for cmd in ["propka3", "propka"]:
            try:
                result = subprocess.run(
                    [cmd, "-q", str(tmp_pdb)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                executable_found = True
                if result.returncode == 0:
                    break
                detail = (result.stderr or result.stdout or "no diagnostic output").strip()
                failures.append(f"{cmd} exited {result.returncode}: {detail[:300]}")
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                executable_found = True
                failures.append(f"{cmd} timed out after 120 seconds")
        else:
            if executable_found:
                raise RuntimeError("PROPKA calculation failed: " + "; ".join(failures))
            # Neither command is installed; the caller can explicitly report
            # that model-pKa fallback is being used.
            return []

        # Find the .pka output file
        pka_files = list(Path(tmpdir).glob("*.pka"))
        if not pka_files:
            raise RuntimeError("PROPKA completed without producing a .pka result file")

        predictions = _parse_propka_output(pka_files[0])

    return predictions


def _parse_propka_output(pka_file: Path) -> list[dict]:
    """Parse PROPKA .pka output file into structured data.

    Only reads the summary table section (after 'SUMMARY OF THIS PREDICTION').
    PROPKA v3.5 format per line:
        RESNAME  RESID  CHAIN  predicted_pKa  model_pKa
    """
    results = []
    with open(pka_file) as fh:
        in_summary = False
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect the summary table header
            if 'SUMMARY' in stripped.upper() and 'PREDICTION' in stripped.upper():
                in_summary = True
                continue

            if not in_summary:
                continue

            # Skip separator lines and non-data
            if stripped.startswith('-') or stripped.startswith('='):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            residue = parts[0].upper()
            if residue not in ("ASP", "GLU", "HIS", "LYS", "CYS", "TYR"):
                continue

            try:
                if len(parts) >= 5:
                    resname = parts[0]
                    resid = int(parts[1])
                    chain = parts[2]
                    predicted_pka = float(parts[3])
                    model_pka = float(parts[4])
                elif len(parts) >= 4:
                    resname = parts[0]
                    resid = int(parts[1])
                    chain = ""
                    predicted_pka = float(parts[2])
                    model_pka = float(parts[3])
                else:
                    continue

                results.append({
                    "residue_name": resname,
                    "chain": chain,
                    "resid": resid,
                    "model_pKa": round(model_pka, 2),
                    "predicted_pKa": round(predicted_pka, 2),
                    "shift": round(predicted_pka - model_pka, 2),
                })
            except (ValueError, IndexError):
                continue

    return results


def assign_protonation_with_propka(
    structure_residues: list[dict],
    pka_predictions: list[dict],
    pH: float = 7.0,
    his_tautomer: str = "HSE",
) -> list[dict]:
    """Combine PROPKA pKa predictions with protonation assignment.

    Parameters
    ----------
    structure_residues : list[dict]
        From _procResidues format: [{resname, chain, resid, index}, ...]
    pka_predictions : list[dict]
        From predict_pka_from_pdb().
    pH : float
    his_tautomer : str

    Returns
    -------
    list of assignment dicts (same format as assign_all_protonations output,
    but with predicted_pKa and pKa_shift fields added).
    """
    # Build a lookup: (resname, chain, resid) → predicted pKa
    pka_lookup: dict[tuple[str, str, int], dict] = {}
    for p in pka_predictions:
        key = (p["residue_name"].upper(), p.get("chain", "").strip(),
               p.get("resid", 0))
        pka_lookup[key] = p

    results = []
    for r in structure_residues:
        rn = r["resname"].strip().upper()
        key = (rn, r.get("chain", "").strip(), r.get("resid", 0))
        pka_data = pka_lookup.get(key)

        # Get the baseline assignment
        base = assign_protonation(rn, pH=pH, his_tautomer=his_tautomer)

        if pka_data and base["is_titratable"]:
            # Use PROPKA-predicted pKa instead of model pKa
            predicted_pka = pka_data["predicted_pKa"]
            pka_shift = pka_data["shift"]

            # Re-determine protonation using predicted pKa
            if rn in ("ASP", "GLU"):
                if pH < predicted_pka:
                    base["assigned_name"] = "ASH" if rn == "ASP" else "GLH"
                    base["charge"] = 0
                    base["state_label"] = f"Neutral (pKa_pred={predicted_pka:.1f})"
                else:
                    base["assigned_name"] = rn
                    base["charge"] = -1
                    base["state_label"] = f"Deprotonated (pKa_pred={predicted_pka:.1f})"
            elif rn == "HIS":
                if pH < predicted_pka:
                    base["assigned_name"] = "HSP"
                    base["charge"] = 1
                    base["state_label"] = f"Protonated +1 (pKa_pred={predicted_pka:.1f})"
                else:
                    base["assigned_name"] = his_tautomer
                    base["charge"] = 0
                    base["state_label"] = f"Neutral {his_tautomer} (pKa_pred={predicted_pka:.1f})"
            elif rn == "LYS":
                if pH < predicted_pka:
                    base["assigned_name"] = "LYS"
                    base["charge"] = 1
                    base["state_label"] = f"Protonated +1 (pKa_pred={predicted_pka:.1f})"
                else:
                    base["assigned_name"] = "LYN"
                    base["charge"] = 0
                    base["state_label"] = f"Neutral (pKa_pred={predicted_pka:.1f})"
            elif rn == "CYS":
                if pH < predicted_pka:
                    base["assigned_name"] = "CYS"
                    base["charge"] = 0
                    base["state_label"] = f"Protonated (pKa_pred={predicted_pka:.1f})"
                else:
                    base["assigned_name"] = "CYM"
                    base["charge"] = -1
                    base["state_label"] = f"Thiolate (pKa_pred={predicted_pka:.1f})"
            elif rn == "TYR":
                if pH < predicted_pka:
                    base["assigned_name"] = "TYR"
                    base["charge"] = 0
                    base["state_label"] = f"Protonated (pKa_pred={predicted_pka:.1f})"
                else:
                    base["assigned_name"] = "TYM"
                    base["charge"] = -1
                    base["state_label"] = f"Tyrosinate (pKa_pred={predicted_pka:.1f})"

            base["predicted_pKa"] = round(predicted_pka, 2)
            base["pKa_shift"] = round(pka_shift, 2)

        base["index"] = r["index"]
        results.append(base)

    return results
