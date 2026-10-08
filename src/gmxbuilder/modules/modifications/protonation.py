"""pH-dependent amino-acid protonation state assignment.

Uses PROPKA 3.x for environment-sensitive pKa prediction when a PDB
structure is available.  Falls back to standard model-pKa values for
sequence-only assignment.

Supports residue renaming for CHARMM/AMBER force-field conventions.
"""

from __future__ import annotations

import dataclasses
import math
import sys
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
    "HIS": {"neutral": 6.0},  # imidazole H⁺ dissociation
    "ASP": {"neutral": 3.9},  # β-COOH → β-COO⁻
    "GLU": {"neutral": 4.3},  # γ-COOH → γ-COO⁻
    "CYS": {"thiolate": 8.3},  # -SH → -S⁻
    "LYS": {"neutral": 10.5},  # ε-NH₃⁺ → ε-NH₂
    "TYR": {"phenolate": 10.1},  # -OH → -O⁻
    # N-terminal NH₃⁺: pKa ~8.0 (model compound)
    # C-terminal COOH: pKa ~3.5 (model compound)
    "NTER": {"neutral": 8.0},
    "CTER": {"neutral": 3.5},
}


# ---------------------------------------------------------------------------
# Free-terminus modelling boundary
# ---------------------------------------------------------------------------
# GMXBUILDER instantiates free termini only as the canonical charged templates
# (NH3+ and COO-); no neutral terminal microstate exists.  How defensible that
# assignment is depends on how populated the canonical form actually is at the
# requested pH, so both boundaries below are derived from the model pKa values
# above rather than written down as pH constants.  Changing a pKa moves the
# boundaries with it.
#
# MD requires one discrete state per titratable group and the convention is to
# assign the dominant one.  Below the pKa the charged assignment is therefore an
# approximation; above it the charged form is the *minority* species, which
# makes the assignment wrong rather than merely approximate.  That is why the
# hard boundary is the pKa itself and not a chosen number.
CANONICAL_TERMINUS_DOMINANT_FRACTION = 0.90
CANONICAL_TERMINUS_MAJORITY_FRACTION = 0.50


def canonical_terminus_fraction(terminus: str, pH: float) -> float:
    """Return the populated fraction of the canonical charged state.

    ``NTER`` is canonical as NH3+, which dominates *below* its pKa; ``CTER`` is
    canonical as COO-, which dominates *above* its pKa.
    """
    key = terminus.strip().upper()
    try:
        pka = _MODEL_PKA[key]["neutral"]
    except KeyError as exc:
        raise KeyError(f"No model pKa for terminus {terminus!r}") from exc
    exponent = (pH - pka) if key == "NTER" else (pka - pH)
    return 1.0 / (1.0 + 10.0**exponent)


def free_terminus_ph_window(fraction: float) -> tuple[float, float]:
    """Return the pH interval where both canonical termini reach *fraction*.

    Inverting the Henderson-Hasselbalch relation: the C-terminus sets the lower
    bound and the N-terminus the upper one.  At ``0.50`` this collapses to the
    two pKa values themselves.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")
    offset = math.log10(fraction / (1.0 - fraction))
    return (
        _MODEL_PKA["CTER"]["neutral"] + offset,
        _MODEL_PKA["NTER"]["neutral"] - offset,
    )


# ---------------------------------------------------------------------------
# Protonation states and residue names (CHARMM / AMBER conventions)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProtonationState:
    """A possible protonation state of a residue at a given pH."""

    state_id: str  # naming-independent identity, e.g. "HIS_ND"
    residue_name: str  # resolved for one force field, e.g. "HSD" or "HID"
    charge: int  # net sidechain charge
    description: str  # human-readable


# Amber and CHARMM disagree on the residue name of almost every non-default
# protonation state, and their .rtp files are the authority:
#
#   amber14sb  ASH  GLH  LYN  CYM  HID  HIE  HIP
#   charmm36m  ASPP GLUP LSN  CYM  HSD  HSE  HSP
#
# A single hard-coded table cannot serve both. The one that used to be here
# mixed them -- CHARMM histidines with Amber acids -- so a CHARMM build failed
# on ASH and an Amber build would have failed on HSD.
_STATE_DEFINITIONS: dict[str, list[tuple[str, dict[str, str], int, str]]] = {
    "HIS": [
        ("HIS_ND", {"amber": "HID", "charmm": "HSD", "opls": "HISD"}, 0, "Neutral (proton on Nδ)"),
        ("HIS_NE", {"amber": "HIE", "charmm": "HSE", "opls": "HISE"}, 0, "Neutral (proton on Nε)"),
        ("HIS_P", {"amber": "HIP", "charmm": "HSP", "opls": "HISH"}, 1, "Doubly protonated (+1)"),
    ],
    "ASP": [
        ("ASP", {"amber": "ASP", "charmm": "ASP"}, -1, "Deprotonated (-1) — aspartate"),
        ("ASP_H", {"amber": "ASH", "charmm": "ASPP"}, 0, "Neutral (0) — aspartic acid"),
    ],
    "GLU": [
        ("GLU", {"amber": "GLU", "charmm": "GLU"}, -1, "Deprotonated (-1) — glutamate"),
        ("GLU_H", {"amber": "GLH", "charmm": "GLUP"}, 0, "Neutral (0) — glutamic acid"),
    ],
    "LYS": [
        ("LYS_P", {"amber": "LYS", "charmm": "LYS"}, 1, "Protonated (+1) — lysine"),
        ("LYS_N", {"amber": "LYN", "charmm": "LSN"}, 0, "Neutral (0) — deprotonated"),
    ],
    "CYS": [
        ("CYS", {"amber": "CYS", "charmm": "CYS"}, 0, "Protonated (0) — free cysteine"),
        ("CYS_M", {"amber": "CYM", "charmm": "CYM"}, -1, "Deprotonated (-1) — thiolate"),
    ],
    "TYR": [
        ("TYR", {"amber": "TYR", "charmm": "TYR"}, 0, "Protonated (0) — tyrosine"),
        # TYM is provided by neither shipped force field, so this state is
        # filtered out below rather than offered and then rejected at grompp.
        ("TYR_M", {"amber": "TYM", "charmm": "TYM"}, -1, "Deprotonated (-1) — tyrosinate"),
    ],
}

# Used when no force field is known. Amber is the project default.
_DEFAULT_FAMILY = "amber"


def _family_for(force_field: str | None) -> str:
    if not force_field:
        return _DEFAULT_FAMILY
    try:
        from gmxbuilder.modules.forcefield.catalog import force_field_family

        family = str(force_field_family(force_field)).strip().lower()
    except Exception:  # noqa: BLE001 - fall back to the name itself
        family = str(force_field).strip().lower()
    if family.startswith("opls"):
        return "opls"
    return "charmm" if family.startswith("charmm") else _DEFAULT_FAMILY


def _force_field_provides(force_field: str | None) -> set[str] | None:
    """Residue names this force field actually defines, or None if unknown."""
    if not force_field:
        return None
    try:
        from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

        rtp = load_force_field_rtp(force_field)
    except Exception:  # noqa: BLE001 - absent parameters must not block the UI
        return None
    residues = getattr(rtp, "_residues", None)
    if not isinstance(residues, dict) or not residues:
        return None
    return {str(name).strip().upper() for name in residues}


def resolve_titratable_states(force_field: str | None = None) -> dict[str, list[ProtonationState]]:
    """Return the protonation states *this* force field can actually build.

    Names are resolved for the force field's family, then filtered against its
    residue templates. Offering a state the force field has no template for is
    what produced "residue ASH has no charmm36m template" several steps later,
    with a message that blamed the input structure.
    """
    family = _family_for(force_field)
    available = _force_field_provides(force_field)
    resolved: dict[str, list[ProtonationState]] = {}
    for parent, definitions in _STATE_DEFINITIONS.items():
        states = []
        for state_id, names, charge, description in definitions:
            name = names.get(family, names[_DEFAULT_FAMILY])
            if available is not None and name.upper() not in available:
                continue
            states.append(ProtonationState(state_id, name, charge, description))
        if states:
            resolved[parent] = states
    return resolved


_TITRATABLE_STATES: dict[str, list[ProtonationState]] = resolve_titratable_states()


# ---------------------------------------------------------------------------
# Protonation assignment
# ---------------------------------------------------------------------------


def get_titratable_residues(force_field: str | None = None) -> dict[str, list[ProtonationState]]:
    """Return the titratable residues, named for *force_field*.

    Passing the force field is what makes the names correct; omitting it falls
    back to Amber naming and is only appropriate where no force field has been
    chosen yet.
    """
    return resolve_titratable_states(force_field)


def assign_protonation(
    residue_name: str,
    pH: float,
    his_tautomer: str = "HSE",
    force_field: str | None = None,
    pka_override: float | None = None,
) -> dict:
    """Determine the protonation state of a single residue at a given pH.

    Parameters
    ----------
    residue_name : str
        The original residue name (3-letter code, e.g. "HIS").
    pH : float
        Target pH.
    his_tautomer : str
        Preferred HIS tautomer when neutral: "HSD" (Nδ) or "HSE" (Nε). These
        are selector tokens, not output names -- under Amber the same choice
        yields HID or HIE.
    force_field : str or None
        Decides the residue names, which differ between Amber and CHARMM.
    pka_override : float or None
        Use this pKa instead of the model value -- an environment-sensitive
        prediction from PROPKA. The selection itself is deliberately shared
        with the model-pKa path: a second copy of it drifted, and hard-coded
        Amber names in that copy are what broke CHARMM builds.

    Returns
    -------
    dict with keys:
        original, assigned_name, charge, state_label, pKa, is_titratable
    """
    rn = residue_name.strip().upper()
    titratable = resolve_titratable_states(force_field)
    if rn not in titratable:
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
    if pka_override is not None:
        pka_val = float(pka_override)
    states = titratable[rn]

    # Assign based on pH vs pKa. The desired state is chosen first and
    # resolved second, because a force field need not provide every state --
    # neither shipped one has a deprotonated tyrosine.
    if rn in ("ASP", "GLU"):
        # Acidic: neutral below pKa, deprotonated (-1) above.
        wanted_charge = 0 if pH < pka_val else -1
        wanted_id = None
    elif rn == "LYS":
        # +1 below pKa, neutral above.
        wanted_charge = 1 if pH < pka_val else 0
        wanted_id = None
    elif rn == "TYR":
        # Neutral below pKa, tyrosinate above.
        wanted_charge = 0 if pH < pka_val else -1
        wanted_id = None
    elif rn == "CYS":
        # Neutral below pKa, thiolate above.
        wanted_charge = 0 if pH < pka_val else -1
        wanted_id = None
    elif rn == "HIS":
        if pH < pka_val:
            wanted_charge, wanted_id = 1, None
        else:
            wanted_charge = 0
            wanted_id = (
                "HIS_ND" if str(his_tautomer).upper() in {"HSD", "HID", "HISD"} else "HIS_NE"
            )
    else:
        wanted_charge, wanted_id = states[0].charge, states[0].state_id

    state = None
    if wanted_id is not None:
        state = next((s for s in states if s.state_id == wanted_id), None)
    if state is None:
        state = next((s for s in states if s.charge == wanted_charge), None)
    unavailable = state is None
    if unavailable:
        # The chemistry calls for a state this force field cannot build. Say
        # so and use what it does have, rather than emitting a residue name
        # that fails at grompp with a message about the input structure.
        state = states[0]

    return {
        "original": rn,
        "assigned_name": state.residue_name,
        "charge": state.charge,
        "state_label": state.description,
        "pKa": round(pka_val, 1),
        "is_titratable": True,
        "force_field_lacks_state": unavailable,
        "ambiguous_at_pka": abs(float(pH) - pka_val) < 1e-9,
        "alternatives": [
            {"name": s.residue_name, "charge": s.charge, "label": s.description} for s in states
        ],
    }


def assign_all_protonations(
    residue_list: list[str],
    pH: float = 7.0,
    his_tautomer: str = "HSE",
    force_field: str | None = None,
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
        result = assign_protonation(rn, pH=pH, his_tautomer=his_tautomer, force_field=force_field)
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
        # PROPKA is a project dependency, so it lives in the environment running
        # this code -- but that environment's bin directory is not necessarily
        # on PATH. The systemd unit inherits a plain system PATH, so a bare
        # "propka3" was never found there and every structure silently fell
        # back to model pKa values while reporting that PROPKA could not
        # produce them. Look beside this interpreter first.
        interpreter_bin = Path(sys.executable).parent
        candidates: list[str] = []
        for name in ("propka3", "propka"):
            beside = interpreter_bin / name
            if beside.exists():
                candidates.append(str(beside))
            candidates.append(name)
        for cmd in candidates:
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
            if "SUMMARY" in stripped.upper() and "PREDICTION" in stripped.upper():
                in_summary = True
                continue

            if not in_summary:
                continue

            # Skip separator lines and non-data
            if stripped.startswith("-") or stripped.startswith("="):
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

                results.append(
                    {
                        "residue_name": resname,
                        "chain": chain,
                        "resid": resid,
                        "model_pKa": round(model_pka, 2),
                        "predicted_pKa": round(predicted_pka, 2),
                        "shift": round(predicted_pka - model_pka, 2),
                    }
                )
            except (ValueError, IndexError):
                continue

    return results


def assign_protonation_with_propka(
    structure_residues: list[dict],
    pka_predictions: list[dict],
    pH: float = 7.0,
    his_tautomer: str = "HSE",
    force_field: str | None = None,
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
        key = (p["residue_name"].upper(), p.get("chain", "").strip(), p.get("resid", 0))
        pka_lookup[key] = p

    results = []
    for r in structure_residues:
        rn = r["resname"].strip().upper()
        key = (rn, r.get("chain", "").strip(), r.get("resid", 0))
        pka_data = pka_lookup.get(key)

        # Get the baseline assignment
        base = assign_protonation(rn, pH=pH, his_tautomer=his_tautomer, force_field=force_field)

        if pka_data and base["is_titratable"]:
            # Use PROPKA-predicted pKa instead of model pKa
            predicted_pka = pka_data["predicted_pKa"]
            pka_shift = pka_data["shift"]

            # Re-run the one selection with the predicted pKa. This used to be
            # a second copy of the same logic carrying hard-coded Amber names,
            # which is how a CHARMM build ended up asking for ASH.
            base = assign_protonation(
                rn,
                pH=pH,
                his_tautomer=his_tautomer,
                force_field=force_field,
                pka_override=predicted_pka,
            )
            base["state_label"] = f"{base['state_label']} (pKa_pred={predicted_pka:.1f})"
            base["predicted_pKa"] = round(predicted_pka, 2)
            base["pKa_shift"] = round(pka_shift, 2)

        base["index"] = r["index"]
        results.append(base)

    return results
