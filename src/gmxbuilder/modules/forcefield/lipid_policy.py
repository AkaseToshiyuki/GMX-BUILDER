"""Force-field policy for exact RTP, Lipid21 and GAFF2 lipid parameters."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# GMXBUILDER uses user-facing lipid names that do not always match the
# residue names used by the CHARMM distribution.  Keep this mapping in one
# place: geometry generation, compatibility checks and topology writing must
# resolve to the same chemical template.
#
# Every entry below is an identity mapping verified against the residue
# description, elemental composition and formal charge in the bundled
# CHARMM36 lipid RTP.  Do not add "nearest" analogues here.
_CHARMM_RTP_IDENTITIES = {
    "BSM": "LSM",  # d18:1/24:0 sphingomyelin (CHARMM BSM is 22:0)
    "CER16": "CER160",  # ceramide d18:1/16:0
    "CER18": "CER180",  # ceramide d18:1/18:0
    "CER24": "CER240",  # ceramide d18:1/24:0
    "CHOL": "CHL1",
    "DPEPE": "DYPE",  # dipalmitoleoyl PE
    "POP2": "POPI24",  # Declared PI(4,5)P2 has its remaining proton on P4, net -4.
    "POP3": "POPI35",  # POPI(3,4,5)P3, protonated on P5, net -6
    "PUPC": "PDOPC",  # 1-palmitoyl-2-docosahexaenoyl PC
    "SAPI": "SAPI24",  # Same declared PI(4,5)P2/P4 protonation, net -4.
    "TMCL": "TMCL2",  # tetramyristoyl cardiolipin, net -2
    "TOCL": "TOCL2",  # tetraoleoyl cardiolipin, net -2
}

# Modular CHARMM lipid residues use the same glycerol/ester atom naming and
# transfer complete acyl subgraphs between official templates.  ``None``
# retains the corresponding tail from the base residue.  This is exact
# parameter reuse within CHARMM36, not a cross-force-field approximation.
_CHARMM_RTP_TAIL_COMBINATIONS = {
    "DLIPA": ("DOPA", "DLIPC", "DLIPC"),
    "DLIPG": ("DOPG", "DLIPC", "DLIPC"),
    "DLIPS": ("DOPS", "DLIPC", "DLIPC"),
    "LPC16": ("LPC14", "DPPC", None),
    "LPC18": ("LPC14", "DSPC", None),
    "LPE16": ("LPC14", "DPPC", None),
    "LYSPG": ("DPPGK", None, "POPG"),
    "PAPC": ("POPC", None, "SAPC"),
    "PAPE": ("POPE", None, "SAPE"),
    "PAPG": ("POPG", None, "SAPG"),
    "PAPI": ("POPI24", None, "SAPI24"),
    "PIPI": ("POPI", None, "SAPI"),
    "PMPC": ("DPPC", None, "DMPC"),
    "SMPC": ("DSPC", None, "DMPC"),
    "SOP2": ("POPI2C", "SOPC", None),  # PI(3,4)P2, protonated on P3.
    "SOP3": ("POPI35", "SOPC", None),
    "SOPI": ("POPI", "SOPC", None),
}

# CHARMM36m changes protein backbone terms; its current lipid stream remains
# compatible with the classic CHARMM36 protein force field.  The bundled
# classic GROMACS port predates these lipid residues, so resolve only this
# audited set from the current lipid namespace and still require real grompp
# parameter resolution.  This is not a cross-family force-field substitution.
_CHARMM_CURRENT_LIPIDS_FOR_CLASSIC = {
    "DLIPA",
    "DLIPC",
    "DLIPE",
    "DLIPG",
    "DLIPS",
    "ERG",
    "LPC16",
    "LPC18",
    "LPE16",
    "LYSPG",
    "PUPC",
    "SITO",
    "STIG",
}
# When a custom lipid requires GAFF2, keep the whole topology in the current
# recommended Amber family instead of silently falling back to ff99SB-ILDN.
GAFF_PROTEIN_FORCE_FIELD = "amber14sb"

# These built-in identities were exercised through full GAFF2/AM1-BCC and
# explicit-solvent NPT library builds. They must not be advertised merely
# because the GAFF executable is installed: the corrected large lipids still
# await V4 physical validation, while the generic-GAFF sterols/other entries
# failed strict orientation, overlap, or hydrophobic-core gates after repeated
# production runs.  CHARMM alternatives remain available where an exact RTP
# exists.
_GAFF_UNAVAILABLE: dict[str, str] = {
    "BSM": (
        "three independent 1000 ps explicit-solvent NPT builds retained only "
        "about 63% of the experimental C24:0 sphingomyelin DHH"
    ),
    "GM1": "the corrected ganglioside GAFF2 model awaits V4 physical validation",
    "PAPI": "the corrected phosphoinositide GAFF2 model awaits V4 physical validation",
    "POP3": "the corrected phosphoinositide GAFF2 model awaits V4 physical validation",
    "SOP3": "the corrected phosphoinositide GAFF2 model awaits V4 physical validation",
    "20AHC": "repeated GAFF2 NPT builds fail the sterol orientation gate",
    "25OHC": "repeated GAFF2 NPT builds fail the sterol orientation gate",
    "27OHC": "repeated GAFF2 NPT builds fail the sterol orientation gate",
    "CAMP": "repeated GAFF2 NPT builds fail the sterol orientation gate",
    "CHOL": (
        "a corrected 1000 ps explicit-solvent NPT rebuild retained only "
        "93.8% correctly oriented membrane molecules"
    ),
    "SITO": "repeated GAFF2 NPT builds fail the sterol orientation gate",
    "MGDG": "the corrected GAFF2 structure retains unresolved minimization overlaps",
    "SOP2": "the corrected GAFF2 bilayer does not pass the hydrophobic-core seal gate",
}


_CHARMM_LIBRARY_UNAVAILABLE: dict[tuple[str, str], str] = {
    (
        "charmm36m",
        "SAPI",
    ): "the explicit-solvent NPT conformer set fails DHH and hydrophobic-core gates",
}


#: Cache for the superseded-entry check, keyed by (lipid, force field, lipid ff)
#: *and by the identity of the metadata file the answer was read from*.
#:
#: The check reads a metadata file and re-canonicalises two structures, which is
#: milliseconds -- but it is asked once per lipid per compatibility report, so
#: it is worth caching. Keying on the lipid alone was wrong twice over. The
#: docstring below promises the report "clears itself the moment a rebuilt entry
#: is installed", and with no entry in the key it never did: a long-running web
#: service would go on refusing a lipid after its replacement had landed, until
#: someone restarted it -- which is precisely what happens when V4 lands. And a
#: task-scoped custom library resolves the same lipid name to a different
#: directory, so without the path in the key the two would share an answer.
#:
#: The identity is the resolved path with its size and modification time, so
#: publishing a new entry invalidates the answer by construction.
_SUPERSEDED_CACHE: dict[tuple[str, str, str, str, int, int], str] = {}


#: Set while the library builder is producing a replacement entry.
#:
#: The superseded report exists to stop a stale entry being served. The process
#: that *rebuilds* that entry must not be stopped by it, or the report blocks
#: its own remedy: the V4 queue asks for an Amber/GAFF2 backend for exactly the
#: lipids whose GAFF2 entries are superseded, and would be told they are
#: unavailable for that reason. Nothing stale is served inside this scope --
#: the builder is writing a new entry from the corrected structure.
_REBUILDING: ContextVar[bool] = ContextVar("gmxbuilder_rebuilding_lipid_entry", default=False)
_RETRY_FAILED: ContextVar[bool] = ContextVar("gmxbuilder_retry_failed_lipid_entry", default=False)


@contextmanager
def rebuilding_library_entry(*, retry_failed_validation: bool = False) -> Iterator[None]:
    """Permit replacement preparation without advertising a failed entry as usable.

    V4 can explicitly retry historical preparation/MD failures with corrected
    identities and reviewed conditions. This changes admission to an experiment,
    never its identity, parameter, geometry or production acceptance gates.
    Ordinary builders retain the historical quarantine.
    """
    token = _REBUILDING.set(True)
    retry_token = _RETRY_FAILED.set(_RETRY_FAILED.get() or retry_failed_validation)
    try:
        yield
    finally:
        _RETRY_FAILED.reset(retry_token)
        _REBUILDING.reset(token)


def _explicit_atoms(name: str) -> int:
    """Atom count of a registry lipid with explicit hydrogens, cached."""
    from gmxbuilder.modules.membrane.equilibrated_library import _explicit_atom_count
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    try:
        return _explicit_atom_count(LipidRegistry.get(name).smiles)
    except (KeyError, ValueError, TypeError):
        return -1


def library_entry_superseded(lipid_name: str, force_field: str, lipid_ff: str | None = None) -> str:
    """Why a pre-equilibrated entry exists but may not be served, or ''.

    An entry records the structure it was built from. When that structure is
    later corrected, the entry describes a molecule the project no longer
    believes in, and the library refuses it -- correctly, but silently, so the
    lipid would either stop working with no stated reason or, worse, quietly
    fall back to a generated conformer in place of an equilibrated one.

    Reported dynamically rather than listed, so it clears itself the moment a
    rebuilt entry is installed. There is no list to remember to unwind.

    Deliberately cheap. It compares the one field that a structure correction
    changes rather than re-running the library's full validation, and it uses
    the shared library rather than constructing one: `coverage()` asks this for
    every lipid against every force field, and building a library per question
    re-enters `ensure_prebuilt_assets()` each time.
    """
    from gmxbuilder.modules.membrane.equilibrated_library import (
        get_equilibrated_lipid_library,
    )
    from gmxbuilder.modules.membrane.lipids import LipidRegistry, canonical_lipid_identity

    if _REBUILDING.get():
        return ""
    name = str(lipid_name).strip().upper()
    prefix = (name, str(force_field).strip().lower(), str(lipid_ff or "").strip().lower())
    key = None

    reason = ""
    try:
        current = _registry_identity(name, LipidRegistry, canonical_lipid_identity)
        if current:
            library = get_equilibrated_lipid_library()
            for directory in library._candidate_dirs(name, force_field, lipid_ff):
                metadata_path = directory / "metadata.json"
                if not metadata_path.is_file():
                    continue
                try:
                    stat = metadata_path.stat()
                except OSError:
                    break
                key = (*prefix, str(metadata_path), stat.st_size, stat.st_mtime_ns)
                cached = _SUPERSEDED_CACHE.get(key)
                if cached is not None:
                    return cached
                try:
                    metadata = json.loads(metadata_path.read_text())
                except (OSError, ValueError):
                    break
                if str(metadata.get("status")) != "ready":
                    break
                stored = str(metadata.get("canonical_smiles", ""))
                # Same test the library itself applies: a stale recorded SMILES
                # only matters if the entry really holds a different molecule.
                # CHARMM and Lipid21 entries were built from their own force
                # field's definition and merely copied the note in, so their
                # atoms still match; a GAFF2 entry was built from the SMILES and
                # its atoms do not.
                superseded_here = bool(stored) and (
                    canonical_lipid_identity(stored)["canonical_smiles"] != current
                    and len(metadata.get("atom_names", [])) != _explicit_atoms(name)
                )
                if superseded_here:
                    reason = (
                        "its pre-equilibrated entry was built from a lipid structure that has "
                        "since been corrected against the force field's own residue, so the "
                        "entry no longer describes this molecule; a rebuilt entry is in "
                        "preparation and the lipid becomes available again as soon as it lands"
                    )
                break
    except (RuntimeError, OSError, ValueError, KeyError):
        return ""
    if key is not None:
        _SUPERSEDED_CACHE[key] = reason
    return reason


#: Canonical identity per registry lipid, computed once. Canonicalising a
#: lipid SMILES costs a couple of milliseconds and the answer never changes
#: within a process.
_REGISTRY_IDENTITY: dict[str, str] = {}


def _registry_identity(name: str, registry, canonicalise) -> str:
    cached = _REGISTRY_IDENTITY.get(name)
    if cached is not None:
        return cached
    try:
        smiles = registry.get(name).smiles
        identity = canonicalise(smiles)["canonical_smiles"] if smiles else ""
    except (KeyError, ValueError, TypeError):
        identity = ""
    _REGISTRY_IDENTITY[name] = identity
    return identity


def charmm_lipid_capability(lipid_name: str, force_field: str) -> tuple[bool, str]:
    """Return release-specific CHARMM lipid support including NPT quality."""
    name = str(lipid_name).strip().upper()
    selected = str(force_field).strip().lower()
    if not lipid_has_rtp(name, selected):
        return False, f"{name} has no exact {selected} lipid topology"
    reason = None if _RETRY_FAILED.get() else _CHARMM_LIBRARY_UNAVAILABLE.get((selected, name))
    if reason:
        from gmxbuilder.modules.membrane.equilibrated_library import get_equilibrated_lipid_library
        from gmxbuilder.modules.membrane.v4_availability import accepted_entry

        if accepted_entry(get_equilibrated_lipid_library(), name, selected, selected):
            reason = None
    if not reason:
        superseded = library_entry_superseded(name, selected, selected)
        if superseded:
            return False, f"{name} is temporarily unavailable: {superseded}"
        return True, ""
    alternatives = [
        label
        for candidate, label in (("charmm36m", "CHARMM36m"), ("charmm36", "CHARMM36"))
        if candidate != selected
        and lipid_has_rtp(name, candidate)
        and (candidate, name) not in _CHARMM_LIBRARY_UNAVAILABLE
    ]
    suffix = (
        f"; use {' or '.join(alternatives)} for this lipid"
        if alternatives
        else "; no validated bundled alternative is currently available"
    )
    return False, f"{selected} unavailable: {reason}{suffix}"


def gaff_lipid_capability(lipid_name: str) -> tuple[bool, str]:
    """Return whether a built-in lipid has validated GAFF2 production support."""
    name = str(lipid_name).strip().upper()
    reason = _GAFF_UNAVAILABLE.get(name, "") if not _RETRY_FAILED.get() else ""
    if _RETRY_FAILED.get() and name in {"GM1", "PAPI", "POP3", "SOP3"}:
        from gmxbuilder.modules.forcefield.gaff_backend import (
            cached_gaff_template,
            gaff_charge_method,
        )
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        lipid = LipidRegistry.get(name)
        if (
            gaff_charge_method() != "bcc"
            or cached_gaff_template(name, lipid.smiles, lipid.charge, charge_method="bcc") is None
        ):
            reason = (
                "a current identity-valid AM1-BCC parameter cache is required before V4 sampling"
            )
    if reason and not _RETRY_FAILED.get():
        from gmxbuilder.modules.membrane.equilibrated_library import get_equilibrated_lipid_library
        from gmxbuilder.modules.membrane.v4_availability import accepted_entry

        if accepted_entry(get_equilibrated_lipid_library(), name, "amber14sb", "gaff2"):
            reason = ""
    if not reason:
        superseded = library_entry_superseded(name, "amber14sb", "gaff2")
        if superseded:
            return False, f"{name} is temporarily unavailable: {superseded}"
        return True, ""
    alternatives = [
        label
        for force_field, label in (("charmm36m", "CHARMM36m"), ("charmm36", "CHARMM36"))
        if lipid_has_rtp(name, force_field)
    ]
    suffix = (
        f"; use {' or '.join(alternatives)} for this lipid"
        if alternatives
        else "; no validated bundled alternative is currently available"
    )
    return False, f"Amber/GAFF2 unavailable: {reason}{suffix}"


def lipid_backend_for(lipid_name: str, selected_backend: str | None) -> str:
    """Resolve a canonical mixed Amber selection for one molecular species."""
    selected = str(selected_backend or "").strip().lower()
    if selected != "amber-mixed":
        return selected
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability

    if lipid21_capability(lipid_name)[0]:
        return "lipid21"
    # Routing is parameter identity, not a production-availability decision.
    # Callers enforce capability; provenance must be evaluable outside rebuild scope.
    return "gaff2"


def amber_mixed_validation_reason(lipid_names) -> str:
    """User builds require current mixed-host initialization evidence.

    The offline builder may generate that evidence. Parameter completeness or a
    successful grompp alone cannot make a new mixed model production-ready.
    """
    if _REBUILDING.get():
        return ""
    from gmxbuilder.modules.membrane.equilibrated_library import get_equilibrated_lipid_library

    missing = []
    library = get_equilibrated_lipid_library()
    for name in sorted(set(lipid_names)):
        if lipid_backend_for(name, "amber-mixed") != "gaff2":
            continue
        entry = library.inspect(name, "amber14sb", "gaff2")
        if (
            entry is None
            or entry.metadata.get("lipid_ff") != "amber-mixed"
            or (
                entry.metadata.get("equilibration_host")
                != {
                    "lipid_name": "POPC",
                    "ratio_percent": 90,
                    "target_ratio_percent": 10,
                }
            )
        ):
            missing.append(name)
    if missing:
        return (
            "Mixed Lipid21/GAFF2 initialization is unavailable pending physical validation "
            f"with the corrected parameters: {', '.join(missing)}. "
            "The administrator's V4 workflow must qualify these entries first."
        )
    return ""


def amber_lipid_backend_candidates(
    lipid_names: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Canonical per-species assignment first; full GAFF2 only by explicit choice."""
    from gmxbuilder.modules.forcefield.gaff_backend import gaff_available
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability

    names = sorted({str(value).strip().upper() for value in lipid_names if str(value).strip()})
    if not names:
        return ("none",)
    native = {name for name in names if lipid21_capability(name)[0]}
    missing = set(names) - native
    candidates = []
    if not missing:
        candidates.append("lipid21")
    elif native and gaff_available() and all(gaff_lipid_capability(n)[0] for n in missing):
        candidates.append("amber-mixed")
    if gaff_available() and all(gaff_lipid_capability(name)[0] for name in names):
        candidates.append("gaff2")
    return tuple(candidates)


def amber_lipid_backend(lipid_names: list[str] | tuple[str, ...]) -> tuple[str | None, str]:
    """Prefer Lipid21 for each supported species and GAFF2 for missing species.

    Library availability does not change molecular parameters. In particular,
    historical failures under superseded coefficients cannot authorize a silent
    switch of the entire membrane to a different force field.
    """
    candidates = amber_lipid_backend_candidates(lipid_names)
    if candidates:
        backend = candidates[0]
        reasons = {
            "none": "no membrane lipids selected",
            "lipid21": "all selected lipids use exact Amber Lipid21 v1.0 parameters",
            "amber-mixed": (
                "Lipid21 is retained for supported species; GAFF2 supplies only missing "
                "species, with independent molecular 1-4 parameters"
            ),
            "gaff2": "selected lipids lack Lipid21 coverage and use GAFF2 parameters",
        }
        return backend, reasons[backend]
    from gmxbuilder.modules.forcefield.lipid21_backend import lipid21_capability

    reasons = []
    for name in sorted(set(lipid_names)):
        if not lipid21_capability(name)[0]:
            supported, reason = gaff_lipid_capability(name)
            if not supported:
                reasons.append(f"{name}: {reason}")
    return None, "No validated Amber parameter assignment covers the membrane. " + "; ".join(
        reasons
    )


def membrane_lipid_names(config: dict) -> tuple[str, ...]:
    """Extract selected lipid names from a MembraneBuilder config."""
    names: set[str] = set()
    composition = config.get("lipid_composition")
    if isinstance(composition, dict):
        for leaflet in (composition.get("upper") or [], composition.get("lower") or []):
            for entry in leaflet:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(str(entry["name"]).strip().upper())
    elif config.get("lipid_type"):
        names.add(str(config["lipid_type"]).strip().upper())
    return tuple(sorted(name for name in names if name))


def lipid_rtp_name(lipid_name: str, force_field: str) -> str:
    """Return the chemically equivalent RTP residue name for a lipid."""
    name = lipid_name.strip().upper()
    if force_field.strip().lower().startswith("charmm"):
        return _CHARMM_RTP_IDENTITIES.get(name, name)
    return name


def _tail_subtree(template: dict, root: str, bridge: str) -> set[str]:
    adjacency: dict[str, set[str]] = {atom[0]: set() for atom in template["atoms"]}
    for left, right in template["bonds"]:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    if root not in adjacency or bridge not in adjacency[root]:
        raise ValueError(f"CHARMM lipid template lacks bridge {bridge}-{root}")
    result: set[str] = set()
    stack = [root]
    while stack:
        atom = stack.pop()
        if atom == bridge or atom in result:
            continue
        result.add(atom)
        stack.extend(adjacency[atom] - result - {bridge})
    return result


def _compose_charmm_lipid_template(
    parser,
    base_name: str,
    tail1_name: str | None,
    tail2_name: str | None,
) -> dict | None:
    base = parser.get_residue(base_name)
    if base is None:
        return None
    result = copy.deepcopy(base)
    # CHARMM's D-glycero naming uses the C31 branch for the user-facing sn-1
    # acyl chain and C21 for sn-2 (e.g. POPC is 3-palmitoyl/2-oleoyl).
    for root, bridge, donor_name in (
        ("C31", "O31", tail1_name),
        ("C21", "O21", tail2_name),
    ):
        if donor_name is None or donor_name == base_name:
            continue
        donor = parser.get_residue(donor_name)
        if donor is None:
            return None
        result = _replace_charmm_subtree(result, donor, root, bridge)

    atom_names = [atom[0] for atom in result["atoms"]]
    if len(atom_names) != len(set(atom_names)):
        raise ValueError("Composed CHARMM lipid template has duplicate atom names")
    atom_set = set(atom_names)
    if any(left not in atom_set or right not in atom_set for left, right in result["bonds"]):
        raise ValueError("Composed CHARMM lipid template has dangling bonds")
    return result


def _replace_charmm_subtree(
    base: dict,
    donor: dict,
    root: str,
    bridge: str,
) -> dict:
    """Replace one covalent branch while retaining the shared bridge atom."""
    result = copy.deepcopy(base)
    removed = _tail_subtree(result, root, bridge)
    inserted = _tail_subtree(donor, root, bridge)
    result["atoms"] = [atom for atom in result["atoms"] if atom[0] not in removed] + [
        copy.deepcopy(atom) for atom in donor["atoms"] if atom[0] in inserted
    ]
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    donor_scope = inserted | {bridge}
    for section, size in term_sizes.items():
        result[section] = [
            term for term in result[section] if not any(atom in removed for atom in term[:size])
        ] + [
            copy.deepcopy(term)
            for term in donor[section]
            if all(atom in donor_scope for atom in term[:size])
            and any(atom in inserted for atom in term[:size])
        ]
    return result


def _make_decanoyl_sphingomyelin(parser) -> dict | None:
    """Truncate the official PSM N-acyl chain to saturated 10:0 DSM."""
    base = parser.get_residue("PSM")
    if base is None:
        return None
    result = copy.deepcopy(base)
    removed = _tail_subtree(result, "C11F", "C10F")
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    result["atoms"] = [atom for atom in result["atoms"] if atom[0] not in removed]
    for section, size in term_sizes.items():
        result[section] = [
            term for term in result[section] if not any(atom in removed for atom in term[:size])
        ]

    terminal_c = next(atom for atom in base["atoms"] if atom[0] == "C16F")
    terminal_h = next(atom for atom in base["atoms"] if atom[0] == "H16F")
    extra_h = next(atom for atom in base["atoms"] if atom[0] == "H16H")
    converted = []
    terminal_group = 0
    for atom in result["atoms"]:
        if atom[0] == "C10F":
            terminal_group = atom[3]
            converted.append(("C10F", terminal_c[1], terminal_c[2], terminal_group))
        elif atom[0] in {"H10F", "H10G"}:
            converted.append((atom[0], terminal_h[1], terminal_h[2], atom[3]))
        else:
            converted.append(atom)
    converted.append(("H10H", extra_h[1], extra_h[2], terminal_group))
    result["atoms"] = converted
    result["bonds"].append(("C10F", "H10H"))
    return result


def _make_charmm_diacylglycerol(parser, phosphatidic_acid: str) -> dict | None:
    """Convert an exact CHARMM36 PA tail template to neutral 1,2-DAG.

    The glycerol charges and atom types are the published Wu et al. CHARMM36
    DAG parameters (DODG/DPDG), not charges inherited from phosphatidic acid.
    Both acyl branches remain unchanged from the matching PA residue.
    """
    base = parser.get_residue(phosphatidic_acid)
    if base is None:
        return None
    result = copy.deepcopy(base)
    removed = {"P", "O12", "H12", "O13", "O14"}
    result["atoms"] = [atom for atom in result["atoms"] if atom[0] not in removed]
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    for section, size in term_sizes.items():
        result[section] = [
            term for term in result[section] if not any(atom in removed for atom in term[:size])
        ]

    converted = []
    head_group = 0
    for atom in result["atoms"]:
        name, atom_type, charge, group = atom
        if name == "C1":
            converted.append((name, "CTL2", 0.05, group))
        elif name == "O11":
            head_group = group
            converted.append((name, "OHL", -0.65, group))
        else:
            converted.append(atom)
    if not head_group or not any(atom[0] == "C1" for atom in converted):
        raise ValueError(
            f"CHARMM phosphatidic-acid template {phosphatidic_acid} lacks DAG head atoms"
        )
    converted.append(("HO1", "HOL", 0.42, head_group))
    result["atoms"] = converted
    result["bonds"].append(("O11", "HO1"))
    return result


def _make_charmm_plasmalogen(parser, headgroup: str) -> dict | None:
    """Build P-16:0/18:1 plasmalogen from the West et al. CHARMM36 model."""
    base = parser.get_residue("POPE")
    if base is None:
        return None
    result = copy.deepcopy(base)
    removed = {"O32", "H2Y"}
    result["atoms"] = [atom for atom in result["atoms"] if atom[0] not in removed]
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    for section, size in term_sizes.items():
        result[section] = [
            term for term in result[section] if not any(atom in removed for atom in term[:size])
        ]

    # West 2020 original PLA18, not the later reparameterized ether stream:
    # https://terpconnect.umd.edu/~jbklauda/download/toppar_all36_lipid_ether_AL.str
    # SHA256 67d3f8090ff58c29b4c6c58bd265b10fddf68ac4e8ed3216d58be6995750cb0f.
    # SOLE/PLA18 is P-18:0/18:1.  Its C31-C318 vinyl-ether branch uses the
    # same local charges as the requested P-16:0 branch; POPE already supplies
    # the exact C31-C316 carbon count and the unchanged 18:1 ester branch.
    west_atoms = {
        "C3": ("CTL2", 0.08),
        "HX": ("HAL2", 0.08),
        "HY": ("HAL2", 0.08),
        "O31": ("OG301", -0.36),
        "C31": ("CEL1", 0.00),
        "C32": ("CEL1", -0.20),
        "H2X": ("HEL1", 0.08),
        "C23": ("CTL2", 0.00),
        "H3R": ("HAL2", 0.08),
        "H3S": ("HAL2", 0.08),
        "C33": ("CTL2", -0.18),
        "H3X": ("HAL2", 0.09),
        "H3Y": ("HAL2", 0.09),
    }
    converted = []
    vinyl_group: int | None = None
    for atom in result["atoms"]:
        name, atom_type, charge, group = atom
        if name in west_atoms:
            atom_type, charge = west_atoms[name]
        if name == "C31":
            vinyl_group = group
        converted.append((name, atom_type, charge, group))
    if vinyl_group is None or not all(
        any(atom[0] == name for atom in converted) for name in west_atoms
    ):
        raise ValueError("CHARMM POPE template lacks plasmalogen branch atoms")
    converted.append(("H1X", "HEL1", 0.08, vinyl_group))
    result["atoms"] = converted
    result["bonds"].append(("C31", "H1X"))
    result["impropers"].append(("C31", "C32", "O31", "H1X"))

    if headgroup == "PC":
        pc = parser.get_residue("POPC")
        if pc is None:
            return None
        result = _replace_charmm_subtree(result, pc, "O12", "P")
    # The supplementary type names also occur in ordinary unsaturated tails.
    # Apply their parameters only at this molecule's actual vinyl-ether motif.
    result["local_bonded_scope"] = ("O31", "C31", "C32")
    result["source_provenance"] = {
        "model": "West2020-PLA18-local-motif",
        "doi": "10.1021/acs.jpcb.9b08850",
        "sha256": "67d3f8090ff58c29b4c6c58bd265b10fddf68ac4e8ed3216d58be6995750cb0f",
        "adaptation": "P-18:0 to P-16:0 terminal chain shortening; PC head from native POPC",
    }
    return result


def _remove_charmm_atoms(template: dict, removed: set[str]) -> dict:
    """Remove atoms and every bonded term that references them."""
    result = copy.deepcopy(template)
    result["atoms"] = [atom for atom in result["atoms"] if atom[0] not in removed]
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    for section, size in term_sizes.items():
        result[section] = [
            term for term in result[section] if not any(atom in removed for atom in term[:size])
        ]
    return result


def _copy_sn2_tail_to_sn1(template: dict) -> dict:
    """Replace the sn-1 branch with an exact renamed copy of the sn-2 branch."""
    result = copy.deepcopy(template)
    removed = _tail_subtree(result, "C31", "O31")
    source = _tail_subtree(template, "C21", "O21")

    def rename(name: str) -> str:
        if name == "O21":
            return "O31"
        if name == "O22":
            return "O32"
        if name.startswith("C2") and name[2:].isdigit():
            return f"C3{name[2:]}"
        match = re.fullmatch(r"H(\d+)([RST])", name)
        if match:
            suffix = {"R": "X", "S": "Y", "T": "Z"}[match.group(2)]
            return f"H{match.group(1)}{suffix}"
        raise ValueError(f"Cannot map CHARMM sn-2 tail atom {name!r} to sn-1")

    mapping = {name: rename(name) for name in source | {"O21"}}
    result = _remove_charmm_atoms(result, removed)
    result["atoms"].extend(
        (mapping[name], atom_type, charge, group)
        for name, atom_type, charge, group in template["atoms"]
        if name in source
    )
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    scope = source | {"O21"}
    for section, size in term_sizes.items():
        result[section].extend(
            tuple(mapping.get(atom, atom) for atom in term[:size]) + tuple(term[size:])
            for term in template[section]
            if all(atom in scope for atom in term[:size])
            and any(atom in source for atom in term[:size])
        )
    return result


def _suffix_charmm_template(template: dict, suffix: str) -> dict:
    """Return a covalent component with unique, element-leading atom names."""
    result = copy.deepcopy(template)
    mapping = {atom[0]: f"{atom[0]}{suffix}" for atom in result["atoms"]}
    result["atoms"] = [
        (mapping[name], atom_type, charge, group)
        for name, atom_type, charge, group in result["atoms"]
    ]
    term_sizes = {
        "bonds": 2,
        "angles": 3,
        "dihedrals": 4,
        "impropers": 4,
    }
    for section, size in term_sizes.items():
        result[section] = [
            tuple(mapping.get(atom, atom) for atom in term[:size]) + tuple(term[size:])
            for term in result[section]
        ]
    return result


def _set_charmm_atoms(template: dict, replacements: dict[str, tuple[str, float]]) -> None:
    template["atoms"] = [
        (name, *replacements.get(name, (atom_type, charge)), group)
        for name, atom_type, charge, group in template["atoms"]
    ]


def _merge_charmm_templates(base: dict, component: dict) -> None:
    base["atoms"].extend(copy.deepcopy(component["atoms"]))
    for section in ("bonds", "angles", "dihedrals", "impropers"):
        base[section].extend(copy.deepcopy(component[section]))


def _make_charmm_galactolipid(parser, digalactosyl: bool) -> dict | None:
    """Apply CHARMM DAGB and optional 16AT patches to 18:3/18:3 DAG."""
    beta_gal = parser.get_residue("BGAL")
    alpha_gal = parser.get_residue("AGAL")
    if beta_gal is None or alpha_gal is None or parser.get_residue("LLPA") is None:
        return None

    # LLPA provides an exact alpha-linolenoyl (18:3) sn-2 branch.  Copy that
    # branch to sn-1, then use the published neutral DAG glycerol charges.
    dag = _make_charmm_diacylglycerol(parser, "LLPA")
    if dag is None:
        return None
    result = _copy_sn2_tail_to_sn1(dag)
    result = _remove_charmm_atoms(result, {"O11", "HO1"})
    _set_charmm_atoms(result, {"C1": ("CTO2", 0.00)})

    # DAGB: beta-D-galactose O1 becomes the glycosidic oxygen to glycerol C1.
    first = _suffix_charmm_template(beta_gal, "G")
    first = _remove_charmm_atoms(first, {"HO1G"})
    _set_charmm_atoms(
        first,
        {
            "C1G": ("CC3162", 0.29),
            "H1G": ("HCA1", 0.09),
            "O1G": ("OC301", -0.36),
        },
    )
    _merge_charmm_templates(result, first)
    result["bonds"].append(("O1G", "C1"))

    if digalactosyl:
        # 16AT: alpha-D-Gal-(1->6)-beta-D-Gal.  The acceptor O6 remains;
        # the donor anomeric O1 and both displaced hydroxyl hydrogens leave.
        result = _remove_charmm_atoms(result, {"HO6G"})
        _set_charmm_atoms(
            result,
            {"C6G": ("CC321", 0.00), "O6G": ("OC301", -0.36)},
        )
        second = _suffix_charmm_template(alpha_gal, "A")
        second = _remove_charmm_atoms(second, {"HO1A", "O1A"})
        _set_charmm_atoms(second, {"C1A": ("CC3162", 0.29)})
        _merge_charmm_templates(result, second)
        result["bonds"].append(("O6G", "C1A"))
    return result


def _make_charmm_campesterol(parser) -> dict | None:
    """Build campesterol from exact CHARMM36 plant-sterol fragments.

    Campesterol is beta-sitosterol with the C24 ethyl substituent shortened to
    the same C24 methyl substituent already parameterized in ergosterol.
    """
    base = parser.get_residue("SITO")
    ergosterol = parser.get_residue("ERG")
    if base is None or ergosterol is None:
        return None
    result = _remove_charmm_atoms(base, {"C29", "H29A", "H29B", "H29C"})
    erg_atoms = {atom[0]: atom for atom in ergosterol["atoms"]}
    if not all(name in erg_atoms for name in ("C28", "H28A", "H28B", "H28C")):
        raise ValueError("CHARMM ergosterol template lacks its C24 methyl branch")
    _set_charmm_atoms(
        result,
        {
            "C28": erg_atoms["C28"][1:3],
            "H28A": erg_atoms["H28A"][1:3],
            "H28B": erg_atoms["H28B"][1:3],
        },
    )
    c28_group = next(atom[3] for atom in result["atoms"] if atom[0] == "C28")
    result["atoms"].append(("H28C", *erg_atoms["H28C"][1:3], c28_group))
    result["bonds"].append(("C28", "H28C"))
    return result


def _make_charmm_gm1(parser) -> dict | None:
    """Build deprotonated GM1(d18:1/18:0) from native CHARMM36 blocks.

    The sequence is Gal(beta1-3)GalNAc(beta1-4)[Neu5Ac(alpha2-3)]
    Gal(beta1-4)Glc(beta1-1)Cer.  Atom-type and charge changes below are the
    native CERB, 14bb, 13bb and SA23AB CHARMM carbohydrate/glycolipid patches;
    no CGenFF atom typing or nearest-lipid substitution is used.
    """
    required = {
        name: parser.get_residue(name) for name in ("CER180", "BGLC", "BGAL", "BGALNA", "ANE5AC")
    }
    if any(template is None for template in required.values()):
        return None

    result = copy.deepcopy(required["CER180"])

    # CERB: beta-D-glucose O1 to the primary alcohol carbon of ceramide.
    result = _remove_charmm_atoms(result, {"O1", "HO1"})
    _set_charmm_atoms(
        result,
        {
            "C1S": ("CTO2", 0.00),
            "H1S": ("HAL2", 0.09),
            "H1T": ("HAL2", 0.09),
        },
    )
    # Use one-character suffixes absent from CER180 (which uses F/G/S/T/U)
    # so every atom name stays unique and within GROMACS' five-character GRO
    # field.  X=Glc, Y=central Gal, Z=GalNAc, Q=terminal Gal, A=Neu5Ac.
    glucose = _suffix_charmm_template(required["BGLC"], "X")
    glucose = _remove_charmm_atoms(glucose, {"HO1X"})
    _set_charmm_atoms(
        glucose,
        {"C1X": ("CC3162", 0.29), "O1X": ("OC301", -0.36)},
    )
    _merge_charmm_templates(result, glucose)
    result["bonds"].append(("O1X", "C1S"))

    def add_beta_sugar(
        template: dict,
        donor_suffix: str,
        acceptor_suffix: str,
        position: int,
    ) -> None:
        """Apply the native 13bb/14bb equatorial-equatorial linkage."""
        nonlocal result
        acceptor_c = f"C{position}{acceptor_suffix}"
        acceptor_o = f"O{position}{acceptor_suffix}"
        result = _remove_charmm_atoms(result, {f"HO{position}{acceptor_suffix}"})
        _set_charmm_atoms(
            result,
            {
                acceptor_c: ("CC3161", 0.09),
                acceptor_o: ("OC301", -0.36),
            },
        )
        donor = _suffix_charmm_template(template, donor_suffix)
        donor = _remove_charmm_atoms(donor, {f"HO1{donor_suffix}", f"O1{donor_suffix}"})
        _set_charmm_atoms(donor, {f"C1{donor_suffix}": ("CC3162", 0.29)})
        _merge_charmm_templates(result, donor)
        result["bonds"].append((acceptor_o, f"C1{donor_suffix}"))

    # Gal(beta1-4)Glc, GalNAc(beta1-4)Gal, Gal(beta1-3)GalNAc.
    add_beta_sugar(required["BGAL"], "Y", "X", 4)
    add_beta_sugar(required["BGALNA"], "Z", "Y", 4)
    add_beta_sugar(required["BGAL"], "Q", "Z", 3)

    # SA23AB: Neu5Ac(alpha2-3) branches from the central galactose.  ANE5AC
    # is the native -1 sialate residue used by CHARMM36 at physiological pH.
    result = _remove_charmm_atoms(result, {"HO3Y"})
    _set_charmm_atoms(
        result,
        {"C3Y": ("CC3161", 0.09), "O3Y": ("OC301", -0.36)},
    )
    sialate = _suffix_charmm_template(required["ANE5AC"], "A")
    sialate = _remove_charmm_atoms(sialate, {"HO2A", "O2A"})
    _set_charmm_atoms(sialate, {"C2A": ("CC3062", 0.28)})
    _merge_charmm_templates(result, sialate)
    result["bonds"].append(("O3Y", "C2A"))

    return result


def lipid_rtp_template(lipid_name: str, force_field: str) -> tuple[str, dict | None]:
    """Resolve or construct one exact CHARMM lipid RTP template."""
    from gmxbuilder.modules.forcefield.rtp_parser import load_force_field_rtp

    name = lipid_name.strip().upper()
    template_name = lipid_rtp_name(name, force_field)
    parser = load_force_field_rtp(force_field)
    template = parser.get_residue(template_name)
    if template is not None:
        return template_name, template
    if not force_field.strip().lower().startswith("charmm"):
        return template_name, None
    combination = _CHARMM_RTP_TAIL_COMBINATIONS.get(name)
    if combination is not None:
        generated = _compose_charmm_lipid_template(parser, *combination)
        if (
            generated is None
            and force_field.strip().lower() == "charmm36"
            and name in _CHARMM_CURRENT_LIPIDS_FOR_CLASSIC
        ):
            modern_name, modern_template = lipid_rtp_template(name, "charmm36m")
            template_name = modern_name
            generated = copy.deepcopy(modern_template)
    elif name == "DSM":
        generated = _make_decanoyl_sphingomyelin(parser)
    elif name == "DOPGD":
        generated = _make_charmm_diacylglycerol(parser, "DOPA")
    elif name == "DPPGD":
        generated = _make_charmm_diacylglycerol(parser, "DPPA")
    elif name == "PPCPL":
        generated = _make_charmm_plasmalogen(parser, "PC")
    elif name == "PPEPL":
        generated = _make_charmm_plasmalogen(parser, "PE")
    elif name in {"MGDG", "DGDG"}:
        source_parser = parser
        if force_field.strip().lower() == "charmm36":
            source_parser = load_force_field_rtp("charmm36m")
        generated = _make_charmm_galactolipid(source_parser, digalactosyl=name == "DGDG")
    elif name == "CAMP":
        source_parser = parser
        if force_field.strip().lower() == "charmm36":
            source_parser = load_force_field_rtp("charmm36m")
        generated = _make_charmm_campesterol(source_parser)
    elif name == "GM1":
        generated = _make_charmm_gm1(parser)
    elif force_field.strip().lower() == "charmm36" and name in _CHARMM_CURRENT_LIPIDS_FOR_CLASSIC:
        modern_name, modern_template = lipid_rtp_template(name, "charmm36m")
        template_name = modern_name
        generated = copy.deepcopy(modern_template)
    else:
        return template_name, None
    if generated is None:
        return template_name, None
    if name == "LPE16":
        pe = parser.get_residue("DPPE")
        if pe is None:
            return template_name, None
        generated = _replace_charmm_subtree(generated, pe, "O12", "P")
    return template_name, generated


def _formula_elements(formula: str) -> Counter[str]:
    return Counter(
        {element: int(count or 1) for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
    )


def lipid_rtp_identity_issues(lipid_name: str, force_field: str) -> tuple[str, ...]:
    """Return chemical-identity mismatches between registry and RTP.

    A shared residue name is not enough: lipid databases commonly reuse short
    names for different tails or protonation states.  Reject a template unless
    its elemental composition and net charge match the selected lipid.
    """
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    name = lipid_name.strip().upper()
    template_name, template = lipid_rtp_template(name, force_field)
    if template is None:
        return (f"RTP residue {template_name} is absent",)
    try:
        lipid = LipidRegistry.get(name)
    except KeyError:
        return (f"Lipid {name} has no registered chemical identity",)

    issues: list[str] = []
    if lipid.formula:
        expected = _formula_elements(lipid.formula)
        observed = Counter()
        for atom_name, _atom_type, _charge, _group in template["atoms"]:
            element = atom_name.strip()[:1].upper()
            if element in {"C", "H", "N", "O", "P", "S"}:
                observed[element] += 1
        if observed != expected:
            issues.append(f"elemental composition {dict(observed)} != {lipid.formula}")
    template_charge = sum(atom[2] for atom in template["atoms"])
    if abs(template_charge - lipid.charge) > 0.05:
        issues.append(f"net charge {template_charge:+.3f} != registry {lipid.charge:+d}")
    if not issues:
        from gmxbuilder.geometry.molecular_identity import reference_mappings
        from gmxbuilder.modules.membrane.lipid_orientation import atom_element

        names = [atom[0] for atom in template["atoms"]]
        indices = {atom: index for index, atom in enumerate(names)}
        try:
            if len(indices) != len(names):
                raise ValueError("duplicate topology atom names")
            bonds = tuple((indices[a], indices[b]) for a, b in template["bonds"])
            reference_mappings(lipid.smiles, tuple(atom_element(n) for n in names), bonds)
        except (ValueError, KeyError) as exc:
            issues.append(f"bond graph differs from registered identity: {exc}")
    return tuple(issues)


def lipid_has_rtp(lipid_name: str, force_field: str) -> bool:
    return not lipid_rtp_identity_issues(lipid_name, force_field)


@dataclass(frozen=True)
class LipidForceFieldResolution:
    requested_force_field: str
    protein_force_field: str
    lipid_force_field: str
    lipid_names: tuple[str, ...]
    gaff_lipids: tuple[str, ...]

    @property
    def switched_to_amber(self) -> bool:
        return self.protein_force_field != self.requested_force_field


def resolve_lipid_force_field(
    requested_force_field: str, lipid_names: list[str] | tuple[str, ...]
) -> LipidForceFieldResolution:
    """Select one internally compatible protein/lipid parameter family."""
    from gmxbuilder.modules.forcefield.gaff_backend import gaff_available

    requested = requested_force_field.strip().lower()
    names = tuple(sorted({name.strip().upper() for name in lipid_names if name.strip()}))
    missing = tuple(name for name in names if not lipid_has_rtp(name, requested))
    if not missing:
        return LipidForceFieldResolution(
            requested,
            requested,
            requested,
            names,
            (),
        )
    if not gaff_available():
        raise RuntimeError(
            "Selected lipids require GAFF2, but the isolated AmberTools/ACPYPE "
            "environment is unavailable"
        )
    # Amber force fields and GAFF2 share the same combination and 1-4 scaling
    # rules. All membrane lipids use GAFF2 when this policy is active; mixing
    # CHARMM lipid RTPs with GAFF in one topology is intentionally forbidden.
    return LipidForceFieldResolution(
        requested,
        GAFF_PROTEIN_FORCE_FIELD,
        "gaff2",
        names,
        names,
    )
