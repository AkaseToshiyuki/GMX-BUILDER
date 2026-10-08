"""Reviewed V4 candidates and immutable simulation-condition provenance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from gmxbuilder.modules.membrane.local_relaxation import construction_policy
from gmxbuilder.modules.membrane.v4_stationarity import FAMILY_ALPHA, METHOD, analysis_policy

SCHEMA = "v4-scientific-repair-2"
MIN_TEMPERATURE_K = 308.15
HOST_TEMPERATURE_K = 315.15


def resolve_protocol(name: str, family: str) -> dict:
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    name = name.upper()
    path = Path(__file__).parents[2] / "data" / "v4_protocols.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported reviewed V4 protocol schema")
    records = payload.get("records")
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or not {"lipid", "family", "status"} <= row.keys()
        for row in records
    ):
        raise ValueError("Malformed reviewed V4 protocol records")
    record = next((r for r in records if r["lipid"] == name and r["family"] == family), None)
    if record is None or record["status"] != "candidate":
        raise ValueError(
            record.get("reason", "V4 protocol is not a reviewed candidate")
            if record
            else "No reviewed V4 protocol"
        )
    if LipidRegistry.get(name).smiles != record["smiles"]:
        raise ValueError("Reviewed temperature protocol belongs to a different molecular identity")
    hosted = record["environment"] == "host"
    if record["environment"] not in {"host", "pure"}:
        raise ValueError("Unknown reviewed simulation environment")
    for key in ("reference_temperature_K", "bilayer_transition_K", "hexagonal_transition_K"):
        if key in record and not math.isfinite(float(record[key])):
            raise ValueError("Invalid reviewed simulation temperature")
    if hosted:
        temperature = HOST_TEMPERATURE_K
    elif "reference_temperature_K" in record:
        temperature = max(MIN_TEMPERATURE_K, float(record["reference_temperature_K"]))
    elif "bilayer_transition_K" in record:
        temperature = max(MIN_TEMPERATURE_K, float(record["bilayer_transition_K"]) + 15.0)
    else:
        raise ValueError("Pure protocol lacks a reference temperature or reviewed bilayer Tm")
    if not math.isfinite(temperature) or temperature < MIN_TEMPERATURE_K:
        raise ValueError("Invalid reviewed simulation temperature")
    if not hosted and temperature >= record.get("hexagonal_transition_K", math.inf):
        raise ValueError("Pure protocol exceeds the reviewed lamellar temperature range")
    protocol = {
        **record,
        "schema": SCHEMA,
        "temperature_K": temperature,
        "temperature_adjusted": (temperature != record.get("reference_temperature_K")),
        "lipids_per_leaflet": 200 if hosted else 64,
        "composition": {"POPC": 90, name: 10} if hosted else {name: 100},
        "minimum_effective_time_samples": 10,
        "observable_selection_method": "bond-graph-head-tail-1",
        "stationarity_method": METHOD,
        "stationarity_policy": analysis_policy(),
        "stationarity_family_alpha": FAMILY_ALPHA,
        "construction_policy": construction_policy(),
        "quantitative_area_relative_ci95_half_width": 0.02,
        "trajectory_interval_ps": 10,
        "host_smiles": LipidRegistry.get("POPC").smiles if hosted else None,
    }
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return protocol


def protocol_refusal(entry: dict) -> str | None:
    try:
        resolve_protocol(entry["lipid"], entry["family"])
    except (ValueError, KeyError) as exc:
        return str(exc)
    return None
