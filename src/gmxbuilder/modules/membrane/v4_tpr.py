"""Verify effective compiled inputs rather than trusting an MDP filename."""

import hashlib
import json
import math
import re
import subprocess
import tempfile


def validate_inputrec(text, temperature, force_field, stage):
    def field(name):
        match = re.search(r"^\s*" + re.escape(name) + r"\s*[=:]\s*(.+)$", text, re.M)
        if not match:
            raise ValueError(f"Compiled TPR is missing {name}")
        return match.group(1).strip()

    expected = {
        "integrator": "md",
        "cutoff-scheme": "Verlet",
        "pbc": "xyz",
        "coulombtype": "PME",
        "tcoupl": "V-rescale",
        "vdw-modifier": "Force-switch" if force_field.startswith("charmm") else "Potential-shift",
        "DispCorr": "No" if force_field.startswith("charmm") else "EnerPres",
        "pcoupl": "C-rescale" if stage == "npt" else "No",
    }
    for key, value in expected.items():
        if field(key).lower() != value.lower():
            raise ValueError(f"Compiled TPR {key} differs from {value}")
    numeric = {
        "dt": 0.002,
        "nstxout-compressed": 5000,
        "rvdw": 1.2 if force_field.startswith("charmm") else 1.0,
        "rcoulomb": 1.2 if force_field.startswith("charmm") else 1.0,
    }
    if force_field.startswith("charmm"):
        numeric["rvdw-switch"] = 1.0
    for key, value in numeric.items():
        actual = float(field(key))
        if not math.isfinite(actual) or abs(actual - value) > 1e-6:
            raise ValueError(f"Compiled TPR {key} differs from {value}")
    temperatures = [float(v) for v in field("ref-t").split()]
    if not temperatures or any(
        not math.isfinite(v) or abs(v - temperature) > 0.005 for v in temperatures
    ):
        raise ValueError("Compiled thermostat differs from the approved temperature")
    if stage == "npt" and field("pcoupltype").lower() != "semiisotropic":
        raise ValueError("Compiled membrane barostat is not semi-isotropic")
    return {"passed": True, "effective": {**expected, **numeric, "ref-t": temperatures}}


def verify_tpr(gmx, tpr, protocol, force_field, stage):
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
        header = output.read(65536)
    result = validate_inputrec(header, protocol["temperature_K"], force_field, stage)
    result.update(
        tpr_sha256=hashlib.sha256(tpr.read_bytes()).hexdigest(), protocol_sha256=protocol["sha256"]
    )
    tpr.with_suffix(".verification.json").write_text(json.dumps(result, indent=2))
    return result
