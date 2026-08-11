#!/usr/bin/env python3
"""Exercise the membrane builder through the same HTTP flow as the web UI.

The run is deliberately reproducible: it samples 20 entries from a curated
membrane-protein pool, keeps one protein chain, applies one supported random
Amber modification, builds a POPC-major Lipid21 membrane, confirms that the
Ion Check viewer contains the real complete system, downloads the final ZIP,
and validates ``mini.mdp`` with the configured GROMACS ``grompp``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

from gmxbuilder.runtime.hardware import find_gromacs_executable


PDB_POOL = (
    "1GZM", "2RH1", "3SN6", "4N6H", "5C1M", "6IIK", "4EIY", "5G53",
    "6CM4", "6DDF", "1BL8", "3JYC", "4HFI", "5KXI", "6COF", "4M48",
    "5I6C", "6Y5W", "7E3Y", "1QLE", "2BS2", "3CAP", "4GD3", "6N3W",
    "7L1V", "1C3W", "2A65", "3WBN", "4U15", "5X29", "6B73", "6VMS",
)

PATCHES = {
    "SER": ("PHOS_SER", "PHOS1_SER"),
    "THR": ("PHOS_THR", "PHOS1_THR"),
    "TYR": ("PHOS_TYR", "PHOS1_TYR"),
    "ASN": ("DEA_ASN",),
    "GLN": ("DEG_GLN",),
}

MINOR_LIPIDS = ("POPE", "POPG", "POPS", "CHOL", "DOPC", "DPPC")


def _json_response(response: httpx.Response, action: str) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{action}: HTTP {response.status_code}, non-JSON response") from exc
    if response.status_code >= 400 or data.get("status") == "error" or data.get("error"):
        raise RuntimeError(f"{action}: {data.get('error') or data}")
    return data


def _post_json(session: httpx.Client, url: str, payload: dict, action: str) -> dict:
    return _json_response(session.post(url, json=payload, timeout=1800), action)


def _composition(rng: random.Random) -> tuple[list[dict], list[dict]]:
    def leaflet() -> list[dict]:
        minor = rng.choice(MINOR_LIPIDS)
        minor_ratio = rng.randrange(10, 31, 5)
        return [
            {"name": "POPC", "ratio": 100 - minor_ratio},
            {"name": minor, "ratio": minor_ratio},
        ]

    upper = leaflet()
    lower = leaflet() if rng.random() < 0.5 else [dict(item) for item in upper]
    return upper, lower


def _select_chain_and_modification(
    upload: dict, rng: random.Random
) -> tuple[str, list[dict], dict]:
    chains = [chain for chain in upload.get("sequences", []) if chain.get("residues")]
    if not chains:
        raise RuntimeError("upload returned no protein chains")
    chain = max(chains, key=lambda item: len(item["residues"]))
    candidates = [
        (index, residue)
        for index, residue in enumerate(chain["residues"])
        if residue.get("resname") in PATCHES
    ]
    if not candidates:
        raise RuntimeError(f"chain {chain['chain_id']} has no supported modification target")
    index, residue = rng.choice(candidates)
    patch_id = rng.choice(PATCHES[residue["resname"]])
    return str(chain["chain_id"]), list(chain["residues"]), {
        "index": index,
        "patch_id": patch_id,
        "source_resname": residue["resname"],
        "source_resid": residue["resid"],
    }


def _payload(
    task_id: str,
    composition: tuple[list[dict], list[dict]],
    modification: dict,
    protonation: list[dict],
) -> dict:
    upper, lower = composition
    lipid_names = sorted({item["name"] for item in upper + lower})
    modules = {
        "input": {"task_id": task_id},
        "forcefield": {
            "name": "amber14sb", "lipid_ff": "lipid21", "ligand_ff": "none",
            "water_model": "tip3p", "lipid_names": lipid_names,
        },
        "structure": {
            "protonation": protonation, "modifications": [{
                "index": modification["index"], "patch_id": modification["patch_id"],
            }],
            "termini": {}, "pH": 7.0, "skip_protonation": False,
        },
        "orient": {"method": "ppm"},
        "membrane": {
            "lipid_composition": {"upper": upper, "lower": lower},
            "n_lipids_per_leaflet": 64,
        },
        "solvation": {"box_padding": 1.2, "overlap_scale": 0.8},
        "ions": {
            "cations": ["NA"], "anions": ["CL"],
            "concentration": {"NA": 0.15, "CL": 0.15},
            "neutralize": True, "neutralize_cation": "NA", "neutralize_anion": "CL",
            "ion_method": "replace", "exclusion_radius": 0.35,
        },
        "topology": {},
        "verify": {},
        "simparams": {
            "constraints": "h-bonds", "pcoupl_type": "auto", "temperature": 310,
            "gen_seed": 20260722,
            "eq_stages": [{
                "bb": 1000, "sc": 500, "lipid": 400, "dih": 200,
                "dt": 2.0, "nsteps": 5000, "ensemble": "npt",
                "tcoupl": "v-rescale", "tau_t": "1.0", "pcoupl": "C-rescale",
                "tau_p": "5.0", "ref_p": "1.0", "compress": "4.5e-5",
                "nstcomm": 100, "comm_mode": "linear",
                "nstxout_compressed": 1000, "nstxout": 0, "nstvout": 0,
                "nstfout": 0, "nstcalcenergy": 100, "nstenergy": 500, "nstlog": 500,
            }],
            "prod_iters": [{
                "dt": 2.0, "nsteps": 10000, "tcoupl": "v-rescale", "tau_t": "1.0",
                "pcoupl": "C-rescale", "tau_p": "5.0", "ref_p": "1.0",
                "compress": "4.5e-5", "nstcomm": 100, "comm_mode": "linear",
                "nstxout_compressed": 1000, "nstxout": 0, "nstvout": 0,
                "nstfout": 0, "nstcalcenergy": 100, "nstenergy": 500, "nstlog": 500,
            }],
            "em_nsteps": 5000, "em_ftol": 1000.0, "em_step": 0.01,
            "em_nstlist": 10, "em_integrator": "steep",
        },
        "export": {"write_mdp": True, "mdp_params": {}},
    }
    return {
        "task_id": task_id, "task_type": "membrane-bilayer",
        "system_name": "membrane_system", "seed": 20260722,
        "modules": modules,
    }


def _validate_ion_viewer(content: str) -> dict:
    if not content.startswith(("HEADER", "REMARK")) or "CRYST1" not in content:
        raise RuntimeError("Ion Check viewer is missing a valid CRYST1 system box")
    atom_lines = [line for line in content.splitlines() if line.startswith(("ATOM", "HETATM"))]
    resnames = {line[17:20].strip() for line in atom_lines}
    if "SOL" not in resnames or not ({"NA", "CL"} & resnames):
        raise RuntimeError("Ion Check viewer does not contain real solvent and ions")
    return {"atoms": len(atom_lines), "resnames": sorted(resnames)}


def _grompp_validate(zip_path: Path, gmx: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-e2e-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        top = next(root.rglob("topol.top"))
        gro = next(root.rglob("input.gro"))
        mdp = next(root.rglob("mini.mdp"))
        ndx = next(root.rglob("index.ndx"))
        output = root / "mini.tpr"
        proc = subprocess.run(
            [gmx, "grompp", "-f", str(mdp), "-c", str(gro), "-p", str(top),
             "-r", str(gro), "-n", str(ndx), "-o", str(output)],
            cwd=top.parent, text=True, capture_output=True, timeout=300,
        )
        if proc.returncode:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
            raise RuntimeError(f"grompp failed:\n{tail}")
        return {"tpr_bytes": output.stat().st_size}


def run_one(base_url: str, pdb_id: str, seed: int, output_dir: Path, gmx: str) -> dict:
    started = time.time()
    rng = random.Random(f"{seed}:{pdb_id}")
    session = httpx.Client(follow_redirects=True)
    pdb_response = session.get(
        f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=120,
        headers={"User-Agent": "GMXBUILDER-E2E/0.6"},
    )
    pdb_response.raise_for_status()
    upload = _json_response(
        session.post(
            f"{base_url}/api/upload-pdb",
            files={"file": (f"{pdb_id}.pdb", pdb_response.content, "chemical/x-pdb")},
            timeout=300,
        ),
        f"{pdb_id} upload",
    )
    task_id = upload["task_id"]
    chain, residues, modification = _select_chain_and_modification(upload, rng)
    exclusions = sorted({item["resname"] for item in upload.get("small_molecules", [])})
    _post_json(
        session, f"{base_url}/api/filter-pdb/{task_id}",
        {"include_chains": [chain], "exclude_resnames": exclusions, "small_molecule_labels": {}},
        f"{pdb_id} filter",
    )
    structure_residues = [
        {
            "resname": residue["resname"], "chain": chain,
            "resid": residue["resid"], "index": index,
        }
        for index, residue in enumerate(residues)
    ]
    protonation_result = _post_json(
        session, f"{base_url}/api/protonate",
        {
            "residues": [item["resname"] for item in structure_residues],
            "pH": 7.0, "his_tautomer": "HSE", "task_id": task_id,
            "structure_residues": structure_residues,
        },
        f"{pdb_id} protonation",
    )
    protonation = [
        assignment for assignment in protonation_result.get("assignments", [])
        if assignment.get("is_titratable")
    ]
    composition = _composition(rng)
    payload = _payload(task_id, composition, modification, protonation)

    # Exact sequence of frontend Check actions through the Ion page.
    for step in ("input", "forcefield", "structure", "orient", "membrane", "solvation", "ions"):
        result = _post_json(
            session, f"{base_url}/api/step/{task_id}/{step}",
            {"pipeline_type": "membrane-bilayer", "config": payload["modules"][step], "seed": seed},
            f"{pdb_id} Check {step}",
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"{pdb_id} Check {step}: {result}")

    viewer_response = session.get(
        f"{base_url}/api/step/{task_id}/ions/viewer.pdb", timeout=300
    )
    viewer_response.raise_for_status()
    viewer = _validate_ion_viewer(viewer_response.text)

    submitted = _post_json(session, f"{base_url}/api/build", payload, f"{pdb_id} final build")
    if submitted.get("status") not in {"started", "queued"}:
        raise RuntimeError(f"unexpected build submission response: {submitted}")
    since = 0
    log_lines: list[str] = []
    while True:
        status = _json_response(
            session.get(f"{base_url}/api/build/{task_id}/log", params={"since": since}, timeout=60),
            f"{pdb_id} build log",
        )
        log_lines.extend(status.get("lines", []))
        since = status.get("total", since)
        if status.get("done"):
            break
        time.sleep(2)
    if not any("Build complete" in line for line in log_lines):
        raise RuntimeError("final build failed: " + " | ".join(log_lines[-10:]))

    download = session.get(f"{base_url}/api/task/{task_id}/download", timeout=600)
    download.raise_for_status()
    zip_path = output_dir / f"{pdb_id}_{task_id}.zip"
    zip_path.write_bytes(download.content)
    grompp = _grompp_validate(zip_path, gmx)
    return {
        "pdb_id": pdb_id, "task_id": task_id, "status": "passed", "chain": chain,
        "modification": modification, "upper": composition[0], "lower": composition[1],
        "viewer": viewer, "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size,
        "grompp": grompp, "elapsed_s": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7788")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, default=Path("output/e2e_membrane_20"))
    parser.add_argument("--gmx", default=find_gromacs_executable())
    args = parser.parse_args()
    if not 1 <= args.count <= len(PDB_POOL):
        parser.error(f"--count must be in [1, {len(PDB_POOL)}]")
    if not args.gmx:
        parser.error("GROMACS was not found; set GMX_BIN or pass --gmx")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    selected = random.Random(args.seed).sample(PDB_POOL, args.count)
    report = {"seed": args.seed, "selected": selected, "results": []}
    for index, pdb_id in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {pdb_id}", flush=True)
        try:
            result = run_one(args.base_url.rstrip("/"), pdb_id, args.seed, args.output, args.gmx)
        except Exception as exc:
            result = {"pdb_id": pdb_id, "status": "failed", "error": str(exc)}
            print(f"  FAILED: {exc}", flush=True)
        else:
            print(f"  passed in {result['elapsed_s']} s", flush=True)
        report["results"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    passed = sum(item["status"] == "passed" for item in report["results"])
    print(f"Completed: {passed}/{len(selected)} passed; report: {report_path}")
    return 0 if passed == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
