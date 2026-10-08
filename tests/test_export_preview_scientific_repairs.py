import json
import os
import subprocess

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from gmxbuilder.modules.coarse_grained.protocol import (
    normalize_protocol,
    write_mdp_files,
    write_run_script,
)
from gmxbuilder.web.server_parts.modification_preview import build_modification_preview


@pytest.mark.parametrize("membrane", [False, True])
@pytest.mark.parametrize("eq1,eq2", [(True, True), (False, True), (False, False)])
def test_export_script_only_consumes_checkpoints_written_by_dynamics(tmp_path, membrane, eq1, eq2):
    fake = tmp_path / "fake-gmx"
    fake.write_text("""#!/usr/bin/env python3
import sys, json
from pathlib import Path
args = sys.argv[1:]
with Path('calls.jsonl').open('a') as f:
    f.write(json.dumps(args) + '\\n')
if args[0] == 'grompp':
    if '-t' in args and not Path(args[args.index('-t')+1]).is_file():
        sys.exit(2)
elif args[0] == 'mdrun':
    stage = args[args.index('-deffnm')+1]
    Path(stage+'.gro').write_text('fake coordinates')
    if stage != 'mini':
        Path(stage+'.cpt').write_text('fake checkpoint')
else:
    sys.exit(3)
""")
    fake.chmod(0o755)
    config = normalize_protocol(
        {"equilibration_1": eq1, "equilibration_2": eq2}, has_membrane=membrane
    )
    stages = write_mdp_files(tmp_path / "mdp", config)
    write_run_script(tmp_path / "run_md.sh", stages, config, {"use_gpu": False})
    run = subprocess.run(
        ["bash", "run_md.sh"],
        cwd=tmp_path,
        env={**os.environ, "GMX": str(fake)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    preprocessing = [args for args in calls if args[0] == "grompp"]
    assert len(preprocessing) == len(stages)
    assert "-t" not in preprocessing[1]
    for previous, args in zip(stages[1:], preprocessing[2:]):
        assert args[args.index("-t") + 1] == previous[0] + ".cpt"


@pytest.mark.parametrize("sequence,expected", [("AKA", 1), ("ARA", 1), ("AAA", 0)])
@pytest.mark.parametrize("ncap,ccap,shift", [(None, None, 0), ("ACE", None, -1), (None, "NME", 1)])
def test_preview_counts_charges_once_per_residue_and_tracks_free_ends(
    tmp_path, sequence, expected, ncap, ccap, shift
):
    molecule = Chem.AddHs(Chem.MolFromFASTA(sequence))
    assert AllChem.EmbedMolecule(molecule, randomSeed=872) == 0
    path = tmp_path / "protein.pdb"
    Chem.MolToPDBFile(molecule, str(path))
    result = build_modification_preview(path, 7.0, "HSE", [], "amber14sb", ncap, ccap)
    assert result["net_charge"] == pytest.approx(expected + shift)
    assert all(row["index"] < 3 for row in result["residue_changes"])
