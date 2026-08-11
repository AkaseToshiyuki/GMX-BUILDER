<p align="center">
  <img src="src/gmxbuilder/web/static/assets/gmxbuilder-logo.png" alt="GMXBUILDER logo" width="520">
</p>

<h1 align="center">GMXBUILDER</h1>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><strong>Checkpointed molecular-system preparation for GROMACS</strong></p>

GMXBUILDER prepares GROMACS input packages for membrane-protein, lipid-bilayer,
and solution-phase molecular dynamics systems. The Web interface uses
checkpointed steps so that the coordinates confirmed in the viewer are the
coordinates used for final topology assignment and export.

## Supported workflows

- **Bilayer Builder** — process and orient a membrane protein, construct a
  compatible mixed or single-component bilayer, solvate it, and add ions.
- **Pure Bilayer System** — construct a protein-free bilayer, with optional
  solvation and ions. A dry export contains coordinates and topology only.
- **Solvator** — prepare protein, canonical DNA/RNA, protein–nucleic-acid,
  protein–ligand, or nucleic-acid–ligand systems in an aqueous box without a
  membrane.
- **Martini 3 Builder** — independently map a standard protein and build a
  Martini 3 aqueous, flat-bilayer, or protein–bilayer coarse-grained system.

Each Check creates a task-owned coordinate checkpoint. Finalization assigns
topology and packages the last confirmed coordinates without rebuilding the
membrane, solvent, or ions.

Available force fields, water models, lipids, and molecular parameter backends
are filtered by compatibility. Query the installed build instead of assuming
that every molecule is available with every force field:

```bash
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
```

## Requirements

- Linux and Python 3.10 or later.
- Git LFS for the release-bundled lipid assets.
- A working GROMACS installation for topology checks and simulations.
- A CUDA-capable GROMACS build only when GPU execution is required.
- AmberTools/ACPYPE in a separate environment when parameterizing new
  GAFF2 molecules. These executables are not bundled.
- The pinned Martinize2/Vermouth, COBY and MDTraj versions declared by the
  package when using the Martini 3 workflow.
- `uv` is recommended for exact, hash-checked installation from `uv.lock`.

## Installation

For an interactive local installation and persistent user service, run:

```bash
./install-local.sh
```

The installer defaults to loopback-only `127.0.0.1:7788`, half of the detected CPU cores, and
an exact number of concurrent task slots close to one slot per four allocated
cores. It creates a project virtual environment, installs dependencies and
prebuilt assets, and starts `gmxbuilder.service`. Every task remains serial;
the calculated per-task thread share is used only inside the active step.

Selecting a non-loopback address requires explicit `trusted-lan` mode. This
mode has no end-user login and must stay behind a private-network firewall.

Manual installation remains available:

```bash
git clone https://github.com/AkaseToshiyuki/GMX-BUILDER.git
cd GMX-BUILDER

git lfs install
git lfs pull

uv sync --frozen --no-dev
source .venv/bin/activate

gmxbuilder --version
gmxbuilder prebuilt-assets status
gmxbuilder prebuilt-assets install
```

`prebuilt-assets install` verifies the bundled archive before installing
missing files into the user cache. It does not overwrite existing cache files.
The public repository and its public release assets do not require a GitHub
access token.

## Web interface

Start a local service:

```bash
gmxbuilder serve
```

Open <http://127.0.0.1:7788/>. To listen on another interface:

```bash
GMXBUILDER_DEPLOYMENT_MODE=trusted-lan \
  gmxbuilder serve --host 0.0.0.0 --port 7788
```

The server refuses non-loopback binding in the default `local` mode. Public
mode additionally requires global Basic/Bearer authentication, an HTTPS origin,
and an explicitly trusted TLS reverse proxy.

At startup GMXBUILDER discovers GROMACS and checks whether the selected GPU
configuration is executable. Resource limits can be set explicitly:

```bash
gmxbuilder serve \
  --cpu-cores <allocated-threads> \
  --task-threads <threads-per-task> \
  --max-builds <concurrent-tasks> \
  --gpu-count <count>
```

`--task-threads` must divide `--cpu-cores` exactly. GMXBUILDER keeps the
checkpoint steps of each Task strictly serial; the per-task thread limit is
used only by safe numerical kernels and external tools inside the active step.
If the finalization slots are occupied, the task is saved in a restart-safe
queue and the Web UI reports its position and estimated start time.

Set `CUDA_VISIBLE_DEVICES` before starting the service when a specific subset
or ordering of physical GPUs is required. Use `--gpu-count 0` for CPU-only
operation.

### Web workflow

The Bilayer Builder presents these user-facing steps:

1. Input Structure
2. Force Field
3. Structure Processing
4. Protein Orientation
5. Membrane Builder
6. Solvent & Box
7. Ions
8. Simulation Parameters and export

Each **Check** operation saves a checkpoint. Final export assigns topology and
generates the package from the last confirmed coordinate checkpoint; it does
not rebuild the membrane, solvent, or ions.

During Input Structure Check, unambiguous registered modified-residue names
are recorded and reduced to their standard parent residues. Step 3 proposes
the matching modification only when the selected force field has a validated
atom-complete patch and complete bonded geometry parameters. Newly introduced
heavy atoms are constructed from the selected force field's equilibrium bond
lengths and angles, then checked for local geometry and hard overlaps before
the checkpoint is accepted. Ambiguous, unrecognized, or unavailable
modifications remain visible as warnings and require user review; they are
never guessed.

The built-in matrix is force-field-specific and includes validated
phosphorylation, deamidation and selected native-template modifications.
Explicit R-methionine sulfoxide, trans-(2S,4R)-hydroxyproline and defined
hydroxylysine stereocentres are constructed and checked numerically. Supported
Amber protein families also expose a dedicated paired-CYS disulfide workflow
that preserves the SG-SG bond through checkpoints and topology export. Query
`/api/patches?force_field=...` and
`/api/crosslink-capabilities?force_field=...` for the authoritative lists.
Names with competing chemical meanings are resolved conservatively; unsupported
glycosylation, long-chain lipidation and CHARMM disulfides remain unavailable
instead of being represented by an approximate topology.

## Command-line interface

Show the commands available in the installed version:

```bash
gmxbuilder --help
gmxbuilder build --help
gmxbuilder serve --help
```

Common inspection commands:

```bash
gmxbuilder info --pdb protein.pdb
gmxbuilder list-ff
gmxbuilder list-water
gmxbuilder list-lipids
gmxbuilder list-modules
gmxbuilder lipid-library status
gmxbuilder coarse-grained --help
```

### Martini 3 example

Build and interactively confirm a solvated asymmetric mixed bilayer:

```bash
gmxbuilder coarse-grained \
  --mode bilayer \
  --upper POPC:3 --upper CHOL:1 \
  --lower POPE:1 --lower POPG:1 \
  --box-xy 12 --box-z 14 --salt 0.15 \
  --threads 8 --mpi-ranks 1 --gpu-ids 0 \
  --output ./martini-system
```

Use `--pdb protein.pdb` for a protein–bilayer system, or `--mode solution
--pdb protein.pdb` for an aqueous protein. The first release accepts standard
protein residues only and intentionally rejects ligands, PTMs, glycans,
nucleic acids, custom CG molecules, curved membranes, and backmapping. The
authoritative installed capability list is available at
`GET /api/coarse-grained/capabilities`.

### YAML build example

The following example is for a membrane protein without retained ligands:

```yaml
system_name: membrane_system
output_dir: ./output
seed: 42

modules:
  input:
    pdb: ./protein.pdb

  forcefield:
    name: amber14sb
    lipid_ff: lipid21
    ligand_ff: none
    water_model: tip3p

  structure:
    pH: 7.0
    prepare_standard_termini: true

  orient:
    method: ppm

  membrane:
    lipid_type: POPC
    box_padding: 2.0

  solvation:
    box_padding: 2.0

  ions:
    cation: NA
    anion: CL
    concentration: 0.15
    neutralize: true
    ion_method: random

  topology: {}

  simparams:
    hardware:
      mode: thread-mpi
      cpu_threads: 8
      mpi_ranks: 2
      use_gpu: true
      gpu_count: 1
      gpu_ids: [0]
      gmx_command: gmx

  export:
    write_mdp: true
```

Run it with:

```bash
gmxbuilder build --config build.yaml
gmxbuilder build --config build.yaml --output ./another-output
```

`--output` overrides the top-level `output_dir`. For mixed bilayers, replace
`lipid_type` with the validated composition form:

```yaml
membrane:
  lipid_composition:
    upper:
      - {name: POPC, ratio: 70}
      - {name: POPE, ratio: 30}
    lower:
      - {name: POPC, ratio: 70}
      - {name: POPS, ratio: 30}
```

All selected lipids must be supported by the chosen lipid backend. Retained
small molecules also require an explicitly compatible backend and, for GAFF2,
an integer net charge.

Canonical linear DNA/RNA polymers are recognized separately from small molecules and
are currently enabled with **CHARMM36m + CHARMM TIP3P** in the Solvator. Native
GROMACS topology generation validates 5′/3′ termini, O3′–P polymer bonds,
hydrogens, and integral chain charge before solvation. Modified nucleotides,
Amber nucleic-acid combinations, membrane-embedded nucleic acids, and
coarse-grained nucleic acids are explicitly unavailable; they are never routed
through GAFF2/CGenFF as independent residues.

`simparams.hardware` configures the generated `run_md.sh`; it does not change
MDP physics. CPU threads must divide evenly across MPI ranks, and enabled GPU
IDs must be unique. Simulation defaults are selected by system class and
force-field family, while every exported stage remains explicitly editable.
See the [user manual](docs/GMXBUILDER_USER_MANUAL_V1.0.1.md) for the complete
parameter contract.

## HTTP API

When the service is running, interactive API documentation is available at:

- Swagger UI: <http://127.0.0.1:7788/docs>
- OpenAPI schema: <http://127.0.0.1:7788/openapi.json>

Useful discovery endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service and runtime status |
| `GET` | `/health/live` | Minimal unauthenticated liveness only |
| `GET` | `/api/hardware` | Public hardware capability summary |
| `GET` | `/api/task-types` | Available workflows |
| `GET` | `/api/options` | Force-field, water, lipid, and ion options |

Create a Solvator task:

```bash
API=http://127.0.0.1:7788

curl -sS -X POST "$API/api/upload-pdb" \
  -F "file=@protein.pdb" \
  -F "task_type=solvator"
```

Use the returned `task_id` for all subsequent calls. The step API is
checkpoint-based and must be called in workflow order:

```bash
TASK_ID=<returned-task-id>

curl -sS -X POST "$API/api/step/$TASK_ID/input" \
  -H "Content-Type: application/json" \
  -d '{"config":{}}'

curl -sS -X POST "$API/api/step/$TASK_ID/forcefield" \
  -H "Content-Type: application/json" \
  -d '{"config":{"name":"amber14sb","lipid_ff":"none","ligand_ff":"none","water_model":"tip3p"}}'

curl -sS -X POST "$API/api/step/$TASK_ID/structure" \
  -H "Content-Type: application/json" \
  -d '{"config":{"pH":7.0,"prepare_standard_termini":true}}'

curl -sS -X POST "$API/api/step/$TASK_ID/solvation" \
  -H "Content-Type: application/json" \
  -d '{"config":{"box_padding":2.0}}'

curl -sS -X POST "$API/api/step/$TASK_ID/ions" \
  -H "Content-Type: application/json" \
  -d '{"config":{"cation":"NA","anion":"CL","concentration":0.15,"neutralize":true,"ion_method":"random"}}'
```

After the required Check checkpoint exists, `/api/build` performs topology
assignment and packaging only:

```bash
curl -sS -X POST "$API/api/build" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"'"$TASK_ID"'",
    "task_type":"solvator",
    "system_name":"solution_system",
    "modules":{
      "simparams":{"schema_version":2},
      "export":{"write_mdp":true}
    }
  }'

curl -sS "$API/api/build/$TASK_ID/queue-status"
curl -fL "$API/api/task/$TASK_ID/download" -o result.zip
```

Save the Task ID before leaving the page. A queued or interrupted finalization
is recovered from disk after a service restart, and opening the task resumes at
the first incomplete visible step.

Bilayer API clients additionally execute `orient` and `membrane` before
`solvation`. Use `GET /api/steps/{task_id}` to inspect saved checkpoints.
`GET /api/tasks` is an administrator endpoint and requires the
`X-Admin-Token` header.

## Output package

Atomistic ZIP packages use a flat parameter layout. A typical solvated
atomistic package contains:

```text
input.gro
input.pdb                    # optional; omitted for systems beyond PDB limits
topol.top
index.ndx
run_md.sh
README.txt
forcefield.itp
ffbonded.itp
ffnonbonded.itp
<force-field database files>
topol_Protein_chain_A.itp    # generated only for chains that are present
<lipid-or-ligand>.itp        # generated only when present
<water-model>.itp
<ion-parameters>.itp
mdp/
  mini.mdp
  equili_<n>.mdp             # only enabled equilibration stages
  production.mdp             # one unsplit production stage, when requested
  production_<n>.mdp         # restart-friendly segmented production
```

The exact parameter filenames depend on the selected force field, water model,
lipids, ligands, chains, and ions. `topol.top` is the authoritative include
list. `index.ndx` contains only groups applicable to the exported system.

A dry Pure Bilayer export intentionally omits solvent, ions, MDP files, and
`run_md.sh`.

Martini 3 packages keep their versioned force-field and protein ITP files in
`toppar/`, and additionally contain `manifest.json` and `CITATIONS.json`.
Solvated CG packages contain `mdp/` and `run_md.sh`; a dry CG bilayer is a
coordinate/topology geometry package and deliberately omits both.

Run a solvated package with:

```bash
unzip result.zip -d simulation
cd simulation
chmod +x run_md.sh
./run_md.sh
```

The exported coordinates are an input system for minimization and
equilibration. They are not a claim that production equilibration has already
been achieved.

## Deployment

For a public installation, keep GMXBUILDER bound to loopback behind a TLS
reverse proxy and set `GMXBUILDER_DEPLOYMENT_MODE=public`. Public mode refuses
startup/request handling unless it has an HTTPS origin, trusted proxy addresses,
and either strong Basic credentials or a 32-character Bearer token. Forwarded
client/protocol headers are trusted only from the configured proxy.

Hardened examples are provided in [deploy/](deploy/). Application rate limits
use a private SQLite database and therefore survive restarts; the reverse-proxy
example adds an independent network-edge limit. The generated local user service
also enables systemd filesystem, privilege, kernel and syscall protections while
leaving CUDA/GROMACS devices available.

## Documentation

- [Documentation index](docs/README.md)
- [User manual V1.0.1](docs/GMXBUILDER_USER_MANUAL_V1.0.1.md)
  ([PDF](docs/GMXBUILDER_USER_MANUAL_V1.0.1.pdf))
- [Scientific compatibility and limitations](docs/SCIENTIFIC_COMPATIBILITY.md)

## License and third-party data

Original GMXBUILDER code is licensed under the MIT License. Force-field data,
generated parameters, and external tools retain their own licenses, citation
requirements, and scientific limitations. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.
