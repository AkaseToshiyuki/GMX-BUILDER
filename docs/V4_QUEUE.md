# V4 queue operation

Run the queue from a dedicated, clean checkout pinned to a commit. Install its
required force-field assets before starting. Do not edit that checkout during
the run. Web development and deployment can then continue in the main checkout.

The queue requires physically valid artifacts and at least **10 effective area
samples in each replica**. Area convergence and cross-replica disagreement are
separate diagnostics; neither silently increases this stopping threshold.
Analysis uses the complete selected stationary window, not just ten records.
The 200 ns cap is a resource limit and does not establish a physical phase.

Current V4 admission uses retained trajectory frames and separately records
local-conformation readiness and area diagnostics. Preserve the original
trajectories, checkpoints and parameter/source fingerprints when extending a run.
Older entries without the required evidence cannot be promoted by changing a
saved ready flag. The shipped 1.0.0 assets are initialization conformers only;
see [release support](release-support.md).

The offline queue performs molecular dynamics and is an administrator workflow.
Ordinary installation, library status queries and Web construction do not start it.

## Launch and evidence

Create `~/.config/gmxbuilder/run-v4-library.sh` with explicit absolute paths to
the pinned checkout, its Python environment and the intended GROMACS executable.
Use the same GROMACS release as a retained simulation when continuing it. Example:

```bash
#!/usr/bin/env bash
set -euo pipefail
export GMX_BIN=/absolute/path/to/pinned/gmx
unset GMXBUILDER_LIPID_LIBRARY
cd /absolute/path/to/pinned/checkout
exec /absolute/path/to/python scripts/build_v4_library.py \
  --families charmm36m-lipid,amber-lipid21,charmm36-lipid \
  --replicas 2 --gpus 0,1 --threads 12
```

Install `deploy/gmxbuilder-v4-library.service.example` as the user unit
`gmxbuilder-v4-library.service`; adapt writable paths if the cache root changes.
Before starting it, inspect existing queue/replica/GROMACS processes and disable
the legacy V3 library service and watchdog timer. The new lock cannot protect
against an already-running older queue that predates that lock.

`queue.lock` excludes another current scheduler. Replica Python children inherit
it. The systemd control group owns their GROMACS processes as well; manage the
service as a whole rather than killing individual replica Python processes.
The unit does not automatically restart a failed queue or overwrite partial data.

`runs/*.json` records the source/parameter hashes, commit, executable hash,
GROMACS version and queue configuration. Adjacent `.operations.jsonl` files
record each replica launch, seed, target time and previous metadata hash.
`queue-progress.jsonl` separates queue completion from area diagnostics.
These records establish new launches, not the origin of older trajectories.

## Maintenance and recovery

For a planned update, create `STOP_AFTER_ENTRY` in the V4 root and wait until
the service and all its descendants have exited. The current entry may require
extensions before reaching its stopping condition. A stopped queue with pending
entries exits nonzero; this does not invalidate completed entries. Preserve the
run records, prepare the new pinned checkout and review its dry run. Remove the
stop marker only when dispatch is intended, then start the service.

For an unexpected failure, inspect the replica logs, retained `work/` files and
metadata before restarting. Ready replicas are preserved. Incomplete or corrupt
directories are refused for manual recovery, not silently rebuilt. A continuation
requires `npt.tpr`, `npt.cpt`, `npt.edr` and `topol.top`; an existing published run
is copied to a temporary work directory before extension. Do not delete the
original artifacts to make a retry pass. Reuse between force fields also refuses
to replace existing simulation work or replica directories.
