# Anonymous Web resource policy

The managed Linux installation accepts anonymous jobs without a queue-count,
daily job-count or rolling CPU-hour limit. It schedules expensive upload parsing,
Check operations, previews, protonation, ligand preparation and finalization in
one persistent FIFO. Requests waiting for the same task's active operation are
serialized; other tasks can use available slots.

The defaults, exposed by `install-local.sh`, are:

| Setting | Installer option | Default |
| --- | --- | --- |
| `GMXBUILDER_TASK_MEMORY_GIB` | `--task-memory-gib` | 16 GiB per operation and all children |
| `GMXBUILDER_TASK_MAX_PERSISTENT_GIB` | `--task-storage-gib` | 2 GiB per task directory |
| `GMXBUILDER_STORAGE_MAX_GIB` | `--storage-gib` | 100 GiB of Web writable data |
| `GMXBUILDER_TASK_TTL_HOURS` | Installer fixes this to 24 | 24 hours from creation |
| `GMXBUILDER_CPU_CORES` | `--cpu-cores` | Installation CPU allocation |
| `GMXBUILDER_MAX_BUILDS` | `--queue-slots` | Installation concurrent operation slots |

GiB means 1,073,741,824 bytes. Per-task accounting includes uploads, checkpoints,
operation responses, temporary work and ZIP copies in that task directory.
The total includes Web caches, per-operation worker logs, the queue database and
temporary uploads. The main Web process logs to inherited stderr/journald outside
the FUSE quota; host journal retention must bound those logs separately.
Installed scientific assets and the offline V4 library are not copied into or
modified by this volume.

Expiry is computed from the original `created_at`, including queue and user
interaction time. Reads, downloads, retries and resume do not extend it. On
expiry the coordinator stops the operation's entire cgroup, confirms exit, and
then removes its files. Existing downloads are also ended at the deadline;
termination can include a short process-exit grace period. Queued expired work
never starts. Under storage pressure the oldest non-running task is removed
first; active uploads/downloads hold file leases. If no space can be released,
new writes and starts pause while reads and queue status remain available.

Each running process group receives a systemd memory ceiling and no swap.
Memory admission reserves the unused part of existing operations' allowances.
CPU quotas are redistributed when operations start and finish, within the
installation allocation. External scientific programs already running retain
their thread counts; increasing their allowed CPU time does not guarantee a
linear speedup. There is no additional fixed anonymous compute pool or
60-minute operation timeout. Necessary tool-specific timeouts still apply.

## Installation and upgrade

Requires Linux with Landlock ABI 3 or later, cgroup v2, a working systemd user
manager, and FUSE3. On Ubuntu the build dependencies for the optional `managed`
Python extra are `libfuse3-dev` and `pkg-config`; the runtime also needs
`fusermount3` and access to `/dev/fuse`. The installer fails explicitly if these
are unavailable. It must not silently offer unbounded anonymous computation.

`gmxbuilder-storage.service` mounts a private FUSE volume. File sizes are rounded
to 4 KiB and charged an additional 4 KiB per inode. Sparse files and files
deleted while still open remain accounted for. A small reserve within the total
limit is kept for queue cancellation/error records. Landlock confines Web and
scientific-process file writes to the managed mount. The Web service's own
memory is bounded separately from its operation cgroups; it is not a second
anonymous computation pool.

For an existing installation, inspect Web workers and stop the Web service
before migration. `python -m gmxbuilder.web.resource_setup --source-tasks PATH
--inventory` only reports affected tasks. `--install` prepares the volume;
`--migrate` copies retained tasks and verifies all file hashes before removing
their original copies and expired tasks. An oversized unexpired task or
insufficient space stops migration with original data retained. Do not run the
full scientific installer against a pinned scientific checkout while V4 runs;
the resource setup helper is independent of those paths.
The old task-root path becomes an alias into the managed volume, preserving
server-authored absolute paths inside retained checkpoints without bypassing quotas.

The Web unit depends on the storage unit. The launcher sets
`GMXBUILDER_MANAGED_ROOT`, its `tasks` directory, managed temporary/cache paths,
and `GMXBUILDER_SERVICE_UNIT=gmxbuilder.service`. On Web shutdown its managed
operations stop; waiting tickets persist. An interrupted scientific operation
is reported for review/retry and is not silently rerun against partial output.

For a TLS reverse proxy, `GMXBUILDER_DEPLOYMENT_MODE=public-anonymous` explicitly
selects anonymous resource-limited access, requiring managed storage, configured
trusted proxies and explicit HTTPS origins. `public` retains its authenticated
behavior. Do not infer proxy addresses from untrusted forwarding headers. The
local installer retains its existing `local` / explicitly opted-in `trusted-lan`
listener configuration; changing a reverse proxy is a separate deployment step.

## HTTP and browser behavior

Managed expensive requests return HTTP 202 with an `X-GMXBUILDER-Operation`
header and a capability `operation_id`. Poll `/api/operations/{id}`; when
`response_ready` is true, retrieve the original status/body from
`/api/operations/{id}/result`. A disconnected browser does not cancel work.
The browser client performs this exchange automatically and can restore its
pending tickets after a reload. XHR structure uploads use the same final-result
exchange as fetch. Progress, queue details and Copy Task ID are shown inline;
there is no modal acknowledgement. Keep the task ID for later resume.

Status includes queue length, position, number ahead, elapsed wait, expiry,
resource pause reason and an estimate based on comparable completed operations.
Without enough history the estimate is explicitly unavailable. It is not a
promise that a task will start before expiry. HTTP 410 means expired/removed;
507 reports unavailable storage; a failed operation reports its error while
valid checkpoints remain available until expiry or pressure cleanup.

`/api/resource-policy` distinguishes configured defaults from whether managed
isolation is active. `/health` reports managed running/waiting operations so the
deployment script can avoid interrupting a live Check or ligand preparation.
The plain developer server can run without managed isolation for local tests;
it must not be presented as enforcing production resource quotas.
