# Web ingress and quota-storage recovery

The application validates Host on all requests and Origin on state-changing
browser requests. Configure exact deployment origins and hostnames; an arbitrary
matching Host/Origin pair does not authorize a request.

For public anonymous deployments, set `GMXBUILDER_DEPLOYMENT_MODE=public-anonymous`,
`GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=0`, the managed storage root, explicit HTTPS
`GMXBUILDER_CORS_ORIGINS`, and exact `GMXBUILDER_TRUSTED_PROXIES`. Set
`FORWARDED_ALLOW_IPS=` (empty) so Uvicorn preserves the socket peer for the
application's own validation. The trusted proxy must replace untrusted forwarded
headers and supply valid `X-Forwarded-For` and `X-Forwarded-Proto: https`.
Do not trust all private networks just because a proxy runs on one private IP.

Optional direct LAN HTTP requires both settings, for example:

```sh
GMXBUILDER_LAN_ORIGINS=http://192.168.1.10:7788
GMXBUILDER_LAN_NETWORKS=192.168.1.0/24
```

The HTTP exception requires an exact configured LAN origin and a direct peer in
the configured LAN range. A trusted reverse proxy cannot claim this exception.
The public hostname still requires HTTPS. `GMXBUILDER_ALLOWED_HOSTS` can add exact
hostnames for monitoring; adding a Host does not add an allowed browser Origin.
Host identifies the requested API/site, whereas Origin identifies the browser
page making a request. When these differ, configure both explicitly, for example
`GMXBUILDER_ALLOWED_HOSTS=api.example.org` and
`GMXBUILDER_CORS_ORIGINS=https://app.example.org`. Allowed hosts are the union of
explicit hostnames, configured public/LAN origin hostnames, and local listener
addresses; the explicit setting adds hosts rather than replacing that union.
LAN origins and networks are optional and are needed together only when direct
LAN HTTP is required. Preserve these settings in the generated launcher when
reinstalling; an existing launcher outside the repository is not a deployment
configuration backup.

## Request budgets

JSON endpoints default to 2 MiB. Only POST to `/api/upload-pdb`,
`/api/ligand-chemistry-upload/{id}` and `/api/cgenff-upload/{id}` receives the
128 MiB multipart budget. Content type cannot enlarge a JSON endpoint's budget.
Fixed-length and streamed bodies use the same contract, including managed
worker ingress. Wrong media types return 415; excessive bodies return 413.

The default receive idle deadline is 30 seconds and total receive deadline is
300 seconds (`GMXBUILDER_BODY_IDLE_TIMEOUT`, `GMXBUILDER_BODY_TOTAL_TIMEOUT`).
These bound receipt of a request, not a scientific operation's runtime.
Managed ingress waits at most 10 seconds (`GMXBUILDER_INGRESS_WAIT_TIMEOUT`),
with two concurrent pending/active uploads per client
(`GMXBUILDER_CLIENT_UPLOADS`). Control requests use separate admission slots.
Timeout or cancellation releases its reservations.

`GMXBUILDER_HTTP_BODY_MEMORY_MIB` defaults to 2048. Reservations cover four
endpoint budgets for multipart and 32 for JSON, allowing for copies and parsed
objects. Uploads can reserve at most three quarters of the pool. This is an
admission estimate, not a measured peak-memory guarantee; the process cgroup
remains the hard memory boundary. Excess admission returns 503 with Retry-After.
Tune body limits and this pool together: one request's reservation must fit.

## Readiness and recovery

`/health/live` reports process liveness; `/health/ready` reports quota mount,
policy, write-fault, free-space and coordinator readiness. Both support GET and
HEAD. Readiness is bounded to one outstanding storage probe even if FUSE hangs.
Use readiness for service monitoring; a 200 on the home page alone does not
establish storage health. Application logs use inherited stderr/journald, outside
the FUSE volume, with capability redaction including formatted tracebacks.

Install recovery **after storage installation, task migration and Web service
configuration are complete**:

```sh
.venv/bin/python -m gmxbuilder.web.resource_setup \
  --root "$HOME/.local/share/gmxbuilder/web-storage" --install-web-recovery
```

This installs and enables the user `gmxbuilder-online.target`, a recovery timer,
a bounded oneshot recovery service and a Web unit drop-in. The timer checks
storage readiness and restarts Web with bounded backoff after BindsTo stopped
it. `Restart=on-failure` or `always` alone does not restart a dependent service
after its dependency recovers. The recovery service does not start a manually
stopped storage service or reset systemd failure limits.
Storage is checked even when the Web process is active. A `storage-write-fault`
event is recorded on a state transition, without restarting storage or clearing
the fault. Liveness or an active systemd unit does not imply writable storage.

For deliberate maintenance, stop `gmxbuilder-online.target`; after maintenance,
start the target. Stopping only Web while the online target remains active asks
the supervisor to restore Web. The target's Requisite/PartOf relationship keeps
late recovery attempts from undoing maintenance. Do not infer library-queue
permission from this Web target; library services and timers are separate.

### Recovering a latched storage write fault

`storage-write-fault` means the daemon has stopped accepting mutations after an
unexpected invariant or data-integrity failure. It stays latched for the life of
that daemon. The supervisor reports it but deliberately does not restart storage:
a restart invalidates open files and can interrupt scientific operations.

1. Preserve the storage and Web journal around the first error. Inspect active
   `gmxbuilder-op-*.service` workers and scientific processes using the volume;
   wait for safe completion or deliberately stop the affected work. Do not assume
   that stopping Web also stops its separate worker units.
2. Stop `gmxbuilder-online.target` for maintenance. Once the volume's users have
   stopped, restart `gmxbuilder-storage.service`. The new daemon rescans backing
   data and checks quotas; it must mount successfully before proceeding. Do not
   clear the backing directory, delete task data, or manually reset `write_fault`.
3. Verify the mount and quota policy using the storage readiness probe. If
   startup or the probe fails, retain maintenance mode and investigate the
   reported error; repeated restarts are not a repair for damaged data.
4. Start `gmxbuilder-online.target`, verify `/health/ready`, public HTTPS HEAD and
   direct LAN HTTP, and review interrupted operations before retrying them.

After step 1, the service commands are:

```sh
systemctl --user stop gmxbuilder-online.target
systemctl --user restart gmxbuilder-storage.service
.venv/bin/python -m gmxbuilder.web.resource_setup \
  --wait-mounted "$HOME/.local/share/gmxbuilder/web-storage/mounted"
systemctl --user start gmxbuilder-online.target
```

Run each step only after the previous one succeeds. This does not resume the
lipid-library queue or its watchdog.

After any rollout, verify both `curl -I https://your-host.example.org` and
`curl --fail https://your-host.example.org/health/ready`, then check direct LAN access
when configured. Inspect active scientific workers before service changes.
