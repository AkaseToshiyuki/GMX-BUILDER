#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_HOST=127.0.0.1
DEFAULT_PORT=7788
INTERACTIVE=0
ROTATE_ADMIN_TOKEN=0
CLI_BIND_HOST=""
CLI_DEPLOYMENT_MODE=""
CLI_PORT=""
CLI_CPU_CORES=""
CLI_QUEUE_SLOTS=""
CLI_GMX_BIN=""
CLI_TASK_MEMORY_GIB=""
CLI_TASK_STORAGE_GIB=""
CLI_STORAGE_GIB=""
CLI_ALLOW_UNSAFE_DEPLOYMENT=""

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./install-local.sh [options]

With no options, installation is unattended and uses safe local defaults.
Environment variables with the corresponding GMXBUILDER_* names are also
accepted. Command-line options take precedence over environment variables.

  --bind-host ADDRESS       Listener address (default: 127.0.0.1)
  --deployment-mode MODE    local or trusted-lan
  --allow-unsafe-deployment Explicitly permit an unauthenticated non-loopback listener
  --port PORT               Web port (default: 7788)
  --cpu-cores COUNT         CPU cores exposed to GMXBUILDER (default: half)
  --queue-slots COUNT       Concurrent task slots (default: divisor near cores/4)
  --task-memory-gib GiB     Memory per operation, including children (default: 16)
  --task-storage-gib GiB    Persistent storage per task (default: 2)
  --storage-gib GiB         Total Web writable storage (default: 100)
  --gmx-bin PATH            GROMACS executable (default: GMX_BIN or PATH lookup)
  --rotate-admin-token      Explicitly rotate the existing admin token
  --interactive             Ask for each deployment value
  -h, --help                Show this help
EOF
}

while (($#)); do
  case "$1" in
    --rotate-admin-token) ROTATE_ADMIN_TOKEN=1; shift ;;
    --bind-host)
      (($# >= 2)) || fail "--bind-host requires a value."
      CLI_BIND_HOST="$2"
      shift 2
      ;;
    --deployment-mode)
      (($# >= 2)) || fail "--deployment-mode requires a value."
      CLI_DEPLOYMENT_MODE="$2"
      shift 2
      ;;
    --allow-unsafe-deployment)
      CLI_ALLOW_UNSAFE_DEPLOYMENT=1
      shift
      ;;
    --port)
      (($# >= 2)) || fail "--port requires a value."
      CLI_PORT="$2"
      shift 2
      ;;
    --cpu-cores)
      (($# >= 2)) || fail "--cpu-cores requires a value."
      CLI_CPU_CORES="$2"
      shift 2
      ;;
    --queue-slots)
      (($# >= 2)) || fail "--queue-slots requires a value."
      CLI_QUEUE_SLOTS="$2"
      shift 2
      ;;
    --gmx-bin)
      (($# >= 2)) || fail "--gmx-bin requires a value."
      CLI_GMX_BIN="$2"
      shift 2
      ;;
    --task-memory-gib)
      (($# >= 2)) || fail "--task-memory-gib requires a value."
      CLI_TASK_MEMORY_GIB="$2"
      shift 2
      ;;
    --task-storage-gib)
      (($# >= 2)) || fail "--task-storage-gib requires a value."
      CLI_TASK_STORAGE_GIB="$2"
      shift 2
      ;;
    --storage-gib)
      (($# >= 2)) || fail "--storage-gib requires a value."
      CLI_STORAGE_GIB="$2"
      shift 2
      ;;
    --interactive)
      INTERACTIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown installer option: $1"
      ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python 3 was not found."
"$PYTHON_BIN" - <<'PY' || fail "Python 3.10 or newer is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

AVAILABLE_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || printf '1')"
[[ "$AVAILABLE_CORES" =~ ^[1-9][0-9]*$ ]] || AVAILABLE_CORES=1

normalize_boolean() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off|'') printf '0\n' ;;
    *) return 1 ;;
  esac
}

bind_scope() {
  "$PYTHON_BIN" - "$1" <<'PY'
import ipaddress
import sys

address = ipaddress.ip_address(sys.argv[1])
if address.is_unspecified:
    print("wildcard")
elif address.is_loopback:
    print("loopback")
else:
    print("nonloopback")
PY
}

# Refuse an unsafe bind before installing anything. The authoritative check
# stays below with the rest of the deployment configuration, but repeating it
# here means a rejected address fails in about a second instead of after a
# full GROMACS build. Interactive runs are skipped: the operator has not been
# asked to confirm yet at this point.
if (( ! INTERACTIVE )); then
  EARLY_BIND_HOST="${CLI_BIND_HOST:-${GMXBUILDER_BIND_HOST:-$DEFAULT_HOST}}"
  EARLY_BIND_SCOPE="$(bind_scope "$EARLY_BIND_HOST")" || \
    fail "Bind address must be a valid IPv4 or IPv6 address."
  EARLY_ALLOW_UNSAFE="$(normalize_boolean "${CLI_ALLOW_UNSAFE_DEPLOYMENT:-${GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT:-0}}")" || \
    fail "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT must be 1/0, true/false, yes/no, or on/off."
  if [[ "$EARLY_BIND_SCOPE" != "loopback" && "$EARLY_ALLOW_UNSAFE" != "1" ]]; then
    fail "Binding $EARLY_BIND_HOST starts an unauthenticated listener that is reachable from outside this machine. Re-run with --allow-unsafe-deployment (or GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=1) to confirm, or keep the default 127.0.0.1."
  fi
fi

# Refuse before downloads, dependency changes, migrations or launcher writes.
command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 || \
  fail "Managed Web installation requires an active systemd user session."
command -v flock >/dev/null 2>&1 || fail "The installer requires flock."
INSTALLER_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/gmxbuilder"
mkdir -p "$INSTALLER_STATE"
chmod 700 "$INSTALLER_STATE"
exec 9>"${GMXBUILDER_INSTALL_LOCK:-$INSTALLER_STATE/install.lock}"
flock -n 9 || fail "An installer or HTTP submission currently holds the installation lock."
"$PYTHON_BIN" "$ROOT_DIR/scripts/installer_state.py" idle || fail "Installation would interrupt Web work."

gmx_version() {
  "$1" --version 2>&1 | sed -n 's/^GROMACS version:[[:space:]]*//p' | head -n 1
}

gmx_is_compatible() {
  local version
  [[ -n "$1" && -x "$1" ]] || return 1
  version="$(gmx_version "$1")"
  "$PYTHON_BIN" - "$version" <<'PY'
import re
import sys

match = re.search(r"\d+(?:\.\d+)+", sys.argv[1])
if not match:
    raise SystemExit(1)
parts = tuple(int(part) for part in match.group(0).split("."))
raise SystemExit(0 if parts >= (2026, 0) else 1)
PY
}

REQUESTED_GMX_BIN="${CLI_GMX_BIN:-${GMX_BIN:-}}"
if [[ -n "$REQUESTED_GMX_BIN" ]]; then
  gmx_is_compatible "$REQUESTED_GMX_BIN" || \
    fail "The explicitly selected GROMACS must be version 2026.0 or newer."
  GMX_BIN="$REQUESTED_GMX_BIN"
else
  GMX_BIN="$(command -v gmx || true)"
  if ! gmx_is_compatible "$GMX_BIN"; then
    printf '\nInstalling the required GROMACS runtime from the verified official source\n'
    BUILD_JOBS=$((AVAILABLE_CORES / 2))
    (( BUILD_JOBS >= 1 )) || BUILD_JOBS=1
    GROMACS_ARGS=(
      --target-root "${GMXBUILDER_RUNTIME_DIR:-$HOME/.local/share/gmxbuilder/runtime}"
      --jobs "$BUILD_JOBS"
    )
    if [[ "${GMXBUILDER_GROMACS_FORCE_CPU:-0}" == "1" ]]; then
      GROMACS_ARGS+=(--force-cpu)
    fi
    "$PYTHON_BIN" "$ROOT_DIR/scripts/install_gromacs.py" "${GROMACS_ARGS[@]}" || \
      fail "The required GROMACS runtime could not be installed automatically."
    GROMACS_VERSION="$("$PYTHON_BIN" "$ROOT_DIR/scripts/install_gromacs.py" --version)"
    GMX_BIN="${GMXBUILDER_RUNTIME_DIR:-$HOME/.local/share/gmxbuilder/runtime}/gromacs-$GROMACS_VERSION/bin/gmx"
  fi
fi
gmx_is_compatible "$GMX_BIN" || \
  fail "GROMACS 2026.0 or newer is required by the bundled Amber ff14SB port."
GMX_VERSION="$(gmx_version "$GMX_BIN")"
GMX_BIN="$(cd -- "$(dirname -- "$GMX_BIN")" && pwd)/$(basename -- "$GMX_BIN")"
printf 'Using GROMACS %s at %s\n' "$GMX_VERSION" "$GMX_BIN"

GAFF_ENV="${GMXBUILDER_GAFF_ENV:-$HOME/.local/share/gmxbuilder/gaff-env-24.8-lock1}"

# Optional deployment extensions, carried through from the environment so a
# private feature module survives a reinstall instead of having to be re-added
# to the generated runner by hand. Empty in a public installation, where no
# extension exists; the service behaves identically either way.
EXTENSIONS="${GMXBUILDER_EXTENSIONS:-}"
EXTENSION_PATH="${GMXBUILDER_EXTENSION_PATH:-}"
# A module that lives beside the application rather than being installed needs
# its parent directory on the import path. Default to this checkout, which is
# where a private module sits.
if [[ -n "$EXTENSIONS" && -z "$EXTENSION_PATH" ]]; then
  EXTENSION_PATH="$ROOT_DIR"
fi
printf '\nInstalling the GAFF2/AM1-BCC runtime from conda-forge\n'
"$PYTHON_BIN" "$ROOT_DIR/scripts/install_gaff_runtime.py" --prefix "$GAFF_ENV" || \
  fail "The GAFF2/AM1-BCC runtime could not be installed automatically."

printf '\nInstalling separately distributed force-field assets from official sources\n'
"$PYTHON_BIN" "$ROOT_DIR/scripts/install_external_assets.py" \
  --target "$ROOT_DIR/src/gmxbuilder/data/forcefields" || \
  fail "Required external force-field assets could not be installed."

printf '\nHydrating the verified prebuilt lipid library\n'
"$PYTHON_BIN" "$ROOT_DIR/scripts/fetch_prebuilt_assets.py" || \
  fail "The prebuilt lipid library could not be downloaded or verified."

DEFAULT_CPU_CORES=$((AVAILABLE_CORES / 2))
(( DEFAULT_CPU_CORES >= 1 )) || DEFAULT_CPU_CORES=1

choose_default_slots() {
  # Half the allocated cores, not a quarter. Per-task threads are now decided
  # when a task starts rather than fixed here, so a higher ceiling no longer
  # means every task is permanently squeezed: one task on an idle machine
  # still gets the whole allocation.
  local cores="$1"
  local target=$((cores / 2))
  (( target >= 1 )) || target=1
  local candidate
  for ((candidate=target; candidate>=1; candidate--)); do
    if (( cores % candidate == 0 )); then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '1\n'
}

prompt_value() {
  local label="$1"
  local default_value="$2"
  local value=""
  if (( INTERACTIVE )) && [[ -t 0 ]]; then
    read -r -p "$label [$default_value]: " value
  fi
  printf '%s\n' "${value:-$default_value}"
}

prompt_boolean() {
  local label="$1"
  local default_value="$2"
  local default_label=no
  local value=""
  [[ "$default_value" == "1" ]] && default_label=yes
  if (( INTERACTIVE )) && [[ -t 0 ]]; then
    read -r -p "$label (yes/no) [$default_label]: " value
  fi
  if [[ -z "$value" ]]; then
    printf '%s\n' "$default_value"
  else
    normalize_boolean "$value"
  fi
}

BIND_HOST="$(prompt_value 'Bind IP address' "${CLI_BIND_HOST:-${GMXBUILDER_BIND_HOST:-$DEFAULT_HOST}}")"
BIND_SCOPE="$(bind_scope "$BIND_HOST")" || \
  fail "Bind address must be a valid IPv4 or IPv6 address."
DEFAULT_DEPLOYMENT_MODE=local
if [[ "$BIND_SCOPE" != "loopback" ]]; then
  DEFAULT_DEPLOYMENT_MODE=trusted-lan
fi
DEPLOYMENT_MODE="$(prompt_value 'Deployment mode (local or trusted-lan)' "${CLI_DEPLOYMENT_MODE:-${GMXBUILDER_DEPLOYMENT_MODE:-$DEFAULT_DEPLOYMENT_MODE}}")"
RAW_ALLOW_UNSAFE="${CLI_ALLOW_UNSAFE_DEPLOYMENT:-${GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT:-0}}"
DEFAULT_ALLOW_UNSAFE="$(normalize_boolean "$RAW_ALLOW_UNSAFE")" || \
  fail "GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT must be 1/0, true/false, yes/no, or on/off."
ALLOW_UNSAFE_DEPLOYMENT="$(prompt_boolean 'Allow unsafe deployment without end-user authentication' "$DEFAULT_ALLOW_UNSAFE")" || \
  fail "Unsafe-deployment selection must be yes or no."
PORT="$(prompt_value 'Web port' "${CLI_PORT:-${GMXBUILDER_PORT:-$DEFAULT_PORT}}")"
CPU_CORES="$(prompt_value 'CPU cores exposed to GMXBUILDER' "${CLI_CPU_CORES:-${GMXBUILDER_CPU_CORES:-$DEFAULT_CPU_CORES}}")"
[[ "$CPU_CORES" =~ ^[1-9][0-9]*$ ]] || fail "CPU core count must be a positive integer."
(( CPU_CORES <= AVAILABLE_CORES )) || fail "Only $AVAILABLE_CORES CPU cores are available."
DEFAULT_QUEUE_SLOTS="$(choose_default_slots "$CPU_CORES")"
QUEUE_SLOTS="$(prompt_value 'Concurrent task slots' "${CLI_QUEUE_SLOTS:-${GMXBUILDER_MAX_BUILDS:-$DEFAULT_QUEUE_SLOTS}}")"
TASK_MEMORY_GIB="$(prompt_value 'Memory per operation (GiB)' "${CLI_TASK_MEMORY_GIB:-${GMXBUILDER_TASK_MEMORY_GIB:-16}}")"
TASK_STORAGE_GIB="$(prompt_value 'Persistent storage per task (GiB)' "${CLI_TASK_STORAGE_GIB:-${GMXBUILDER_TASK_MAX_PERSISTENT_GIB:-2}}")"
STORAGE_GIB="$(prompt_value 'Total Web writable storage (GiB)' "${CLI_STORAGE_GIB:-${GMXBUILDER_STORAGE_MAX_GIB:-100}}")"
export GMXBUILDER_TASK_MEMORY_GIB="$TASK_MEMORY_GIB"
export GMXBUILDER_TASK_MAX_PERSISTENT_GIB="$TASK_STORAGE_GIB"
export GMXBUILDER_STORAGE_MAX_GIB="$STORAGE_GIB"
export GMXBUILDER_TASK_TTL_HOURS=24
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
  'from gmxbuilder.web.resource_policy import ResourcePolicy; ResourcePolicy.from_environment()' || \
  fail "Invalid resource limits. Values must be positive; task storage must be below total storage."
printf 'Resource policy: %s GiB memory/operation, %s GiB/task, %s GiB total; 24 hours from creation; no queue-count limit.\n' \
  "$TASK_MEMORY_GIB" "$TASK_STORAGE_GIB" "$STORAGE_GIB"

[[ "$DEPLOYMENT_MODE" == "local" || "$DEPLOYMENT_MODE" == "trusted-lan" ]] || \
  fail "The local installer supports local or trusted-lan mode. Use the documented TLS reverse-proxy deployment for public mode."
# The installer only configures local and trusted-lan mode, neither of which
# sets up end-user authentication, so any non-loopback listener it starts is
# unauthenticated. The opt-in is therefore required in both modes; trusted-lan
# is not a way around it, and a wildcard address is not consent to itself.
if [[ "$BIND_SCOPE" != "loopback" && "$ALLOW_UNSAFE_DEPLOYMENT" != "1" ]]; then
  fail "Binding $BIND_HOST starts an unauthenticated listener that is reachable from outside this machine. Re-run with --allow-unsafe-deployment (or GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=1) to confirm, or keep the default 127.0.0.1."
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || fail "Port must be 1-65535."
[[ "$QUEUE_SLOTS" =~ ^[1-9][0-9]*$ ]] || fail "Concurrent task slots must be a positive integer."
(( QUEUE_SLOTS <= CPU_CORES )) || fail "Concurrent task slots cannot exceed allocated CPU cores."
(( CPU_CORES % QUEUE_SLOTS == 0 )) || fail "Concurrent task slots must divide allocated CPU cores exactly."
TASK_THREADS=$((CPU_CORES / QUEUE_SLOTS))

VENV_DIR="$ROOT_DIR/.venv"
printf '\nCreating/updating Python environment at %s\n' "$VENV_DIR"
UV_BIN="$(command -v uv || true)"
BOOTSTRAP_DIR=""
cleanup_bootstrap() {
  if [[ -n "$BOOTSTRAP_DIR" ]]; then
    rm -rf "$BOOTSTRAP_DIR"
  fi
}
trap cleanup_bootstrap EXIT
if [[ -z "$UV_BIN" ]]; then
  BOOTSTRAP_DIR="$(mktemp -d "$ROOT_DIR/.gmxbuilder-uv-bootstrap.XXXXXX")"
  rm -rf "$BOOTSTRAP_DIR"
  "$PYTHON_BIN" -m venv "$BOOTSTRAP_DIR" || \
    fail "Python's venv module is required to bootstrap the locked installer."
  "$BOOTSTRAP_DIR/bin/python" -m pip install \
    --disable-pip-version-check --no-input --only-binary=:all: --require-hashes \
    -r "$ROOT_DIR/scripts/uv-bootstrap.txt" || \
    fail "The locked uv installer could not be bootstrapped from PyPI."
  UV_BIN="$BOOTSTRAP_DIR/bin/uv"
fi
pkg-config --exists fuse3 || fail "Managed Web storage requires FUSE3 development headers. On Ubuntu install libfuse3-dev pkg-config, then rerun this installer."
"$UV_BIN" sync --project "$ROOT_DIR" --frozen --no-dev --extra managed --python "$PYTHON_BIN"
if [[ -n "$BOOTSTRAP_DIR" ]]; then
  rm -rf "$BOOTSTRAP_DIR"
  BOOTSTRAP_DIR=""
fi

"$VENV_DIR/bin/gmxbuilder" prebuilt-assets install

CONFIG_DIR="$HOME/.config/gmxbuilder"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gmxbuilder"
LEGACY_TASK_DIR="${GMXBUILDER_TASK_DIR:-$HOME/.local/share/gmxbuilder/tasks}"
RESOURCE_DIR="${GMXBUILDER_STORAGE_DIR:-$HOME/.local/share/gmxbuilder/web-storage}"
MANAGED_ROOT="$RESOURCE_DIR/mounted"
TASK_DIR="$MANAGED_ROOT/tasks"
"$VENV_DIR/bin/python" -m gmxbuilder.web.resource_setup --root "$RESOURCE_DIR" --install
if [[ -d "$LEGACY_TASK_DIR" && "$LEGACY_TASK_DIR" != "$TASK_DIR" ]]; then
  "$VENV_DIR/bin/python" -m gmxbuilder.web.resource_setup --root "$RESOURCE_DIR" \
    --source-tasks "$LEGACY_TASK_DIR" --inventory
  "$VENV_DIR/bin/python" -m gmxbuilder.web.resource_setup --root "$RESOURCE_DIR" \
    --source-tasks "$LEGACY_TASK_DIR" --migrate
fi
mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$TASK_DIR" "$HOME/.config/systemd/user"
chmod 700 "$CONFIG_DIR" "$STATE_DIR" "$TASK_DIR"

TOKEN_ARGS=(token --config "$CONFIG_DIR")
(( ROTATE_ADMIN_TOKEN )) && TOKEN_ARGS+=(--rotate)
ADMIN_TOKEN="$("$PYTHON_BIN" "$ROOT_DIR/scripts/installer_state.py" "${TOKEN_ARGS[@]}")"
RUNNER="$CONFIG_DIR/run-local.sh"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'export GMXBUILDER_TASK_DIR=%q\n' "$TASK_DIR"
  printf 'export GMXBUILDER_TASK_TTL_HOURS=%q\n' "24"
  printf 'export GMXBUILDER_TASK_MEMORY_GIB=%q\n' "$TASK_MEMORY_GIB"
  printf 'export GMXBUILDER_TASK_MAX_PERSISTENT_GIB=%q\n' "$TASK_STORAGE_GIB"
  printf 'export GMXBUILDER_STORAGE_MAX_GIB=%q\n' "$STORAGE_GIB"
  printf 'export GMXBUILDER_MANAGED_ROOT=%q\n' "$MANAGED_ROOT"
  printf 'export GMXBUILDER_SERVICE_UNIT=gmxbuilder.service\n'
  printf 'export PYTHONDONTWRITEBYTECODE=1\n'
  printf 'export GMXBUILDER_GPU_COUNT=0\n'
  printf 'export GMXBUILDER_PREBUILT_AUTO_INSTALL=0\n'
  printf 'export GMXBUILDER_LIPID_LIBRARY=%q\n' "$HOME/.cache/gmxbuilder/lipid_equilibrated_v4/library"
  printf 'export TMPDIR=%q\n' "$MANAGED_ROOT/tmp"
  printf 'export XDG_CACHE_HOME=%q\n' "$MANAGED_ROOT/cache"
  printf 'export GMXBUILDER_ADMIN_TOKEN=%q\n' "$ADMIN_TOKEN"
  printf 'export GMXBUILDER_DEPLOYMENT_MODE=%q\n' "$DEPLOYMENT_MODE"
  printf 'export GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT=%q\n' "$ALLOW_UNSAFE_DEPLOYMENT"
  printf 'export GMX_BIN=%q\n' "$GMX_BIN"
  printf 'export GMXBUILDER_GAFF_ENV=%q\n' "$GAFF_ENV"
  if [[ -n "$EXTENSIONS" ]]; then
    printf 'export GMXBUILDER_EXTENSIONS=%q\n' "$EXTENSIONS"
    printf 'export PYTHONPATH=%q\n' "$EXTENSION_PATH"
  fi
  printf 'exec %q serve --host %q --port %q --cpu-cores %q --task-threads %q --max-builds %q\n' \
    "$VENV_DIR/bin/gmxbuilder" "$BIND_HOST" "$PORT" "$CPU_CORES" "$TASK_THREADS" "$QUEUE_SLOTS"
} > "$RUNNER"
chmod 700 "$RUNNER"

SERVICE_FILE="$HOME/.config/systemd/user/gmxbuilder.service"
{
  printf '[Unit]\nDescription=GMXBUILDER local web service\nAfter=network.target gmxbuilder-storage.service\nBindsTo=gmxbuilder-storage.service\n\n'
  printf '[Service]\nType=simple\nExecStart=/usr/bin/env bash %%h/.config/gmxbuilder/run-local.sh\n'
  printf 'Restart=on-failure\nRestartSec=3\nUMask=0077\n'
  printf 'MemoryMax=%s\nMemorySwapMax=0\n' "$("$VENV_DIR/bin/python" -c 'from gmxbuilder.web.resource_policy import ResourcePolicy; print(ResourcePolicy.from_environment().task_memory_bytes)')"
  printf 'NoNewPrivileges=true\nPrivateTmp=true\nProtectSystem=strict\nProtectHome=read-only\n'
  printf 'ReadWritePaths=%s %s %s %s\n' "$TASK_DIR" "$STATE_DIR" "$HOME/.cache/gmxbuilder" "$HOME/.local/share/gmxbuilder"
  printf 'RestrictSUIDSGID=true\nLockPersonality=true\n'
  printf 'RestrictRealtime=true\nSystemCallArchitectures=native\n'
  printf 'SystemCallFilter=~@clock @cpu-emulation @debug @module @mount @obsolete @privileged @raw-io @reboot @swap\n'
  printf 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK\n'
  printf '\n[Install]\nWantedBy=default.target\n'
} > "$SERVICE_FILE"

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/installer_state.py" idle || fail "Web work started during installation; restart deferred."
  systemctl --user daemon-reload
  systemctl --user enable --now gmxbuilder.service
  # V3's one-replica hourly queue must never repopulate the V4 publication root.
  if systemctl --user cat gmxbuilder-lipid-library-watchdog.timer >/dev/null 2>&1; then
    systemctl --user disable --now gmxbuilder-lipid-library-watchdog.timer
  fi
  if systemctl --user cat gmxbuilder-lipid-library.service >/dev/null 2>&1; then
    systemctl --user disable gmxbuilder-lipid-library.service
  fi
  systemctl --user restart gmxbuilder.service
  printf '\nGMXBUILDER is running as the user service gmxbuilder.service.\n'
  printf 'Status: systemctl --user status gmxbuilder.service\n'
else
  LOG_FILE="$STATE_DIR/server.log"
  nohup "$RUNNER" > "$LOG_FILE" 2>&1 &
  printf '%s\n' "$!" > "$STATE_DIR/server.pid"
  printf '\nA user systemd session was unavailable; GMXBUILDER was started in the background.\n'
  printf 'Log: %s\n' "$LOG_FILE"
fi

DISPLAY_HOST="$BIND_HOST"
if [[ "$BIND_HOST" == "0.0.0.0" || "$BIND_HOST" == "::" ]]; then
  DISPLAY_HOST=localhost
fi
if [[ "$DISPLAY_HOST" == *:* ]]; then
  DISPLAY_HOST="[$DISPLAY_HOST]"
fi
printf 'URL: http://%s:%s/\n' "$DISPLAY_HOST" "$PORT"
printf 'Resources: %s/%s CPU cores, %s task threads, %s concurrent task slots.\n' \
  "$CPU_CORES" "$AVAILABLE_CORES" "$TASK_THREADS" "$QUEUE_SLOTS"
if [[ "$DEPLOYMENT_MODE" == "trusted-lan" ]]; then
  printf 'Security notice: trusted-lan mode has no end-user login. Keep it behind a private-network firewall and never expose it to the Internet.\n'
fi
if [[ "$ALLOW_UNSAFE_DEPLOYMENT" == "1" ]]; then
  printf 'WARNING: unsafe deployment was explicitly enabled and persisted in %s. This service may have no end-user authentication; never expose it to the Internet.\n' "$RUNNER"
fi
