#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_HOST=127.0.0.1
DEFAULT_PORT=7788

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python 3 was not found."
"$PYTHON_BIN" - <<'PY' || fail "Python 3.10 or newer is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

AVAILABLE_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || printf '1')"
[[ "$AVAILABLE_CORES" =~ ^[1-9][0-9]*$ ]] || AVAILABLE_CORES=1
DEFAULT_CPU_CORES=$((AVAILABLE_CORES / 2))
(( DEFAULT_CPU_CORES >= 1 )) || DEFAULT_CPU_CORES=1

choose_default_slots() {
  local cores="$1"
  local target=$((cores / 4))
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
  if [[ -t 0 ]]; then
    read -r -p "$label [$default_value]: " value
  fi
  printf '%s\n' "${value:-$default_value}"
}

BIND_HOST="$(prompt_value 'Bind IP address' "${GMXBUILDER_BIND_HOST:-$DEFAULT_HOST}")"
DEFAULT_DEPLOYMENT_MODE=local
if [[ "$BIND_HOST" != "127.0.0.1" && "$BIND_HOST" != "::1" ]]; then
  DEFAULT_DEPLOYMENT_MODE=trusted-lan
fi
DEPLOYMENT_MODE="$(prompt_value 'Deployment mode (local or trusted-lan)' "${GMXBUILDER_DEPLOYMENT_MODE:-$DEFAULT_DEPLOYMENT_MODE}")"
PORT="$(prompt_value 'Web port' "${GMXBUILDER_PORT:-$DEFAULT_PORT}")"
CPU_CORES="$(prompt_value 'CPU cores exposed to GMXBUILDER' "${GMXBUILDER_CPU_CORES:-$DEFAULT_CPU_CORES}")"
DEFAULT_QUEUE_SLOTS="$(choose_default_slots "$CPU_CORES")"
QUEUE_SLOTS="$(prompt_value 'Concurrent task slots' "${GMXBUILDER_MAX_BUILDS:-$DEFAULT_QUEUE_SLOTS}")"

"$PYTHON_BIN" - "$BIND_HOST" <<'PY' || fail "Bind address must be a valid IPv4 or IPv6 address."
import ipaddress
import sys
ipaddress.ip_address(sys.argv[1])
PY
[[ "$DEPLOYMENT_MODE" == "local" || "$DEPLOYMENT_MODE" == "trusted-lan" ]] || \
  fail "The local installer supports local or trusted-lan mode. Use the documented TLS reverse-proxy deployment for public mode."
if [[ "$DEPLOYMENT_MODE" == "local" && "$BIND_HOST" != "127.0.0.1" && "$BIND_HOST" != "::1" ]]; then
  fail "Local mode is loopback-only. Select trusted-lan for an explicitly firewalled private network."
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || fail "Port must be 1-65535."
[[ "$CPU_CORES" =~ ^[1-9][0-9]*$ ]] || fail "CPU core count must be a positive integer."
(( CPU_CORES <= AVAILABLE_CORES )) || fail "Only $AVAILABLE_CORES CPU cores are available."
[[ "$QUEUE_SLOTS" =~ ^[1-9][0-9]*$ ]] || fail "Concurrent task slots must be a positive integer."
(( QUEUE_SLOTS <= CPU_CORES )) || fail "Concurrent task slots cannot exceed allocated CPU cores."
(( CPU_CORES % QUEUE_SLOTS == 0 )) || fail "Concurrent task slots must divide allocated CPU cores exactly."
TASK_THREADS=$((CPU_CORES / QUEUE_SLOTS))

VENV_DIR="$ROOT_DIR/.venv"
printf '\nCreating/updating Python environment at %s\n' "$VENV_DIR"
if command -v uv >/dev/null 2>&1; then
  uv sync --project "$ROOT_DIR" --frozen --no-dev --python "$PYTHON_BIN"
else
  printf 'Warning: uv is unavailable; falling back to an unlocked pip install.\n' >&2
  printf 'Install uv and rerun this installer to reproduce the checked uv.lock environment exactly.\n' >&2
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR"
fi

if command -v git >/dev/null 2>&1 && git -C "$ROOT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  if git lfs version >/dev/null 2>&1; then
    git -C "$ROOT_DIR" lfs pull
  else
    printf 'Warning: Git LFS is not installed; large prebuilt assets may be unavailable.\n' >&2
  fi
fi

"$VENV_DIR/bin/gmxbuilder" prebuilt-assets install

CONFIG_DIR="$HOME/.config/gmxbuilder"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gmxbuilder"
TASK_DIR="${GMXBUILDER_TASK_DIR:-$HOME/.local/share/gmxbuilder/tasks}"
mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$TASK_DIR" "$HOME/.config/systemd/user"
chmod 700 "$CONFIG_DIR" "$STATE_DIR" "$TASK_DIR"

ADMIN_TOKEN="$($VENV_DIR/bin/python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
RUNNER="$CONFIG_DIR/run-local.sh"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
  printf 'export GMXBUILDER_TASK_DIR=%q\n' "$TASK_DIR"
  printf 'export GMXBUILDER_TASK_TTL_HOURS=%q\n' "168"
  printf 'export GMXBUILDER_ADMIN_TOKEN=%q\n' "$ADMIN_TOKEN"
  printf 'export GMXBUILDER_DEPLOYMENT_MODE=%q\n' "$DEPLOYMENT_MODE"
  printf 'exec %q serve --host %q --port %q --cpu-cores %q --task-threads %q --max-builds %q\n' \
    "$VENV_DIR/bin/gmxbuilder" "$BIND_HOST" "$PORT" "$CPU_CORES" "$TASK_THREADS" "$QUEUE_SLOTS"
} > "$RUNNER"
chmod 700 "$RUNNER"

SERVICE_FILE="$HOME/.config/systemd/user/gmxbuilder.service"
{
  printf '[Unit]\nDescription=GMXBUILDER local web service\nAfter=network.target\n\n'
  printf '[Service]\nType=simple\nExecStart=/usr/bin/env bash %%h/.config/gmxbuilder/run-local.sh\n'
  printf 'Restart=on-failure\nRestartSec=3\nUMask=0077\n'
  printf 'NoNewPrivileges=true\nPrivateTmp=true\nProtectSystem=strict\nProtectHome=read-only\n'
  printf 'ReadWritePaths=%s %s %s %s\n' "$TASK_DIR" "$STATE_DIR" "$HOME/.cache/gmxbuilder" "$HOME/.local/share/gmxbuilder"
  printf 'RestrictSUIDSGID=true\nLockPersonality=true\n'
  printf 'RestrictRealtime=true\nSystemCallArchitectures=native\n'
  printf 'SystemCallFilter=~@clock @cpu-emulation @debug @module @mount @obsolete @privileged @raw-io @reboot @swap\n'
  printf 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK\n'
  printf '\n[Install]\nWantedBy=default.target\n'
} > "$SERVICE_FILE"

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user daemon-reload
  systemctl --user enable --now gmxbuilder.service
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
