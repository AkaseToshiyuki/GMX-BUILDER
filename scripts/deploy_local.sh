#!/usr/bin/env bash
# Bring the local service up to the working tree and prove that it happened.
#
# The service runs an editable install, so the source needs no reinstallation --
# but a running process keeps the modules it imported at start-up. Without a
# restart the deployment silently serves whatever version was current when it
# last started, which is exactly how it came to be fourteen releases behind.
#
# The policy this enforces: the private repository and the local deployment
# always hold the same version. Public releases are separate and happen on the
# maintainer's command -- see scripts/prepare_public_release.py.
#
# Usage:
#     scripts/deploy_local.sh              restart and verify
#     scripts/deploy_local.sh --check      verify only, change nothing
#     scripts/deploy_local.sh --allow-dirty  deploy an uncommitted working tree
#
# Exit status is non-zero when the deployment does not match the repository, so
# this can gate a release step rather than merely report.
set -euo pipefail

SERVICE="${GMXBUILDER_SERVICE:-gmxbuilder.service}"
HEALTH_URL="${GMXBUILDER_HEALTH_URL:-http://127.0.0.1:7788/health}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check_only=0
allow_dirty=0
for argument in "$@"; do
    case "$argument" in
        --check) check_only=1 ;;
        --allow-dirty) allow_dirty=1 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done

source_version() {
    sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$REPO_ROOT/src/gmxbuilder/__version__.py"
}

live_version() {
    # Tolerate a dead service here: pipefail would otherwise let curl's exit
    # code escape through set -e, aborting with a bare status and none of the
    # explanation below.
    local body
    body="$(curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null || true)"
    sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$body"
}

expected="$(source_version)"
if [[ -z "$expected" ]]; then
    echo "Cannot read the source version from src/gmxbuilder/__version__.py" >&2
    exit 2
fi

# The editable install means the service runs the working tree, not the
# commit. Matching version strings therefore do not prove the deployment
# matches the repository: an uncommitted change is served while both report the
# same number. That is the same shape as the problem this script exists for,
# so it is checked rather than assumed.
dirty="$(git -C "$REPO_ROOT" status --porcelain -- src scripts 2>/dev/null || true)"
if [[ -n "$dirty" ]]; then
    if (( allow_dirty )); then
        echo "Warning: deploying an uncommitted working tree." >&2
        sed 's/^/  /' <<<"$dirty" >&2
    else
        echo "The working tree has uncommitted changes under src/ or scripts/:" >&2
        sed 's/^/  /' <<<"$dirty" >&2
        echo >&2
        echo "The service runs the working tree, so deploying now would serve code" >&2
        echo "that is not in the repository. Commit first, or pass --allow-dirty." >&2
        exit 1
    fi
fi

if (( ! check_only )); then
    worker_units="$(systemctl --user list-units --type=service --state=active,activating \
        'gmxbuilder-op-*.service' --no-legend --plain 2>/dev/null || true)"
    if [[ -n "$worker_units" ]]; then
        echo "Refusing to restart while isolated Web operations are active:" >&2
        echo "$worker_units" >&2
        exit 1
    fi
    # Refuse to interrupt work in progress. A restart drops in-flight builds,
    # and a build can represent an hour of GROMACS time.
    health="$(curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null || true)"
    if [[ -n "$health" ]]; then
        active="$(sed -n 's/.*"builds_active"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$health")"
        queued="$(sed -n 's/.*"builds_queued"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$health")"
        operations="$(sed -n 's/.*"operations_active"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$health")"
        operation_queue="$(sed -n 's/.*"operations_queued"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$health")"
        if [[ "${active:-0}" != "0" || "${queued:-0}" != "0" || "${operations:-0}" != "0" || "${operation_queue:-0}" != "0" ]]; then
            echo "Refusing to restart: ${active:-?} build(s) running, ${queued:-?} queued." >&2
            echo "Wait for them to finish, or restart deliberately with systemctl." >&2
            echo "Managed operations: ${operations:-0} running, ${operation_queue:-0} queued." >&2
            exit 1
        fi
    fi

    # The unit runs a generated launcher, and install-local.sh is what
    # generates it. A restart therefore picks up source changes but never
    # installer changes: the launcher on disk can predate settings the
    # installer now writes, which is how this deployment ended up without
    # GMXBUILDER_ALLOW_UNSAFE_DEPLOYMENT long after it was added. Report the
    # drift rather than silently restarting into it.
    runner="${GMXBUILDER_RUNNER:-$HOME/.config/gmxbuilder/run-local.sh}"
    if [[ -f "$runner" ]]; then
        missing=()
        while read -r setting; do
            [[ -n "$setting" ]] || continue
            grep -q "export $setting=" "$runner" || missing+=("$setting")
        done < <(
            # Only the settings this launcher always receives. The scrape is
            # scoped to the run-local.sh block, so the lipid-queue runner's
            # variables are not counted, and to two-space indentation, which
            # excludes the conditional block that writes extension settings
            # only when they are configured. Reporting either as missing would
            # fire on every healthy install, and a check that always fires is
            # one people learn to ignore.
            sed -n '/^RUNNER=/,/^} > "\$RUNNER"/p' "$REPO_ROOT/install-local.sh" |
                sed -n "s/^  printf 'export \([A-Z0-9_]*\)=%q.*/\1/p" | sort -u
        )
        if (( ${#missing[@]} )); then
            echo "Note: $runner predates the current installer and is missing:" >&2
            printf '  %s\n' "${missing[@]}" >&2
            echo "Re-run ./install-local.sh to regenerate it." >&2
        fi
    fi

    echo "Restarting $SERVICE ..."
    systemctl --user restart "$SERVICE"

    # Poll rather than sleep a fixed interval: start-up time varies with the
    # hardware probe, and a fixed wait is either too slow or occasionally short.
    for _ in $(seq 1 30); do
        [[ -n "$(live_version)" ]] && break
        sleep 0.5
    done
fi

actual="$(live_version)"
if [[ -z "$actual" ]]; then
    echo "The service is not answering at $HEALTH_URL" >&2
    exit 1
fi

if [[ "$actual" != "$expected" ]]; then
    echo "Version mismatch: source is $expected, the service is serving $actual" >&2
    echo "The service is running code other than this working tree." >&2
    exit 1
fi

if (( allow_dirty )) && [[ -n "$dirty" ]]; then
    echo "Deployed: $SERVICE is serving $actual from an uncommitted working tree."
else
    echo "Deployed: $SERVICE is serving $actual, matching the repository."
fi
