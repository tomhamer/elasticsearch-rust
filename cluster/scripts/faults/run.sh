#!/usr/bin/env bash
# Jepsen-style fault suite for the cluster-aware `elasticsearch` binary.
# Runs ON the Linux test box (needs passwordless sudo, iptables, cgroup v2).
#
#   cluster/scripts/faults/run.sh                       # short suite: every scenario, seed 1, 5 nodes
#   cluster/scripts/faults/run.sh --nodes 3             # same on 3 nodes
#   cluster/scripts/faults/run.sh --mode long --seeds 5 # long suite (ES default fault detection)
#   cluster/scripts/faults/run.sh --scenario pause-primary --seed 7
#   cluster/scripts/faults/run.sh --mode long --repeat-until-fail   # soak
#   cluster/scripts/faults/run.sh list | cleanup
#
# From the laptop:
#   E2E_REMOTE_DIR=repo-faults cluster/scripts/ec2-e2e.sh sync
#   E2E_REMOTE_DIR=repo-faults cluster/scripts/ec2-e2e.sh run 'cluster/scripts/faults/run.sh'
#
# Env: PROFILE=debug|release (default release), BIN=<node binary> (default:
# built from this checkout), PORT_BASE (default 9600), SKIP_BUILD=1.
# Artifacts (histories, nemesis logs, node logs, verdicts, REPRO.md on
# failure) go to /tmp/es-faults-$PORT_BASE unless --out is given.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ws="$(cd "$here/../../.." && pwd)"   # repository root (one workspace)
PROFILE="${PROFILE:-release}"
PORT_BASE="${PORT_BASE:-9600}"
cd "$ws"
flag=(); [[ "$PROFILE" == release ]] && flag=(--release)
if [[ -z "${SKIP_BUILD:-}" ]]; then
  if [[ -z "${BIN:-}" ]]; then
    cargo build "${flag[@]}" -p elasticsearch-faults -p elasticsearch
  else
    cargo build "${flag[@]}" -p elasticsearch-faults
  fi
fi
BIN="${BIN:-$ws/target/$PROFILE/elasticsearch}"
harness="$ws/target/$PROFILE/elasticsearch-faults"
cleanup() { "$harness" cleanup --port-base "$PORT_BASE" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
case "${1:-run}" in
  list) exec "$harness" list ;;
  cleanup) cleanup; echo cleaned ;;
  *)
    [[ "${1:-}" == run ]] && shift
    "$harness" run --bin "$BIN" --port-base "$PORT_BASE" "$@" ;;
esac
