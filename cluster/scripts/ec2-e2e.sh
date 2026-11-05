#!/usr/bin/env bash
# Run the cluster end-to-end suite on a throwaway EC2 builder.
#
#   cluster/scripts/ec2-e2e.sh up        # launch (or reuse) the instance
#   cluster/scripts/ec2-e2e.sh sync      # rsync the repo (no target dirs)
#   cluster/scripts/ec2-e2e.sh run CMD   # run CMD in the repo on the box
#   cluster/scripts/ec2-e2e.sh cargo DIR ARGS  # sync, then `cargo ARGS` in repo/DIR (`.` = the one workspace)
#   cluster/scripts/ec2-e2e.sh ssh       # interactive shell
#   cluster/scripts/ec2-e2e.sh down      # terminate + delete key/SG
#
# Credentials come only from the standard AWS_* environment variables; the
# script never writes them anywhere. The instance gets docker, kind,
# kubectl, helm and a Rust toolchain via cloud-init. Everything it creates
# is tagged Project=es-rust-e2e so `down` can find and remove it.
set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
TYPE="${E2E_INSTANCE_TYPE:-c7g.8xlarge}"   # 32 vCPU / 64 GiB, arm64 (parallel agent builds)
DISK_GB="${E2E_DISK_GB:-150}"
TAG=es-rust-e2e
STATE_DIR="${E2E_STATE_DIR:-$HOME/.es-rust-e2e}"
KEY="$STATE_DIR/key.pem"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Separate checkouts (and target dirs) per agent: E2E_REMOTE_DIR=repo-<name>.
REMOTE_DIR="${E2E_REMOTE_DIR:-repo}"
aws_() { aws --region "$REGION" "$@"; }
mkdir -p "$STATE_DIR"

instance_id() {
  aws_ ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text | awk '{print $1}'
}
instance_ip() {
  aws_ ec2 describe-instances --instance-ids "$1" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

up() {
  local id; id="$(instance_id)"
  if [[ -z "$id" ]]; then
    if [[ ! -f "$KEY" ]]; then
      aws_ ec2 delete-key-pair --key-name "$TAG" >/dev/null 2>&1 || true
      aws_ ec2 create-key-pair --key-name "$TAG" --key-type ed25519 --query KeyMaterial --output text >"$KEY"
      chmod 600 "$KEY"
    fi
    local sg; sg="$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$TAG" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
    if [[ -z "$sg" || "$sg" == "None" ]]; then
      sg="$(aws_ ec2 create-security-group --group-name "$TAG" --description "es-rust e2e builder (ssh only)" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$TAG}]" --query GroupId --output text)"
    fi
    local myip; myip="$(curl -fsS https://checkip.amazonaws.com)"
    aws_ ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$myip/32" >/dev/null 2>&1 || true
    local ami; ami="$(aws_ ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)"
    id="$(aws_ ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --key-name "$TAG" --security-group-ids "$sg" \
      --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$TAG},{Key=Name,Value=$TAG}]" \
      --user-data "file://$(dirname "$0")/ec2-cloud-init.sh" --query 'Instances[0].InstanceId' --output text)"
    echo "launched $id ($TYPE)"
    aws_ ec2 wait instance-running --instance-ids "$id"
  fi
  local ip; ip="$(instance_ip "$id")"
  echo "$ip" >"$STATE_DIR/ip"
  echo "waiting for ssh + cloud-init on $ip ..."
  for _ in $(seq 1 90); do
    if ssh_ true 2>/dev/null && ssh_ 'test -f /var/lib/cloud/instance/boot-finished && test -f /opt/e2e-ready'; then
      echo "ready: $id $ip"; return 0
    fi
    sleep 10
  done
  echo "instance did not become ready" >&2; exit 1
}

ssh_() {
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10 \
    "ubuntu@$(cat "$STATE_DIR/ip")" "$@"
}

# Serialise concurrent syncs from several agents (macOS has no flock).
with_sync_lock() {
  local lock="$STATE_DIR/sync-$REMOTE_DIR.lock" i=0
  until mkdir "$lock" 2>/dev/null; do
    i=$((i + 1)); [[ $i -gt 600 ]] && rm -rf "$lock"; sleep 0.5
  done
  trap 'rm -rf "$lock"' RETURN
  "$@"
}

sync() { sync_tree; }

sync_tree() {
  rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    --exclude target/ --exclude 'target-*/' --exclude .git/ --exclude bench/tools/ --exclude bench/results/ --exclude bench/indexes/ --exclude bench/corpus/ --exclude "*.mov" \
    --exclude 'bench/**/data/' --exclude __pycache__/ --exclude .overlay_ref "$REPO_ROOT/" "ubuntu@$(cat "$STATE_DIR/ip"):$REMOTE_DIR/"
}

down() {
  local ids; ids="$(aws_ ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)"
  if [[ -n "$ids" ]]; then
    aws_ ec2 terminate-instances --instance-ids $ids >/dev/null
    aws_ ec2 wait instance-terminated --instance-ids $ids
    echo "terminated $ids"
  fi
  local sg; sg="$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$TAG" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  [[ -n "$sg" && "$sg" != "None" ]] && aws_ ec2 delete-security-group --group-id "$sg" && echo "deleted $sg"
  aws_ ec2 delete-key-pair --key-name "$TAG" >/dev/null 2>&1 || true
  rm -f "$KEY" "$STATE_DIR/ip"
}

case "${1:-}" in
  up) up ;;
  sync) sync ;;
  run) shift; ssh_ "source ~/.cargo/env; cd ~/$REMOTE_DIR && $*" ;;
  cargo)
    shift; dir="$1"; shift
    with_sync_lock sync
    # printf %q keeps argument boundaries across ssh.
    ssh_ "source ~/.cargo/env; cd ~/$REMOTE_DIR/$dir && cargo $(printf '%q ' "$@")" ;;
  ssh) ssh_ ;;
  down) down ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
