#!/bin/bash
# cloud-init user data for the e2e builder (Ubuntu 24.04 arm64).
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io build-essential pkg-config libssl-dev curl jq rsync git clang
usermod -aG docker ubuntu
systemctl enable --now docker
# kind needs these for many pods per node
sysctl -w fs.inotify.max_user_watches=524288 fs.inotify.max_user_instances=512 vm.max_map_count=262144
ARCH=arm64
curl -fsSLo /usr/local/bin/kind "https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-$ARCH" && chmod +x /usr/local/bin/kind
curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/v1.31.1/bin/linux/$ARCH/kubectl" && chmod +x /usr/local/bin/kubectl
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
sudo -u ubuntu bash -c 'curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.87.0 --profile minimal'
touch /opt/e2e-ready
