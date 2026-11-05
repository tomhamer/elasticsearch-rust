#!/usr/bin/env bash
# Type-check the workspace for Windows from a Linux build host (no Windows
# machine needed). Run it on the EC2 build box, not a laptop.
#
# One-time setup (Ubuntu):
#   sudo apt-get install -y mingw-w64 clang-19 lld-19 llvm-19
#   rustup target add x86_64-pc-windows-gnu x86_64-pc-windows-msvc
#   cargo install --locked cargo-xwin
#
# MSVC headers need clang >= 19, hence the PATH below. bench/path-poc and
# the benches are left out: they do not build on Linux either.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/usr/lib/llvm-19/bin:$PATH
args=(--workspace --exclude es-path-poc --lib --bins --tests --examples)
cargo check "${args[@]}" --target x86_64-pc-windows-gnu
XWIN_ACCEPT_LICENSE=1 cargo xwin check "${args[@]}" --target x86_64-pc-windows-msvc
echo "Windows (gnu + msvc) check passed"
