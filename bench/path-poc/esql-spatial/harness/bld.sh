#!/bin/bash
# bld.sh PROFILE OUTNAME: build the branch server and copy it out (forces the uplift).
set -e
cd ~/esql-poc && source ~/.cargo/env
touch src/main.rs
taskset -c 0-15 cargo build --profile $1 -j 16 --bin elasticsearch 2>&1 | grep -E "^error" -A8 | head -40
cp target/$1/elasticsearch ~/esql-bins/$2
ls -la ~/esql-bins/$2
