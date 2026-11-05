#!/bin/bash
# Unit tests (logs kept), single-node YAML suites, two-node fastcheck.
cd ~/esql-poc && source ~/.cargo/env
out=~/esql-verify.txt; : > $out
for c in rustic-core elasticsearch-engine elasticsearch-node elasticsearch-http; do
  taskset -c 0-15 cargo test --profile fast -j 16 -p $c --lib > ~/esql-unit-$c.log 2>&1
  echo "$c: $(grep -E "test result|^test .*FAILED" ~/esql-unit-$c.log | head -5)" >> $out
done
taskset -c 0-15 cargo build --profile fast -j 16 -p es-yaml-runner -p es-cluster-yaml -p elasticsearch > ~/esql-ybuild.log 2>&1 || echo "yaml build failed" >> $out
for s in oss main plug; do x=""; [ $s = main ] && x=--security
  taskset -c 0-15 target/fast/es-yaml-runner --suite ~/es955-yaml/$s --jobs 12 $x > ~/esql-yaml-$s.log 2>&1
  echo "$s: $(grep "^tests:" ~/esql-yaml-$s.log)" >> $out
done
taskset -c 0-15 python3 bench/path-poc/es9-parity/fastcheck.py --root ~/esql-poc --tag=-eq --http-base 36000 --all --restart --suite main esql/ spatial/ vector-tile/ > ~/esql-fastcheck.log 2>&1
echo "fastcheck: $(grep "pass,.*fail" ~/esql-fastcheck.log | tail -n 1)" >> $out
touch ~/esql-verify.done
