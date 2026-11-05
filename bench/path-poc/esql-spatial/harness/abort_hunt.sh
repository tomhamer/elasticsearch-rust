#!/bin/bash
# abort_hunt.sh BIN N CRATE_DIR TAG: run a test binary N times; keep the output of abnormal exits.
bin=$1; n=${2:-30}; dir=${3:-$HOME/esql-poc/crates/rustic-core}; tag=${4:-ah}
cd $dir
for i in $(seq 1 $n); do
  taskset -c 0-15 $bin > /tmp/esql-$tag-$i.log 2>&1; rc=$?
  if [ $rc -ne 0 ]; then echo "run $i rc=$rc" >> /tmp/esql-$tag.txt; else rm /tmp/esql-$tag-$i.log; fi
done
touch /tmp/esql-$tag.done
