#!/bin/bash
# 1% ES|QL comparison and latency: ES9 (19201), branch release (19203), master release (19202).
T=$HOME/esql-tools; S=$HOME/esql-poc/bench/path-poc/esql-spatial
flock ~/.bench.lock bash -c "
  $T/es9.sh 19201 19301 > /dev/null
  $T/rs.sh \$HOME/esql-bins/poc-release 19203 19303 \$HOME/esql-poc-data > /dev/null
  $T/rs.sh \$HOME/esql-bins/master-release 19202 19302 \$HOME/esql-master-data > /dev/null
  sleep 5
  for p in 19201 19202 19203; do curl -s localhost:\$p/_cat/count/osmgeopoints; done
  taskset -c 14,15 python3 $S/compare.py --a http://127.0.0.1:19201 --b http://127.0.0.1:19203 --sorted > ~/esql-compare-1pct.txt 2>&1
  taskset -c 14,15 python3 $S/latency.py $T/queries.tsv http://127.0.0.1:19201 http://127.0.0.1:19203 http://127.0.0.1:19202 --warmup 5 --runs 30 --max-seconds 1 > ~/esql-latency-1pct.txt 2>&1
  kill \$(cat ~/esql-es9-conf-19201/pid) \$(cat ~/esql-poc-data.pid) \$(cat ~/esql-master-data.pid)
"
