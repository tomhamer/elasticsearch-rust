#!/bin/bash
# full_parity.sh: index the full geopoint corpus on ES9, save its ES|QL responses,
# then on the Rust release build, and compare. Each engine under the bench lock.
set -u
T=$HOME/esql-tools; C=$HOME/esql-poc/bench/path-poc/esql-spatial/compare.py
setup() { # port
  taskset -c 14,15 $HOME/rally-venv/bin/esrally race --pipeline=benchmark-only --offline --target-hosts=127.0.0.1:$1 \
    --track-path=$HOME/esql-tools/tracks/geopoint --include-tasks="tag:setup" --on-error=abort --report-file=/dev/null
}
flock ~/.bench.lock bash -c "
  rm -rf ~/esql-es9-full; $T/es9.sh 19201 19301 ~/esql-es9-full || exit 1
  $(declare -f setup); setup 19201 > ~/esql-full-es9-setup.log 2>&1 || { $T/kill_es9_19201.sh; exit 1; }
  curl -s localhost:19201/_cat/count/osmgeopoints > ~/esql-full-es9-count.txt
  taskset -c 14,15 python3 $C --a http://127.0.0.1:19201 --b http://127.0.0.1:19201 --save-a ~/esql-es9-full-responses.json > ~/esql-full-es9-self.txt 2>&1
  $T/kill_es9_19201.sh; sleep 15; rm -rf ~/esql-es9-full
" || exit 1
flock ~/.bench.lock bash -c "
  rm -rf ~/esql-rs-full; $T/rs.sh \$HOME/esql-bins/poc-release 19203 19303 ~/esql-rs-full || exit 1
  $(declare -f setup); setup 19203 > ~/esql-full-rs-setup.log 2>&1 || { kill \$(cat ~/esql-rs-full.pid); exit 1; }
  curl -s localhost:19203/_cat/count/osmgeopoints > ~/esql-full-rs-count.txt
  taskset -c 14,15 python3 $C --a file:\$HOME/esql-es9-full-responses.json --b http://127.0.0.1:19203 --sorted > ~/esql-full-compare.txt 2>&1
  kill \$(cat ~/esql-rs-full.pid); sleep 10; rm -rf ~/esql-rs-full
" || exit 1
touch ~/esql-full-parity.done
