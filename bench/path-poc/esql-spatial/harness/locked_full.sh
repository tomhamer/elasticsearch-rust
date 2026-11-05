#!/bin/bash
# locked_full.sh: full-corpus geopoint races (setup + ES|QL tasks), each engine under the bench lock.
export TRACK_ROOT=$HOME/esql-tools/tracks RALLY_EXTRA="--include-tasks=tag:setup,tag:esql"
flock ~/.bench.lock env TRACKS=geopoint RES_TAG=full-es9 bash ~/esql-tools/run_geo.sh es9 || exit 1
flock ~/.bench.lock env TRACKS=geopoint RES_TAG=full-rust-poc RUST_BIN=$HOME/esql-bins/poc-release bash ~/esql-tools/run_geo.sh rust || exit 1
touch ~/esql-rally/full-all.done
