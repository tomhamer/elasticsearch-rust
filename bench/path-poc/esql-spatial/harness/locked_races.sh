#!/bin/bash
# locked_races.sh TAG_PREFIX [TRACK_ROOT] [RALLY_EXTRA]: the ES9 then Rust geopoint races,
# both under the box-wide benchmark lock (one paired measurement); stops if a server fails.
prefix=$1; export TRACK_ROOT=${2:-} RALLY_EXTRA=${3:-}
flock ~/.bench.lock bash -c "
  TRACKS=geopoint RES_TAG=${prefix}es9 bash ~/esql-tools/run_geo.sh es9 || exit 1
  TRACKS=geopoint RES_TAG=${prefix}rust-poc RUST_BIN=\$HOME/esql-bins/poc-release bash ~/esql-tools/run_geo.sh rust || exit 1
  touch ~/esql-rally/${prefix}all.done"
