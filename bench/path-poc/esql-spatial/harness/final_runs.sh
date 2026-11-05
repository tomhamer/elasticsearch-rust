#!/bin/bash
# Release build, then every measurement in order (each timed phase under the bench lock).
T=$HOME/esql-tools
$T/bld.sh release poc-release > ~/esql-rel-final.log 2>&1 || { echo build failed > ~/esql-final.failed; exit 1; }
sha256sum ~/esql-bins/poc-release ~/esql-bins/master-release > ~/esql-final-sha.txt
$T/lat_1pct.sh > ~/esql-lat-final.log 2>&1
$T/kill_es9_19201.sh
$T/full_parity.sh > ~/esql-parity.log 2>&1
rm -rf ~/esql-rally/full-*
$T/locked_full.sh > ~/esql-full.log 2>&1
touch ~/esql-final.done
