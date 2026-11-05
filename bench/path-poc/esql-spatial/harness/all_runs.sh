#!/bin/bash
# The remaining measurements, in order, each phase under the bench lock.
~/esql-tools/locked_races.sh p2- > ~/esql-p2.log 2>&1
~/esql-tools/full_parity.sh > ~/esql-parity.log 2>&1
rm -rf ~/esql-rally/full-*
~/esql-tools/locked_full.sh > ~/esql-full.log 2>&1
touch ~/esql-all-runs.done
