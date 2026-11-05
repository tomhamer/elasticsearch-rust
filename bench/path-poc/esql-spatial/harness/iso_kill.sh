#!/bin/bash
# Stop my isolated race (namespace, runner, servers it started); never touches other agents' processes.
for p in $(pgrep -f "^sudo unshare --net --pid --fork --mount-proc /tmp/esql-iso-inner.sh"); do sudo kill -TERM -- $p; done
for p in $(pgrep -f "^bash /home/ubuntu/esql-tools/run_geo.sh"); do kill $p; done
for p in $(pgrep -f "^/home/ubuntu/esql-bins/"); do kill $p; done
for p in $(pgrep -f "ES_PATH_CONF=/home/ubuntu/esql-rally"); do kill $p; done
for j in $(pgrep -u ubuntu java); do tr "\0" "\n" < /proc/$j/environ 2>/dev/null | grep -q "ES_PATH_CONF=/home/ubuntu/esql-rally" && kill $j; done
for p in $(pgrep -f "track-path=/home/ubuntu/(rally-1pct|esql-tools)/tracks/geopoint"); do kill $p; done
sleep 3
ps -eo pid,args | grep -E "esql-iso|run_geo|esql-bins|esql-rally|tracks/geopoint" | grep -v -E "grep|iso_kill" | cut -c1-160
