#!/bin/bash
for j in $(pgrep -u ubuntu java); do tr "\0" "\n" < /proc/$j/environ 2>/dev/null | grep -q "ES_PATH_CONF=/home/ubuntu/esql-es9-conf-19201" && { echo "killing $j"; kill $j; }; done
