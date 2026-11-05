#!/bin/bash
# start a Rust server binary $1 on http $2 / transport $3, data $4
bin=$1; http=$2; tr=$3; data=$4; mkdir -p $data
curl -s -m 2 localhost:$http >/dev/null && { echo "port $http already in use" >&2; exit 1; }
setsid taskset -c 0-13 $bin -E cluster.name=esql-rs -E node.name=rs-$http -E network.host=127.0.0.1 -E http.port=$http -E transport.port=$tr -E discovery.type=single-node -E path.data=$data -E xpack.security.enabled=false > $data.log 2>&1 < /dev/null &
echo $! > $data.pid
for i in $(seq 1 120); do curl -sf "localhost:$http/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null && break; sleep 1; done
curl -s -XPOST "localhost:$http/_license/start_trial?acknowledge=true"; echo
