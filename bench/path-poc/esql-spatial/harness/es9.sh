#!/bin/bash
# start ES 9.5.5 on $1 (http) / $2 (transport), data dir $3
http=${1:-19201}; tr=${2:-19301}; data=${3:-$HOME/esql-es9-data}
conf=$HOME/esql-es9-conf-$http; rm -rf $conf; cp -r $HOME/ref/elasticsearch-9.5.5/config $conf
curl -s -m 2 localhost:$http >/dev/null && { echo "port $http already in use" >&2; exit 1; }
mkdir -p $data
cat > $conf/elasticsearch.yml <<YML
cluster.name: esql-ref
node.name: es9-$http
network.host: 127.0.0.1
http.port: $http
transport.port: $tr
discovery.type: single-node
path.data: $data
path.logs: $conf/logs
xpack.security.enabled: false
YML
ES_PATH_CONF=$conf ES_JAVA_OPTS="-Xms16g -Xmx16g" setsid taskset -c 0-13 $HOME/ref/elasticsearch-9.5.5/bin/elasticsearch > $conf/server.log 2>&1 < /dev/null &
echo $! > $conf/pid
for i in $(seq 1 120); do curl -sf "localhost:$http/_cluster/health?wait_for_status=yellow&timeout=1s" >/dev/null && break; sleep 1; done
curl -s -XPOST "localhost:$http/_license/start_trial?acknowledge=true"; echo
