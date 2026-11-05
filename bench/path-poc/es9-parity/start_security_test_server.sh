#!/usr/bin/env bash
# AWS-only fixture for the official ES9 security REST suite.
set -euo pipefail
root="${1:-$HOME/es9-feature-work}"
fixture="${2:-/tmp/es9-secured-product}"
mkdir -p "$fixture/conf" "$fixture/data" "$fixture/repo"
python3 - "$fixture/conf" <<'PY'
import pathlib,hashlib,base64,secrets,sys
conf=pathlib.Path(sys.argv[1])
def hash_value(password):
    salt=secrets.token_bytes(32)
    key=hashlib.pbkdf2_hmac('sha512',password.encode(),salt,1000,32)
    return '{PBKDF2}1000$'+base64.b64encode(salt).decode()+'$'+base64.b64encode(key).decode()
(conf/'users').write_text('x_pack_rest_user:'+hash_value('x-pack-test-password')+'\n')
(conf/'users_roles').write_text('superuser:x_pack_rest_user\n')
(conf/'service_tokens').write_text('elastic/fleet-server/token1:'+hash_value('r5wdbdboQSe9vGOKwaJGAw')+'\n')
PY
# Two actual nodes: replicas can be allocated and YAML can round-robin reads.
python3 - "$root" "$fixture" <<'PYFIX'
import json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);fixture=pathlib.Path(sys.argv[2])
pids=[]
conf=fixture/"conf"
# `bootstrap.password` is a secure setting (refused in elasticsearch.yml /
# -E): it goes into the shared keystore before the nodes start.
tool=root/"target/fast/elasticsearch-keystore"
secure={}
if tool.exists():
    ks_env=dict(os.environ,ES_PATH_CONF=str(conf))
    if not (conf/"elasticsearch.keystore").exists():
        subprocess.run([str(tool),"create"],env=ks_env,stdin=subprocess.DEVNULL,check=True,capture_output=True)
    subprocess.run([str(tool),"add","-x","-f","bootstrap.password"],env=ks_env,input="x-pack-test-password\n",text=True,check=True,capture_output=True)
else:
    secure["bootstrap.password"]="x-pack-test-password"
for i in range(2):
    settings={"cluster.name":"es9-secured-parity","node.name":f"yamlRestTest-{i}",
        "path.data":str(fixture/f"node-{i}"),"path.repo":str(fixture/"repo"),
        "network.host":"127.0.0.1","http.port":str(19540+i),"transport.port":str(19640+i),
        "discovery.seed_hosts":"127.0.0.1:19640,127.0.0.1:19641",
        "cluster.initial_master_nodes":"yamlRestTest-0,yamlRestTest-1","node.attr.testattr":"test",
        "xpack.security.enabled":"true","xpack.security.authc.token.enabled":"true",
        "xpack.security.authc.api_key.enabled":"true","xpack.license.self_generated.type":"trial",
        "action.destructive_requires_name":"false",
        "cluster.routing.allocation.disk.watermark.low":"1b","cluster.routing.allocation.disk.watermark.high":"1b",
        "cluster.routing.allocation.disk.watermark.flood_stage":"1b"}
    settings.update(secure)
    args=[str(root/"target/fast/elasticsearch")]
    for key,value in settings.items():args.extend(["-E",key+"="+value])
    process=subprocess.Popen(args,env=dict(os.environ,ES_PATH_CONF=str(fixture/"conf")),stdin=subprocess.DEVNULL,
        stdout=open(fixture/f"node-{i}.log","w"),stderr=subprocess.STDOUT,start_new_session=True)
    (fixture/f"node-{i}.pid").write_text(str(process.pid));pids.append(process.pid)
(fixture/"pids.json").write_text(json.dumps(pids))
print("secured two-node fixture",pids)
print("YAML URLs: http://127.0.0.1:19540,http://127.0.0.1:19541")
PYFIX
