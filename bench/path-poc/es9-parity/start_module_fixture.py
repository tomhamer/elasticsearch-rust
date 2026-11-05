#!/usr/bin/env python3
"""Start an isolated AWS-only module fixture, with real two-node allocation.
Uses the YAML runner's exported module realms rather than guessed users.
No binaries are built here. --nodes defaults to two; official module-specific
single-node or remote-cluster exceptions must be specified explicitly.
"""
import argparse,base64,hashlib,json,os,pathlib,shutil,socket,subprocess,sys,time,urllib.request
p=argparse.ArgumentParser()
p.add_argument('--root',default='/home/ubuntu/es9-feature-work')
p.add_argument('--runner',required=True)
p.add_argument('--module',required=True)
p.add_argument('--directory',required=True)
p.add_argument('--http-base',type=int,default=19560)
p.add_argument('--transport-base',type=int,default=19660)
p.add_argument('--nodes',type=int,default=2)
p.add_argument('--binary',help='Immutable executable to test; defaults to root/target/fast/elasticsearch')
a=p.parse_args()
root=pathlib.Path(a.root);fixture=pathlib.Path(a.directory)
if a.nodes<1:raise RuntimeError('At least one node is required')
if (fixture/'pids.json').exists():raise RuntimeError('Fixture already registered; stop and inspect it before restarting')
for port in [a.http_base+i for i in range(a.nodes)]+[a.transport_base+i for i in range(a.nodes)]:
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} is in use; refusing overlapping fixture')
spec=json.loads(subprocess.check_output([a.runner,'--export-fixture',a.module],text=True))
conf=fixture/'conf';conf.mkdir(parents=True,exist_ok=True);(fixture/'repo').mkdir()
(fixture/'executed-launcher.py').write_text(pathlib.Path(__file__).read_text())
(conf/'users').write_text(''.join(f'{u}:{h}\n' for u,h in spec['file_users'].items()))
roles={}
for user,grants in spec['file_user_roles'].items():
 for role in grants:roles.setdefault(role,[]).append(user)
(conf/'users_roles').write_text(''.join(f'{role}:{",".join(users)}\n' for role,users in roles.items()))
(conf/'roles.yml').write_text(json.dumps(spec['file_roles'],indent=2))
(conf/'service_tokens').write_text(''.join(f'{name}:{hashed}\n' for name,hashed in spec['file_service_tokens'].items()))
# Module config files (the x-pack suite's transport TLS certificate and key).
for name,text in spec.get('config_files',{}).items():(conf/name).write_text(text)
pids=[]
processes=[]
# Pin both nodes to one executable, even if another AWS build completes.
binary=fixture/'elasticsearch'
source_binary=pathlib.Path(a.binary) if a.binary else root/'target/fast/elasticsearch'
shutil.copy2(source_binary,binary)
binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
settings=dict(spec['settings'])
# Secure settings (ES refuses them in elasticsearch.yml / -E) go into the
# shared config directory's keystore, written by the build's own
# elasticsearch-keystore before any node starts. A missing tool fails the
# fixture loudly (a silent fallback to elasticsearch.yml made every node of a
# keystore build refuse to start); only a build older than the keystore may
# take them as plain settings, by setting ES_FIXTURE_PLAIN_SECURE_SETTINGS=1.
secure=dict(spec.get('secure_settings') or {})
secure['bootstrap.password']='x-pack-test-password'
keystore_tool=source_binary.with_name('elasticsearch-keystore')
if keystore_tool.exists():
 tool=fixture/'elasticsearch-keystore';shutil.copy2(keystore_tool,tool)
 ks_env=dict(os.environ,ES_PATH_CONF=str(conf))
 subprocess.run([str(tool),'create'],env=ks_env,stdin=subprocess.DEVNULL,check=True,capture_output=True)
 for key,value in secure.items():
  subprocess.run([str(tool),'add','-x','-f',key],env=ks_env,input=str(value)+'\n',text=True,check=True,capture_output=True)
elif os.environ.get('ES_FIXTURE_PLAIN_SECURE_SETTINGS')=='1':settings.update(secure)
else:sys.exit(f'{keystore_tool} is missing: build it beside the node binary (cargo build -p elasticsearch --bin elasticsearch-keystore), or set ES_FIXTURE_PLAIN_SECURE_SETTINGS=1 for a build older than the keystore')
settings.update({'cluster.name':'yaml-'+a.module,'network.host':'127.0.0.1','path.repo':str(fixture/'repo'),
 'node.attr.testattr':'test','action.destructive_requires_name':'false',
 'cluster.routing.allocation.disk.watermark.low':'1b','cluster.routing.allocation.disk.watermark.high':'1b',
 'cluster.routing.allocation.disk.watermark.flood_stage':'1b',
 # The rest of ES's test-cluster defaults (DefaultSettingsProvider): fixed
 # processors so output does not depend on the host, no script compilation
 # rate limit, no real-memory breaker, no deprecation indexing.
 'node.processors':'2','script.disable_max_compilations_rate':'true',
 'indices.breaker.total.use_real_memory':'false','discovery.initial_state_timeout':'0s',
 'cluster.service.slow_task_logging_threshold':'5s','cluster.service.slow_master_task_logging_threshold':'5s',
 'discovery.seed_hosts':','.join(f'127.0.0.1:{a.transport_base+i}' for i in range(a.nodes))})
# Per-node settings (e.g. the stateless cluster's index / search roles);
# only master-eligible nodes bootstrap the cluster.
per_node=spec.get('per_node_settings') or []
def node_extra(i):return per_node[i] if i<len(per_node) else {}
def master_eligible(i):
 roles=node_extra(i).get('node.roles',settings.get('node.roles'))
 return roles is None or 'master' in (roles if isinstance(roles,list) else [r.strip() for r in str(roles).strip('[]').split(',')])
settings['cluster.initial_master_nodes']=','.join(f'yamlRestTest-{i}' for i in range(a.nodes) if master_eligible(i))
# Default (non integ-test) distributions also disable deprecation indexing.
if a.module!='oss':settings.setdefault('cluster.deprecation_indexing.enabled','false')
env=dict(os.environ,ES_PATH_CONF=str(conf))
if spec['inference_test_services']:env['ES_INFERENCE_TEST_SERVICES']='1'
for i in range(a.nodes):
 opts=dict(settings,**node_extra(i),**{'node.name':f'yamlRestTest-{i}','path.data':str(fixture/f'node-{i}'),
 'http.port':str(a.http_base+i),'transport.port':str(a.transport_base+i)})
 args=[str(binary)]
 for key,value in opts.items():args+=['-E',key+'='+str(value)]
 proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=open(fixture/f'node-{i}.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
 pids.append(proc.pid);(fixture/'pids.json').write_text(json.dumps(pids))
 # Record kernel start identity while Popen still owns the unreaped child.
 start_ticks=pathlib.Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 processes.append({'pid':proc.pid,'start_ticks':start_ticks})
 (fixture/'processes.json').write_text(json.dumps(processes))
urls=','.join(f'http://127.0.0.1:{a.http_base+i}' for i in range(a.nodes))
(fixture/'fixture.json').write_text(json.dumps({'module':a.module,'urls':urls,'security':spec['security'],
 'nodes':a.nodes,'per_node_settings':spec.get('per_node_settings',[]),'pids':pids,'binary_sha256':binary_hash,'binary':str(binary),'settings':settings,'transport_tls':settings.get('xpack.security.transport.ssl.enabled')=='true','http_tls':False},indent=2))
auth='Basic '+base64.b64encode(b'x_pack_rest_user:x-pack-test-password').decode()
for _ in range(120):
 try:
  req=urllib.request.Request(f'http://127.0.0.1:{a.http_base}/_cluster/health?wait_for_nodes={a.nodes}&timeout=1s')
  if spec['security']:req.add_header('Authorization',auth)
  v=json.load(urllib.request.urlopen(req,timeout=3))
  if v.get('number_of_nodes')==a.nodes and not v.get('timed_out'):
   initialized=True
   for port in range(a.http_base,a.http_base+a.nodes):
    state_req=urllib.request.Request(f'http://127.0.0.1:{port}/_cluster/state/blocks')
    if spec['security']:state_req.add_header('Authorization',auth)
    state=json.load(urllib.request.urlopen(state_req,timeout=3))
    initialized &= '1' not in state.get('blocks',{}).get('global',{})
   if initialized:break
 except Exception:pass
 time.sleep(1)
else:raise RuntimeError('Fixture failed to form; retain logs and do not run tests')
print(json.dumps({'module':a.module,'urls':urls,'security':spec['security'],'pids':pids}),flush=True)
