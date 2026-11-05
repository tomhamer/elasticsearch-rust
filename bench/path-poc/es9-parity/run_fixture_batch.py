#!/usr/bin/env python3
"""AWS-only sequential module verification; one fresh fixture owns each port pair."""
import argparse,json,os,pathlib,signal,subprocess,time,fcntl
p=argparse.ArgumentParser()
p.add_argument('--root',default='/home/ubuntu/es9-feature-work')
p.add_argument('--runner',default='/tmp/es-cluster-yaml-default-coverage')
p.add_argument('--modules',default='fleet,ccr,inference,inference-2,async-search')
p.add_argument('--suite',default='plug',choices=['oss','main','plug','sec'])
p.add_argument('--filter',action='append',dest='filters')
p.add_argument('--all',action='store_true',help='Select the whole suite')
p.add_argument('--run-name',default='default-coverage-fixtures')
p.add_argument('--http-base',type=int,default=19560)
p.add_argument('--transport-base',type=int,default=19660)
a=p.parse_args();root=pathlib.Path(a.root);out=root/'parity-results'/a.run_name;out.mkdir(parents=True,exist_ok=True)
lock=open(f'/tmp/es9-fixture-{a.http_base}.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for module in a.modules.split(','):
 fixture=pathlib.Path('/tmp')/(a.run_name+'-'+module)
 subprocess.run(['python3',str(root/'bench/path-poc/es9-parity/start_module_fixture.py'),'--root',str(root),'--runner',a.runner,'--module',module,'--directory',str(fixture),'--http-base',str(a.http_base),'--transport-base',str(a.transport_base)],check=True)
 spec=json.load(open(fixture/'fixture.json'));args=[a.runner,'--suite',str(pathlib.Path.home()/'es955-yaml'/a.suite),
 '--url',spec['urls'],'--module-security-fixtures','--json',str(out/(module+'.json'))]
 if not a.all:
  for selected in a.filters or [module+'/']:args+=['--filter',selected]
 if spec['security']:args+=['--security']
 try:
  with open(out/(module+'.log'),'w') as log:code=subprocess.call(args,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (out/(module+'.exit')).write_text(str(code));print(module,'runner exit',code,flush=True)
  result=out/(module+'.json')
  if not result.exists() or not json.load(open(result)):raise RuntimeError('No completed selected cases; this run does not establish coverage')
 finally:
  for pid in spec['pids']:
   try:os.kill(pid,signal.SIGTERM)
   except ProcessLookupError:pass
  for _ in range(30):
   if not any(pathlib.Path('/proc/'+str(pid)).exists() for pid in spec['pids']):break
   time.sleep(1)
  else:raise RuntimeError('Nodes did not stop; do not start another wiping fixture')
