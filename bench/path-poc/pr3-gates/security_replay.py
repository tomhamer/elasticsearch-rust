#!/usr/bin/env python3
"""EC2-only unchanged security downsample replay; consumes pinned builds, never builds."""
import argparse, fcntl, hashlib, json, os, pathlib, socket, subprocess, sys, tempfile, uuid

p=argparse.ArgumentParser()
for name in ('root','runner','binary','manifest','manifest-sha256','runner-manifest-sha256','runner-sha256','binary-sha256','revision','completed-results','out'):
    p.add_argument('--'+name, required=True)
p.add_argument('--port-base',type=int,default=28000)
p.add_argument('--completed-service',default='pr3-round25')
a=p.parse_args()
if sys.platform != 'linux' or not pathlib.Path('/sys/devices/virtual/dmi/id/product_uuid').exists():
    raise SystemExit('Linux EC2 launcher only')
if os.environ.get('ES_YAML_CLAIM_ALL'): raise SystemExit('Forced gates forbidden')
root=pathlib.Path(a.root).resolve(); runner=pathlib.Path(a.runner).resolve(); binary=pathlib.Path(a.binary).resolve()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
completed=pathlib.Path(a.completed_results).resolve()
# A failed completed suite can still supply a pinned diagnostic build. Missing
# completion, however, is not permission to race an active workload.
exit_code=int((completed/'exit.code').read_text().strip())
state=subprocess.run(['systemctl','show',a.completed_service,'--property=ActiveState','--value'],check=True,capture_output=True,text=True).stdout.strip()
if state not in ('inactive','failed'): raise SystemExit(f'Prerequisite service still {state}')
for process in ('cargo','rustc'):
    probe=subprocess.run(['pgrep','-x',process],capture_output=True)
    if probe.returncode != 1: raise SystemExit(f'Cannot establish idle compiler: {process}')
if sha(a.manifest)!=a.manifest_sha256 or a.runner_manifest_sha256!=a.manifest_sha256:
    raise SystemExit('Server/runner source manifest proof mismatch')
manifest=json.loads(pathlib.Path(a.manifest).read_text())
if not manifest: raise SystemExit('Empty source manifest')
for rel,expected in manifest.items():
    path=(root/rel).resolve()
    if not path.is_relative_to(root) or sha(path)!=expected: raise SystemExit(f'Source mismatch: {rel}')
for path,expected in ((binary,a.binary_sha256),(runner,a.runner_sha256)):
    if sha(path)!=expected: raise SystemExit(f'Binary hash mismatch: {path}')
selected='security/authz/80_downsample.yml'
fixture_yaml=pathlib.Path.home()/'es955-yaml/sec/test'/selected
inventory=json.loads((root/'bench/path-poc/pr3-gates/yaml18-files.json').read_text())
if selected not in inventory['sec'] or not fixture_yaml.is_file(): raise SystemExit('Missing verified security fixture')
spec=json.loads(subprocess.check_output([str(runner),'--export-fixture','sec'],text=True))
if spec.get('nodes',2)!=2 or not spec.get('security'): raise SystemExit('Expected two-node security module fixture')
ports=range(a.port_base,a.port_base+200)
low,high=map(int,pathlib.Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
if a.port_base<1024 or a.port_base+199>65535 or not (a.port_base+199<low or a.port_base>high): raise SystemExit('Invalid/ephemeral port block')
lockroot=pathlib.Path('/var/lock/es-pr3-faults');lockroot.mkdir(parents=True,exist_ok=True)
locks=[]
for port in ports:
    fd=os.open(lockroot/f'port-{port}.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
owner=lockroot/f'security-owner-{a.port_base}.json'
if owner.exists(): raise SystemExit(f'Existing fixture ownership: {owner}; inspect explicitly')
# Refuse a matching process even if it has temporarily closed its listener.
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    if int(proc.name)==os.getpid(): continue
    try: argv=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
    except FileNotFoundError: continue
    if not argv or not any('elasticsearch' in x or 'start_module_fixture.py' in x or 'parity_all.py' in x for x in argv): continue
    for i,arg in enumerate(argv):
        if any(key in arg for key in ('http.port','transport.port','http-base','transport-base','port-base')):
            value=arg.split('=',1)[-1] if '=' in arg else (argv[i+1] if i+1<len(argv) else '')
            if value.isdigit() and int(value) in ports: raise SystemExit(f'Matching existing cluster process {proc.name}')
for port in ports:
    with socket.socket() as sock: sock.bind(('0.0.0.0',port))
parent=pathlib.Path(a.out).resolve();parent.mkdir(parents=True,exist_ok=True)
out=pathlib.Path(tempfile.mkdtemp(prefix='security-replay-',dir=parent))
run='pr3-security-replay-'+uuid.uuid4().hex
fixture=pathlib.Path('/tmp')/(run+'-sec-sec');reports=root/'parity-results'/run
if fixture.exists() or reports.exists(): raise SystemExit('Unique fixture collision')
proof=dict(revision=a.revision,source_manifest_sha256=a.manifest_sha256,source_files=len(manifest),binary_sha256=a.binary_sha256,runner_sha256=a.runner_sha256,runner_source_manifest_sha256=a.runner_manifest_sha256,prerequisite_exit=exit_code,fixture=str(fixture),reports=str(reports),fixture_spec=spec,yaml_sha256=sha(fixture_yaml),selected=selected,nodes=2,profiling=False)
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
with owner.open('x') as f: json.dump(dict(pid=os.getpid(),out=str(out),fixture=str(fixture),run=run),f)
command=['python3',str(root/'bench/path-poc/es9-parity/parity_all.py'),'--root',str(root),'--runner',str(runner),'--binary',str(binary),'--run-name',run,'--suites','sec','--filter',selected,'--parallel','1','--port-base',str(a.port_base)]
(out/'command.json').write_text(json.dumps(command,indent=2)+'\n');print(out,flush=True)
code=1
try:
    with (out/'replay.log').open('w') as log: runner_code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    cases=json.loads((reports/'sec-sec.json').read_text())
    if not cases or any(c['file']!=selected for c in cases): raise RuntimeError('Missing or unexpected selected coverage')
    if not any('Downsample admin user with field level security' in c['name'] for c in cases): raise RuntimeError('Required failing case did not execute')
    counts={s:sum(c['status']==s for c in cases) for s in ('pass','fail','skip')}
    (out/'counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    code=runner_code if not counts['fail'] and not counts['skip'] else 1
finally:
    (out/'exit.code').write_text(str(code)+'\n')
    # Keep ownership and all fixture logs even on success. Operator verifies
    # no surviving fixture processes before clearing ownership for port reuse.
sys.exit(code)
