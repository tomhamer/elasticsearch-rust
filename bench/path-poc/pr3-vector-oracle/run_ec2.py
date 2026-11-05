#!/usr/bin/env python3
"""Explicit, never scheduled automatically; preserve all output on any failure."""
import argparse, hashlib, json, os, pathlib, subprocess, sys, fcntl, tempfile
p=argparse.ArgumentParser()
for flag in ('source','source-manifest','source-manifest-sha256','revision','out','cargo'):
    p.add_argument('--'+flag,required=True)
p.add_argument('--completed',action='append',required=True,help='systemd-service=/absolute/results-directory (repeat for full suite and security replay)')
a=p.parse_args()
if sys.platform!='linux' or not pathlib.Path('/sys/devices/virtual/dmi/id/product_uuid').exists():raise SystemExit('EC2 Linux only')
root=pathlib.Path(a.source).resolve(); here=pathlib.Path(__file__).resolve().parent
if not here.is_relative_to(root):raise SystemExit('Launcher must be part of the verified source snapshot')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read_manifest():
    if sha(a.source_manifest)!=a.source_manifest_sha256:raise RuntimeError('Source manifest hash mismatch')
    files=json.loads(pathlib.Path(a.source_manifest).read_text())
    if not files:raise RuntimeError('Empty source manifest')
    for rel,digest in files.items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:raise RuntimeError('Source mismatch: '+rel)
    for path in (here/'run_ec2.py',here/'EncodedScoreOracle.java',here/'es955-distribution-manifest.json',root/'crates/rustic-core/src/vector/osq.rs'):
        if str(path.relative_to(root)) not in files:raise RuntimeError('Critical source absent from manifest: '+str(path))
    return files
source_files=read_manifest()
prerequisites=[]
services=set(); result_dirs=set()
for item in a.completed:
    service,path=item.split('=',1); results=pathlib.Path(path)
    if not results.is_absolute():raise SystemExit('Absolute prerequisite result path required')
    results=results.resolve()
    # Resolve actual systemd identity: bare names and unit aliases must not
    # allow one completed workload to masquerade as two prerequisites.
    properties=dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','show',service,'--property=Id','--property=ActiveState'],text=True).splitlines() if '=' in line)
    service=properties.get('Id','')
    if not service:raise SystemExit('Missing prerequisite service identity')
    if service in services or results in result_dirs:raise SystemExit('Prerequisites require distinct services and resolved result directories')
    services.add(service); result_dirs.add(results)
    code=int((results/'exit.code').read_text().strip())
    state=properties.get('ActiveState','')
    if state not in ('inactive','failed'):raise SystemExit('Prerequisite still active: '+service)
    prerequisites.append({'service':service,'results':str(results),'exit':code,'state':state})
if len(prerequisites)<2:raise SystemExit('Require full-suite and security-replay completion records')
lock=open('/var/lock/es-pr3-vector-oracle.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    if int(proc.name)==os.getpid():continue
    try: args=proc.joinpath('cmdline').read_bytes().split(b'\0');name=proc.joinpath('comm').read_text().strip()
    except FileNotFoundError:continue
    if name in ('cargo','rustc','javac','elasticsearch-n') or any(pathlib.PurePath(x.decode(errors='replace')).name in ('elasticsearch-node','es-cluster-yaml','parity_all.py','start_module_fixture.py') for x in args) or any(b'org.elasticsearch.bootstrap.Elasticsearch' in x for x in args):
        raise SystemExit('Competing workload: '+proc.name)
distribution=json.loads((here/'es955-distribution-manifest.json').read_text());dist=pathlib.Path(distribution['root'])
actual={str(x.relative_to(dist)) for x in dist.rglob('*') if x.is_file() or x.is_symlink()}
if actual!=set(distribution['files']):raise SystemExit('Distribution file inventory mismatch')
for rel,entry in distribution['files'].items():
    path=dist/rel
    if 'symlink' in entry:
        if not path.is_symlink() or os.readlink(path)!=entry['symlink'] or not path.resolve().is_relative_to(dist):raise SystemExit('Distribution symlink mismatch: '+rel)
    elif path.is_symlink() or sha(path)!=entry['sha256']:raise SystemExit('Distribution hash mismatch: '+rel)
parent=pathlib.Path(a.out).resolve();parent.mkdir(parents=True,exist_ok=True)
out=pathlib.Path(tempfile.mkdtemp(prefix='encoded-oracle-',dir=parent));print(out,flush=True)
jar=dist/'lib/lucene-core-10.5.1.jar';jdk=dist/'jdk/bin'
env=dict(os.environ);env['CARGO_TARGET_DIR']=str(out/'target');env['CARGO_BUILD_JOBS']='16';env['RUSTFLAGS']='-Ctarget-cpu=native'
env['PR3_ENCODED_SCORE_ORACLE']=str(out/'golden.json')
proof={'revision_label':a.revision,'source_manifest_sha256':a.source_manifest_sha256,'source_files':len(source_files),'distribution_manifest_sha256':sha(here/'es955-distribution-manifest.json'),'lucene_jar_sha256':sha(jar),'cargo_sha256':sha(a.cargo),'prerequisites':prerequisites,'instrumentation':False,'test':'vector::osq::upstream_encoded_oracle::lucene_1051_encoded_score_oracle'}
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n');code=1
commands=[]
def run(argv,filename):
    commands.append(argv);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/filename).open('w') as log:subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    run([str(jdk/'java'),'-version'],'java-version.log')
    (out/'classes').mkdir()
    run([str(jdk/'javac'),'-cp',str(jar),'-d',str(out/'classes'),str(here/'EncodedScoreOracle.java')],'javac.log')
    argv=[str(jdk/'java'),'-ea','-cp',str(out/'classes')+':'+str(jar),'EncodedScoreOracle'];commands.append(argv)
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/'golden.json').open('w') as data,(out/'java.log').open('w') as err:subprocess.run(argv,cwd=root,env=env,stdout=data,stderr=err,check=True)
    golden=json.loads((out/'golden.json').read_text());cases=golden['cases']
    if golden['lucene']!='10.5.1' or len(cases)!=192:raise RuntimeError('Wrong oracle version/count')
    for bits in (7,4,1):
        for sim in ('l2','dot','cosine','mip'):
            if sum(c['bits']==bits and c['sim']==sim for c in cases)!=16:raise RuntimeError('Missing oracle category')
    run([a.cargo,'test','--locked','--offline','-p','rustic-core','--lib',proof['test'],'--','--ignored','--exact'],'rust-test.log')
    if 'test result: ok. 1 passed; 0 failed; 0 ignored;' not in (out/'rust-test.log').read_text():raise RuntimeError('Expected exactly one explicit oracle test')
    read_manifest()
    (out/'summary.json').write_text(json.dumps({'oracle_cases':192,'rust_pass':1,'rust_fail':0,'rust_skip':0,'golden_sha256':sha(out/'golden.json')},indent=2)+'\n');code=0
finally:(out/'exit.code').write_text(str(code)+'\n')
