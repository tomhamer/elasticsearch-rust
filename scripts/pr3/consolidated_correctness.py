#!/usr/bin/env python3
"""Reviewed external launcher for an immutable candidate; EC2 execution only.
No work on import. Never infers unknown artifact hashes or clears failed owners.
"""
import argparse, fcntl, getpass, hashlib, json, os, pathlib, re, shutil, signal, socket, subprocess, time, urllib.request

def readable_ec2_identity():
    """Unreadable DMI fields are not evidence; require a readable EC2 identity."""
    for name in ('/sys/hypervisor/uuid', '/sys/devices/virtual/dmi/id/product_uuid',
                 '/sys/devices/virtual/dmi/id/board_asset_tag'):
        try:
            identity = pathlib.Path(name).read_text().strip().lower()
        except (OSError, UnicodeError):
            continue
        if identity.startswith('ec2') or identity.startswith('i-'):
            return True
    return False


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def pinned(path,digest):
    assert isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Unresolved artifact hash'
    assert sha(path)==digest,('Hash mismatch',str(path))
def save(path,value):pathlib.Path(path).write_text(json.dumps(value,indent=2)+'\n')
def run(argv,log,env=None):
    with pathlib.Path(log).open('xb') as stream:code=subprocess.call(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,env=env)
    pathlib.Path(str(log)+'.exit').write_text(str(code)+'\n');return code
def idle():
    blocked={'cargo','rustc','javac','java','elasticsearch-node','elasticsearch','elasticsearch-faults','es-cluster-yaml','esrally','parity_all.py','start_module_fixture.py','fault_matrix.py','perf'}
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0');comm=proc.joinpath('comm').read_text().strip()
        except FileNotFoundError:continue
        assert comm not in blocked and not {pathlib.PurePath(a).name for a in args if a}&blocked and not any('/target/' in a and '/deps/' in a for a in args),('Competing workload',proc.name,comm)
def inventory(root,path,digest):
    pinned(path,digest);rows=json.loads(pathlib.Path(path).read_text());assert isinstance(rows,dict) and rows
    root=pathlib.Path(root).resolve()
    for name,expected in rows.items():
        target=(root/name).resolve();assert target.is_relative_to(root);pinned(target,expected)
    return len(rows)
def ports_free(ports):
    for port in ports:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
def fault_port_base(summary):
    """The first fault port block without an ownership record. fault_matrix.py
    keeps the owner file of a failed or interrupted run for operator
    inspection and refuses that block from then on; a fixed block let one
    interrupted run (candidate 13, 26 Oct) refuse the fault stage of every
    later candidate. Stale owners are never cleared here, only skipped and
    recorded. Blocks stay below the YAML stage's 24000-24599."""
    lock_root=pathlib.Path('/var/lock/es-pr3-faults');skipped=[]
    for base in range(22000,24000,200):
        owner=lock_root/f'owner-{base}.json'
        if owner.exists() or pathlib.Path(f'/sys/fs/cgroup/es-faults-{base}').exists():
            try:skipped.append({'port_base':base,'owner':json.loads(owner.read_text())})
            except (OSError,ValueError):skipped.append({'port_base':base,'owner':None})
            continue
        summary['fault_port_base']=base
        if skipped:summary['fault_skipped_owners']=skipped
        return base
    summary['fault_skipped_owners']=skipped
    raise RuntimeError('Every fault port block has a retained owner; inspect /var/lock/es-pr3-faults')
def stamp(pid):return pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
def terminate_owned(owned):
    for pid,start in owned:
        try:
            if stamp(pid)==start and os.getpgid(pid)==pid:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
        except FileNotFoundError:pass
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        alive=[]
        for pid,start in owned:
            try:
                if stamp(pid)==start:alive.append((pid,start))
            except FileNotFoundError:pass
        if not alive:return
        time.sleep(.5)
    for pid,start in alive:
        try:
            if stamp(pid)==start and os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):pass

def differential(cfg,source,out,node,runner):
    fixture=out/'fixtures';fixture.mkdir();owned=[];logs=[];children=[]
    ports_free([25000,25001,25100,25101,25200,25201,25300,25301])
    try:
        rust=fixture/'rust';rust.mkdir()
        try:
            code=run(['python3',source/'bench/path-poc/es9-parity/start_module_fixture.py','--root',source,'--runner',runner,'--binary',node,'--module','oss','--directory',rust,'--nodes','2','--http-base','25000','--transport-base','25100'],out/'rust-launch.log')
        finally:
            if (rust/'pids.json').exists():
                for pid in json.loads((rust/'pids.json').read_text()):
                    try:
                        assert pathlib.Path(f'/proc/{pid}/exe').resolve()==(rust/'elasticsearch').resolve(),'Fixture PID executable changed'
                        owned.append((pid,stamp(pid)))
                    except FileNotFoundError:pass
        assert code==0,'Rust differential fixture failed'
        es=pathlib.Path(cfg['es_distribution'])
        inventory(es,cfg['es_inventory'],cfg['es_inventory_sha256'])
        for ordinal in range(2):
            root=fixture/f'es-{ordinal}';root.mkdir();conf=root/'conf';shutil.copytree(es/'config',conf)
            env=dict(os.environ,ES_PATH_CONF=str(conf),ES_JAVA_HOME=str(es/'jdk'),ES_JAVA_OPTS='-Xms1g -Xmx1g')
            settings={'cluster.name':out.name,'node.name':f'pr3-diff-{ordinal}','network.host':'127.0.0.1','path.data':str(root/'data'),'path.logs':str(root/'logs'),'http.port':str(25200+ordinal),'transport.port':str(25300+ordinal),'discovery.seed_hosts':'127.0.0.1:25300,127.0.0.1:25301','cluster.initial_master_nodes':'pr3-diff-0,pr3-diff-1','xpack.security.enabled':'false','xpack.ml.enabled':'false'}
            argv=[str(es/'bin/elasticsearch')]+['-E'+k+'='+v for k,v in settings.items()]
            stream=(root/'process.log').open('xb');logs.append(stream)
            process=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children.append(process);owned.append((process.pid,stamp(process.pid)))
            save(root/'launch.json',{'argv':argv,'pid':process.pid,'start_ticks':owned[-1][1]})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            try:
                with opener.open('http://127.0.0.1:25200/_cluster/health?wait_for_nodes=2&timeout=1s',timeout=3) as r:health=json.load(r)
                if health['number_of_nodes']==2 and not health.get('timed_out'):break
            except (OSError,ValueError):pass
            time.sleep(1)
        else:raise RuntimeError('ES differential fixture formation failed')
        proof={'source_revision':cfg['revision'],'node_sha256':cfg['candidate_sha256'],'fixtures':{'rust':json.loads((rust/'fixture.json').read_text()),'es_distribution_inventory_sha256':cfg['es_inventory_sha256'],'owned_pids':owned}}
        save(out/'provenance.json',proof)
        return run(['python3',source/'bench/path-poc/pr3-gates/replicated_differential.py','--rust','http://127.0.0.1:25000,http://127.0.0.1:25001','--es','http://127.0.0.1:25200,http://127.0.0.1:25201','--provenance',out/'provenance.json','--out',out/'runs'],out/'differential.log')
    finally:
        terminate_owned(owned)
        for child in children:child.wait(timeout=10)
        for stream in logs:stream.close()

def main():
    if not __debug__:raise RuntimeError('Validation requires Python assertions enabled')
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(pathlib.Path(a.config).read_text())
    assert os.uname().sysname=='Linux' and readable_ec2_identity(),'EC2 only'
    assert not any(os.environ.get(k) for k in ('ES_YAML_CLAIM_ALL','RUSTIC_PROFILE','RUSTIC_TRACING'))
    assert getpass.getuser()=='ubuntu','Run fixture launcher as ubuntu; only fault harness uses sudo'
    root=pathlib.Path(cfg['root']);source=root/'source';results=root/'results';target=pathlib.Path(cfg['target']);node=results/'elasticsearch-node'
    assert re.fullmatch('[0-9a-f]{40}',cfg['revision']) and (source/'.pr3-source-revision').read_text().strip()==cfg['revision']
    assert isinstance(cfg['candidate_sha256'],str) and re.fullmatch('[0-9a-f]{64}',cfg['candidate_sha256']),'Resolve candidate hash from successful build before queuing'
    out=pathlib.Path(cfg['out']);out.mkdir(parents=True,exist_ok=False);summary={'complete':False,'source_revision':cfg['revision'],'stages':[]}
    try:
        while subprocess.call(['systemctl','is-active','--quiet',cfg['prerequisite_service']])==0:time.sleep(15)
        state=subprocess.check_output(['systemctl','show',cfg['prerequisite_service'],'--property=ActiveState','--value'],text=True).strip();assert state in ('inactive','failed')
        assert (results/'exit.code').exists() and (results/'binary.sha256').exists()
        shutil.copy2(results/'exit.code',out/'prerequisite-exit.code')
        # A failed full suite remains failed evidence; independent gates may
        # still diagnose the successfully built matching immutable executable.
        assert run(['sha256sum','-c',results/'binary.sha256'],out/'suite-binary-check.log')==0;pinned(node,cfg['candidate_sha256'])
        with open('/tmp/es-pr3-validation-host.lock','a+') as host,open(target/'.pr3-build.lock','a+') as lock:
            fcntl.flock(host,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle()
            inventory(source,cfg['manifest'],cfg['manifest_sha256'])
            inventory(cfg['yaml_root'],cfg['yaml_inventory'],cfg['yaml_inventory_sha256'])
            assert pathlib.Path(cfg['yaml_root']).resolve()==pathlib.Path('/home/ubuntu/es955-yaml')
            # parity_all --root selects product source, while its corpus is
            # Path.home()/es955-yaml. Bind both independently, never substitute
            # the materialized corpus path for the product source root.
            assert (pathlib.Path.home()/'es955-yaml').resolve()==pathlib.Path(cfg['yaml_root']).resolve(),'YAML launcher HOME would select a different fixture corpus'
            assert all((pathlib.Path(cfg['yaml_root'])/suite/sub).is_dir() for suite in ('oss','main','sec','plug') for sub in ('api','test'))
            env=dict(os.environ,HOME='/home/ubuntu',CARGO_TARGET_DIR=str(target),CARGO_BUILD_JOBS='24',RUSTFLAGS='-C target-cpu=native',PATH='/home/ubuntu/.cargo/bin:'+os.environ['PATH'])
            old=os.getcwd();os.chdir(source)
            try:code=run(['cargo','+stable','build','--locked','--profile','fast','-p','es-cluster-yaml','-p','elasticsearch-faults'],out/'helpers-build.log',env)
            finally:os.chdir(old)
            assert code==0,'Matching helper build failed'
            runner=out/'es-cluster-yaml';harness=out/'elasticsearch-faults';shutil.copy2(target/'fast/es-cluster-yaml',runner);shutil.copy2(target/'fast/elasticsearch-faults',harness)
            inventory(source,cfg['manifest'],cfg['manifest_sha256']);pinned(node,cfg['candidate_sha256'])
            save(out/'provenance.json',dict(cfg,runner_sha256=sha(runner),harness_sha256=sha(harness),launcher_sha256=sha(__file__)))
            for stage in ('yaml','fault','differential'):
                work=out/stage;work.mkdir();row={'stage':stage,'exit':1};summary['stages'].append(row)
                try:
                    idle();pinned(node,cfg['candidate_sha256'])
                    if stage=='yaml':code=run(['python3',source/'bench/path-poc/es9-parity/parity_all.py','--root',source,'--runner',runner,'--binary',node,'--run-name',cfg['run_name'],'--suites','oss,main,sec,plug','--parallel','3','--port-base','24000'],work/'matrix.log')
                    elif stage=='fault':code=run(['sudo','-n','env','HOME=/home/ubuntu','python3',source/'bench/path-poc/pr3-gates/fault_matrix.py','--node',node,'--node-sha256',cfg['candidate_sha256'],'--harness',harness,'--harness-sha256',sha(harness),'--source-revision',cfg['revision'],'--out',work/'runs','--port-base',str(fault_port_base(summary)),'--seeds','3'],work/'matrix.log')
                    else:code=differential(cfg,source,work,node,runner)
                    row['exit']=code
                except Exception as e:row['error']=str(e)
                (work/'exit.code').write_text(str(row['exit'])+'\n');save(out/'summary.json',summary)
            summary['complete']=all(row['exit']==0 for row in summary['stages'])
    except Exception as error:summary['error']=str(error);raise
    finally:save(out/'summary.json',summary);(out/'exit.code').write_text('0\n' if summary['complete'] else '1\n')
    if not summary['complete']:raise SystemExit(1)
if __name__=='__main__':main()
