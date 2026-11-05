#!/usr/bin/env python3
"""EC2-only explicit supplemental coverage; immutable source, no automatic launch."""
import argparse, fcntl, hashlib, json, os, pathlib, re, shlex, stat, subprocess, time
from consolidated_correctness import idle, inventory, pinned, run, save, sha

TESTS = (
 ('coordination-soak','elasticsearch-cluster','--lib','coordination::sim_scenarios::soak',{'COORD_SEEDS':'100000','COORD_FIRST_SEED':'1000000'}),
 ('dictionary-above-four-gib','rustic-core','--lib','codec::terms::tests::dictionary_above_four_gib_round_trips',{}),
 ('hnsw-four-workers','rustic-core','--lib','vector::hnsw::mutable::tests::adaptive_distances_preserve_serial_arithmetic_and_order',{'RUSTIC_HNSW_QUERY_WORKERS':'4'}),
 ('real-s3-cycle','elasticsearch-snapshots','s3_repository','s3_full_cycle',{}),
)
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


def prerequisites(rows):
    found=[];services=set();directories=set()
    for row in rows:
        path=pathlib.Path(row['results']).resolve()
        props=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','show',row['service'],'--property=Id','--property=ActiveState'],text=True).splitlines() if '=' in line)
        assert props.get('Id') and props['ActiveState'] in ('inactive','failed'),'Prerequisite not completed'
        assert props['Id'] not in services and str(path) not in directories,'Duplicate prerequisite'
        services.add(props['Id']);directories.add(str(path))
        found.append(dict(service=props['Id'],results=str(path),exit=int((path/'exit.code').read_text())))
    assert len(found)==2,'Require full suite and scheduled correctness completion'
    return found

def protected_s3(path):
    path=pathlib.Path(path);info=path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_mode&0o077==0,'S3 environment must be private owned regular file'
    allowed={'AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN','AWS_REGION','ES_TEST_S3_BUCKET'}
    result={}
    for line in path.read_text().splitlines():
        words=shlex.split(line,comments=True)
        if not words:continue
        if words[0]=='export':words=words[1:]
        assert len(words)==1 and '=' in words[0],'Environment assignment only'
        key,value=words[0].split('=',1);assert key in allowed and key not in result
        result[key]=value
    assert all(result.get(k) for k in ('AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_REGION','ES_TEST_S3_BUCKET')),'Missing protected S3 binding'
    assert re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]',result['ES_TEST_S3_BUCKET'])
    return result

def native_result(log,selector,target):
    text=log.read_text(errors='replace')
    rows=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    assert rows==[('ok','1','0','0')],'Expected exactly one explicit native pass'
    assert re.search(r'^test '+re.escape(selector)+r' \.\.\.',text,re.M),'Expected exact test identity'
    executables=[]
    for line in text.splitlines():
        try:item=json.loads(line)
        except ValueError:continue
        if item.get('reason')=='compiler-artifact' and item.get('profile',{}).get('test') and item.get('executable'):
            executable=pathlib.Path(item['executable']).resolve()
            assert executable.is_relative_to(target.resolve())
            executables.append({'path':str(executable),'sha256':sha(executable)})
    assert executables,'Missing exact test binary provenance'
    return executables

def main():
    if not __debug__:raise RuntimeError('Assertions required')
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(pathlib.Path(a.config).read_text())
    assert os.uname().sysname=='Linux' and readable_ec2_identity(),'EC2 only'
    pinned(__file__,cfg['launcher_sha256']);pinned(pathlib.Path(__file__).with_name('consolidated_correctness.py'),cfg['helper_sha256'])
    source=pathlib.Path(cfg['source']);target=pathlib.Path(cfg['target']);out=pathlib.Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    summary={'complete':False,'revision':cfg['revision'],'stages':[]}
    try:
        summary['prerequisites']=prerequisites(cfg['completed'])
        assert (source/'.pr3-source-revision').read_text().strip()==cfg['revision']
        inventory(source,cfg['manifest'],cfg['manifest_sha256']);pinned(cfg['candidate'],cfg['candidate_sha256']);pinned(cfg['cargo'],cfg['cargo_sha256'])
        ledger=json.loads(pathlib.Path(cfg['ledger']).read_text());pinned(cfg['ledger'],cfg['ledger_sha256'])
        assert ledger['source_revision']==cfg['revision']
        required=[r for r in ledger['records'] if r.get('required')];assert len(required)==5
        for row in required:pinned(source/row['source_path'],row['source_sha256'])
        # Original native ignored inventory must be present and contain each
        # selector; filtered zero-test output is never equivalent coverage.
        logs=pathlib.Path(cfg['completed'][0]['results'])
        for row in required:
            name,test=row['id'].split('::',1)
            assert re.search(r'^test '+re.escape(test)+r' \.\.\. ignored', (logs/name).read_text(errors='replace'),re.M),'Missing native ignored inventory'
        env=dict(os.environ,HOME='/home/ubuntu',CARGO_TARGET_DIR=str(target),CARGO_BUILD_JOBS='24',RUSTFLAGS='-C target-cpu=native',RUSTUP_TOOLCHAIN='stable')
        assert not any(env.get(k) for k in ('RUSTIC_PROFILE','RUSTIC_TRACING','ES_YAML_CLAIM_ALL','ES_TEST_S3_ENDPOINT','ES_TEST_S3_PATH_STYLE'))
        assert not any(value for key,value in env.items() if key.startswith('AWS_ENDPOINT_URL')), 'Custom AWS endpoints forbidden'
        s3=protected_s3(cfg['s3_environment'])
        with open('/tmp/es-pr3-validation-host.lock','a+') as host:
            fcntl.flock(host,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with open(target/'.pr3-build.lock','a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                for key,package,kind,selector,extra in TESTS:
                    idle();inventory(source,cfg['manifest'],cfg['manifest_sha256'])
                    row={'stage':key,'test':selector,'exit':1};summary['stages'].append(row)
                    argv=[cfg['cargo'],'+stable','test','--locked','--profile','fast','--message-format=json','-p',package]+(['--lib'] if kind=='--lib' else ['--test',kind])+[selector,'--','--exact','--ignored','--nocapture','--test-threads=1']
                    row['argv']=argv
                    test_env=dict(env,**extra)
                    if key=='real-s3-cycle':test_env.update(s3)
                    old=os.getcwd();os.chdir(source)
                    try:code=run(argv,out/(key+'.log'),test_env)
                    finally:os.chdir(old)
                    row['exit']=code;save(out/'summary.json',summary)
                    assert code==0,'Supplemental test failed: '+key
                    row['binaries']=native_result(out/(key+'.log'),selector,target)
                    if key=='real-s3-cycle':
                        text=(out/(key+'.log')).read_text(errors='replace')
                        prefixes=re.findall(r'cleanup of s3://'+re.escape(s3['ES_TEST_S3_BUCKET'])+r'/(es-snapshots-test/[0-9a-f]{16}): Ok\(',text)
                        assert len(prefixes)==1,'Missing unique test-owned S3 cleanup proof'
                        row['owned_prefix']=prefixes[0]
                    save(out/'summary.json',summary)
            # Oracle owns its independent lock/isolated target. Do not hold the
            # shared build lock recursively; host lock still serializes schedules.
            idle();oracle=out/'oracle';oracle.mkdir()
            argv=['python3',str(source/'bench/path-poc/pr3-vector-oracle/run_ec2.py'),'--source',str(source),'--source-manifest',cfg['manifest'],'--source-manifest-sha256',cfg['manifest_sha256'],'--revision',cfg['revision'],'--out',str(oracle),'--cargo',cfg['cargo']]
            for row in cfg['completed']:argv+=['--completed',row['service']+'='+row['results']]
            code=run(argv,out/'oracle.log',env);summary['stages'].append({'stage':'lucene-oracle','exit':code});assert code==0
            runs=list(oracle.glob('encoded-oracle-*'));assert len(runs)==1
            result=json.loads((runs[0]/'summary.json').read_text());assert result['oracle_cases']==192 and (result['rust_pass'],result['rust_fail'],result['rust_skip'])==(1,0,0)
            inventory(source,cfg['manifest'],cfg['manifest_sha256']);summary['complete']=True
    finally:
        save(out/'summary.json',summary);(out/'exit.code').write_text('0\n' if summary['complete'] else '1\n')
if __name__=='__main__':main()
