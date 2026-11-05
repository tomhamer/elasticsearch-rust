#!/usr/bin/env python3
"""Explicit real-hardware GPU correctness gate; no work on import.
Root supplies immutable source, complete YAML and CUDA-library pins. This never
provisions hardware, downloads software, mutates source, or runs performance.
"""
import argparse, fcntl, json, os, pathlib, re, shutil, subprocess, sys
from consolidated_correctness import inventory, pinned, run, save, sha, idle, ports_free

JOBS={'plug-gpu-single','plug-gpu-multi','plug-gpu-mixed','plug-gpu-disabled','plug-gpu-unsupported'}
REQUIRED_TEST='vector::gpu::tests::hardware_required_builds_graphs'

def artifact(log,name,out):
    found=[]
    for line in log.read_text().splitlines():
        row=json.loads(line)
        if row.get('reason')=='compiler-artifact' and row.get('target',{}).get('name')==name and row.get('executable'):found.append(pathlib.Path(row['executable']))
    assert len(set(found))==1,('Ambiguous/missing Cargo executable',name)
    shutil.copy2(found[0],out);return {'path':str(out),'sha256':sha(out)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    if not __debug__:raise RuntimeError('Assertions are required')
    cfg=json.loads(pathlib.Path(a.config).read_text());source=pathlib.Path(cfg['source']).resolve(strict=True)
    sys.path.insert(0,str(source/'bench/path-poc/pr3-gates/consolidated'))
    from common import ec2
    machine=ec2();assert re.fullmatch('[0-9a-f]{40}',cfg['source_revision'])
    assert (source/'.pr3-source-revision').read_text().strip()==cfg['source_revision']
    out=pathlib.Path(cfg['out']);assert out.is_absolute();out.mkdir(parents=True,exist_ok=False);out=out.resolve()
    os.chdir(source)
    state={'complete':False,'passed':False,'source_revision':cfg['source_revision'],'machine':machine,'runtime_validation':True}
    try:
        with open('/tmp/es-pr3-validation-host.lock','a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);idle()
            inventory(source,cfg['source_manifest']['path'],cfg['source_manifest']['sha256'])
            yaml=pathlib.Path(cfg['yaml_root']).resolve(strict=True)
            inventory(yaml,cfg['yaml_manifest']['path'],cfg['yaml_manifest']['sha256'])
            assert yaml==(pathlib.Path.home()/'es955-yaml').resolve(strict=True) and (yaml/'plug/test/gpu').is_dir() and cfg['jobs']>0
            corpus=json.loads(pathlib.Path(cfg['yaml_manifest']['path']).read_text())
            actual={str(p.relative_to(yaml)) for p in (yaml/'plug/test/gpu').rglob('*') if p.is_file()}
            assert actual and actual <= set(corpus),'GPU fixture inventory incomplete'
            assert not os.environ.get('ES_YAML_CLAIM_ALL'),'Forced YAML gates forbidden'
            env=dict(os.environ);env.update(CARGO_BUILD_JOBS=str(cfg['jobs']),RUSTFLAGS=cfg['rustflags'])
            assert not env.get('LD_PRELOAD'),'Preloaded libraries invalidate runtime binding'
            libdirs=[str(pathlib.Path(v).resolve(strict=True)) for v in cfg['library_directories']]
            assert libdirs;env['LD_LIBRARY_PATH']=':'.join(libdirs)
            libraries=cfg['runtime_libraries'];assert libraries
            for item in libraries:pinned(item['path'],item['sha256'])
            # Explicit full paths bind installed cuVS/driver/runtime objects;
            # child loader sees exactly these ordered directory inputs.
            names=[pathlib.Path(v['path']).name for v in libraries]
            assert any(n.startswith('libcuda.so') for n in names) and any(n.startswith('libcudart.so') for n in names) and any(n.startswith('libcuvs_c.so') for n in names)
            probe="""import ctypes,hashlib,json,pathlib,sys
expected={str(pathlib.Path(v['path']).resolve()):v['sha256'] for v in json.loads(sys.argv[1])}
ctypes.CDLL('libcuda.so.1');loaded=None
for name in ('libcuvs_c.so','libcuvs_c.so.26','libcuvs_c.so.25'):
 try:loaded=ctypes.CDLL(name);break
 except OSError:pass
assert loaded is not None,'cuVS unavailable'
runtime=None
for name in ('libcudart.so.12','libcudart.so.13','libcudart.so'):
 try:runtime=ctypes.CDLL(name);break
 except OSError:pass
assert runtime is not None,'CUDA runtime unavailable'
paths={line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if '/' in line}
observed=[]
for name in sorted(paths):
 p=pathlib.Path(name)
 if not p.name.startswith(('libcuda','libcuvs')):continue
 p=p.resolve(strict=True);digest=hashlib.sha256(p.read_bytes()).hexdigest()
 assert expected.get(str(p))==digest,('Unpinned loaded GPU library',str(p))
 observed.append({'path':str(p),'sha256':digest})
assert all(any(pathlib.Path(v['path']).name.startswith(prefix) for v in observed) for prefix in ('libcuda.so','libcudart.so','libcuvs_c.so'))
print(json.dumps(observed))
"""
            assert run([sys.executable,'-c',probe,json.dumps(libraries)],out/'runtime-library-proof.log',env)==0
            assert run(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,compute_cap','--format=csv'],out/'hardware.log',env)==0
            save(out/'inputs.json',cfg)
            binaries={}
            for mode in ('cpu','gpu'):
                idle();target=out/f'target-{mode}';assert not target.exists();env['CARGO_TARGET_DIR']=str(target)
                cargo=['cargo','+stable','build','--locked','--profile','fast','-p','elasticsearch-node','--message-format=json-render-diagnostics']
                if mode=='gpu':cargo+=['-p','es-cluster-yaml','--features','elasticsearch-node/gpu']
                # --manifest-path avoids changing the caller/source tree.
                cargo+=['--manifest-path',str(source/'Cargo.toml')]
                # Rust JSON diagnostics are retained separately from stdout.
                with (out/f'{mode}.jsonl').open('xb') as stdout,(out/f'{mode}.log').open('xb') as stderr:
                    code=subprocess.call(cargo,cwd=source,stdout=stdout,stderr=stderr,env=env)
                (out/f'{mode}.exit').write_text(str(code)+'\n');assert code==0,(mode,code)
                binaries[mode]=artifact(out/f'{mode}.jsonl','elasticsearch-node',out/f'elasticsearch-node-{mode}')
                if mode=='gpu':binaries['runner']=artifact(out/'gpu.jsonl','es-cluster-yaml',out/'es-cluster-yaml')
            assert binaries['cpu']['sha256']!=binaries['gpu']['sha256'],'Feature binaries unexpectedly identical'
            save(out/'binary-provenance.json',{'source_revision':cfg['source_revision'],'binaries':binaries,'libraries':libraries})
            # --include-ignored explicitly executes the hardware-required test;
            # normal portable tiny_segments test may otherwise silently return.
            tests=['cargo','+stable','test','--locked','--profile','fast','--manifest-path',str(source/'Cargo.toml'),'-p','rustic-core','--features','mutable-hnsw-poc,gpu','--lib','--no-run','--message-format=json-render-diagnostics']
            with (out/'unit-build.jsonl').open('xb') as stdout,(out/'unit-build.log').open('xb') as stderr:
                code=subprocess.call(tests,cwd=source,stdout=stdout,stderr=stderr,env=env)
            (out/'unit-build.exit').write_text(str(code)+'\n');assert code==0
            unit=artifact(out/'unit-build.jsonl','rustic_core',out/'rustic-core-gpu-tests')
            save(out/'unit-binary-provenance.json',{'source_revision':cfg['source_revision'],'binary':unit})
            assert run([unit['path'],'vector::gpu::tests::','--include-ignored','--test-threads=1'],out/'gpu-unit.log',env)==0
            text=(out/'gpu-unit.log').read_text();assert f'test {REQUIRED_TEST} ... ok' in text
            totals=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text);assert len(totals)==1 and int(totals[0][0])>0 and totals[0][1:]==('0','0')
            idle();port=int(cfg['port_base']);ports_free(list(range(port,port+2))+list(range(port+100,port+102)))
            name=cfg['run_name'];assert re.fullmatch('[a-z0-9-]+',name)
            reports=source/'parity-results'/name;assert not reports.exists()
            argv=[sys.executable,str(source/'bench/path-poc/es9-parity/parity_all.py'),'--root',str(source),'--binary',binaries['gpu']['path'],'--cpu-binary',binaries['cpu']['path'],'--runner',binaries['runner']['path'],'--run-name',name,'--suites','plug','--modules','gpu','--parallel','1','--port-base',str(port)]
            # parity writes relative to its configured root; preserve raw output.
            code=run(argv,out/'yaml.log',env)
            assert reports.is_dir(),'Missing complete GPU YAML reports'
            shutil.copytree(reports,out/'reports');summary=json.loads((out/'reports/summary.json').read_text())
            assert set(summary['jobs'])==JOBS and summary['complete'] and summary['all_selected_cases_passed'] and code==0
            assert all(v.get('exit')==0 for v in summary['jobs'].values())
            counts=summary['suites']['plug'];assert counts.get('pass',0)>0 and counts.get('fail',0)==counts.get('skip',0)==0
            inventory(source,cfg['source_manifest']['path'],cfg['source_manifest']['sha256'])
            assert (source/'.pr3-source-revision').read_text().strip()==cfg['source_revision']
            state.update(complete=True,passed=True,unit_passed=int(totals[0][0]),yaml=counts,topologies=sorted(JOBS))
    except Exception as e:state['error']=str(e);raise
    finally:save(out/'summary.json',state);(out/'exit.code').write_text('0\n' if state['passed'] else '1\n')
if __name__=='__main__':main()
