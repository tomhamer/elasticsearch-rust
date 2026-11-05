#!/usr/bin/env python3
"""Create the exact standalone PR3 IVF fixtures on an idle EC2 host.
This generator never reads or samples Rally data and never contacts an engine.
"""
import argparse,fcntl,hashlib,json,os,pathlib,platform,shutil,sys,time
from common import ec2,idle_compilers,save,sha

DOCUMENTS=1_000_000
DIMENSIONS=(128,768)
CHUNK=4096

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--queries',type=int,default=100);p.add_argument('--seed',type=int,default=73);p.add_argument('--query-seed',type=int,default=7301);a=p.parse_args()
    machine=ec2();idle_compilers()
    lease=open('/tmp/es-pr3-validation-host.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:name=(proc/'comm').read_text().strip();argv=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert name not in ('java','elasticsearch','elasticsearch-n','rustic-node') and not any(b'esrally' in arg for arg in argv),'Fixture generation must not overlap engine/Rally workloads'
    assert 100<=a.queries<=10000 and a.seed>=0 and a.query_seed>=0 and a.seed!=a.query_seed
    root=pathlib.Path(a.out).resolve();assert root.is_absolute() and root.parent.is_dir() and not root.exists()
    required=sum((DOCUMENTS+a.queries)*dim*4 for dim in DIMENSIONS);assert shutil.disk_usage(root.parent).free>required+1_000_000_000,'Insufficient fixture space'
    # Set before importing NumPy; generation is separate from all timings.
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    import numpy as np
    root.mkdir();status={'complete':False,'scope':'standalone synthetic IVF, never Rally','started':time.time(),'dimensions':{},'machine':machine}
    save(root/'status.json',status)
    try:
        for dim in DIMENSIONS:
            directory=root/str(dim);directory.mkdir();rng=np.random.Generator(np.random.PCG64(a.seed));qrng=np.random.Generator(np.random.PCG64(a.query_seed))
            centers=rng.normal(size=(100,dim)).astype('<f4');centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            query=(centers[np.arange(a.queries)%100]+qrng.normal(0,.07,(a.queries,dim)).astype('<f4')).astype('<f4')
            query/=np.linalg.norm(query,axis=1,keepdims=True);assert np.isfinite(query).all()
            query_path=directory/'queries.f32'
            with query_path.open('xb') as stream:stream.write(query.astype('<f4',copy=False).tobytes());stream.flush();os.fsync(stream.fileno())
            partial=directory/'vectors.f32.partial';digest=hashlib.sha256()
            with partial.open('xb') as stream:
                for start in range(0,DOCUMENTS,CHUNK):
                    count=min(CHUNK,DOCUMENTS-start);ids=np.arange(start,start+count)
                    rows=(centers[ids%100]+rng.normal(0,.07,(count,dim)).astype('<f4')).astype('<f4');rows/=np.linalg.norm(rows,axis=1,keepdims=True)
                    assert np.isfinite(rows).all() and (np.linalg.norm(rows,axis=1)>0).all()
                    data=rows.astype('<f4',copy=False).tobytes();stream.write(data);digest.update(data)
                stream.flush();os.fsync(stream.fileno())
            vectors=directory/'vectors.f32';partial.rename(vectors);assert vectors.stat().st_size==DOCUMENTS*dim*4 and query_path.stat().st_size==a.queries*dim*4
            manifest={'scope':'complete requested standalone synthetic IVF corpus; not a prefix or Rally dataset','documents':DOCUMENTS,'dimensions':dim,'query_count':a.queries,'format':'little-endian float32 row-major without header','seed':a.seed,'query_seed':a.query_seed,'rng':'NumPy PCG64','chunk_rows':CHUNK,'centers':100,'noise_sigma':.07,'normalization':'float32 L2','numpy_version':np.__version__,'python_version':platform.python_version(),'generator':{'path':str(pathlib.Path(__file__).resolve()),'sha256':sha(__file__)},'vectors':{'path':str(vectors),'sha256':digest.hexdigest()},'queries':{'path':str(query_path),'sha256':sha(query_path)},'metadata':{'id':'decimal ordinal0..999999','tag':'ordinal %10 as keyword','tick_initial':0,'filters':['all','tag=3','tag=3 AND250000<=ordinal<750000'],'delete_rule':'ordinal %17 ==0','mixed_update_rule':'tick only; vectors remain unchanged','ground_truth':'computed independently by ivf_compare over every float vector, including filter/deletion masks'},'generation_instrumentation':'no engine process or timings; one NumPy worker'}
            save(directory/'manifest.json',manifest);status['dimensions'][str(dim)]={'manifest':{'path':str(directory/'manifest.json'),'sha256':sha(directory/'manifest.json')},'vectors':manifest['vectors'],'queries':manifest['queries']};save(root/'status.json',status)
        idle_compilers();status.update(complete=True,finished=time.time());save(root/'fixture-bindings.json',status)
    finally:
        save(root/'status.json',status);(root/'exit.code').write_text('0\n' if status['complete'] else '1\n')
if __name__=='__main__':main()
