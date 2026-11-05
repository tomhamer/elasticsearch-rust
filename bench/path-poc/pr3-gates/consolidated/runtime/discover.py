#!/usr/bin/env python3
"""Root-invoked read-only EC2 fixture discovery. Never launches a workload.
Only hashes explicitly known binaries/manifests; large corpus hashes are checked
separately by the gate pins so discovery does not contend with active tests.
"""
import argparse,json,pathlib,re,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from common import ec2,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--known',required=True);p.add_argument('--out',required=True);a=p.parse_args();machine=ec2();known=json.loads(pathlib.Path(a.known).read_text());result={'machine':machine,'status':'discovery_only','files':{},'directories':{},'manifest_candidates':[]}
    paths=known['expected_paths']
    for key in ('candidate_binary','master_binary'):
        path=pathlib.Path(paths[key]);result['files'][key]={'path':str(path),'exists':path.is_file()}
        if path.is_file():result['files'][key].update(bytes=path.stat().st_size,sha256=sha(path))
    for key,path in [('source',paths['source']),('master_source',paths['master_source']),('yaml',known['yaml_root_candidate'])]+[(f'es-{i}',p) for i,p in enumerate(known['es_distribution_candidates'])]+[(f'tracks-{i}',p) for i,p in enumerate(known['tracks']['rendered_candidates'])]:
        root=pathlib.Path(path);result['directories'][key]={'path':path,'exists':root.is_dir()}
        if root.is_dir():result['directories'][key]['immediate_entries']=sorted(p.name for p in root.iterdir())
    for root in (pathlib.Path('/home/ubuntu/perf-validation'),pathlib.Path('/home/ubuntu/recovery-comparison')):
        if not root.is_dir():continue
        # Bounded two-level fixture manifests; do not recursively walk node data.
        for path in sorted(set(root.glob('*/manifest.json'))|set(root.glob('*/*/manifest.json'))):
            if path.stat().st_size>1_000_000:continue
            try:value=json.loads(path.read_text())
            except (ValueError,OSError):continue
            if any(key in value for key in ('dims','dimensions','docs','documents')):
                result['manifest_candidates'].append({'path':str(path),'sha256':sha(path),'manifest':value})
    for kind in ('compressed','uncompressed'):
        item=known['full_dense_corpus'][kind];path=pathlib.Path(item['path']);result['files']['dense-'+kind]={'path':str(path),'exists':path.is_file(),'expected_sha256':item['sha256'],'hash_checked':False}
        if path.is_file():result['files']['dense-'+kind]['bytes']=path.stat().st_size
    native=pathlib.Path(paths['results']);coverage=[]
    suites=native/'suites.tsv'
    if suites.is_file():
        for line in suites.read_text().splitlines():
            package,kind,target=line.split('\t');log=native/f'{package}-{kind}-{target}.log'
            if not log.is_file():continue
            for name in re.findall(r'^test (.+?) \.\.\. ignored(?:,.*)?$',log.read_text(),re.M):coverage.append({'id':log.name+'::'+name,'required':None,'classification':None,'status':'needs exact scope review or explicit execution'})
    omitted=native/'feature-required-targets.json'
    if omitted.is_file():
        for item in json.loads(omitted.read_text()):coverage.append({'id':'feature:'+item['package']+'::'+item['target']['name'],'required':None,'classification':None,'target':item['target'],'status':'needs requested-scope review; not a pass'})
    result['coverage_candidates']=coverage
    out=pathlib.Path(a.out);assert not out.exists();save(out,result)
if __name__=='__main__':main()
