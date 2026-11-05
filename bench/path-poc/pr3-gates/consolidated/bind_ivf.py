#!/usr/bin/env python3
"""Bind already generated full IVF fixtures to reviewed policy/live engines.
Source preparation only: writes configs/argv records, never invokes a workload.
"""
import argparse,json,math,pathlib
from common import ec2,pin,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--policy',required=True);p.add_argument('--engines',required=True);p.add_argument('--out',required=True);p.add_argument('--engine',choices=['candidate','master','es9'],help='Bind only the currently running engine; permits clean sequential fixtures');a=p.parse_args();ec2()
    fixtures=json.loads(pathlib.Path(a.fixtures).read_text());policy=json.loads(pathlib.Path(a.policy).read_text());engines=json.loads(pathlib.Path(a.engines).read_text());out=pathlib.Path(a.out).resolve();assert not out.exists() and out.parent.is_dir();assert fixtures['complete'] and set(fixtures['dimensions'])=={'128','768'}
    expected={a.engine} if a.engine else {'candidate','master','es9'}
    assert set(engines)==expected and policy['review_reference'];assert 0<policy['k']<=policy['num_candidates']<=10000 and policy['repetitions']>=3 and policy['mixed_seconds']>=120 and policy['mixed_updates_per_second']>0
    measure_only=policy.get('measurements_only',False);assert isinstance(measure_only,bool)
    if not measure_only:assert math.isfinite(policy['minimum_recall']) and 0<policy['minimum_recall']<=1
    groups={f'{phase}/{variant}' for phase in ('initial','deleted','mixed') for variant in ('all','tag','tag-range')}
    for dim in (() if measure_only else ('128','768')):
        assert set(policy['thresholds'][dim])==groups
        for thresholds in policy['thresholds'][dim].values():
            assert all(math.isfinite(v) for v in thresholds.values()) and 0<thresholds['minimum_recall']<=1 and 0<=thresholds['max_recall_loss']<=1 and thresholds['max_p99_ratio']>0
    out.mkdir();runs=[];candidate=engines.get('candidate',{});comparison={'source_revision':candidate.get('source_revision'),'binary_sha256':candidate.get('binary_sha256'),'out':str(out/'comparison-result'),'dimensions':{},'measurements_only':measure_only}
    for dim in ('128','768'):
        manifest=json.loads(pin(fixtures['dimensions'][dim]['manifest']).read_text());assert manifest['documents']==1000000 and manifest['dimensions']==int(dim) and manifest['query_count']>=100
        vectors=pin(manifest['vectors']);queries=pin(manifest['queries']);assert vectors.stat().st_size==1000000*int(dim)*4 and queries.stat().st_size==manifest['query_count']*int(dim)*4
        item={'thresholds':policy.get('thresholds',{}).get(dim,{})}
        for name,engine in engines.items():
            proof=json.loads(pin(engine['fixture_provenance']).read_text());assert len(proof['nodes'])==2
            executable=pin(engine['process_executable']);assert all(n['binary_sha256']==sha(executable) for n in proof['nodes'])
            assert engine['distribution_files'];[pin(artifact) for artifact in engine['distribution_files']];pin(engine['conditions'])
            assert sha(executable)==engine['binary_sha256']
            if name in ('candidate','master'):
                build=json.loads(pin(engine['build_provenance']).read_text());assert build['source_revision']==engine['source_revision'] and build['binary_sha256']==engine['binary_sha256']
            else:assert engine['source_revision']=='es9-9.5.5'
            config={key:policy[key] for key in ('k','num_candidates','repetitions','mixed_seconds','mixed_updates_per_second')}
            config.update(measurements_only=measure_only,minimum_recall=policy.get('minimum_recall'),source_revision=engine['source_revision'],binary_sha256=engine['binary_sha256'],build_provenance=engine.get('build_provenance'));config.update(documents=1000000,dimensions=int(dim),query_count=manifest['query_count'],vectors=manifest['vectors'],queries=manifest['queries'],fixture_provenance=engine['fixture_provenance'],distribution_files=engine['distribution_files'])
            config_path=out/f'{name}-{dim}.json';save(config_path,config);result=out/f'{name}-{dim}-result'
            runs.append({'engine':name,'dimensions':int(dim),'config':{'path':str(config_path),'sha256':sha(config_path)},'result':str(result),'argv':['python3',str(pathlib.Path(__file__).with_name('ivf_compare.py').resolve()),'--config',str(config_path),'--engine',name,'--url',engine['url'],'--out',str(result)]})
            item[name]={'path':str(result/'summary.json'),'sha256':None};item[name+'_provenance']={'path':str(result/'provenance.json'),'sha256':None};item[name+'_conditions']=engine['conditions']
        comparison['dimensions'][dim]=item
    save(out/'runs.json',{'status':'prepared_not_executed','fixture_input_sha256':sha(a.fixtures),'policy_input_sha256':sha(a.policy),'engines_input_sha256':sha(a.engines),'runs':runs,'comparison_instruction':'After all six exact runs complete, hash actual summary/provenance files into comparison.pending.json; never predeclare hashes or a pass.'})
    if not a.engine:save(out/'comparison.pending.json',comparison)
if __name__=='__main__':main()
