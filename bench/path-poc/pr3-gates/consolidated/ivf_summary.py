#!/usr/bin/env python3
"""Compare pinned full-corpus candidate/ES IVF results without rerunning them."""
import argparse,json,math,pathlib
from common import ec2,pin,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();ec2();cfg=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    result={'complete':False,'passed':0,'failed':1,'skipped':0,'excluded':[],'source_revision':cfg['source_revision'],'binary_sha256':cfg['binary_sha256'],'details':{},'raw_artifacts':[]}
    try:
        rows=[];raw=[];measure_only=cfg.get('measurements_only',False);assert isinstance(measure_only,bool)
        for dim in (128,768):
            item=cfg['dimensions'][str(dim)];candidate=json.loads(pin(item['candidate']).read_text());reference=json.loads(pin(item['es9']).read_text());raw.extend([pin(item['candidate']),pin(item['es9'])])
            proofs=[json.loads(pin(item[k]).read_text()) for k in ('candidate_provenance','es9_provenance')];raw.extend(pin(item[k]) for k in ('candidate_provenance','es9_provenance'))
            a,b=[v['config'] for v in proofs]
            assert proofs[0]['engine']=='candidate' and proofs[1]['engine']=='es9'
            assert a['source_revision']==cfg['source_revision'] and a['binary_sha256']==cfg['binary_sha256']
            assert b['source_revision']=='es9-9.5.5'
            build_path=pin(a['build_provenance']);build=json.loads(build_path.read_text());raw.append(build_path)
            assert build['source_revision']==cfg['source_revision'] and build['binary_sha256']==cfg['binary_sha256']
            for engine,proof in zip(('candidate','es9'),proofs):
                workload=proof['config'];fixture_path=pin(workload['fixture_provenance']);fixture=json.loads(fixture_path.read_text());raw.append(fixture_path)
                assert len(fixture['nodes'])==2 and all(node['binary_sha256']==workload['binary_sha256'] for node in fixture['nodes'])
                native=pin(item[engine]).parent
                for boundary in ('live-before.json','live-after.json'):
                    observed_path=native/boundary;observed=json.loads(observed_path.read_text());raw.append(observed_path)
                    assert len(observed)==2
                    for actual,expected in zip(observed,fixture['nodes']):
                        assert actual['pid']==expected['pid'] and actual['instance_id']==expected['instance_id'] and actual['binary_sha256']==workload['binary_sha256']
                for artifact in workload['distribution_files']:raw.append(pin(artifact))
                raw.extend([pin(workload['vectors']),pin(workload['queries'])])
            conditions=[json.loads(pin(item[k]).read_text()) for k in ('candidate_conditions','es9_conditions')];raw.extend(pin(item[k]) for k in ('candidate_conditions','es9_conditions'))
            assert conditions[0]==conditions[1] and conditions[0]['instrumentation']=='off' and conditions[0]['dedicated_host']
            for key in ('vectors','queries','dimensions','k','num_candidates','repetitions','mixed_seconds','mixed_updates_per_second'):
                assert a[key]==b[key],('Unmatched IVF conditions',key)
            assert a['dimensions']==dim and a['repetitions']>=3
            for value in (candidate,reference):assert (value['passed'] or value.get('measurements_complete')) and value['documents']==1_000_000 and value['dimensions']==dim and len(value['groups'])==9
            assert candidate['groups'].keys()==reference['groups'].keys()
            for group,value in candidate['groups'].items():
                baseline=reference['groups'][group];threshold=None if measure_only else item['thresholds'][group]
                if not measure_only:assert all(math.isfinite(v) for v in threshold.values()) and 0<threshold['minimum_recall']<=1 and 0<=threshold['max_recall_loss']<=1 and threshold['max_p99_ratio']>0
                assert all(math.isfinite(row[key]) for row in (value,baseline) for key in ('mean_recall','p99_seconds'))
                if not measure_only:assert value['mean_recall']>=threshold['minimum_recall'] and value['mean_recall']>=baseline['mean_recall']-threshold['max_recall_loss']
                assert baseline['p99_seconds']>0
                ratio=value['p99_seconds']/baseline['p99_seconds']
                if not measure_only:assert ratio<=threshold['max_p99_ratio']
                rows.append({'dimensions':dim,'group':group,'p99_ratio':ratio,'candidate_recall':value['mean_recall'],'es_recall':baseline['mean_recall'],'thresholds':threshold})
        save(out/'comparisons.json',rows);raw.append(out/'comparisons.json');result.update(complete=not measure_only,measurements_complete=True,passed=0 if measure_only else len(rows),failed=0,details={'dimensions':[128,768],'documents':1_000_000,'thresholds_passed':not measure_only,'acceptance_pending':measure_only,'engines':['candidate','es9']})
        result['raw_artifacts']=[{'path':str(p.resolve()),'sha256':sha(p)} for p in raw]
    finally:save(out/'summary.json',result);(out/'exit.code').write_text('0\n' if result.get('measurements_complete') else '1\n')
if __name__=='__main__':main()
