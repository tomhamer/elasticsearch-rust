#!/usr/bin/env python3
"""Run pinned performance commands and compare every repetition, never averages
that hide failures. Thresholds are explicit reviewed configuration, not defaults.
"""
import argparse, csv, json, math, pathlib
from common import command, ec2, idle_compilers, pin, save, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();ec2();cfg=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    result={'complete':False,'source_revision':cfg['source_revision'],'binary_sha256':cfg['binary_sha256'],'passed':0,'failed':1,'skipped':0,'excluded':[],'raw_artifacts':[],'details':{}}
    try:
        for artifact in cfg['artifacts']:pin(artifact)
        if cfg['kind']=='profile':
            prior=[json.loads(pin(item).read_text()) for item in cfg['completed_timing_reports']]
            assert prior and all(r['complete'] and not r['failed'] for r in prior),'Authoritative timing gates not complete'
        for i,argv in enumerate(cfg['commands']):idle_compilers();command(argv,out/f'command-{i}.log');idle_compilers()
        kind=cfg['kind'];measure_only=cfg.get('measurements_only',False);assert isinstance(measure_only,bool);runs=cfg['runs'];raw=[];measurements={};conditions={}
        assert kind in ('mixed','idle','segments','rally','profile')
        if kind=='profile':
            profile=pin(cfg['profile']);assert profile.stat().st_size>0
            prior=[json.loads(pin(item).read_text()) for item in cfg['completed_timing_reports']]
            assert prior and all(r['complete'] and not r['failed'] for r in prior)
            # perf.data is retained on EC2; text report may be collected locally.
            command(['perf','report','--stdio','--input',str(profile)],out/'perf-report.txt')
            result.update(complete=True,passed=1,failed=0);result['details']={'diagnostic_only':True,'after_authoritative_timings':True,'profile_artifact_sha256':sha(profile)};raw=[profile,out/'perf-report.txt']
        else:
            assert len(runs)>=9
            for run in runs:
                engine=run['engine'];rep=int(run['repetition']);assert engine in ('candidate','master','es9') and rep>=1
                native=pathlib.Path(run['root']);assert (native/'exit.code').read_text().strip()=='0'
                expected_engine=pin(cfg['engine_artifacts'][engine]);assert engine!='candidate' or sha(expected_engine)==cfg['binary_sha256']
                if kind=='rally':
                    expected=pin(run['process_executable']);assert sha(expected)==sha(expected_engine)
                    for boundary in ('live-before.json','live-after.json'):
                        live=json.loads((native/boundary).read_text());assert len(live)==2 and all(row['binary_sha256']==sha(expected) for row in live);raw.append(native/boundary)
                else:
                    for boundary in ('live-before.json','live-after.json'):
                        live=json.loads((native/boundary).read_text());assert live['engine']==engine and live['instrumentation']=='off' and len(live['nodes'])==2
                        expected=pin(run['binary']);assert sha(expected)==sha(expected_engine) and live['launcher_sha256']==sha(expected)
                        assert all(node['exe_sha256']==live['expected_executable_sha256'] for node in live['nodes'])
                        if engine=='es9':assert live['distribution_sha256']==sha(pin(run['distribution_manifest']))
                        else:assert live['expected_executable_sha256']==sha(expected)
                        raw.append(native/boundary)
                condition=json.loads(pin(run['conditions']).read_text())
                assert condition['instrumentation']=='off' and condition['dedicated_host'] and condition['cpu_affinity'] and condition['effective_memory_limit']>0
                if kind!='rally':
                    for boundary in ('live-before.json','live-after.json'):
                        live=json.loads((native/boundary).read_text());assert condition['instance_id'] in live['machine']['identity'] and condition['kernel']==live['machine']['kernel'] and condition['storage']==live['storage_device']
                        assert all(node['affinity']==condition['cpu_affinity'] and node['memory_max']==condition['effective_memory_limit'] for node in live['nodes'])
                else:
                    for boundary in ('live-before.json','live-after.json'):
                        facts=json.loads((native/boundary).read_text());assert all(row['instance_id']==condition['instance_id'] and row['kernel']==condition['kernel'] and row['storage_device']==condition['storage'] and row['affinity']==condition['cpu_affinity'] and row['memory_max']==condition['effective_memory_limit'] for row in facts)
                match={k:condition[k] for k in ('instance_id','cpu_affinity','driver_affinity','effective_memory_limit','storage','kernel','workload')}
                source=run['source_revision'];assert source==({'candidate':cfg['source_revision'],'master':cfg['master_revision'],'es9':'es9-9.5.5'}[engine])
                if kind=='rally':
                    summary=json.loads((native/'summary.json').read_text());assert summary['passed'] and not summary['failed'] and not summary['skipped']
                    provenance=json.loads((native/'provenance.json').read_text());assert provenance['driver_affinity']==condition['driver_affinity']
                    assert provenance['full_original_corpus'] and provenance['config']['query_fraction']==1
                    for corpus in provenance['config']['original_corpora']:pin(corpus['artifact'])
                    metrics={}
                    for row in csv.DictReader((native/'rally.csv').open()):
                        if row.get('Task') and row.get('Metric') in ('99th percentile latency','Median Throughput'):
                            metrics[row['Task']+'/'+row['Metric']]=float(row['Value'])
                    assert set(metrics)==set(cfg['required_metrics'][run['track']]) and metrics,'Missing required Rally metrics'
                    assert all(math.isfinite(v) and v>0 for v in metrics.values())
                    scope=run['track'];key=(engine,rep,scope);assert key not in measurements,'Duplicate measurement';measurements[key]=metrics;raw += [native/'rally.csv',native/'summary.json',native/'provenance.json',native/'health.jsonl']
                else:
                    measurement=native/'measurement';value=json.loads((measurement/'result.json').read_text());proof=json.loads((measurement/'provenance.json').read_text())
                    assert proof['driver_affinity']==condition['driver_affinity']
                    assert value['complete'] and proof['instrumentation']=='off' and proof['source']==source and proof['mode']==kind
                    phases=value['phases'];assert all(row['errors']==0 for row in phases)
                    if kind=='mixed':
                        assert [r['label'] for r in phases]==['0-r1','1-r0','2-r1']
                        for i,replicas in enumerate([1,0,1]):
                            ready=json.loads((measurement/f'ready-{i}.json').read_text());assert ready['active_shards']==proof['shards']*(replicas+1) and ready['status']=='green'
                    elif kind=='idle':assert [r['label'] for r in phases]==['indexing','idle-collect','post-idle-indexing']
                    else:
                        def segments(path):
                            s=json.loads(path.read_text())['segments'];return sum(len(copy['segments']) for index in s['indices'].values() for copies in index['shards'].values() for copy in copies)
                        assert segments(measurement/'small-segments-before.json')>segments(measurement/'merged-segments-before.json'),'No observed segment-count difference'
                    metrics={}
                    for row in phases:
                        required=('p99_ms','docs_per_s') if kind=='mixed' else (('p99_ms',) if kind=='segments' else (('docs_per_s',) if row['label']!='idle-collect' else ()))
                        for name in required:
                            value=row[name];assert isinstance(value,(int,float)) and math.isfinite(value) and value>0,(kind,row['label'],name,value)
                            metrics[row['label']+'/'+name]=value
                    assert metrics,'No authoritative metrics'
                    scope=str(proof['shards']);key=(engine,rep,scope);assert key not in measurements,'Duplicate measurement';measurements[key]=metrics;raw += [measurement/'result.json',measurement/'provenance.json',measurement/'events.jsonl']
                if scope not in conditions:conditions[scope]=match
                assert match==conditions[scope],'Engine measurement conditions differ within scope'
                raw.append(pin(run['conditions']))
            scopes={key[2] for key in measurements};reps={key[1] for key in measurements};assert len(reps)>=3
            assert set(measurements)=={(engine,rep,scope) for engine in ('candidate','master','es9') for rep in reps for scope in scopes}
            if kind=='rally':assert scopes==set(cfg['required_tracks']) and cfg['required_tracks']
            if kind=='segments':assert scopes=={'2','16'}
            ratios=[]
            for rep in reps:
                for scope in scopes:
                    candidate=measurements[('candidate',rep,scope)]
                    for engine in ('master','es9'):
                        reference=measurements[(engine,rep,scope)];assert set(candidate)==set(reference)
                        for metric,value in candidate.items():
                            denominator=reference[metric];assert denominator>0
                            ratio=value/denominator;threshold=None if measure_only else cfg['thresholds'][engine][metric]
                            if not measure_only:assert math.isfinite(threshold) and threshold>0
                            if not measure_only and ('Throughput' in metric or metric.endswith('docs_per_s')):assert ratio>=threshold,(engine,rep,scope,metric,ratio,threshold)
                            elif not measure_only:assert ratio<=threshold,(engine,rep,scope,metric,ratio,threshold)
                            ratios.append({'engine':engine,'rep':rep,'scope':scope,'metric':metric,'ratio':ratio,'threshold':threshold})
            assert ratios,'No comparisons executed'
            save(out/'ratios.json',ratios);raw.append(out/'ratios.json');result.update(complete=not measure_only,measurements_complete=True,passed=0 if measure_only else len(ratios),failed=0)
            d=result['details'];d.update(engines=['candidate','master','es9'],master_revision=cfg['master_revision'],es_version='9.5.5',matching_conditions=True,repetitions=len(reps),thresholds_passed=not measure_only,acceptance_pending=measure_only,tracing=False,profiling=False,build_overlap=False)
            if kind=='mixed':d.update(replica_phases=[1,0,1],active_copy_checks=True,query_p99=True,achieved_write_rate=True)
            if kind=='rally':d.update(full_original_corpus=True,test_mode=False,all_original_tracks_accounted=True,final_counts_verified=True)
            if kind=='idle':d.update(pre_idle_post=True,hook_comparison_scope='current-master whole-product comparison; not isolated hook causality')
            if kind=='segments':d.update(observed_segment_count_difference=True,shard_counts=[2,16])
        result['raw_artifacts']=[{'path':str(path.resolve()),'sha256':sha(path)} for path in dict.fromkeys(raw)]
    finally:save(out/'summary.json',result);(out/'exit.code').write_text('0\n' if (result['complete'] or result.get('measurements_complete')) and not result['failed'] else '1\n')
if __name__=='__main__':main()
