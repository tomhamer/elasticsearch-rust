#!/usr/bin/env python3
"""Execute a reviewed complete PR3 plan on EC2; fail closed on absent gates.
Source preparation only until explicitly invoked. Commands are JSON argv arrays,
never shell fragments; all gate adapters and fixtures must be hash-pinned.
"""
import argparse, contextlib, fcntl, json, os, pathlib, re, select, subprocess, sys, time
from common import command, ec2, idle_compilers, pin, save, sha

REQUIRED={'unit-integration','yaml','fault','differential','rolling','pressure','ivf-128-candidate','ivf-128-es9','ivf-768-candidate','ivf-768-es9','ivf-comparison','mixed','rally','idle','segments','profile'}
PERFORMANCE={'ivf-128-candidate','ivf-128-es9','ivf-768-candidate','ivf-768-es9','mixed','rally','idle','segments'}

def counts(report):
    return {k:int(report[k]) for k in ('passed','failed','skipped')}

def validate_report(stage,path):
    r=json.loads(path.read_text());kind=stage['kind']
    assert r['complete'] is True and r['source_revision']==stage['source_revision']
    assert r['binary_sha256']==stage['binary_sha256']
    # Adapter reports must reference original raw artifacts, not just counts.
    assert r['raw_artifacts'];[pin(a) for a in r['raw_artifacts']]
    c=counts(r);assert c['passed']>0 and c['failed']==0
    # Exclusions remain visible and require an explicit scope decision outside
    # this driver; no passing shell exit can turn omitted coverage into a pass.
    if c['skipped'] or r.get('excluded'):
        coverage=r['details']['coverage'];assert coverage['required_uncovered']==0
        assert coverage['skipped_accounted']==c['skipped'] and coverage['excluded_accounted']==len(r.get('excluded',[]))
        assert coverage['scope_ledger_sha256'] and coverage['records'],'Skipped/excluded cases need exact reviewed scope accounting, never pass credit'
    detail=r['details']
    if kind=='unit-integration':assert detail['workspace_targets_enumerated'] and detail['default_features'] and detail['required_feature_targets_accounted']
    if kind=='yaml':assert detail['suites']==['oss','main','sec','plug'] and detail['real_nodes']==2 and detail['module_fixtures'] and not detail['forced_gates'] and not detail['filters']
    if kind=='fault':assert detail['scenarios']==11 and detail['seeds']==3 and detail['nodes']==5 and detail['search_aggregation_requests']>0 and detail['violations']==0
    if kind=='differential':assert detail['rust_nodes']==2 and detail['es_nodes']==2 and detail['es_version']=='9.5.5' and detail['cluster_uuids_distinct']
    if kind=='rolling':assert detail['old_binary_sha256']!=r['binary_sha256'] and detail['old_master_then_new_master'] and detail['writes_under_upgrade'] and detail['downgrade_refused']
    if kind=='pressure':assert len(set(detail['instance_ids']))==3 and detail['accepted_writes']>0 and detail['refused_requests']>0 and detail['lost_acknowledged_writes']==0
    if kind.startswith('ivf-') and kind!='ivf-comparison':assert detail['documents']==1_000_000 and detail['dimensions']==int(kind.split('-')[1]) and detail['filters'] and detail['deletes'] and detail['mixed'] and detail['fixed_num_candidates']
    if kind=='ivf-comparison':assert detail['dimensions']==[128,768] and detail['documents']==1_000_000 and detail['thresholds_passed'] and detail['engines']==['candidate','es9']
    if kind in {'mixed','rally','idle','segments'}:
        assert set(detail['engines'])=={'candidate','master','es9'} and detail['master_revision']==stage['master_revision'] and detail['es_version']=='9.5.5'
        assert detail['matching_conditions'] and detail['repetitions']>=3 and detail['thresholds_passed']
    if kind=='mixed':assert detail['replica_phases']==[1,0,1] and detail['active_copy_checks'] and detail['query_p99'] and detail['achieved_write_rate']
    if kind=='rally':assert detail['full_original_corpus'] and not detail['test_mode'] and detail['all_original_tracks_accounted'] and detail['final_counts_verified']
    if kind=='idle':assert detail['pre_idle_post'] and detail['hook_comparison_scope']
    if kind=='segments':assert detail['observed_segment_count_difference'] and detail['shard_counts']==[2,16]
    if kind=='profile':assert detail['diagnostic_only'] and detail['after_authoritative_timings'] and detail['profile_artifact_sha256']
    if kind in PERFORMANCE:assert not detail['tracing'] and not detail['profiling'] and not detail['build_overlap']
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    summary={'complete':False,'passed':0,'failed':0,'skipped':0,'stages':[]};locks=[];remote=[]
    try:
        machine=ec2();save(out/'machine.json',machine)
        assert re.fullmatch('[0-9a-f]{40}',plan['candidate_revision']) and re.fullmatch('[0-9a-f]{40}',plan['master_revision'])
        assert plan['master_revision']==subprocess.check_output(['git','ls-remote',plan['repository'],'refs/heads/master'],text=True).split()[0],'Master moved; reconcile and repin before validation'
        kinds=[s['kind'] for s in plan['stages']];assert len(kinds)==len(set(kinds)) and set(kinds)==REQUIRED
        assert kinds.index('unit-integration')==0 and kinds[-1]=='profile'
        for item in plan['source_files']:pin(item)
        assert plan['source_files'] and plan['source_manifest_sha256']==sha(pin(plan['source_manifest']))
        for artifact in plan['artifacts']:pin(artifact)
        assert not any(os.environ.get(k) for k in ('ES_YAML_CLAIM_ALL','RUSTIC_HNSW_DIAGNOSTIC_EVERY','RUSTIC_HNSW_WRITER_DIAGNOSTIC_EVERY'))
        for host in plan['hosts']:
            assert host['instance_id'].startswith('i-') and host['purpose'] in ('correctness','performance','pressure')
            if host['local']:
                handle=open('/tmp/es-pr3-validation-host.lock','a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
            else:
                # A live SSH process retains the remote flock until orchestration
                # exits. This never starts/stops an instance or kills workloads.
                argv=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-i',host['ssh_key'],host['ssh_target'],'flock -n /tmp/es-pr3-validation-host.lock sh -c \'echo LOCKED; cat >/dev/null\'']
                process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(out/(host['instance_id']+'-lock.log'),'wb'),text=True)
                remote.append(process)
                assert select.select([process.stdout],[],[],30)[0],'Remote lock handshake timed out'
                assert process.stdout.readline().strip()=='LOCKED','Remote host is busy'
        save(out/'plan.json',plan)
        for number,stage in enumerate(plan['stages']):
            record={'kind':stage['kind'],'passed':False,'start':time.time()};summary['stages'].append(record);save(out/'summary.json',summary)
            try:
                assert stage['source_revision']==plan['candidate_revision'] and stage['master_revision']==plan['master_revision']
                for fixture in stage['fixtures']:pin(fixture)
                pin(stage['adapter'])
                assert stage['argv'][0]==str(pathlib.Path(stage['adapter']['path']).resolve()),'Executable adapter must be pinned'
                if stage['kind'] in PERFORMANCE:idle_compilers()
                # Explicit dedicated-fixture ownership and cleanup commands are
                # reviewed/pinned adapters, not broad pkill/delete operations.
                command(stage['argv'],out/f'{number:02d}-{stage["kind"]}.log')
                report=validate_report(stage,pathlib.Path(stage['report']))
                record.update(passed=True,end=time.time(),report=stage['report'],report_sha256=sha(stage['report']),counts=counts(report));summary['passed']+=1
                if stage['kind'] in PERFORMANCE:idle_compilers()
                save(out/'summary.json',summary)
            except Exception as error:
                record.update(end=time.time(),error=str(error));summary['failed']+=1;save(out/'summary.json',summary)
        summary['complete']=summary['failed']==0
    except Exception as error:summary['failed']+=1;summary['error']=str(error);raise
    finally:
        for process in remote:
            if process.stdin:process.stdin.close()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
        for handle in locks:handle.close()
        save(out/'summary.json',summary);(out/'exit.code').write_text('0\n' if summary['complete'] and not summary['failed'] else '1\n')
if __name__=='__main__':main()
