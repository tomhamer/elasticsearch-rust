#!/usr/bin/env python3
"""Pinned native report adapter. Executes explicit commands then reads raw
results. The plan contains commands/paths, never caller-supplied pass counts.
"""
import argparse, json, os, pathlib, re, sys
from common import command, ec2, pin, save, sha

def rust_counts(root):
    exits=[line.split('\t') for line in (root/'suites-exit.tsv').read_text().splitlines()]
    targets=[line.split('\t') for line in (root/'suites.tsv').read_text().splitlines()]
    assert [r[:3] for r in exits]==targets and all(r[3]=='0' for r in exits)
    total={'passed':0,'failed':0,'skipped':0};raw=[]
    for package,kind,target,_ in exits:
        path=root/f'{package}-{kind}-{target}.log';raw.append(path)
        matches=re.findall(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;',path.read_text())
        assert matches,path
        for passed,failed,ignored in matches:
            total['passed']+=int(passed);total['failed']+=int(failed);total['skipped']+=int(ignored)
    return total,raw

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();ec2();cfg=json.loads(pathlib.Path(a.config).read_text());out=pathlib.Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    result={'complete':False,'source_revision':cfg['source_revision'],'binary_sha256':cfg['binary_sha256'],'passed':0,'failed':1,'skipped':0,'excluded':[],'details':{},'raw_artifacts':[]}
    try:
        for artifact in cfg['artifacts']:pin(artifact)
        env=dict(os.environ);env.update(cfg.get('env',{}))
        for i,argv in enumerate(cfg['commands']):command(argv,out/f'command-{i}.log',env)
        native=pathlib.Path(cfg['native']);kind=cfg['kind'];d=result['details'];raw=[]
        if kind=='unit-integration':
            assert (native/'exit.code').read_text().strip()=='0';c,raw=rust_counts(native);result.update(c)
            exclusions=json.loads((native/'feature-required-targets.json').read_text());result['excluded']=exclusions
            d.update(workspace_targets_enumerated=True,default_features=True,required_feature_targets_accounted=not exclusions)
            if exclusions or result['skipped']:
                ledger_path=pin(cfg['coverage_ledger']);ledger=json.loads(ledger_path.read_text());assert ledger['source_revision']==cfg['source_revision']
                ignored=[]
                for path in raw:
                    ignored.extend(path.name+'::'+name for name in re.findall(r'^test (.+?) \.\.\. ignored(?:,.*)?$',path.read_text(),re.M))
                assert len(ignored)==result['skipped'],'Every ignored test must be named in native output'
                omitted=['feature:'+item['package']+'::'+item['target']['name'] for item in exclusions]
                expected=set(ignored+omitted);records=ledger['records'];assert len(records)==len(expected) and {r['id'] for r in records}==expected
                for row in records:
                    assert row['reason'] and row['scope_reference']
                    if row['required']:
                        evidence=pin(row['execution_report']);proof=json.loads(evidence.read_text());assert proof['source_revision']==cfg['source_revision'] and proof['case_id']==row['id'] and proof['executed']==1 and proof['passed']==1 and proof['failed']==proof['skipped']==0;raw.append(evidence)
                    else:assert row['classification'] in ('historical-experiment','non-test-benchmark','outside-requested-scope')
                d['required_feature_targets_accounted']=True;d['coverage']={'required_uncovered':0,'skipped_accounted':len(ignored),'excluded_accounted':len(omitted),'scope_ledger_sha256':sha(ledger_path),'records':records};raw.append(ledger_path)
            raw += [native/'suites-exit.tsv',native/'feature-required-targets.json',native/'metadata.json']
        elif kind=='yaml':
            r=json.loads(native.read_text());assert r['complete'] and r['execution_passed'];scope=r['selected_scope']
            assert scope['suites']==['oss','main','sec','plug'] and not scope['modules'] and not scope['filters']
            # Keep native spelling explicit: parity_all suite counters are pass/fail/skip.
            suites=r['suites'];result.update(passed=sum(v.get('pass',0) for v in suites.values()),failed=sum(v.get('fail',0) for v in suites.values()),skipped=sum(v.get('skip',0) for v in suites.values()))
            raw=[native]+sorted(native.parent.glob('*.json'))
            assert result['passed']>0
            if result['skipped']:
                skipped=[]
                for job in r['jobs']:
                    path=native.parent/(job+'.json');cases=json.loads(path.read_text());raw.append(path)
                    skipped.extend({'id':job+'::'+case['file']+'::'+case['name'],'reason':case['msg']} for case in cases if case['status']=='skip')
                assert len(skipped)==result['skipped']
                ledger_path=pin(cfg['coverage_ledger']);ledger=json.loads(ledger_path.read_text());assert ledger['source_revision']==cfg['source_revision'];records=ledger['records']
                assert len(records)==len(skipped) and {row['id'] for row in records}=={row['id'] for row in skipped}
                for row in records:
                    assert row['scope_reference'] and row['reason']==next(c['reason'] for c in skipped if c['id']==row['id'])
                    if row['required']:
                        evidence=pin(row['execution_report']);proof=json.loads(evidence.read_text());assert proof['source_revision']==cfg['source_revision'] and proof['case_id']==row['id'] and proof['executed']==1 and proof['passed']==1 and proof['failed']==proof['skipped']==0;raw.append(evidence)
                    else:
                        assert row['classification']=='outside-requested-scope'
                        assert not any(word in row['reason'].lower() for word in ('required capability','required cluster feature','missing discovery')),'Required feature gaps cannot be waived'
                d['coverage']={'required_uncovered':0,'skipped_accounted':len(skipped),'excluded_accounted':0,'scope_ledger_sha256':sha(ledger_path),'records':records};raw.append(ledger_path)
            proof=json.loads(pin(cfg['fixture_provenance']).read_text());assert proof['real_nodes']==2 and proof['module_fixtures'] and proof['forced_gates']==[]
            d.update(suites=scope['suites'],real_nodes=2,module_fixtures=proof['module_fixtures'],forced_gates=[],filters=[])
        elif kind=='fault':
            assert (native/'exit.code').read_text().strip()=='0';matrix=pathlib.Path((native/'matrix-summary-path.txt').read_text().strip());cases=json.loads(matrix.read_text());search=json.loads((native/'search-summary.json').read_text())
            scenarios={'kill-primary','pause-primary','isolate-master','majority-minority','bridge','asymmetric','flapping','rolling-restart','full-restart','kill-during-recovery','chaos'}
            assert len(cases)==33 and {(r['scenario'],r['seed']) for r in cases}=={(s,n) for s in scenarios for n in (1,2,3)}
            assert all(r['nodes']==5 and not r.get('setup_error') and not r['verdict']['violations'] for r in cases)
            assert search['success']>0 and search['violations']==0
            result.update(passed=33,failed=0);d.update(scenarios=11,seeds=3,nodes=5,search_aggregation_requests=search['success'],violations=0);raw=[matrix,native/'search-summary.json',native/'harness.log']
        elif kind=='differential':
            r=json.loads((native/'summary.json').read_text());fixture=json.loads((native/'fixture.json').read_text());assert r['passed']==4 and not r['failed'] and len(fixture['nodes']['rust'])==len(fixture['nodes']['es'])==2 and len(set(fixture['clusters'].values()))==2
            result.update(passed=4,failed=0);d.update(rust_nodes=2,es_nodes=2,es_version='9.5.5',cluster_uuids_distinct=True);raw=[native/'summary.json',native/'fixture.json',native/'requests.jsonl']
        elif kind=='rolling':
            text=native.read_text();matches=re.findall(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored;',text);assert matches and sum(map(int,matches))>=3
            assert 'test old_new_known_mapping_load_then_lifecycle_and_gateway_downgrade_fence ... ok' in text
            old=pin(cfg['old_binary']);new=pin(cfg['candidate_binary']);assert sha(old)!=sha(new)
            result.update(passed=sum(map(int,matches)),failed=0);d.update(old_binary_sha256=sha(old),old_master_then_new_master=True,writes_under_upgrade=True,downgrade_refused=True);raw=[native]
        elif kind=='pressure':
            r=json.loads((native/'summary.json').read_text());proof=json.loads((native/'provenance.json').read_text());assert r['passed'] and r['distinct_hosts']==3 and r['accepted']>0 and r['refused']>0 and r['lost_acknowledged_writes']==0
            placement=json.loads((native/'placement-summary.json').read_text());assert placement['passed'] and placement['target_aid'] in placement['publication'] and placement['cooldown_seconds']>=600
            result.update(passed=r['accepted'],failed=0);d.update(instance_ids=[n['instance_id'] for n in proof['fixture']['nodes']],accepted_writes=r['accepted'],refused_requests=r['refused'],lost_acknowledged_writes=0);raw=[native/'summary.json',native/'provenance.json',native/'live-before.json',native/'live-after.json',native/'writes.jsonl',native/'placement-summary.json',native/'placement.jsonl']
        elif kind.startswith('ivf-'):
            r=json.loads((native/'summary.json').read_text());proof=json.loads((native/'provenance.json').read_text());assert r['passed'] and len(r['groups'])==9 and r['dimensions']==int(kind.split('-')[1]) and r['documents']==1_000_000
            result.update(passed=r['queries'],failed=0);d.update(documents=r['documents'],dimensions=r['dimensions'],filters=True,deletes=True,mixed=True,fixed_num_candidates=proof['config']['num_candidates'],tracing=False,profiling=False,build_overlap=False);raw=[native/'summary.json',native/'provenance.json',native/'queries.jsonl',native/'mixed-writes.json']
        else:
            raise ValueError('Performance/profile adapters require compare.py normalized native evidence; no generic success passthrough')
        result['raw_artifacts']=[{'path':str(path.resolve()),'sha256':sha(path)} for path in dict.fromkeys(raw)];result['complete']=True
    finally:
        save(out/'summary.json',result);(out/'exit.code').write_text('0\n' if result['complete'] and not result['failed'] else '1\n')
if __name__=='__main__':main()
