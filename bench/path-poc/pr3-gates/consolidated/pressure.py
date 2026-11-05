#!/usr/bin/env python3
"""Distinct-host pressure/refusal and acknowledged-write preservation gate.
Dedicated three-node fixture is supplied by the orchestrator; never starts or
stops machines. Uses recorded per-host lock holders and effective memory limits.
"""
import argparse, concurrent.futures, json, pathlib, re, threading, time, urllib.request, uuid
from common import API, ec2, idle_compilers, pin, save, command, remote_fact

def placement(cfg, nodes, api, out):
    """Observe a real sustained-pressure grant and its physical completion.
    Preparation must retain an actual resident shard on one source, with a hard
    allocation filter preventing early movement. This driver removes that one
    fixture filter, never synthesizes pressure samples or invokes reroute.
    """
    spec=cfg['placement'];index=spec['index'];assert index.startswith('pr3-pressure-placement-')
    assert spec['sustain_seconds']>=65 and spec['cooldown_seconds']==600 and spec['timeout_seconds']>=180
    source=spec['source_node'];target=spec['target_node'];assert source!=target
    def routing():
        value,_=api[0].call('GET',f'/_cluster/state/routing_table/{index}')
        copies=value['routing_table']['indices'][index]['shards']['0'];assert len(copies) in (1,2)
        return copies
    initial=routing();assert len(initial)==1 and initial[0]['state']=='STARTED' and initial[0]['node']==source and initial[0]['primary']
    count,_=api[0].call('GET',f'/{index}/_count');assert spec['documents']>0 and count['count']==spec['documents'] and not count['_shards']['failed']
    source_aid=initial[0]['allocation_id']['id'];baseline=[remote_fact(n) for n in nodes]
    event_base={event for fact in baseline for event in fact['pressure_events']}
    def sample():
        facts=[remote_fact(n) for n in nodes];values={}
        for node,fact in zip(nodes,facts):
            anon=int(next(line.split()[1] for line in fact['status'].splitlines() if line.startswith('RssAnon:')))*1024
            local,_=api[nodes.index(node)].call('GET','/_nodes/_local/stats/process');assert set(local['nodes'])=={node['node_id']}
            identity,_=api[nodes.index(node)].call('GET','/_nodes/_local/process');assert identity['nodes'][node['node_id']]['process']['id']==node['pid']
            process=local['nodes'][node['node_id']]['process']
            budget=process['mem']['native_budget_in_bytes'];assert budget>0 and budget==node['effective_engine_budget_bytes']
            values[node['node_id']]=anon/budget
        row={'unix_time':time.time(),'facts':facts,'resident_ratios':values,'routing':routing()}
        with (out/'placement.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        return row
    start=time.monotonic()
    while time.monotonic()-start<spec['sustain_seconds']:
        row=sample();assert row['resident_ratios'][source]>.85 and row['resident_ratios'][target]<=.60,'Actual sustained source/target pressure condition missing'
        assert row['routing']==initial,'Allocation moved despite source-only fixture filter';time.sleep(5)
    api[0].call('PUT',f'/{index}/_settings',{'index.routing.allocation.require._id':None})
    deadline=time.monotonic()+spec['timeout_seconds'];publication=None;target_aid=None
    while time.monotonic()<deadline:
        row=sample();new={event for fact in row['facts'] for event in fact['pressure_events']}-event_base
        for event in new:
            # Default tracing and JSON-format logs both preserve these values.
            if 'capacity_pressure_published' in event and index in event and source_aid in event and target in event:
                publication=event
        complete=[copy for copy in row['routing'] if copy['state']=='STARTED' and copy['node']==target]
        if publication and len(row['routing'])==1 and len(complete)==1:
            target_aid=complete[0]['allocation_id']['id'];assert target_aid in publication;break
        time.sleep(5)
    assert publication and target_aid,'No exact admitted pressure grant followed by target STARTED'
    api[0].healthy(index,3,1)
    count,_=api[0].call('GET',f'/{index}/_count');assert count['count']==spec['documents'] and not count['_shards']['failed']
    aid_match=re.fullmatch(r'capacity-(\d+)-.+',target_aid);assert aid_match,'Missing exact policy cooldown origin in target allocation ID'
    cooldown_end=int(aid_match.group(1))+600_000
    source_position=next(i for i,n in enumerate(nodes) if n['node_id']==source)
    observed_cooldown=0
    while True:
        row=sample()
        if row['facts'][source_position]['unix_millis']>=cooldown_end:break
        observed_cooldown+=1
        assert len(row['routing'])==1 and row['routing'][0]['node']==target and row['routing'][0]['allocation_id']['id']==target_aid,'Copy oscillated inside cooldown'
        new={event for fact in row['facts'] for event in fact['pressure_events']}-event_base
        assert new=={publication},'Another pressure publication occurred inside cooldown'
        time.sleep(5)
    assert observed_cooldown>0,'Relocation completed after policy cooldown; no cooldown observation evidence'
    save(out/'placement-summary.json',{'passed':True,'publication':publication,'source':source,'target':target,'source_aid':source_aid,'target_aid':target_aid,'cooldown_seconds':spec['cooldown_seconds'],'cooldown_started_unix_millis':int(aid_match.group(1)),'cooldown_samples':observed_cooldown,'documents':spec['documents']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);status={'passed':False}
    try:
        ec2();idle_compilers();cfg=json.loads(pathlib.Path(a.config).read_text());proof=json.loads(pin(cfg['fixture_provenance']).read_text());nodes=proof['nodes']
        assert len(nodes)==3 and len({n['instance_id'] for n in nodes})==3 and len({n['machine_id'] for n in nodes})==3
        assert all(n['instance_id'].startswith('i-') and n['binary_sha256']==cfg['binary_sha256'] and n['effective_memory_max_bytes']>0 and n['host_lock_pid']>0 for n in nodes)
        save(out/'live-before.json',[remote_fact(n) for n in nodes])
        api=[API(n['url'],out/f'requests-{i}.jsonl') for i,n in enumerate(nodes)];node_ids=[];cluster_ids=[]
        for client in api:
            root,_=client.call('GET','/');cluster_ids.append(root['cluster_uuid']);local,_=client.call('GET','/_nodes/_local');assert len(local['nodes'])==1;node_ids.extend(local['nodes'])
            process,_=client.call('GET','/_nodes/_local/process');assert next(iter(process['nodes'].values()))['process']['id']==nodes[len(node_ids)-1]['pid']
        assert len(set(cluster_ids))==1 and len(set(node_ids))==3
        assert [n['node_id'] for n in nodes]==node_ids
        placement(cfg,nodes,api,out)
        index='pr3-pressure-'+uuid.uuid4().hex
        api[0].call('PUT','/'+index,{'settings':{'number_of_shards':3,'number_of_replicas':1},'mappings':{'properties':{'ordinal':{'type':'long'},'payload':{'type':'keyword','index':False,'doc_values':False}}}});api[0].healthy(index,3,6)
        save(out/'provenance.json',{'config':cfg,'fixture':proof,'node_ids':node_ids,'index':index});save(out/'before.json',api[0].call('GET','/_nodes/stats')[0])
        docs=int(cfg['documents']);workers=int(cfg['writers']);size=int(cfg['document_bytes']);assert docs>=1000 and 1<=workers<=256 and size>=4096
        accepted=[];rejected=[];errors=[];lock=threading.Lock();done=threading.Event();started=time.monotonic()
        def write(i):
            body=json.dumps({'ordinal':i,'payload':('pressure-%08d-'%i)*((size+18)//19)}).encode();client=api[i%3]
            req=urllib.request.Request(client.url+f'/{index}/_doc/{i}',data=body,method='PUT',headers={'Content-Type':'application/json'})
            begin=time.monotonic()
            try:
                with client.opener.open(req,timeout=120) as response:code=response.status;value=json.load(response)
            except urllib.error.HTTPError as error:code=error.code;value=json.loads(error.read())
            except Exception as error:
                with lock:errors.append({'id':i,'error':str(error)})
                return
            row={'id':i,'status':code,'seconds':time.monotonic()-begin,'response':value}
            with lock:
                with (out/'writes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if code in (200,201):accepted.append(i)
                elif code==429:rejected.append(i)
                else:errors.append(row)
        def queries():
            try:
                while not done.is_set():
                    value,_=api[0].search(index,{'size':0,'track_total_hits':True,'aggs':{'n':{'value_count':{'field':'ordinal'}}}})
                    assert value['aggregations']['n']['value']==value['hits']['total']['value'] and value['hits']['total']['relation']=='eq'
                    done.wait(.1)
            except Exception as error:errors.append({'search_error':str(error)})
        reader=threading.Thread(target=queries);reader.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:list(executor.map(write,range(docs)))
        finally:done.set();reader.join(timeout=185);assert not reader.is_alive()
        assert not errors,errors[:10];assert accepted and rejected,'Pressure must produce both accepted writes and explicit 429 refusals'
        api[0].call('POST',f'/{index}/_refresh');api[0].healthy(index,3,6)
        for client in api:
            for begin in range(0,len(accepted),500):
                found,_=client.call('POST',f'/{index}/_mget',{'ids':[str(i) for i in accepted[begin:begin+500]]})
                requested={str(i) for i in accepted[begin:begin+500]};assert len(found['docs'])==len(requested) and {d['_id'] for d in found['docs']}==requested
                assert all(d.get('found') and d['_source']['ordinal']==int(d['_id']) and not d.get('error') for d in found['docs']),found
            count,_=client.call('GET',f'/{index}/_count');assert count['count']==len(accepted) and count['_shards']['failed']==0,count
        save(out/'after.json',api[0].call('GET','/_nodes/stats')[0]);save(out/'live-after.json',[remote_fact(n) for n in nodes]);idle_compilers()
        status.update(passed=True,lost_acknowledged_writes=0,accepted=len(accepted),refused=len(rejected),failed=0,skipped=0,distinct_hosts=3,seconds=time.monotonic()-started)
    except Exception as error:status['error']=str(error);raise
    finally:save(out/'summary.json',status);(out/'exit.code').write_text('0\n' if status['passed'] else '1\n')
if __name__=='__main__':main()
