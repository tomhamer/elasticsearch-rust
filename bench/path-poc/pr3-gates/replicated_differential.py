#!/usr/bin/env python3
"""Run deterministic replicated search/aggregation comparisons on two dedicated clusters.
Requires two real HTTP nodes per cluster, distinct cluster UUIDs, ES9.5.5 reference.
Never deletes pre-existing indices; retains the uniquely named fixture on failure/success.
"""
import argparse, json, pathlib, tempfile, urllib.request, uuid
p=argparse.ArgumentParser(); p.add_argument('--rust',required=True); p.add_argument('--es',required=True); p.add_argument('--provenance',required=True,help='JSON with source_revision, node_sha256, fixture PID/config evidence'); p.add_argument('--out',required=True); a=p.parse_args()
root=pathlib.Path(a.out); root.mkdir(parents=True,exist_ok=True); out=pathlib.Path(tempfile.mkdtemp(prefix='replicated-differential-',dir=root)); print(out,flush=True)
proof=json.loads(pathlib.Path(a.provenance).read_text())
for key in ('source_revision','node_sha256','fixtures'):
    if not proof.get(key): raise SystemExit(f'Missing provenance {key}')
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
log=(out/'requests.jsonl').open('w'); opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def req(base,method,path,body=None,raw=False):
    data=(body.encode() if raw else json.dumps(body).encode()) if body is not None else None
    with opener.open(urllib.request.Request(base+path,data=data,method=method,headers={'Content-Type':'application/x-ndjson' if raw else 'application/json'}),timeout=120) as r: v=json.load(r)
    log.write(json.dumps({'url':base+path,'method':method,'body':body,'response':v})+'\n'); log.flush(); return v
clusters={k:[x.rstrip('/') for x in urls.split(',')] for k,urls in [('rust',a.rust),('es',a.es)]}; ids={}; uuids={}
for kind,urls in clusters.items():
    if len(urls)!=2 or urls[0]==urls[1]: raise SystemExit('Exactly two distinct HTTP nodes required per cluster')
    nodes=[]
    for url in urls:
        info=req(url,'GET','/'); uuids.setdefault(kind,info['cluster_uuid']); assert info['cluster_uuid']==uuids[kind]
        if kind=='es': assert info['version']['number']=='9.5.5',info
        own=req(url,'GET','/_nodes/_local'); assert len(own['nodes'])==1; nodes.extend(own['nodes'])
    assert len(set(nodes))==2; ids[kind]=nodes
assert uuids['rust']!=uuids['es']
index='pr3-diff-'+uuid.uuid4().hex
(out/'fixture.json').write_text(json.dumps({'index':index,'nodes':ids,'clusters':uuids},indent=2)+'\n')
body={'settings':{'number_of_shards':2,'number_of_replicas':1},'mappings':{'properties':{'k':{'type':'keyword'},'v':{'type':'long'}}}}
for urls in clusters.values():
    req(urls[0],'PUT','/'+index,body)
    bulk=''.join(json.dumps({'index':{'_id':str(i)}})+'\n'+json.dumps({'k':str(i%17),'v':i})+'\n' for i in range(2000))
    result=req(urls[0],'POST',f'/{index}/_bulk?refresh=true',bulk,True); assert result['errors'] is False
    health=req(urls[0],'GET',f'/_cluster/health/{index}?wait_for_status=green&wait_for_no_relocating_shards=true&timeout=90s'); assert not health['timed_out'] and health['status']=='green'
    assert health['number_of_nodes']==2 and health['active_primary_shards']==2 and health['active_shards']==4,health
    for url in urls:
        count=req(url,'GET',f'/{index}/_count?preference=_only_local'); assert count['count']==2000 and count['_shards']['failed']==0,count
queries=[{'match_all':{}},{'term':{'k':'3'}},{'range':{'v':{'gte':400,'lt':1700}}},{'bool':{'filter':[{'term':{'k':'7'}},{'range':{'v':{'gt':200}}}]}}]
checks=[]
for n,q in enumerate(queries):
    results={}
    for kind,urls in clusters.items():
        for node,url in enumerate(urls):
            query={'query':q,'size':10,'sort':[{'v':'asc'}],'track_total_hits':True,'aggs':{'values':{'stats':{'field':'v'}},'keys':{'terms':{'field':'k','size':30,'order':{'_key':'asc'}}}}}
            r=req(url,'POST',f'/{index}/_search?preference=_only_local&allow_partial_search_results=false',query)
            assert r['_shards']['failed']==0 and not r['timed_out']
            results[f'{kind}-{node}']={'total':r['hits']['total'],'ids':[h['_id'] for h in r['hits']['hits']],'aggregations':r['aggregations']}
    reference=results['es-0']; assert all(v==reference for v in results.values()),results
    checks.append({'query':n,'passed':True})
(out/'summary.json').write_text(json.dumps({'checks':checks,'passed':len(checks),'failed':0,'scope':'static replicated search and aggregation oracle; fault history gate is separate'},indent=2)+'\n')
(out/'exit.code').write_text('0\n')
