#!/usr/bin/env python3
"""Small float-HNSW comparison; ground-truth recall and per-hit exact scores."""
import argparse, collections, hashlib, http.client, json, math, pathlib, random, statistics, struct, time
from compare import request, percentile


def generate(root, n=5000, nq=20):
    rng = random.Random(73)
    def unit(v):
        norm = math.sqrt(sum(x*x for x in v))
        return [struct.unpack('<f', struct.pack('<f', x/norm))[0] for x in v]
    centers = [unit([rng.gauss(0,1) for _ in range(128)]) for _ in range(100)]
    def vector(i):
        return unit([x + rng.gauss(0,.07) for x in centers[i % 100]])
    docs = [dict(id=i,title='vector document',body='small knn experiment',price=float(i),ts=i,tag=f't{i%10}',v=vector(i)) for i in range(n)]
    queries = [vector(i) for i in range(nq)]
    root.mkdir(parents=True, exist_ok=True)
    (root/'corpus.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in docs))
    (root/'queries.json').write_text(json.dumps(queries))
    return docs, queries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True); ap.add_argument('--generate',action='store_true'); ap.add_argument('--load',action='store_true')
    ap.add_argument('--before-port',type=int); ap.add_argument('--poc-port',type=int,default=19220); ap.add_argument('--our-port',type=int,default=19213); ap.add_argument('--es-port',type=int,default=19211)
    ap.add_argument('--quant',choices=['hnsw','int8_hnsw'],default='hnsw'); ap.add_argument('--rounds',type=int,default=10); ap.add_argument('--warmup',type=int,default=10); ap.add_argument('--out',required=True)
    a=ap.parse_args(); root=pathlib.Path(a.root)
    if a.generate: generate(root)
    docs=list(map(json.loads,(root/'corpus.jsonl').read_text().splitlines())); queries=json.loads((root/'queries.json').read_text())
    systems={'es':(a.es_port,'/path-poc-knn/_search'),'full':(a.our_port,'/path-poc-knn/_search'),'baseline':(a.poc_port,'/baseline/_search'),'candidate':(a.poc_port,'/candidate/_search')}
    if a.before_port: systems['full_before']=(a.before_port,'/path-poc-knn/_search')
    conns={m:http.client.HTTPConnection('127.0.0.1',p,timeout=180) for m,(p,_) in systems.items()}
    if a.load:
        for mode in [m for m in ('es','full','full_before') if m in systems]:
            c=conns[mode]; c.request('DELETE','/path-poc-knn'); c.getresponse().read()
            request(c,'PUT','/path-poc-knn',{'settings':{'number_of_shards':1,'number_of_replicas':0,'refresh_interval':'-1','index.queries.cache.enabled':False,'index.mapping.exclude_source_vectors':False},'mappings':{'dynamic':'strict','properties':{'id':{'type':'long'},'title':{'type':'text'},'body':{'type':'text'},'price':{'type':'double'},'ts':{'type':'long'},'tag':{'type':'keyword'},'v':{'type':'dense_vector','dims':128,'similarity':'cosine','index_options':{'type':a.quant,'m':16,'ef_construction':100,**({'confidence_interval':1.0} if a.quant=='int8_hnsw' else {})}}}}})
            for off in range(0,len(docs),100):
                raw=''.join(json.dumps({'index':{'_index':'path-poc-knn','_id':str(d['id'])}})+'\n'+json.dumps(d)+'\n' for d in docs[off:off+100])
                request(c,'POST','/_bulk',raw,True)
            request(c,'POST','/path-poc-knn/_refresh'); request(c,'POST','/path-poc-knn/_forcemerge?max_num_segments=1')
    truth={}; cosines={}
    for qi,q in enumerate(queries):
        scores=[]
        qnorm=math.sqrt(sum(x*x for x in q))
        for d in docs:
            v=d['v']; score=(1+sum(x*y for x,y in zip(q,v))/(qnorm*math.sqrt(sum(x*x for x in v))))/2
            scores.append((score,str(d['id'])))
        cosines[qi]=dict((i,s) for s,i in scores)
        for filter_name in ('all','ten_percent','half'):
            eligible=[(s,i) for s,i in scores if filter_name=='all' or (int(i)%10==0 if filter_name=='ten_percent' else int(i)%10<5)]
            truth[qi,filter_name]={i for _,i in sorted(eligible,reverse=True)[:10]}
    samples=collections.defaultdict(list); stages=collections.defaultdict(list); recalls=collections.defaultdict(list); verified=0; rng=random.Random(42)
    for filt in ('all','ten_percent','half'):
        for source in (False,True):
            for qi,q in enumerate(queries):
                knn={'field':'v','query_vector':q,'k':10,'num_candidates':500}
                if a.quant=='int8_hnsw': knn['rescore_vector']={'oversample':3.0}
                if filt=='ten_percent': knn['filter']={'term':{'tag':'t0'}}
                if filt=='half': knn['filter']={'bool':{'should':[{'term':{'tag':f't{i}'}} for i in range(5)]}}
                body={'knn':knn,'size':10,'_source':source,'track_total_hits':True}
                results={m:request(conns[m],'POST',path,body) for m,(_,path) in systems.items()}
                for m,res in results.items():
                    hits=res['hits']['hits']; assert len(hits)==10, (m,res)
                    assert res['hits']['total']=={'value':10,'relation':'eq'}, (m,res['hits']['total'])
                    for h in hits:
                        i=h['_id']; assert math.isclose(h['_score'],cosines[qi][i],abs_tol=2e-5,rel_tol=2e-5),(m,h['_score'],cosines[qi][i])
                        assert ('_source' in h)==source
                        if source:
                            got=h['_source']; expected=docs[int(i)]
                            assert got.keys()==expected.keys(), (m,got.keys(),expected.keys())
                            for k in got:
                                if k=='v': assert len(got[k])==len(expected[k]) and all(math.isclose(x,y,abs_tol=2e-6) for x,y in zip(got[k],expected[k]))
                                else: assert got[k]==expected[k],(m,k)
                        assert filt=='all' or (int(i)%10==0 if filt=='ten_percent' else int(i)%10<5)
                    recall=len({h['_id'] for h in hits}&truth[qi,filt])/10
                    recalls[f'{filt}/{m}'].append(recall)
                    verified+=1
                assert results['baseline']['hits']==results['candidate']['hits']
                for r in range(a.warmup+a.rounds):
                    modes=list(systems); rng.shuffle(modes)
                    for m in modes:
                        t=time.perf_counter_ns(); measured=request(conns[m],'POST',systems[m][1],body); us=(time.perf_counter_ns()-t)/1000
                        if r>=a.warmup:
                            samples[f'{filt}/source={source}/{m}'].append(us)
                            for stage in ('parse_us','query_us','fetch_us'):
                                if '_poc' in measured: stages[f'{filt}/source={source}/{m}/{stage}'].append(measured['_poc'][stage])
    recall_stats={k:{'mean':statistics.mean(v),'min':min(v)} for k,v in recalls.items()}
    out={'args':vars(a),'docs':len(docs),'dimensions':128,'index_options':{'type':a.quant,'m':16,'ef_construction':100,**({'confidence_interval':1.0} if a.quant=='int8_hnsw' else {})},'num_candidates':500,'verified_responses':verified,'exact_score_checks':True,'corpus_sha256':hashlib.sha256((root/'corpus.jsonl').read_bytes()).hexdigest(),'queries_sha256':hashlib.sha256((root/'queries.json').read_bytes()).hexdigest(),'recall':recall_stats,'poc_stages':{k:{'p50_us':statistics.median(v),'p95_us':percentile(v,.95)} for k,v in stages.items()},'stats':{k:{'n':len(v),'p50_us':statistics.median(v),'p95_us':percentile(v,.95)} for k,v in samples.items()}}
    pathlib.Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'recall':recall_stats,'verified':verified,'stats':out['stats']},indent=2),flush=True)
    assert all(v['mean']>=.95 for v in recall_stats.values()), 'mean recall gate < .95'

if __name__=='__main__': main()
