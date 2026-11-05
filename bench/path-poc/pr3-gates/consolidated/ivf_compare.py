#!/usr/bin/env python3
"""Full 1M-vector IVF correctness/recall/latency, one engine per invocation.
Use identical pinned corpus/query/config for candidate and ES9.5.5. No sampling,
no implicit fixture creation outside the uniquely named index, no deletion.
"""
import argparse, concurrent.futures, json, pathlib, threading, time, uuid
from common import API, command, ec2, idle_compilers, pin, save, remote_fact

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--engine',choices=['candidate','master','es9'],required=True);p.add_argument('--url',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);status={'passed':False,'scope':'1M IVF corpus, filters/deletes/mixed updates'}
    try:
        machine=ec2();idle_compilers()
        import numpy as np
        cfg=json.loads(pathlib.Path(a.config).read_text());dim=cfg['dimensions'];n=cfg['documents']
        assert dim in (128,768) and n==1_000_000
        data_path=pin(cfg['vectors']);query_path=pin(cfg['queries']);qcount=cfg['query_count']
        assert data_path.stat().st_size==n*dim*4 and query_path.stat().st_size==qcount*dim*4 and qcount>=100
        data=np.memmap(data_path,dtype='<f4',mode='r',shape=(n,dim));queries=np.memmap(query_path,dtype='<f4',mode='r',shape=(qcount,dim))
        k=int(cfg['k']);candidates=int(cfg['num_candidates']);repetitions=int(cfg['repetitions']);assert 0<k<=candidates<=10000 and repetitions>=3
        measure_only=cfg.get('measurements_only',False);assert isinstance(measure_only,bool)
        assert cfg['mixed_seconds']>=120 and cfg['mixed_updates_per_second']>0
        if not measure_only:assert 0<cfg['minimum_recall']<=1
        for artifact in cfg['distribution_files']:pin(artifact)
        assert cfg['distribution_files'],'Pin actual candidate or ES distribution dependencies'
        fixture=json.loads(pin(cfg['fixture_provenance']).read_text());save(out/'live-before.json',[remote_fact(node) for node in fixture['nodes']])
        assert len(fixture['nodes'])==2 and all(node['binary_sha256']==cfg['binary_sha256'] for node in fixture['nodes'])
        if a.engine!='es9':
            build=json.loads(pin(cfg['build_provenance']).read_text());assert build['source_revision']==cfg['source_revision'] and build['binary_sha256']==cfg['binary_sha256']
        else:assert cfg['source_revision']=='es9-9.5.5'
        cluster_ids=[];node_ids=[]
        for i,node in enumerate(fixture['nodes']):
            client=API(node['url'],out/f'fixture-{i}.jsonl');root,_=client.call('GET','/');cluster_ids.append(root['cluster_uuid'])
            process,_=client.call('GET','/_nodes/_local/process');assert len(process['nodes'])==1
            node_id,details=next(iter(process['nodes'].items()));assert node_id==node['node_id'] and details['process']['id']==node['pid'];node_ids.append(node_id)
        assert len(set(cluster_ids))==1 and len(set(node_ids))==2
        api=API(a.url,out/'requests.jsonl');info,_=api.call('GET','/');assert info['cluster_uuid']==cluster_ids[0]
        members,_=api.call('GET','/_nodes/process');assert set(members['nodes'])==set(node_ids)
        assert a.engine!='es9' or info['version']['number']=='9.5.5'
        index='pr3-ivf-'+uuid.uuid4().hex;save(out/'provenance.json',{'config':cfg,'engine':a.engine,'root_info':info,'machine':machine,'index':index})
        api.call('PUT','/'+index,{'settings':{'number_of_shards':2,'number_of_replicas':1},'mappings':{'properties':{'v':{'type':'dense_vector','dims':dim,'similarity':'cosine','index_options':{'type':'bbq_disk'}},'ordinal':{'type':'long'},'tag':{'type':'keyword'},'tick':{'type':'long'}}}})
        # Keep full document corpus; each chunk checked before indexing.
        for start in range(0,n,256):
            rows=np.asarray(data[start:start+256]);assert np.isfinite(rows).all() and (np.linalg.norm(rows,axis=1)>0).all()
            bulk=''.join(json.dumps({'index':{'_id':str(start+j)}})+'\n'+json.dumps({'v':row.tolist(),'ordinal':start+j,'tag':str((start+j)%10),'tick':0})+'\n' for j,row in enumerate(rows))
            r,_=api.call('POST',f'/{index}/_bulk',bulk,True);assert not r['errors'],r
        api.call('POST',f'/{index}/_refresh');api.healthy(index,2,4)
        count,_=api.call('GET',f'/{index}/_count');assert count['count']==n
        # Run after stable build/refresh; do not use force merge to hide segment effects.
        save(out/'segments-before.json',api.call('GET',f'/{index}/_segments')[0])
        variants={'all':None,'tag':{'term':{'tag':'3'}},'tag-range':{'bool':{'filter':[{'term':{'tag':'3'}},{'range':{'ordinal':{'gte':n//4,'lt':3*n//4}}}]}}}
        deleted=np.zeros(n,dtype=bool);deleted_epoch=False;events=[];truth_cache={}
        def truth(qid,variant):
            key=(deleted_epoch,qid,variant)
            if key in truth_cache:return truth_cache[key]
            q=np.asarray(queries[qid],dtype=np.float64);assert np.isfinite(q).all() and np.linalg.norm(q)>0;q=q/np.linalg.norm(q)
            best_scores=np.array([],dtype=np.float64);best_ids=np.array([],dtype=np.int64)
            for start in range(0,n,8192):
                rows=np.asarray(data[start:start+8192],dtype=np.float64);ids=np.arange(start,start+len(rows));mask=~deleted[ids]
                if variant!='all':mask &= ids%10==3
                if variant=='tag-range':mask &= (ids>=n//4)&(ids<3*n//4)
                rows=rows[mask];ids=ids[mask]
                scores=rows@q/np.linalg.norm(rows,axis=1)
                scores=np.concatenate((best_scores,scores));ids=np.concatenate((best_ids,ids))
                order=np.lexsort((ids,-scores))[:k];best_scores=scores[order];best_ids=ids[order]
            truth_cache[key]=set(map(int,best_ids));return truth_cache[key]
        def query(qid,variant,phase,deadline=None):
            expected=truth(qid,variant)
            knn={'field':'v','query_vector':queries[qid].tolist(),'k':k,'num_candidates':candidates}
            if variants[variant] is not None:knn['filter']=variants[variant]
            request_started=time.monotonic()
            response,seconds=api.search(index,{'knn':knn,'size':k,'_source':False})
            ids=[int(hit['_id']) for hit in response['hits']['hits']];assert len(ids)==k and len(set(ids))==k
            assert all(0<=i<n and not deleted[i] and (variant=='all' or i%10==3) and (variant!='tag-range' or n//4<=i<3*n//4) for i in ids)
            completed=time.monotonic()
            # Retain server scores only; calibration work follows all timings.
            observed=[hit['_score'] for hit in response['hits']['hits']]
            row={'phase':phase,'requested':request_started,'completed':completed,'inside_mixed_window':deadline is None or completed<=deadline,'query':qid,'filter':variant,'seconds':seconds,'recall':len(set(ids)&expected)/k,'ids':ids,'truth':sorted(expected),'native_scores':observed};events.append(row)
            with (out/'queries.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        for phase in ('initial','deleted','mixed'):
            if phase=='deleted':
                doomed=list(range(0,n,17))
                for offset in range(0,len(doomed),1000):
                    bulk=''.join(json.dumps({'delete':{'_id':str(i)}})+'\n' for i in doomed[offset:offset+1000]);r,_=api.call('POST',f'/{index}/_bulk',bulk,True);assert not r['errors']
                deleted[doomed]=True;deleted_epoch=True;api.call('POST',f'/{index}/_refresh');api.healthy(index,2,4)
                c,_=api.call('GET',f'/{index}/_count');assert c['count']==n-len(doomed)
            # Exact reference calculation is untimed and ends before mixed load.
            for variant in variants:
                for qid in range(qcount):truth(qid,variant)
            idle_compilers();started=time.monotonic();stop=threading.Event();writer_errors=[];writes=[]
            def writer():
                client=API(a.url,out/'writes.jsonl');seq=0;deadline=started+cfg['mixed_seconds'];rate=cfg['mixed_updates_per_second']
                try:
                    while not stop.is_set() and time.monotonic()<deadline:
                        due=started+seq/rate
                        if stop.wait(max(0,due-time.monotonic())):break
                        if time.monotonic()>=deadline:break
                        i=(seq*97+1)%n
                        while deleted[i]:i=(i+1)%n
                        _,seconds=client.call('POST',f'/{index}/_update/{i}',{'doc':{'tick':seq}});writes.append({'due':due-started,'completed':time.monotonic()-started,'seconds':seconds,'id':i,'tick':seq});seq+=1
                except Exception as e:writer_errors.append(str(e));stop.set()
            thread=threading.Thread(target=writer) if phase=='mixed' else None
            if thread:thread.start()
            try:
                if phase=='mixed':
                    qid=0
                    while time.monotonic()<started+cfg['mixed_seconds'] and not writer_errors:
                        query(qid%qcount,list(variants)[qid%3],phase,started+cfg['mixed_seconds']);qid+=1
                else:
                    for repetition in range(repetitions):
                        for variant in variants:
                            for qid in range(qcount):query(qid,variant,phase)
            finally:
                stop.set()
                if thread:thread.join(timeout=185);assert not thread.is_alive(),'Mixed writer did not drain'
            assert not writer_errors,writer_errors
            if phase=='mixed':
                save(out/'mixed-writes.json',writes)
                api.call('POST',f'/{index}/_refresh')
                expected_ticks={row['id']:row['tick'] for row in writes}
                expected_ids=list(expected_ticks)
                for offset in range(0,len(expected_ids),500):
                    result,_=api.call('POST',f'/{index}/_mget',{'ids':[str(i) for i in expected_ids[offset:offset+500]]})
                    requested={str(i) for i in expected_ids[offset:offset+500]};assert len(result['docs'])==len(requested) and {doc['_id'] for doc in result['docs']}==requested
                    assert all(doc.get('found') and doc['_source']['tick']==expected_ticks[int(doc['_id'])] for doc in result['docs']),result
                assert writes and writes[-1]['completed']>=cfg['mixed_seconds']-2,'Incomplete mixed writer window'
            idle_compilers();api.healthy(index,2,4)
        save(out/'segments-after.json',api.call('GET',f'/{index}/_segments')[0]);save(out/'node-stats.json',api.call('GET','/_nodes/stats')[0])
        # Separate post-timing calibration diagnostics, never measurement load.
        with (out/'native-score-calibration.jsonl').open('x') as diagnostics:
            for row in events:
                observed=np.asarray(row['native_scores'],dtype=np.float64);assert np.isfinite(observed).all()
                q=np.asarray(queries[row['query']],dtype=np.float64);returned=np.asarray(data[row['ids']],dtype=np.float64)
                exact=(1+(returned@q)/(np.linalg.norm(returned,axis=1)*np.linalg.norm(q)))/2
                assert np.isfinite(exact).all()
                row['score_absolute_errors']=np.abs(observed-exact).tolist()
                diagnostics.write(json.dumps({'phase':row['phase'],'query':row['query'],'filter':row['filter'],'ids':row['ids'],'native_scores':row['native_scores'],'exact_cosine_scores':exact.tolist(),'absolute_errors':row['score_absolute_errors']})+'\n')
        groups={}
        for phase in ('initial','deleted','mixed'):
            for variant in variants:
                rows=[r for r in events if r['phase']==phase and r['filter']==variant and r['inside_mixed_window']];assert rows;recall=sum(r['recall'] for r in rows)/len(rows)
                if not measure_only:assert recall>=cfg['minimum_recall'],(phase,variant,recall)
                groups[f'{phase}/{variant}']={'queries':len(rows),'mean_recall':recall,'p99_seconds':float(np.quantile([r['seconds'] for r in rows],.99)),'score_mean_absolute_error':float(np.mean([e for r in rows for e in r['score_absolute_errors']])),'score_max_absolute_error':max(e for r in rows for e in r['score_absolute_errors'])}
        save(out/'live-after.json',[remote_fact(node) for node in fixture['nodes']])
        status.update(passed=not measure_only,measurements_complete=True,acceptance_evaluated=not measure_only,groups=groups,documents=n,dimensions=dim,queries=len(events),failed=0,skipped=0)
    except Exception as error:status['error']=str(error);raise
    finally:save(out/'summary.json',status);(out/'exit.code').write_text('0\n' if status.get('measurements_complete') else '1\n')
if __name__=='__main__':main()
