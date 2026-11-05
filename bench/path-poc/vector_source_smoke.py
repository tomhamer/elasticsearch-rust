#!/usr/bin/env python3
"""Check default vector exclusion and explicit vector fetch in full ES."""
import argparse, http.client, json, math, pathlib
from compare import request
ap=argparse.ArgumentParser(); ap.add_argument('--corpus',required=True); ap.add_argument('--out',required=True)
ap.add_argument('--our-port',type=int,default=19214); ap.add_argument('--before-port',type=int); a=ap.parse_args()
doc=json.loads(next(iter(pathlib.Path(a.corpus).read_text().splitlines())))
results={}
systems=[('es',19211),('full',a.our_port)]
if a.before_port: systems.append(('full_before',a.before_port))
for mode,port in systems:
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
    c.request('DELETE','/path-poc-vector-source'); c.getresponse().read()
    request(c,'PUT','/path-poc-vector-source',{'settings':{'number_of_shards':1,'number_of_replicas':0},'mappings':{'properties':{'v':{'type':'dense_vector','dims':128,'similarity':'cosine','index_options':{'type':'hnsw'}}}}})
    request(c,'PUT','/path-poc-vector-source/_doc/one?refresh=true',{'v':doc['v']})
    for name,opts in [('default',{}),('disabled',{'_source':False}),('include',{'_source':{'exclude_vectors':False}}),('fields',{'_source':False,'fields':['v']})]:
        body={'knn':{'field':'v','query_vector':doc['v'],'k':1,'num_candidates':10},**opts}
        h=request(c,'POST','/path-poc-vector-source/_search',body)['hits']['hits'][0]
        assert h['_id']=='one' and math.isclose(h['_score'],1,abs_tol=2e-5)
        if name in ('disabled','fields'): assert '_source' not in h
        elif name=='default': assert 'v' not in h['_source']
        else:
            vector=h['_source']['v']; assert len(vector)==128 and all(math.isclose(x,y,abs_tol=2e-6) for x,y in zip(vector,doc['v']))
        if name=='fields':
            vector=h['fields']['v']; assert len(vector)==128 and all(math.isclose(x,y,abs_tol=2e-6) for x,y in zip(vector,doc['v']))
        results[f'{mode}/{name}']=h
if a.before_port:
    for name in ('default','disabled','include','fields'): assert results[f'full_before/{name}']==results[f'full/{name}']
pathlib.Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(results)} vector source checks passed',flush=True)
