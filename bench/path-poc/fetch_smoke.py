#!/usr/bin/env python3
"""Snapshot fetch responses before/after integration; actual ES invariants."""
import argparse, http.client, json, pathlib
from compare import request

ap=argparse.ArgumentParser(); ap.add_argument('--our-port',type=int,default=19213); ap.add_argument('--out',required=True); ap.add_argument('--load',action='store_true'); ap.add_argument('--before'); a=ap.parse_args()
requests={
    'id_only':{'_source':False},
    'source':{},
    'version':{'_source':False,'version':True},
    'sequence':{'_source':False,'seq_no_primary_term':True},
    'fields':{'_source':False,'fields':['tag','code']},
    'stored':{'_source':False,'stored_fields':['tag']},
    'docvalues':{'_source':False,'docvalue_fields':['code']},
    'filtered_source':{'_source':{'includes':['body']}},
    'highlight':{'_source':False,'highlight':{'fields':{'body':{}}}},
    'no_stored_fields':{'_source':False,'stored_fields':'_none_'},
    'scripts':{'_source':False,'script_fields':{'constant':{'script':{'source':'1'}}}},
}
results={}
for mode,port in [('es',19211),('full',a.our_port)]:
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=120)
    if a.load:
        c.request('DELETE','/path-poc-fetch'); c.getresponse().read()
        request(c,'PUT','/path-poc-fetch',{'settings':{'number_of_shards':1,'number_of_replicas':0},'mappings':{'properties':{'body':{'type':'text','store':True},'tag':{'type':'keyword','ignore_above':4,'store':True},'code':{'type':'long','ignore_malformed':True}}}})
        request(c,'PUT','/path-poc-fetch/_doc/plain',{'body':'vector document','tag':'ok','code':2})
        request(c,'POST','/path-poc-fetch/_refresh')
        request(c,'PUT','/path-poc-fetch/_doc/routed?routing=route-A',{'body':'vector document','tag':'too-long','code':'invalid'})
        request(c,'POST','/path-poc-fetch/_refresh')
    for stage in ['refresh','merged']:
        if stage=='merged': request(c,'POST','/path-poc-fetch/_forcemerge?max_num_segments=1')
        for name,options in requests.items():
            result=request(c,'POST','/path-poc-fetch/_search',{'query':{'term':{'body':'vector'}},'size':10,**options})
            hits=sorted(result['hits']['hits'],key=lambda h:h.get('_id',''))
            assert len(hits)==2,(mode,name,result)
            if name!='no_stored_fields':
                routed=next(h for h in hits if h['_id']=='routed'); assert routed['_routing']=='route-A'; assert sorted(routed['_ignored'])==['code','tag']
            results[f'{mode}/{stage}/{name}']=hits
if a.before:
    before=json.loads(pathlib.Path(a.before).read_text())
    assert before==results, [(k,before[k],results[k]) for k in results if before[k]!=results[k]]
pathlib.Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(results)} fetch response checks passed',flush=True)
