#!/usr/bin/env python3
"""EC2-only fault gate with concurrent search/aggregation observations; no builds."""
from gate_reports import native_report
import argparse, fcntl, hashlib, json, os, pathlib, socket, subprocess, tempfile, threading, time, urllib.request, urllib.error
p=argparse.ArgumentParser(); p.add_argument('--node',required=True); p.add_argument('--node-sha256',required=True); p.add_argument('--harness',required=True); p.add_argument('--harness-sha256',required=True); p.add_argument('--source-revision',required=True); p.add_argument('--out',required=True); p.add_argument('--port-base',type=int,default=20000); p.add_argument('--seeds',type=int,default=3); a=p.parse_args()
if not pathlib.Path('/sys/devices/virtual/dmi/id/product_uuid').exists(): raise SystemExit('Linux EC2 launcher only')
proof={}
for name,path,expected in [('node',a.node,a.node_sha256),('harness',a.harness,a.harness_sha256)]:
    f=pathlib.Path(path).resolve(); actual=hashlib.sha256(f.read_bytes()).hexdigest()
    if actual!=expected: raise SystemExit(f'{name} hash mismatch')
    proof[name]={'path':str(f),'sha256':actual}
if a.seeds<1 or a.port_base<1024 or a.port_base//100>255: raise SystemExit('Invalid seed/port selection')
# Cooperative per-port locks remain held through native cleanup and reporting.
# Refuse stale ownership/cgroups too: only an operator may clear a failed fixture.
lock_root=pathlib.Path('/var/lock/es-pr3-faults'); lock_root.mkdir(parents=True,exist_ok=True)
ports=sorted({a.port_base+i for i in range(5)} | {a.port_base+100+i for i in range(5)})
locks=[]
for port in ports:
    fd=os.open(lock_root/f'port-{port}.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(fd)
owner=lock_root/f'owner-{a.port_base}.json'
if owner.exists(): raise SystemExit(f'Existing fixture ownership {owner}; inspect before explicit cleanup')
if pathlib.Path(f'/sys/fs/cgroup/es-faults-{a.port_base}').exists():
    raise SystemExit('Existing fault cgroup; refusing native destructive cleanup')
def matching_processes():
    found=[]
    for proc in pathlib.Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid(): continue
        try: args=proc.joinpath('cmdline').read_bytes().split(b'\0')
        except FileNotFoundError: continue
        # cmdline must be readable; inability to inspect is not proof of idleness.
        text=[x.decode(errors='replace') for x in args if x]
        cluster=f'faults-{a.port_base}'
        if any(x==cluster or x.endswith('='+cluster) for x in text): found.append({'pid':int(proc.name),'args':text})
        elif text and 'elasticsearch-faults' in pathlib.Path(text[0]).name:
            if any(x==f'--port-base={a.port_base}' or (x=='--port-base' and i+1<len(text) and text[i+1]==str(a.port_base)) for i,x in enumerate(text)): found.append({'pid':int(proc.name),'args':text})
    return found
existing=matching_processes()
if existing: raise SystemExit(f'Existing matching cluster processes: {existing}')
for port in ports:
    with socket.socket() as sock: sock.bind(('0.0.0.0',port))
with owner.open('x') as f: json.dump({'pid':os.getpid(),'port_base':a.port_base,'ports':ports},f)
root=pathlib.Path(a.out).resolve(); root.mkdir(parents=True,exist_ok=True)
out=pathlib.Path(tempfile.mkdtemp(prefix='fault-gate-',dir=root)); print(out,flush=True)
proof.update(source_revision=a.source_revision, nodes=5,mode='long',seeds=a.seeds,profiling=False,scope='all eleven fault scenarios; supplemental search/aggregation invariants')
(out/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
stop=threading.Event(); counts={'success':0,'unavailable':0,'violations':0}; opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def observe():
    # `by_index` (diagnostic only, not judged): which index's documents an
    # inconsistent response's `values` left out (`_index` doc counts against
    # each index's own `v` count). An inconsistent response is then read
    # again on each node with ?request_cache=false&preference=_only_local per
    # index (`recheck`), to name the copy whose aggregation is short.
    body=json.dumps({'size':0,'track_total_hits':True,'aggs':{'values':{'stats':{'field':'v'}},'by_index':{'terms':{'field':'_index','size':10},'aggs':{'vc':{'value_count':{'field':'v'}}}}}}).encode()
    plain=json.dumps({'size':0,'track_total_hits':True,'aggs':{'values':{'stats':{'field':'v'}}}}).encode()
    def recheck(row,v):
        out=[]
        for b in v.get('aggregations',{}).get('by_index',{}).get('buckets',[]):
            for j in range(5):
                url=f'http://127.0.{a.port_base//100}.{j+1}:{a.port_base+j}/{b["key"]}/_search?request_cache=false&preference=_only_local'
                try:
                    with opener.open(urllib.request.Request(url,data=plain,headers={'Content-Type':'application/json'}),timeout=2) as r:
                        w=json.load(r); out.append({'index':b['key'],'node':j,'total':w['hits']['total']['value'],'count':w['aggregations']['values']['count']})
                except Exception as e: out.append({'index':b['key'],'node':j,'error':repr(e)[:200]})
        row['recheck']=out
    with (out/'search-aggregation.jsonl').open('w') as log:
        while not stop.is_set():
            for i in range(5):
                url=f'http://127.0.{a.port_base//100}.{i+1}:{a.port_base+i}/jepsen-*/_search?allow_partial_search_results=false'
                row={'time':time.time(),'node':i,'url':url}
                try:
                    with opener.open(urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'}),timeout=2) as r:
                        v=json.load(r); row.update(status=r.status,response=v)
                        # Writes continue: only same-response invariants are valid, not cross-request equality.
                        total=v['hits']['total']['value']; stats=v['aggregations']['values']
                        if v.get('timed_out') or v['_shards']['failed'] or total!=stats['count']:
                            counts['violations']+=1; row['violation']='inconsistent complete search'
                            if counts['violations']%50==1: recheck(row,v)
                        else: counts['success']+=1
                except urllib.error.HTTPError as e:
                    row.update(status=e.code,body=e.read().decode(errors='replace')); counts['unavailable']+=1
                    if e.code not in (404,429,502,503,504): counts['violations']+=1; row['violation']='unexpected error status'
                except (OSError,TimeoutError) as e: row['unavailable']=str(e); counts['unavailable']+=1
                except Exception as e: row['violation']=repr(e); counts['violations']+=1
                log.write(json.dumps(row)+'\n'); log.flush()
                if stop.is_set(): break
            stop.wait(.1)
t=threading.Thread(target=observe); t.start()
code=1
try:
    with (out/'harness.log').open('w') as log:
        code=subprocess.call([a.harness,'run','--bin',a.node,'--out',str(out/'matrix'),'--scenario','all','--seeds',str(a.seeds),'--mode','long','--nodes','5','--port-base',str(a.port_base),'--keep-data'],stdout=log,stderr=subprocess.STDOUT)
finally:
    stop.set(); t.join(); (out/'search-summary.json').write_text(json.dumps(counts,indent=2)+'\n')
# Harness exit plus complete 11*seeds verdict accounting is required.
# Current retention-aware harness prints its exact invocation path. Older pinned
# harnesses write directly under --out; neither layout is discovered by glob.
report=native_report((out/'harness.log').read_text(),out/'matrix')
expected={(name,seed) for name in ('kill-primary','pause-primary','isolate-master','majority-minority','bridge','asymmetric','flapping','rolling-restart','full-restart','kill-during-recovery','chaos') for seed in range(1,a.seeds+1)}
if not report.is_file(): code=1
else:
    summary=json.loads(report.read_text()); pairs=[(r['scenario'],r['seed']) for r in summary]
    if len(pairs)!=len(set(pairs)) or set(pairs)!=expected: code=1
    if any(r['nodes']!=5 or r.get('setup_error') or r['verdict']['violations'] for r in summary): code=1
    (out/'matrix-summary-path.txt').write_text(str(report)+'\n')
if not counts['success'] or counts['violations']: code=1
(out/'exit.code').write_text(str(code)+'\n')
if code==0 and not matching_processes(): owner.unlink()
# Failed ownership deliberately survives for explicit operator inspection.
raise SystemExit(code)
