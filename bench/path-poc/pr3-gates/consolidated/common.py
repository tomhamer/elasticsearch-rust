"""Shared fail-closed EC2 gate primitives; importing this module does no work."""
import hashlib, json, os, pathlib, platform, shlex, subprocess, time, urllib.request, urllib.error

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def pin(item):
    path=pathlib.Path(item['path']).resolve(strict=True)
    if sha(path)!=item['sha256']: raise ValueError(f'Artifact hash mismatch: {path}')
    return path

def ec2():
    if not __debug__: raise RuntimeError('Validation requires Python assertions; unset PYTHONOPTIMIZE and omit -O')
    if platform.system()!='Linux': raise RuntimeError('Binary validation is EC2 Linux only')
    values=[]
    for path in ('/sys/hypervisor/uuid','/sys/devices/virtual/dmi/id/product_uuid','/sys/devices/virtual/dmi/id/board_asset_tag'):
        try: values.append(pathlib.Path(path).read_text().strip())
        except OSError: pass
    if not any(v.lower().startswith('ec2') or v.startswith('i-') for v in values):
        raise RuntimeError('EC2 machine identity is required; no local execution override')
    return {'identity':values,'machine':platform.machine(),'kernel':platform.release()}

def idle_compilers():
    bad=[]
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try: name=(entry/'comm').read_text().strip()
        except OSError: continue
        if name in {'rustc','cargo','javac','cc1','cc1plus','clang','clang++','ld','ld.lld','nvcc','perf'}: bad.append((entry.name,name))
    if bad: raise RuntimeError(f'Timing/build/profile overlap: {bad}')

class API:
    def __init__(self,url,log):
        self.url=url.rstrip('/'); self.log=log
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(self,method,path,body=None,raw=False):
        data=(body.encode() if raw else json.dumps(body,separators=(',',':')).encode()) if body is not None else None
        request=urllib.request.Request(self.url+path,data=data,method=method,headers={'Content-Type':'application/x-ndjson' if raw else 'application/json'})
        started=time.monotonic()
        try:
            with self.opener.open(request,timeout=180) as response: status=response.status; result=json.load(response)
        except urllib.error.HTTPError as error:
            status=error.code; result=json.loads(error.read())
        elapsed=time.monotonic()-started
        # Do not duplicate million-vector bulk request bodies in diagnostics.
        with self.log.open('a') as stream:
            stream.write(json.dumps({'method':method,'path':path,'status':status,'seconds':elapsed,'request_sha256':hashlib.sha256(data or b'').hexdigest(),'response':result})+'\n')
        if status>=300: raise RuntimeError(f'{method} {path}: HTTP {status}: {result}')
        return result,elapsed
    def healthy(self,index,nodes,copies):
        h,_=self.call('GET',f'/_cluster/health/{index}?wait_for_status=green&wait_for_no_relocating_shards=true&wait_for_no_initializing_shards=true&timeout=120s')
        assert h['status']=='green' and not h.get('timed_out') and h['number_of_nodes']==nodes and h['active_shards']==copies,h
    def search(self,index,body):
        response,seconds=self.call('POST',f'/{index}/_search?allow_partial_search_results=false',body)
        assert response['_shards']['failed']==0 and not response.get('timed_out'),response
        return response,seconds

def save(path,value): path.write_text(json.dumps(value,indent=2)+'\n')

def command(argv,log,env=None):
    if not isinstance(argv,list) or not argv or not all(isinstance(v,str) for v in argv): raise ValueError('Commands must be argument arrays')
    with log.open('xb') as stream: result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
    log.with_suffix(log.suffix+'.exit').write_text(str(result.returncode)+'\n')
    if result.returncode: raise RuntimeError(f'Command failed ({result.returncode}); retained {log}')

def remote_fact(node):
    """Live machine/process/cgroup evidence; fixed read-only remote program."""
    pid=int(node['pid']);assert pid>1
    code=r'''import collections,hashlib,json,os,pathlib,platform,subprocess,sys,time,urllib.request
pid=int(sys.argv[1]);p=pathlib.Path('/proc')/str(pid)
cgroup=(p/'cgroup').read_text().splitlines();relative=[v.split(':',2)[2] for v in cgroup if v.startswith('0::')]
assert len(relative)==1,'unified cgroup memory envelope required'
root=pathlib.Path('/sys/fs/cgroup')/relative[0].lstrip('/')
limit=(root/'memory.max').read_text().strip();assert limit!='max','finite memory envelope required'
identity=[]
for name in ('/sys/hypervisor/uuid','/sys/devices/virtual/dmi/id/product_uuid','/sys/devices/virtual/dmi/id/board_asset_tag'):
 try:identity.append(pathlib.Path(name).read_text().strip())
 except OSError:pass
assert identity and any(v.lower().startswith('ec2') or v.startswith('i-') for v in identity)
instance=next((v for v in identity if v.startswith('i-')),None)
if instance is None:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 token=opener.open(urllib.request.Request('http://169.254.169.254/latest/api/token',method='PUT',headers={'X-aws-ec2-metadata-token-ttl-seconds':'60'}),timeout=3).read().decode()
 instance=opener.open(urllib.request.Request('http://169.254.169.254/latest/meta-data/instance-id',headers={'X-aws-ec2-metadata-token':token}),timeout=3).read().decode()
h=hashlib.sha256()
with (p/'exe').open('rb') as stream:
 for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
audit=pathlib.Path(sys.argv[2]);events=[]
with audit.open(errors='replace') as stream:
 for line in stream:
  if 'capacity_pressure_published' in line:events.append(line.rstrip())
assert len(events)<=10000,'pressure publication log unexpectedly large'
lockpath=pathlib.Path('/tmp/es-pr3-validation-host.lock');lockstat=lockpath.stat();claimed=int(sys.argv[3]);assert claimed>1
lock_id='%x:%x:%d'%(os.major(lockstat.st_dev),os.minor(lockstat.st_dev),lockstat.st_ino)
held=[line.split() for line in pathlib.Path('/proc/locks').read_text().splitlines()]
assert any(len(row)>5 and row[1]=='FLOCK' and row[3]=='WRITE' and row[4]==str(claimed) and [int(part,16 if i<2 else 10) for i,part in enumerate(row[5].split(':'))]==[os.major(lockstat.st_dev),os.minor(lockstat.st_dev),lockstat.st_ino] for row in held),'Claimed host lock holder does not own flock'
assert subprocess.run(['flock','-n','/tmp/es-pr3-validation-host.lock','true']).returncode==1,'host ownership lock not held'
compilers=[]
for entry in pathlib.Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:name=(entry/'comm').read_text().strip()
 except OSError:continue
 if name in ('cargo','rustc','javac','perf'):compilers.append((entry.name,name))
assert not compilers,compilers
environment=dict(item.split('=',1) for item in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in item)
for key in ('RUSTIC_HNSW_DIAGNOSTIC_EVERY','RUSTIC_HNSW_WRITER_DIAGNOSTIC_EVERY','RUSTIC_PROFILE','RUSTIC_TRACING','OTEL_TRACES_EXPORTER'):
 assert environment.get(key,'').lower() in ('','0','false','off','none'), 'Active instrumentation: '+key
# These real instrumentation switches use presence/path semantics, not booleans.
assert 'RUSTIC_KNN_TRACE' not in environment,'RUSTIC_KNN_TRACE is enabled even when its value is false'
assert not environment.get('ES_TRACE_HTTP',''),'HTTP trace output enabled'
assert 'trace' not in environment.get('RUST_LOG','').lower(),'Trace-level logging enabled'
cmdline=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
assert not any(v.startswith(('-javaagent:','-agentlib:','-agentpath:','-XX:StartFlightRecording')) for v in cmdline),'JVM instrumentation enabled'
print(json.dumps({'affinity':sorted(os.sched_getaffinity(pid)),'kernel':platform.release(),'storage_device':os.stat(sys.argv[4]).st_dev,'unix_millis':int(time.time()*1000),'identity':identity,'instance_id':instance,'binary_sha256':h.hexdigest(),'pressure_events':events,'machine_id':pathlib.Path('/etc/machine-id').read_text().strip(),'pid':pid,'exe':str((p/'exe').resolve(strict=True)),'status':(p/'status').read_text(),'cgroup':str(root),'memory_max':int(limit),'memory_current':int((root/'memory.current').read_text())}))'''
    argv=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-i',node['ssh_key'],node['ssh_target'],'python3','-',str(pid),shlex.quote(node['audit_log']),str(int(node['host_lock_pid'])),shlex.quote(node['data_root'])]
    result=subprocess.run(argv,input=code,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
    if result.returncode:raise RuntimeError(f'Remote pressure fixture evidence failed for {node["instance_id"]}: {result.stderr}')
    fact=json.loads(result.stdout);assert fact['machine_id']==node['machine_id'] and fact['memory_max']==node['effective_memory_max_bytes'] and fact['instance_id']==node['instance_id'] and fact['binary_sha256']==node['binary_sha256']
    return fact
