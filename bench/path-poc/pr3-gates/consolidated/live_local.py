#!/usr/bin/env python3
"""Read live executable and distribution identity at timing boundaries on EC2."""
import argparse,json,os,pathlib,urllib.request
from common import ec2,idle_compilers,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--binary',required=True);p.add_argument('--engine',required=True,choices=['candidate','master','es9']);p.add_argument('--distribution');p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);a=p.parse_args();machine=ec2();idle_compilers()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));nodes=json.load(opener.open(a.url+'/_nodes/process',timeout=30))['nodes'];assert len(nodes)==2
    expected=sha(a.binary);distribution=None
    if a.engine=='es9':
        distribution=json.loads(pathlib.Path(a.distribution).read_text());root=pathlib.Path(distribution['root']);assert pathlib.Path(a.binary).resolve()==pathlib.Path(distribution['launcher']).resolve()
        for item in distribution['files']:assert sha(root/item['path'])==item['sha256']
        expected=next(item['sha256'] for item in distribution['files'] if item['path']=='jdk/bin/java')
    rows=[]
    for node_id,node in nodes.items():
        pid=node['process']['id'];proc=pathlib.Path('/proc')/str(pid);actual=sha(proc/'exe');assert actual==expected
        env=dict(item.split('=',1) for item in (proc/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in item)
        for key in ('RUSTIC_HNSW_DIAGNOSTIC_EVERY','RUSTIC_HNSW_WRITER_DIAGNOSTIC_EVERY','RUSTIC_PROFILE','RUSTIC_TRACING','OTEL_TRACES_EXPORTER'):assert env.get(key,'').lower() in ('','0','off','false','none')
        assert 'RUSTIC_KNN_TRACE' not in env,'Presence enables KNN tracing even for false/0'
        assert not env.get('ES_TRACE_HTTP',''),'HTTP trace output enabled'
        assert 'trace' not in env.get('RUST_LOG','').lower(),'Trace-level logging enabled'
        cmdline=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        assert not any(v.startswith(('-javaagent:','-agentlib:','-agentpath:','-XX:StartFlightRecording')) for v in cmdline),'JVM instrumentation enabled'
        relative=next(line.split(':',2)[2] for line in (proc/'cgroup').read_text().splitlines() if line.startswith('0::'))
        memory=(pathlib.Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.max').read_text().strip();assert memory!='max'
        rows.append({'memory_max':int(memory),'node_id':node_id,'pid':pid,'exe_sha256':actual,'exe':str((proc/'exe').resolve()),'affinity':sorted(os.sched_getaffinity(pid))})
    save(pathlib.Path(a.out),{'machine':machine,'storage_device':os.stat(a.data_root).st_dev,'engine':a.engine,'launcher_sha256':sha(a.binary),'expected_executable_sha256':expected,'distribution_sha256':sha(a.distribution) if a.distribution else None,'nodes':rows,'instrumentation':'off'})
if __name__=='__main__':main()
