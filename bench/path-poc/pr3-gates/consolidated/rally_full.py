#!/usr/bin/env python3
"""Invoke pinned Rally on a fully rendered original-corpus track, never test mode.
The manifest pins every track dependency and original corpus source/count/hash.
No task filters, corpus rewriting or arbitrary Rally flags are accepted.
"""
import argparse, bz2, configparser, contextlib, csv, gzip, hashlib, json, lzma, os, pathlib, sys, zipfile
from common import API, ec2, idle_compilers, pin, save, command, remote_fact

def verify_decompressed(archive, plain, suffix, expected_sha):
    """Stream original archive outside timing; never extract or rewrite caches."""
    with contextlib.ExitStack() as stack:
        if suffix=='.bz2':stream=stack.enter_context(bz2.open(archive,'rb'))
        elif suffix=='.gz':stream=stack.enter_context(gzip.open(archive,'rb'))
        elif suffix=='.xz':stream=stack.enter_context(lzma.open(archive,'rb'))
        elif suffix=='.zip':
            zipped=stack.enter_context(zipfile.ZipFile(archive));members=[v for v in zipped.infolist() if not v.is_dir()]
            assert len(members)==1 and pathlib.PurePosixPath(members[0].filename).name==plain.name,'Ambiguous original corpus ZIP members'
            stream=stack.enter_context(zipped.open(members[0]))
        elif suffix=='.zst':
            import zstandard
            source=stack.enter_context(archive.open('rb'))
            stream=stack.enter_context(zstandard.ZstdDecompressor().stream_reader(source,read_across_frames=True))
        else:raise ValueError('Unsupported corpus compression: '+suffix)
        digest=hashlib.sha256();size=0
        for block in iter(lambda:stream.read(1<<20),b''):
            size+=len(block);assert size<=plain.stat().st_size,'Archive expands beyond original cache size'
            digest.update(block)
        assert size==plain.stat().st_size and digest.hexdigest()==expected_sha,'Plain cache does not match pinned original archive decompression'
        return {'archive':str(archive),'plain':str(plain),'bytes':size,'decompressed_sha256':digest.hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--url',required=True);p.add_argument('--engine',choices=['candidate','master','es9'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);status={'passed':False}
    try:
        ec2();idle_compilers();cfg=json.loads(pathlib.Path(a.config).read_text());track=pin(cfg['track']);root=track.parent
        assert track.name=='track.json' and cfg['query_fraction']==1.0,'Acceptance runs the full query schedule; no corpus or task reduction'
        pinned_files={pin(artifact) for artifact in cfg['track_files']}
        assert all(path.is_relative_to(root) for path in pinned_files)
        assert pinned_files=={path.resolve() for path in root.rglob('*') if path.is_file()}, 'Track dependency inventory incomplete'
        rendered=json.loads(track.read_text());expected={}
        for corpus in rendered['corpora']:
            for document in corpus['documents']:
                key=(corpus['name'],document['source-file']);assert key not in expected;expected[key]=document
        declared={};decompression_proofs=[]
        for document in cfg['original_corpora']:
            key=(document['corpus'],document['source_file']);assert key not in declared;declared[key]=document
            corpus_path=pin(document['artifact']);original=expected[key]
            assert corpus_path==(pathlib.Path(cfg['data_root'])/document['corpus']/document['source_file']).resolve(strict=True), 'Rally cache differs from pinned corpus'
            assert document['document_count']==original['document-count'] and document['document_count']>0
            assert document['artifact']['path'].endswith(document['source_file'])
            # Rally may prefer an already decompressed cache. Pin that exact
            # input too: the compressed archive alone cannot vouch for it.
            compressed_suffix=next((suffix for suffix in ('.bz2','.gz','.zst','.zip','.xz') if document['source_file'].endswith(suffix)),None)
            if compressed_suffix:
                plain=pin(document['uncompressed_artifact'])
                plain_name=document['source_file'][:-len(compressed_suffix)]
                assert plain==(pathlib.Path(cfg['data_root'])/document['corpus']/plain_name).resolve(strict=True),'Uncompressed Rally cache differs from pinned corpus'
                assert corpus_path.stat().st_size==original['compressed-bytes']
                assert plain.stat().st_size==original['uncompressed-bytes']
                decompression_proofs.append(verify_decompressed(corpus_path,plain,compressed_suffix,document['uncompressed_artifact']['sha256']))
            else:
                assert corpus_path.stat().st_size==original['uncompressed-bytes']
            for name in ('compressed-bytes','uncompressed-bytes'):
                if name in original:assert document[name.replace('-','_')]==original[name]
        assert set(declared)==set(expected),'Every original corpus must be pinned and present'
        save(out/'corpus-decompression.json',decompression_proofs)
        fixture=json.loads(pin(cfg['fixture_provenance']).read_text());assert len(fixture['nodes'])==2
        save(out/'live-before.json',[remote_fact(node) for node in fixture['nodes']])
        for artifact in cfg['distribution_files']:pin(artifact)
        assert cfg['distribution_files'],'Actual executable/distribution dependencies must be pinned'
        # Bind the actual Rally endpoints to the verified live process fixture,
        # not merely to another valid process elsewhere on the host.
        assert {node['url'].rstrip('/') for node in fixture['nodes']}=={'http://127.0.0.1:19200','http://127.0.0.1:19201'}
        cluster_ids=[];node_ids=[]
        for i,node in enumerate(fixture['nodes']):
            client=API(node['url'],out/f'fixture-{i}.jsonl');root_info,_=client.call('GET','/');cluster_ids.append(root_info['cluster_uuid'])
            process,_=client.call('GET','/_nodes/_local/process');assert len(process['nodes'])==1
            node_id,details=next(iter(process['nodes'].items()));assert node_id==node['node_id'] and details['process']['id']==node['pid'];node_ids.append(node_id)
        assert len(set(cluster_ids))==1 and len(set(node_ids))==2
        api=API(a.url,out/'requests.jsonl');info,_=api.call('GET','/');assert info['cluster_uuid']==cluster_ids[0]
        members,_=api.call('GET','/_nodes/process');assert set(members['nodes'])==set(node_ids)
        assert a.engine!='es9' or info['version']['number']=='9.5.5'
        save(out/'provenance.json',{'config':cfg,'root_info':info,'engine':a.engine,'full_original_corpus':True,'driver_affinity':sorted(os.sched_getaffinity(0))})
        rally=pin(cfg['rally_python']);health=pin(cfg['health_wrapper'])
        # Health wrapper fixes localhost19200/19201; refuse inconsistent fixture.
        assert a.url.rstrip('/')=='http://127.0.0.1:19200'
        argv=[str(rally),'-m','esrally','race','--pipeline=benchmark-only','--offline','--track-path='+str(root),'--challenge='+cfg['challenge'],'--target-hosts=127.0.0.1:19200,127.0.0.1:19201','--report-format=csv','--report-file='+str(out/'rally.csv'),'--on-error=abort','--user-tags=pr3-engine:'+a.engine]
        env=dict(os.environ);env['RALLY_HOME']=str(out/'rally-home');env['HOME']=str(out/'home');env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        pathlib.Path(env['HOME']).mkdir();pathlib.Path(env['RALLY_HOME']).mkdir()
        # Derive a private config from the pinned existing installation, keeping
        # the full offline corpus cache; do not rely on an unsupported env var.
        template=pin(cfg['rally_config']);settings=configparser.ConfigParser();settings.read(template)
        settings['benchmarks']['local.dataset.cache']=cfg['data_root']
        settings['node']['root.dir']=str(out/'rally-node')
        settings['reporting']['datastore.type']='in-memory'
        private=pathlib.Path(env['HOME'])/'.rally';private.mkdir()
        with (private/'rally.ini').open('w') as stream:settings.write(stream)
        command([sys.executable,str(health),'--out',str(out/'health.jsonl'),'--reject-build-overlap','--setup-index',cfg['setup_index'],'--']+argv,out/'rally.log',env)
        idle_compilers();rows=list(csv.DictReader((out/'rally.csv').open()));assert rows
        error_rows=[r for r in rows if r.get('Metric')=='error rate'];assert error_rows and all(float(r['Value'])==0 for r in error_rows)
        counts={}
        for index,count in cfg['final_index_counts'].items():
            actual,_=api.call('GET',f'/{index}/_count');assert actual['count']==count and actual['_shards']['failed']==0;counts[index]=count
        assert counts,'Full corpus final index counts are required'
        save(out/'live-after.json',[remote_fact(node) for node in fixture['nodes']])
        status.update(passed=True,failed=0,skipped=0,corpora=len(expected),final_counts=counts,report_rows=len(rows))
    except Exception as error:status['error']=str(error);raise
    finally:save(out/'summary.json',status);(out/'exit.code').write_text('0\n' if status['passed'] else '1\n')
if __name__=='__main__':main()
