#!/usr/bin/env python3
"""Load an isolated matched Qdrant fixture, indexing payload before vectors."""
import argparse
import http.client
import json
from pathlib import Path
import time
ap = argparse.ArgumentParser()
ap.add_argument('--corpus', required=True)
ap.add_argument('--collection', default='path-poc-float')
ap.add_argument('--quant', choices=['float', 'int8'], default='float')
ap.add_argument('--port', type=int, default=19290)
ap.add_argument('--out', required=True)
a = ap.parse_args()
assert a.collection.startswith('path-poc-'), 'Dedicated benchmark collections only'
c = http.client.HTTPConnection('127.0.0.1', a.port, timeout=180)
def request(method, path, body=None):
    raw = json.dumps(body, separators=(',', ':')) if body is not None else None
    c.request(method, path, raw, {'Content-Type': 'application/json'})
    r = c.getresponse(); data = json.loads(r.read())
    assert r.status < 300, (r.status, data)
    return data
base = '/collections/' + a.collection
config = {'vectors': {'size': 128, 'distance': 'Cosine', 'on_disk': False},
          'on_disk_payload': False, 'shard_number': 1,
          'hnsw_config': {'m': 16, 'ef_construct': 100, 'payload_m': 16, 'max_indexing_threads': 8},
          'optimizers_config': {'default_segment_number': 1, 'indexing_threshold': 10000,
                                'max_optimization_threads': 1}}
if a.quant == 'int8':
    config['quantization_config'] = {'scalar': {'type': 'int8', 'quantile': 1.0, 'always_ram': True}}
request('PUT', base, config)
request('PUT', base + '/index?wait=true', {'field_name': 'tag', 'field_schema': 'keyword'})
start = time.monotonic(); points = []; count = 0
for line in Path(a.corpus).open():
    doc = json.loads(line)
    points.append({'id': doc['id'], 'vector': doc['v'], 'payload': {'tag': doc['tag']}})
    if len(points) == 1000:
        request('PUT', base + '/points?wait=true', {'points': points})
        count += len(points); points.clear()
        if count % 20000 == 0: print('loaded', count, flush=True)
if points:
    request('PUT', base + '/points?wait=true', {'points': points}); count += len(points)
loaded = time.monotonic(); deadline = loaded + 180
while True:
    state = request('GET', base)['result']
    if (state['status'] == 'green' and state.get('points_count') == count
            and state.get('indexed_vectors_count') == count):
        break
    assert time.monotonic() < deadline, state
    time.sleep(2)
out = {'collection': a.collection, 'documents': count, 'quant': a.quant,
       'load_seconds': loaded - start, 'optimizer_wait_seconds': time.monotonic() - loaded,
       'normal_ingestion_only': True, 'payload_index_before_ingest': True, 'state': state}
Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
print('ready', state['indexed_vectors_count'], 'indexed vectors', flush=True)
