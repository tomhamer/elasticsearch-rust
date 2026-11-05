#!/usr/bin/env python3
"""Bulk + refresh lifecycle probe. Never calls force merge or segments APIs."""
import argparse
import http.client
import json
import pathlib
import time

from compare import request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--index', default='path-poc-live')
    ap.add_argument('--quant', choices=['hnsw', 'int8_hnsw'], default='hnsw')
    args = ap.parse_args()
    c = http.client.HTTPConnection('127.0.0.1', args.port, timeout=180)
    name = args.index
    request(c, 'PUT', '/' + name, {
        'settings': {'number_of_shards': 1, 'number_of_replicas': 0,
                     'refresh_interval': '-1', 'index.queries.cache.enabled': False},
        'mappings': {'properties': {
            'tag': {'type': 'keyword'},
            'v': {'type': 'dense_vector', 'dims': 128, 'similarity': 'cosine',
                  'index_options': {'type': args.quant, 'm': 16, 'ef_construction': 100,
                                    **({'confidence_interval': 1.0} if args.quant == 'int8_hnsw' else {})}}}}})
    start = time.monotonic()
    count = 0
    batch = []
    with (args.root / 'corpus.jsonl').open() as f:
        for line in f:
            doc = json.loads(line)
            batch.extend([json.dumps({'index': {'_index': name, '_id': str(doc['id'])}}), line.rstrip()])
            count += 1
            if count % 500 == 0:
                response = request(c, 'POST', '/_bulk', '\n'.join(batch) + '\n', True)
                assert not response['errors'], response
                batch = []
        if batch:
            response = request(c, 'POST', '/_bulk', '\n'.join(batch) + '\n', True)
            assert not response['errors'], response
    load_seconds = time.monotonic() - start
    request(c, 'POST', '/' + name + '/_refresh')
    q = json.loads((args.root / 'queries.json').read_text())[0]
    body = {'size': 10, '_source': False, 'track_total_hits': True,
            'knn': {'field': 'v', 'query_vector': q, 'k': 10, 'num_candidates': 500}}
    if args.quant == 'int8_hnsw':
        body['knn']['rescore_vector'] = {'oversample': 3.0}
    samples = []
    refreshed = time.monotonic()
    for delay in (0, 2, 6, 12, 20):
        time.sleep(max(0, delay - (time.monotonic() - refreshed)))
        for iteration in range(20):
            t = time.monotonic_ns()
            response = request(c, 'POST', '/' + name + '/_search', body)
            elapsed = (time.monotonic_ns() - t) / 1000
            assert response['hits']['total']['value'] == 10, response
            samples.append({'after_refresh_s': time.monotonic() - refreshed,
                            'iteration': iteration, 'latency_us': elapsed,
                            'ids': [h['_id'] for h in response['hits']['hits']]})
    args.out.write_text(json.dumps({'docs': count, 'load_seconds': load_seconds,
                                   'force_merge': False, 'samples': samples}, indent=2) + '\n')


if __name__ == '__main__':
    main()
