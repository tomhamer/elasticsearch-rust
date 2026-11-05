#!/usr/bin/env python3
"""Check vector visibility while indexing and refreshing; never force merge."""
import argparse
import concurrent.futures
import http.client
import json
import math
import pathlib
import random
import time

from compare import request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    index = 'path-poc-write'
    rng = random.Random(281)
    docs = []
    for i in range(12000):
        vector = [rng.uniform(-1, 1) for _ in range(128)]
        norm = math.sqrt(sum(x*x for x in vector))
        docs.append({'tag': f't{i % 10}', 'v': [x/norm for x in vector]})
    c = http.client.HTTPConnection('127.0.0.1', args.port, timeout=120)
    c.request('DELETE', '/' + index)
    c.getresponse().read()
    request(c, 'PUT', '/' + index, {'settings': {'number_of_shards': 1,
        'number_of_replicas': 0, 'refresh_interval': '-1'}, 'mappings': {'properties': {
        'tag': {'type': 'keyword'}, 'v': {'type': 'dense_vector', 'dims': 128,
            'similarity': 'cosine', 'index_options': {'type': 'hnsw', 'm': 16, 'ef_construction': 100}}}}})

    def bulk(conn, start, end):
        lines = []
        for i in range(start, end):
            lines.extend([json.dumps({'index': {'_index': index, '_id': str(i)}}), json.dumps(docs[i])])
        request(conn, 'POST', '/_bulk', '\n'.join(lines)+'\n', True)
        request(conn, 'POST', '/' + index + '/_refresh')

    for start in range(0, 10000, 500):
        bulk(c, start, start+500)
    # Exercise a persisted graph alongside subsequently refreshed live vectors.
    time.sleep(6)

    def check(conn, i):
        body = {'size': 1, '_source': False, 'knn': {'field': 'v', 'query_vector': docs[i]['v'],
                'k': 1, 'num_candidates': 100, 'filter': {'term': {'tag': docs[i]['tag']}}}}
        value = request(conn, 'POST', '/' + index + '/_search', body)
        assert len(value['hits']['hits']) == 1, value
        hit = value['hits']['hits'][0]
        assert hit['_id'] == str(i) and '_source' not in hit, (i, value)
        assert math.isclose(hit['_score'], 1.0, abs_tol=2e-5), value

    def writer():
        conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=120)
        for start in range(10000, 12000, 200):
            bulk(conn, start, start+200)
            check(conn, start+199)
            time.sleep(0.05)
        conn.close()
        return 10

    start = time.monotonic()
    reads = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(writer)
        while not writing.done():
            check(c, (reads*97) % 10000)
            reads += 1
        write_checks = writing.result()
    for i in range(10000, 12000, 199):
        check(c, i)
    request(c, 'DELETE', f'/{index}/_doc/11999')
    request(c, 'POST', '/' + index + '/_refresh')
    body = {'size': 1, '_source': False,
            'knn': {'field': 'v', 'query_vector': docs[11999]['v'], 'k': 1, 'num_candidates': 100}}
    response = request(c, 'POST', '/' + index + '/_search', body)
    assert all(h['_id'] != '11999' for h in response['hits']['hits']), response
    args.out.write_text(json.dumps({'docs': 12000, 'force_merge': False,
        'refreshed_tail_checks': write_checks, 'concurrent_reads': reads,
        'deleted_vector_excluded': True, 'wall_seconds': time.monotonic()-start}, indent=2)+'\n')
    print(f'{reads} concurrent reads and {write_checks} refreshed-tail checks passed', flush=True)


if __name__ == '__main__':
    main()
