#!/usr/bin/env python3
"""Reproduce the clustered 100K float/int8 comparison inputs (no server calls)."""
import argparse
import json
import pathlib
import struct

from knn_compare import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--docs', type=int, default=100000)
    ap.add_argument('--queries', type=int, default=20)
    args = ap.parse_args()
    assert args.docs >= 100 and args.queries > 0
    docs, queries = generate(args.root, n=args.docs, nq=args.queries)
    with (args.root/'vectors.f32').open('wb') as out:
        for doc in docs:
            out.write(struct.pack('<128f', *doc['v']))
        for query in queries:
            out.write(struct.pack('<128f', *query))
    entries = []
    for qi, query in enumerate(queries):
        for candidates in [100, 500]:
            for label in ['all', 'ten', 'half']:
                knn = {'field': 'v', 'query_vector': query, 'k': 10, 'num_candidates': candidates}
                if label == 'ten':
                    knn['filter'] = {'term': {'tag': 't0'}}
                if label == 'half':
                    knn['filter'] = {'bool': {'should': [{'term': {'tag': f't{i}'}} for i in range(5)]}}
                entries.append({'id': f'{qi}/{candidates}/{label}',
                    'class': f'knn_nc{candidates}/{label}',
                    'request': {'size': 10, '_source': False, 'track_total_hits': True, 'knn': knn}})
    write = lambda name, rows: (args.root/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write('filtered-requests.jsonl', entries)
    write('requests.jsonl', [row for row in entries if row['id'].endswith('/all')])
    for entry in entries:
        entry['request']['knn']['rescore_vector'] = {'oversample': 3.0}
    write('int8-filtered-requests.jsonl', entries)
    print(f'Generated {args.docs} vectors, {args.queries} queries and {len(entries)} filtered requests')


if __name__ == '__main__':
    main()
