#!/usr/bin/env python3
"""Correctness checks for two full servers over the same persisted 100K index.

Run outside latency measurement windows. The previous binary reads only the
base graph; the integrated binary also reads its checked payload footer.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import struct

from compare import request


def base_graph_hash(path):
    data = path.read_bytes()
    offset, size = struct.unpack_from('<QQ', data, 52)
    graph = data[offset:offset + size]
    layers = struct.unpack_from('<I', graph)[0]
    cursor = 8
    for _ in range(layers):
        nodes = struct.unpack_from('<I', graph, cursor)[0]
        cursor += 8 + nodes * 4
        neighbors = struct.unpack_from('<I', graph, cursor + nodes * 4)[0]
        cursor += (nodes + 1 + neighbors) * 4
    return hashlib.sha256(graph[:cursor]).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--integrated-data', type=Path, required=True)
    ap.add_argument('--before-data', type=Path, required=True)
    ap.add_argument('--integrated-port', type=int, default=19285)
    ap.add_argument('--before-port', type=int, default=19286)
    ap.add_argument('--index', default='path-poc-integrated')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = {'vectors': 100000, 'health': {}, 'count': {}, 'base_graphs': {}}
    connections = {name: http.client.HTTPConnection('127.0.0.1', port, timeout=180)
                   for name, port in [('integrated', args.integrated_port), ('before', args.before_port)]}
    for name, connection in connections.items():
        health = request(connection, 'GET', '/_cluster/health')
        count = request(connection, 'GET', '/' + args.index + '/_count')
        assert health['status'] == 'green' and count['count'] == 100000, (health, count)
        result['health'][name] = health
        result['count'][name] = count
    for name, root in [('integrated', args.integrated_data), ('before', args.before_data)]:
        hashes = []
        for metadata in root.rglob('meta.json'):
            value = json.loads(metadata.read_text())
            for segment in value.get('vector_store', {}).get('segments', []):
                hashes.append(base_graph_hash(metadata.parent / f'vec_{segment:08x}.rvec'))
        assert hashes, 'No committed vector graphs'
        result['base_graphs'][name] = sorted(hashes)
    assert result['base_graphs']['integrated'] == result['base_graphs']['before']
    checked = 0
    entries = [json.loads(line) for line in args.requests.read_text().splitlines()]
    for entry in entries:
        if entry['class'].endswith('/half'):
            continue  # This is the intentionally changed approximate path.
        hits = {}
        for name, connection in connections.items():
            body = json.loads(json.dumps(entry['request']))
            budget = entry.get('target_num_candidates', {}).get(name)
            if budget is not None:
                body['knn']['num_candidates'] = budget
            response = request(connection, 'POST', '/' + args.index + '/_search', body)
            hits[name] = [(hit['_id'], hit['_score']) for hit in response['hits']['hits']]
        assert hits['integrated'] == hits['before'], (entry['class'], hits)
        checked += 1
    result['unchanged_path_exact_response_checks'] = checked
    # A partial group must retain ordinary traversal, including when the
    # constraint arrives through an index alias rather than the KNN body.
    entry = next(e for e in entries if e['class'] == 'knn_nc500/half')
    half = entry['request']['knn']['filter']

    def first_tag(value):
        if isinstance(value, dict):
            tag = value.get('term', {}).get('tag')
            if tag is not None:
                return tag
            children = value.values()
        elif isinstance(value, list):
            children = value
        else:
            return None
        return next((tag for child in children if (tag := first_tag(child)) is not None), None)

    # The fixture independently defines tag=t{id%10}.
    excluded = str(int(first_tag(half)[1:]))
    partial = {'bool': {'filter': [half], 'must_not': [{'ids': {'values': [excluded]}}]}}
    alias = args.index + '-partial'
    pairs = {'direct_partial_filter': {}, 'alias_partial_filter': {}}
    for name, connection in connections.items():
        count = request(connection, 'POST', '/' + args.index + '/_count', {'query': partial})['count']
        assert count == 49999, (name, count)
        request(connection, 'PUT', '/' + args.index + '/_alias/' + alias, {'filter': partial})
        assert request(connection, 'GET', '/' + alias + '/_count')['count'] == count
        body = json.loads(json.dumps(entry['request']))
        body['knn']['num_candidates'] = 550
        body['knn']['filter'] = partial
        direct = request(connection, 'POST', '/' + args.index + '/_search', body)
        del body['knn']['filter']
        indirect = request(connection, 'POST', '/' + alias + '/_search', body)
        pairs['direct_partial_filter'][name] = [(h['_id'], h['_score']) for h in direct['hits']['hits']]
        pairs['alias_partial_filter'][name] = [(h['_id'], h['_score']) for h in indirect['hits']['hits']]
        assert pairs['direct_partial_filter'][name] == pairs['alias_partial_filter'][name]
    for values in pairs.values():
        assert values['integrated'] == values['before'], values
    result['partial_group_checks'] = pairs
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{checked} unchanged-path checks passed on identical base graphs; both servers green, 100K vectors')


if __name__ == '__main__':
    main()
