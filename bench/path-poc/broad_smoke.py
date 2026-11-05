#!/usr/bin/env python3
"""Small three-shard compatibility probe using ordinary bulk + refresh."""
import argparse
import http.client
import json
import math
import pathlib
import random
import urllib.parse

from compare import request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--es-port', type=int, default=19211)
    ap.add_argument('--our-port', type=int, default=19214)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    connections = {name: http.client.HTTPConnection('127.0.0.1', port, timeout=120)
                   for name, port in [('es', args.es_port), ('full', args.our_port)]}
    rng = random.Random(149)
    docs = []
    for i in range(768):
        v = [rng.uniform(-1, 1) for _ in range(128)]
        n = math.sqrt(sum(x*x for x in v))
        docs.append({'id': i, 'body': 'alpha beta gamma' if i % 3 else 'alpha delta',
                     'tag': f't{i % 5}', 'price': i / 10, 'v': [x/n for x in v]})
    index = 'path-poc-broad'
    for c in connections.values():
        # The cross-index fixture from a previous run also matches the main
        # wildcard probe. Remove it before testing that probe, then recreate
        # it below for the cross-index checks.
        c.request('DELETE', '/' + index + '-small')
        c.getresponse().read()
        c.request('DELETE', '/' + index)
        c.getresponse().read()
        request(c, 'PUT', '/' + index, {
            'settings': {'number_of_shards': 3, 'number_of_replicas': 0,
                         'refresh_interval': '-1', 'index.queries.cache.enabled': False},
            'mappings': {'properties': {'id': {'type': 'long'}, 'price': {'type': 'double'},
                'body': {'type': 'text'}, 'tag': {'type': 'keyword'},
                'v': {'type': 'dense_vector', 'dims': 128, 'similarity': 'cosine',
                      'index_options': {'type': 'hnsw', 'm': 16, 'ef_construction': 100}}}}})
        lines = []
        for doc in docs:
            lines.extend([json.dumps({'index': {'_index': index, '_id': str(doc['id'])}}),
                          json.dumps(doc)])
        request(c, 'POST', '/_bulk', '\n'.join(lines) + '\n', True)
        request(c, 'POST', '/' + index + '/_refresh')
    queries = {
        'all': {'match_all': {}}, 'term': {'term': {'body': 'beta'}},
        'and': {'bool': {'must': [{'term': {'body': 'alpha'}}, {'term': {'body': 'beta'}}]}},
        'or': {'bool': {'should': [{'term': {'body': 'delta'}}, {'term': {'body': 'beta'}}]}},
        'phrase': {'match_phrase': {'body': 'beta gamma'}},
        'range': {'range': {'price': {'gte': 10, 'lt': 20}}},
        'excluded': {'bool': {'filter': {'term': {'tag': 't1'}},
                             'must_not': {'term': {'body': 'delta'}}}},
        'prefix': {'prefix': {'tag': 't'}},
    }
    results = {}
    for stage in ['refreshed', 'updated']:
        if stage == 'updated':
            for c in connections.values():
                for i in range(0, 768, 97):
                    request(c, 'DELETE', f'/{index}/_doc/{i}')
                request(c, 'PUT', f'/{index}/_doc/5', {**docs[5], 'tag': 'changed', 'body': 'delta'})
                request(c, 'POST', '/' + index + '/_refresh')
        for label, query in queries.items():
            for source in [False, True]:
                body = {'query': query, 'size': 10, 'sort': [{'id': 'asc'}],
                        '_source': source, 'track_total_hits': True}
                pair = {name: request(c, 'POST', '/' + index + '/_search', body)
                        for name, c in connections.items()}
                a, b = pair['es'], pair['full']
                assert a['hits']['total'] == b['hits']['total'], (stage, label, pair)
                for x, y in zip(a['hits']['hits'], b['hits']['hits'], strict=True):
                    assert x['_id'] == y['_id'] and x['sort'] == y['sort'], (stage, label, x, y)
                    assert ('_source' in x) == source == ('_source' in y)
                    if source:
                        # ES omits vector source by default; compare ordinary fields.
                        assert {k:v for k,v in x['_source'].items() if k != 'v'} == {
                            k:v for k,v in y['_source'].items() if k != 'v'}
                results[f'{stage}/{label}/{source}'] = {'total': a['hits']['total']}
        body = {'size': 0, 'aggs': {'tags': {'terms': {'field': 'tag', 'size': 10}},
            'mean': {'avg': {'field': 'price'}}, 'maximum': {'max': {'field': 'price'}},
            'minimum': {'min': {'field': 'price'}}, 'sum': {'sum': {'field': 'price'}}}}
        pair = {name: request(c, 'POST', '/' + index + '/_search', body)['aggregations']
                for name, c in connections.items()}
        assert pair['es']['tags'] == pair['full']['tags'], pair
        for metric in ['mean', 'minimum', 'maximum', 'sum']:
            assert math.isclose(pair['es'][metric]['value'], pair['full'][metric]['value'],
                                rel_tol=1e-9, abs_tol=1e-9), pair
        results[f'{stage}/aggregations'] = pair
        for filtered in [False, True]:
            vector = docs[23]['v']
            knn = {'field': 'v', 'query_vector': vector, 'k': 10, 'num_candidates': 1000}
            if filtered:
                knn['filter'] = {'term': {'tag': 't3'}}
            body = {'knn': knn, 'size': 10, '_source': False, 'track_total_hits': True}
            pair = {name: request(c, 'POST', '/' + index + '/_search', body)
                    for name, c in connections.items()}
            a, b = pair['es']['hits'], pair['full']['hits']
            assert a['total'] == b['total'] and a['total']['value'] == 10, pair
            for x, y in zip(a['hits'], b['hits'], strict=True):
                assert x['_id'] == y['_id'], pair
                assert math.isclose(x['_score'], y['_score'], rel_tol=2e-5, abs_tol=2e-5), pair
            results[f'{stage}/knn/{filtered}'] = {'ids': [x['_id'] for x in b['hits']]}
    # Resolver fallback and context-bearing responses exercise the cluster
    # optimizations' compatibility paths, beyond ordinary search hit bodies.
    ids = lambda value: [hit['_id'] for hit in value['hits']['hits']]
    body = {'size': 10, 'sort': [{'id': 'asc'}], '_source': False, 'track_total_hits': True}
    for c in connections.values():
        request(c, 'PUT', f'/{index}/_alias/path-poc-broad-filtered',
                {'filter': {'term': {'tag': 't1'}}})
    for expression in ['path-poc-broad-filtered', 'path-poc-broa*']:
        pair = {name: request(c, 'POST', '/' + urllib.parse.quote(expression, safe=',*') + '/_search', body)
                for name, c in connections.items()}
        assert pair['es']['hits']['total'] == pair['full']['hits']['total'], pair
        assert ids(pair['es']) == ids(pair['full']), pair
        results['resolver/' + expression] = {'ids': ids(pair['full'])}
    small = index + '-small'
    for c in connections.values():
        c.request('DELETE', '/' + small)
        c.getresponse().read()
        request(c, 'PUT', '/' + small, {'settings': {'number_of_shards': 1, 'number_of_replicas': 0},
                'mappings': {'properties': {'id': {'type': 'long'}, 'tag': {'type': 'keyword'}}}})
        for i in range(10):
            request(c, 'PUT', f'/{small}/_doc/{2000+i}', {'id': 2000+i, 'tag': f't{i%5}'})
        request(c, 'POST', '/' + small + '/_refresh')
        request(c, 'POST', '/_aliases', {'actions': [
            {'add': {'index': index, 'alias': 'path-poc-broad-cross', 'filter': {'term': {'tag': 't1'}}}},
            {'add': {'index': small, 'alias': 'path-poc-broad-cross', 'filter': {'term': {'tag': 't2'}}}},
            {'add': {'index': index, 'alias': 'path-poc-broad-filtered-alt', 'filter': {'term': {'tag': 't2'}}}},
        ]})
    for expression in ['path-poc-broad-cross', 'path-poc-broad-cross,' + index,
                       small + ',path-poc-broad-filtered', 'path-poc-broad-filter*']:
        pair = {name: request(c, 'POST', '/' + expression + '/_search', body)
                for name, c in connections.items()}
        assert pair['es']['hits']['total'] == pair['full']['hits']['total'], (expression, pair)
        assert ids(pair['es']) == ids(pair['full']), (expression, pair)
        counts = {name: request(c, 'POST', '/' + expression + '/_count', {})['count']
                  for name, c in connections.items()}
        assert counts['es'] == counts['full'] == pair['es']['hits']['total']['value'], counts
        results['resolver/' + expression] = {'ids': ids(pair['full']), 'count': counts['full']}
    global_body = {'size': 0, 'query': {'term': {'body': 'delta'}},
                   'aggs': {'all': {'global': {}, 'aggs': {'ids': {'value_count': {'field': 'id'}}}}}}
    global_results = {name: request(c, 'POST', '/path-poc-broad-cross/_search', global_body)
                      for name, c in connections.items()}
    assert global_results['es']['aggregations'] == global_results['full']['aggregations'], global_results
    results['alias/global-aggregation'] = global_results['full']['aggregations']
    significant = {'size': 0, 'aggs': {'tags': {'significant_terms': {
        'field': 'tag', 'min_doc_count': 1, 'size': 10}}}}
    pair = {name: request(c, 'POST', '/path-poc-broad-cross/_search', significant)['aggregations']
            for name, c in connections.items()}
    assert pair['es'] == pair['full'], pair
    results['alias/significant-terms-background'] = pair['full']
    scrolls, pits = {}, {}
    for name, c in connections.items():
        first = request(c, 'POST', '/path-poc-broad-cross/_search?scroll=1m', body)
        assert isinstance(first.get('_scroll_id'), str), first
        second = request(c, 'POST', '/_search/scroll',
                         {'scroll': '1m', 'scroll_id': first['_scroll_id']})
        assert not set(ids(first)).intersection(ids(second)), (first, second)
        scrolls[name] = ids(first) + ids(second)
        request(c, 'DELETE', '/_search/scroll', {'scroll_id': [second['_scroll_id']]})
        pit = request(c, 'POST', '/path-poc-broad-cross/_pit?keep_alive=1m')['id']
        search = request(c, 'POST', '/_search', {**body, 'pit': {'id': pit, 'keep_alive': '1m'}})
        pits[name] = ids(search)
        request(c, 'DELETE', '/_pit', {'id': search.get('pit_id', pit)})
    assert scrolls['es'] == scrolls['full'], scrolls
    assert pits['es'] == pits['full'], pits
    results['scroll'] = scrolls
    results['pit'] = pits
    args.out.write_text(json.dumps({'force_merge': False, 'shards': 3, 'docs': 768,
                                    'checks': results}, indent=2) + '\n')
    print(f'{len(results)} three-shard query checks passed', flush=True)


if __name__ == '__main__':
    main()
