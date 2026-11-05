#!/usr/bin/env python3
"""Build the route universe: spec routes + routes ES registers that the spec lacks
(from javap of the REST action classes), each tagged compat / security.

`compat`: registered only for the previous REST API version (ES 9.x: routes
built with `RestApiVersion.V_8`, e.g. `/{index}/_knn_search`). Such a route
dispatches only `compatible-with=8` requests; natively it only widens `Allow`.

usage: universe.py <api dir> <es_routes.json> <probe.json> <out.json>
"""
import json, glob, os, re, sys, collections

API, ESR, PROBE, OUT = sys.argv[1:5]


def norm(p):
    p = re.sub(r'\{[^}]+\}', '*', p)
    return p.rstrip('/') or '/'


SEC_PREFIX = ('/_security', '/_xpack/security', '/_xpack/ssl', '/_ssl')
SKIP_PREFIX = ('/_idp',)  # identity provider plugin: disabled by default
# Spec routes ES 9.5.5 does not register (verify405.py shows ES's OPTIONS
# answer differs with them in the trie): aliases ES serves through a wildcard
# route (registering them would shadow it), trailing-slash forms, the example
# cat plugin, serverless-only / feature-flagged APIs.
DROP = {('POST', '/_inference/completion/*'), ('POST', '/_inference/rerank/*'), ('POST', '/_inference/sparse_embedding/*'),
        ('POST', '/_inference/text_embedding/*'), ('POST', '/_inference/embedding/*'),
        ('GET', '/_rollup/data'), ('GET', '/_rollup/job'), ('GET', '/_cat/example'),
        ('GET', '/_project/tags'), ('POST', '/_project/tags'),
        ('GET', '/_project_routing'), ('PUT', '/_project_routing'),
        ('GET', '/_project_routing/*'), ('PUT', '/_project_routing/*'), ('DELETE', '/_project_routing/*'),
        ('GET', '/_query/data_source'), ('GET', '/_query/data_source/*'), ('PUT', '/_query/data_source/*'),
        ('DELETE', '/_query/data_source/*'),
        ('GET', '/_query/dataset'), ('GET', '/_query/dataset/*'), ('PUT', '/_query/dataset/*'), ('DELETE', '/_query/dataset/*')}

specs = {}
for f in sorted(glob.glob(os.path.join(API, '*.json'))):
    if f.endswith('_common.json'):
        continue
    name, spec = next(iter(json.load(open(f)).items()))
    specs[name] = spec

probe = {(r['method'], r['path']): r for r in json.load(open(PROBE))} if os.path.exists(PROBE) else {}

routes = collections.OrderedDict()


def add(method, path, api, compat, origin, parts=()):
    k = (method, norm(path))
    r = routes.get(k)
    if r is None:
        r = routes[k] = {'method': method, 'path': norm(path), 'apis': [], 'compat': compat, 'origin': origin, 'parts': set(),
                         'security': norm(path).startswith(SEC_PREFIX)}
    if api and api not in r['apis']:
        r['apis'].append(api)
    r['compat'] = r['compat'] and compat
    r['parts'] |= set(parts)
    return r


for name, spec in specs.items():
    for p in spec['url']['paths']:
        for m in p['methods']:
            if (m, norm(p['path'])) in DROP or norm(p['path']).startswith(SKIP_PREFIX):
                continue
            r = add(m, p['path'], name, False, 'spec', (p.get('parts') or {}).keys())
            if name.startswith('security.') or name.startswith('ssl.'):
                r['security'] = True

# Map javap handler classes to spec apis (for params of non-spec routes).
es = json.load(open(ESR))
cls_api = collections.defaultdict(collections.Counter)
for e in es:
    k = (e['method'], norm(e['path']))
    if k in routes and routes[k]['origin'] == 'spec':
        for a in routes[k]['apis']:
            cls_api[e['cls']][a] += 1
# Spec routes ES registers only for the previous REST API version.
javap_compat = collections.defaultdict(set)
for e in es:
    javap_compat[(e['method'], norm(e['path']))].add(e['compat'])
for k, flags in javap_compat.items():
    if k in routes and flags == {True}:
        routes[k]['compat'] = True
for e in es:
    if norm(e['path']).startswith(SKIP_PREFIX):
        continue
    k = (e['method'], norm(e['path']))
    if k in routes or k in DROP:
        continue
    api = cls_api[e['cls']].most_common(1)[0][0] if cls_api[e['cls']] else None
    parts = [t[1:-1] for t in e['path'].split('/') if t.startswith('{')]
    r = add(e['method'], e['path'], api, e['compat'], 'javap:' + e['cls'].split('.')[-1], parts)

out = []
for r in routes.values():
    r['parts'] = sorted(r['parts'])
    out.append(r)
json.dump(out, open(OUT, 'w'), indent=1)
c = collections.Counter((r['origin'] == 'spec', r['compat'], r['security']) for r in out)
print(len(out), c)
