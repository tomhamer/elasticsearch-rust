#!/usr/bin/env python3
"""Compare the modelled trie (universe.json, security off) with ES OPTIONS answers.

usage: verify405.py universe.json [extra paths...]
"""
import json, sys, http.client, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trie import Trie

U = json.load(open(sys.argv[1]))
ORDER = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD', 'PATCH', 'TRACE', 'CONNECT']
t = Trie()
for i, r in enumerate(U):
    if r['security']:
        continue
    t.insert(r['path'].replace('*', '{x}'), [i])


def model(path):
    vals = [v for v in t.retrieve_all(path) if v]
    if not vals:
        return None
    ms = set()
    for v in vals:
        for i in v:
            ms.add(U[i]['method'])
    return ','.join(m for m in ORDER if m in ms)


def options(path):
    c = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('ES_PORT', '9299')), timeout=30)
    c.request('OPTIONS', path)
    r = c.getresponse()
    r.read()
    return r.status, r.getheader('Allow')


paths = set()
for r in U:
    p = r['path']
    paths.add(p.replace('*', 'x1'))
    # also try the wildcard positions with explicit-looking tokens
paths |= set(sys.argv[2:])
paths |= {'/aa', '/aa/bb', '/aa/bb/cc', '/aa/bb/cc/dd', '/aa/bb/cc/dd/ee', '/_security/user', '/_security', '/_rollup/job', '/_streams/status',
          '/_nodes/hotthreads', '/_cluster/nodes/hotthreads', '/aa/_mapping/bb', '/_xpack', '/_xpack/x1', '/', '/aa/_doc', '/aa/_doc/bb/cc'}
bad = 0
for p in sorted(paths):
    st, allow = options(p)
    es = allow if st == 200 else None
    m = model(p)
    if (es or None) != (m or None):
        bad += 1
        print(f'MISMATCH {p}: es={st} {allow} model={m}')
print('checked', len(paths), 'mismatches', bad)
