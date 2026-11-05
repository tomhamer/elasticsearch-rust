#!/usr/bin/env python3
"""Compare the validation answers of rustic and a reference ES for every route.

For each route in routes.json (security routes skipped): an unknown param,
a near-miss param ("did you mean"), a param only one REST API version
accepts, a body on a body-less route, a missing body on a body-required
route, and an unsupported method. Prints mismatches of (status, error type,
reason, Allow).

With --compat8 every request carries the `compatible-with=8` Accept (and,
with a body, Content-Type) headers, and the v8 rules / v8-only routes are
used; natively, v8-only routes are probed for their "no handler found".

usage: compare.py <es port> <rustic port> [--all] [--compat8]
"""
import json, os, sys, http.client

ES, RU = int(sys.argv[1]), int(sys.argv[2])
SHOW_ALL = '--all' in sys.argv
COMPAT8 = '--compat8' in sys.argv
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'routes.json')))
routes = d['routes']
sets = d['sets']


def send(port, method, url, body=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    h = {}
    if COMPAT8:
        h['Accept'] = 'application/vnd.elasticsearch+json;compatible-with=8'
    if body is not None:
        h['Content-Type'] = 'application/vnd.elasticsearch+json;compatible-with=8' if COMPAT8 else 'application/json'
    c.request(method, url, body=body.encode() if body is not None else None, headers=h)
    r = c.getresponse()
    data = r.read()
    try:
        j = json.loads(data) if data else None
    except Exception:
        j = data[:200].decode('utf-8', 'replace')
    t = reason = None
    if isinstance(j, dict) and isinstance(j.get('error'), dict):
        t, reason = j['error'].get('type'), j['error'].get('reason')
    elif isinstance(j, dict) and isinstance(j.get('error'), str):
        reason = j['error']
    return (r.status, t, reason, r.getheader('Allow'))


cases = []
by_path = {}
for r in routes:
    if r['security']:
        continue
    by_path.setdefault(r['path'], set()).add(r['method'])
    p = r['path'].replace('*', 'x1')
    if r['v8_only'] and not COMPAT8:
        cases.append(('V8', r['method'], p + '?zzqq_unknown=1', None))
        continue
    rules = r.get('v8', r) if COMPAT8 else r
    other = r if COMPAT8 else r.get('v8', r)
    # A param only the other REST API version accepts.
    only_other = sorted(set(sets[other['accepted'][0]]) - set(sets[rules['accepted'][0]]))
    only_this = sorted(set(sets[rules['accepted'][0]]) - set(sets[other['accepted'][0]]))
    for q in only_other[:1] + only_this[:1]:
        cases.append(('VP', r['method'], p + '?' + q + '=false', None))
    if r['method'] == 'HEAD':
        cases.append(('H', r['method'], p + '?zzqq_unknown=1', None))
        continue
    cases.append(('P', r['method'], p + '?zzqq_unknown=1', None))
    acc = sets[rules['accepted'][0]]
    near = [a for a in acc if len(a) > 4 and a not in ('pretty', 'human', 'format', 'error_trace', 'filter_path', 'source')]
    if near:
        cases.append(('P~', r['method'], p + '?' + near[0][:-1] + '=1', None))
    if rules['body'] in ('none',) or rules['body'].startswith('reject'):
        cases.append(('B', r['method'], p, '{}'))
    if rules['body'] in ('required', 'required_or_source'):
        cases.append(('R', r['method'], p + '?zzqq_unknown=1', None))
for path, ms in by_path.items():
    for m in ('GET', 'POST', 'PUT', 'DELETE'):
        if m not in ms:
            cases.append(('405', m, path.replace('*', 'x1'), None))
            break

stats = {}
for kind, m, url, body in cases:
    a = send(ES, m, url, body)
    b = send(RU, m, url, body)
    ok = a == b
    stats.setdefault(kind, [0, 0])[0 if ok else 1] += 1
    if not ok or SHOW_ALL:
        print(('OK  ' if ok else 'DIFF'), kind, m, url, '\n    es:', a, '\n    ru:', b)
for k, (ok, bad) in stats.items():
    print(f'{k}: {ok} same, {bad} differ')
