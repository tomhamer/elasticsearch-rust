#!/usr/bin/env python3
"""Probe a reference ES for per-route request validation behaviour.

For each spec route (method + path template) records:
  - r_probe: response to (no body, sentinel unknown param) -> shows [R] errors
  - b_probe: response to (body {}, no params) -> shows [B] errors
  - accepted[nobody|body]: params consumed (bisection, sentinel param keeps the
    action from ever executing)
  - cands[nobody|body]: "did you mean" candidate set
Writes JSON to argv[2].

With COMPAT8=1 every request carries `Accept` (and, with a body,
`Content-Type`) `application/vnd.elasticsearch+json;compatible-with=8`, so ES
answers as REST API v8 (routes registered only for v8 dispatch, v8-only
params are consumed).

usage: [ES_PORT=9299] [COMPAT8=1] probe.py <api dir> <out.json> [universe.json]
  (with universe.json: probe only the non-spec routes ES registers)
"""
import json, glob, os, re, sys, http.client, urllib.parse, collections
from concurrent.futures import ThreadPoolExecutor

API = sys.argv[1]
OUT = sys.argv[2]
HOST, PORT = '127.0.0.1', int(os.environ.get('ES_PORT', '9299'))
SENT = 'zzqq_unknown'
COMPAT8 = os.environ.get('COMPAT8') == '1'
Q_FAMILY = ['df', 'analyzer', 'analyze_wildcard', 'lenient', 'default_operator']

specs = {}
for f in sorted(glob.glob(os.path.join(API, '*.json'))):
    if f.endswith('_common.json'):
        continue
    d = json.load(open(f))
    name, spec = next(iter(d.items()))
    specs[name] = spec
common = json.load(open(os.path.join(API, '_common.json')))['params']

# name universe + value by type
types = collections.defaultdict(collections.Counter)
enum_opts = {}
for name, spec in specs.items():
    for p, ps in (spec.get('params') or {}).items():
        types[p][ps.get('type', 'string')] += 1
        if ps.get('options'):
            enum_opts.setdefault(p, ps['options'][0])
    for path in spec['url']['paths']:
        for part in (path.get('parts') or {}):
            types[part]['string'] += 1
for p in common:
    types[p]['string'] += 1
EXTRA = ['format', 'source_content_type', 'local', 'master_timeout', 'timeout', 'type', 'include_type_name', 'help', 'v', 'h', 's',
         'bytes', 'time', 'ts', 'rest_total_hits_as_int', 'index', 'id', 'name', 'level', 'flat_settings', 'wait_for_completion',
         'include_defaults', 'size', 'from', 'q', 'pretty', 'human', 'error_trace', 'filter_path', 'source', 'ignore_unavailable',
         'allow_no_indices', 'expand_wildcards', 'routing', 'preference', 'version', 'refresh', 'pipeline', 'fields',
         'include_unloaded_segments', 'all', 'detailed', 'actions', 'nodes', 'parent_task_id', 'group_by', 'metric', 'node_id',
         'settings_filter', 'verify', 'include_bootstrap', 'resource', 'include_source_on_error', 'allow_partial_search_results',
         'cluster_manager_timeout', 'interval', 'snapshots', 'threads', 'ignore_idle_threads', 'sort', 'order', 'local_only']
for p in EXTRA:
    types[p]['string'] += 0 if p in types else 1
NAMES = sorted(types)


def value_for(p):
    t = types[p].most_common(1)[0][0] if types[p] else 'string'
    if t == 'boolean':
        return 'false'
    if t in ('number', 'int', 'long', 'integer', 'double', 'float'):
        return '1'
    if t == 'time':
        return '1s'
    if t == 'enum':
        return enum_opts.get(p, 'x1')
    return 'x1'


def norm(p):
    return re.sub(r'\{[^}]+\}', '*', p)


routes = collections.OrderedDict()  # (method, normpath) -> info
for name, spec in specs.items():
    body = spec.get('body')
    brule = 'none' if body is None else ('required' if body.get('required') else 'optional')
    ser = (body or {}).get('serialize', 'json')
    for path in spec['url']['paths']:
        for m in path['methods']:
            k = (m, norm(path['path']))
            r = routes.setdefault(k, {'method': m, 'path': norm(path['path']), 'apis': [], 'spec_params': set(), 'parts': set(), 'body': brule, 'serialize': ser})
            r['apis'].append(name)
            r['spec_params'] |= set((spec.get('params') or {}).keys())
            r['parts'] |= set((path.get('parts') or {}).keys())
            if r['body'] != brule:
                r['body'] = 'optional'


def send(method, path, params, body):
    q = urllib.parse.urlencode(params)
    url = path + ('?' + q if q else '')
    headers = {}
    if COMPAT8:
        headers['Accept'] = 'application/vnd.elasticsearch+json;compatible-with=8'
    if body is not None:
        if COMPAT8:
            headers['Content-Type'] = 'application/vnd.elasticsearch+%s;compatible-with=8' % ('x-ndjson' if body.endswith('\n') else 'json')
        else:
            headers['Content-Type'] = 'application/x-ndjson' if body.endswith('\n') else 'application/json'
    for attempt in range(3):
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=60)
            c.request(method, url, body=body.encode() if body is not None else None, headers=headers)
            resp = c.getresponse()
            data = resp.read()
            c.close()
            try:
                j = json.loads(data) if data else None
            except Exception:
                j = data.decode('utf-8', 'replace')[:500]
            return resp.status, j
        except Exception as e:
            err = str(e)
    return 0, 'conn error ' + err


UNREC = re.compile(r'^request \[(.*?)\] contains unrecognized parameters?: (.*)$', re.S)
ITEM = re.compile(r'\[([^\]]+)\](?: -> did you mean (?:\[([^\]]+)\]|any of \[([^\]]*)\])\?)?')


def err_reason(j):
    if isinstance(j, dict) and isinstance(j.get('error'), dict):
        return j['error'].get('type'), j['error'].get('reason')
    if isinstance(j, dict) and isinstance(j.get('error'), str):
        return None, j['error']
    return None, None


def parse_unrec(j):
    t, r = err_reason(j)
    if t != 'illegal_argument_exception' or not r:
        return None
    m = UNREC.match(r)
    if not m:
        return None
    out = {}
    for it in ITEM.finditer(m.group(2)):
        sug = it.group(2) or it.group(3)
        out[it.group(1)] = [s.strip() for s in sug.split(',')] if sug else []
    return out


def body_for(r):
    if r['serialize'] == 'bulk':
        if any(a in ('bulk', 'monitoring.bulk') for a in r['apis']):
            return '{"index":{"_index":"x1"}}\n{"a":1}\n'
        if any('find_structure' in a or 'post_data' in a for a in r['apis']):
            return '{"a":1}\n{"a":2}\n'
        return '{}\n{}\n'
    return '{}'


def inst(path):
    return path.replace('*', 'x1')


def consumed(r, body, names):
    path = inst(r['path'])
    st, j = send(r['method'], path, {**{n: value_for(n) for n in names}, SENT: '1'}, body)
    u = parse_unrec(j)
    if u is not None and SENT in u:
        return set(names) - set(u)
    if len(names) == 1:
        return set(names)
    h = len(names) // 2
    return consumed(r, body, names[:h]) | consumed(r, body, names[h:])


def candidates(r, body, acc):
    path = inst(r['path'])
    probes = {p + 'Q': '1' for p in sorted(acc) if len(p) >= 2 and p + 'Q' not in types}
    probes[SENT] = '1'
    st, j = send(r['method'], path, probes, body)
    u = parse_unrec(j)
    if u is None:
        return None
    c = set()
    for v in u.values():
        c |= set(v)
    return sorted(c)


def probe_route(k):
    r = routes[k]
    m, path = r['method'], inst(r['path'])
    res = {'method': m, 'path': r['path'], 'apis': r['apis'], 'spec_body': r['body'], 'spec_params': sorted(r['spec_params']), 'parts': sorted(r['parts'])}
    if m == 'HEAD':
        # No body to read: a param is consumed iff ES does not answer 400 to it
        # alone (gen_routes.py adds back the GET route's params a bad probe
        # value made 400).
        st, _ = send(m, path, {SENT: '1'}, None)
        res['head_unknown_status'] = st
        res['acc_head'] = sorted(n for n in NAMES if send(m, path, {n: value_for(n)}, None)[0] != 400)
        return res
    # [R] probe
    st, j = send(m, path, {SENT: '1'}, None)
    res['r_probe'] = [st, *err_reason(j)]
    # [B] probe (may execute the action when ES consumes the body)
    if not any(a.startswith('license.') for a in r['apis']):
        st, j = send(m, path, {}, body_for(r))
        res['b_probe'] = [st, *err_reason(j)]
    for variant, body in (('nobody', None), ('body', body_for(r))):
        if variant == 'nobody' and r['body'] == 'required' and (res['r_probe'][2] or '').startswith('request body'):
            continue
        if variant == 'body' and r['body'] == 'none':
            continue
        st, j = send(m, path, {SENT: '1'}, body)
        u = parse_unrec(j)
        if u is None or SENT not in u:
            res['base_' + variant] = [st, *err_reason(j)]
            continue
        # `source` turns a body-less request into a "with body" one: probe it apart.
        acc = consumed(r, body, [n for n in NAMES if n not in ('source', 'source_content_type')])
        st, j = send(m, path, {'source': '{}', 'source_content_type': 'application/json', SENT: '1'}, body)
        u = parse_unrec(j)
        if u is None or SENT not in u:
            acc |= {'source', 'source_content_type'}
        else:
            acc |= {'source', 'source_content_type'} - set(u)
        # Bisection can split `q` from the params it enables: re-test them together.
        if 'q' in acc:
            for p in Q_FAMILY:
                if p not in acc:
                    u = parse_unrec(send(m, path, {'q': 'x1', p: value_for(p), SENT: '1'}, body)[1])
                    if u is not None and SENT in u and p not in u:
                        acc.add(p)
        res['acc_' + variant] = sorted(acc)
        # `q`-family params without `q` (RestActions.urlParamsToQueryBuilder).
        st, j = send(m, path, {**{p: value_for(p) for p in Q_FAMILY}, SENT: '1'}, body)
        t, reason = err_reason(j)
        if reason and "but missing query string parameter 'q'" in reason:
            res['qcheck_' + variant] = reason
        # Params consumed only when another param is present.
        cond = {}
        for p in sorted(acc):
            if p in Q_FAMILY and ('qcheck_' + variant) in res:
                continue
            if p in ('source', 'source_content_type'):
                continue
            st, j = send(m, path, {p: value_for(p), SENT: '1'}, body)
            u = parse_unrec(j)
            if u is not None and p in u:
                cond[p] = sorted(t for t in acc if t != p and p not in (parse_unrec(send(m, path, {p: value_for(p), t: value_for(t), SENT: '1'}, body)[1]) or {p: 1}))
        if cond:
            res['cond_' + variant] = cond
        res['cand_' + variant] = candidates(r, body, acc | set(r['spec_params']) | set(r['parts']))
    return res


keys = list(routes)
if len(sys.argv) > 3:
    # Probe only the non-spec routes ES registers (universe.json from universe.py).
    extra = [u for u in json.load(open(sys.argv[3])) if not u['origin'].startswith('spec') and (COMPAT8 or not u['compat']) and not u['security']]
    routes = collections.OrderedDict()
    for u in extra:
        sp = set()
        body = 'optional'
        for a in u['apis']:
            sp |= set((specs[a].get('params') or {}).keys())
            b = specs[a].get('body')
            body = 'none' if b is None else ('required' if b.get('required') else 'optional')
        routes[(u['method'], u['path'])] = {'method': u['method'], 'path': u['path'], 'apis': u['apis'] or ['?'], 'spec_params': sp,
                                            'parts': set(u['parts']), 'body': body, 'serialize': 'json'}
    keys = list(routes)
with ThreadPoolExecutor(12) as ex:
    results = list(ex.map(probe_route, keys))
json.dump(results, open(OUT, 'w'), indent=1)
print('done', len(results))
