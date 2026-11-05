#!/usr/bin/env python3
"""Generate `src/spec_routes.rs` (the REST route/validation table) from `routes.json`.

`routes.json` is the committed, trimmed route table, versioned by the
request's REST API version (native v9, or v8 for requests sent with
`compatible-with=8`). It is rebuilt from three inputs when the target ES
version changes:

  * the REST spec (`~/xp-<V>/api/*.json`),
  * the routes ES actually registers (javap of its route-declaring classes:
    adds non-spec routes like `GET /_mappings`, and flags the routes ES
    registers only for the previous REST API version, `RestApiVersion.V_8`,
    e.g. `/{index}/_knn_search`: they dispatch only compat-8 requests and
    otherwise only widen 405 `Allow` sets),
  * probes of a reference ES (probe.py), once natively and once with the
    compat-8 headers (COMPAT8=1): per route the accepted query params and
    "did you mean" candidates with and without a body, and the exact
    body-required / body-not-supported behaviour.

Each route records `v8_only` and the native rules (for a v8-only route: its
v8 rules); `v8` holds the compat-8 rules where they differ from the native
ones (e.g. `?local` on `GET /_alias`, consumed only under compat-8).

Regenerating (reference ES 9.5.5 started with security off, trial license,
`-Ehttp.max_initial_line_length=64kb`, on $ES_PORT, default 9299; on the ES9
box; S=crates/elasticsearch-http/spec, API=~/xp-9.5.5/api):
  javap_routes.py ~/ref/elasticsearch-9.5.5 cls javap.txt   # javap of every class declaring routes
  routes_from_javap.py javap.txt es_routes.json             # V_8 routes -> compat
  universe.py $API es_routes.json none universe.json
  verify405.py universe.json          # model of ES PathTrie vs ES OPTIONS answers: 0 mismatches
  probe.py $API probe9.json
  COMPAT8=1 probe.py $API probe8.json
  probe.py $API probe9_extra.json universe.json
  COMPAT8=1 probe.py $API probe8_extra.json universe.json
  gen_routes.py build $API universe.json probe9.json probe9_extra.json probe8.json probe8_extra.json
  gen_routes.py rust
  compare.py <es port> <rustic port> [--compat8]   # validation answers, rustic vs ES, every route
    # <rustic port>: an in-process `elasticsearch_http::serve_listener` server
    # (as es-yaml-runner starts it; a few-line scratch crate). The `elasticsearch`
    # binary's cluster front answers some routes (_cat, _internal, ...) itself.

usage:
  gen_routes.py build <api dir> <universe.json> <probe9.json> <probe9_extra.json> <probe8.json> <probe8_extra.json>  # -> routes.json
  gen_routes.py rust                                                                                                 # routes.json -> ../src/spec_routes.rs
"""
import json, glob, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES_JSON = os.path.join(HERE, 'routes.json')
OUT_RS = os.path.join(HERE, '..', 'src', 'spec_routes.rs')

STD_R = {'request body is required': 'required', 'request body or source parameter is required': 'required_or_source'}
# Routes whose handler accepts a parameter in place of the body (the probe,
# sending neither, sees "request body is required"): the handler checks it.
BODY_OPTIONAL = {('POST', '/_ml/trained_models/*/deployment/_update')}
# Response params every handler consumes (AbstractRestChannel + RestController).
FALLBACK_COMMON = {'pretty', 'human', 'error_trace', 'filter_path', 'format'}
RULE_KEYS = ('body', 'probed', 'qcheck', 'q_only', 'accepted', 'candidates')
SET_KEYS = ('accepted', 'candidates', 'q_only')


def load_probes(*paths):
    probes = {}
    for p in paths:
        for r in json.load(open(p)):
            probes[(r['method'], r['path'])] = r
    return probes


def rules_for(u, spec_params, spec_body, pr):
    """The validation rules of one route from one probe run."""
    key = (u['method'], u['path'])
    rp = pr.get('r_probe')
    bp = pr.get('b_probe')
    probed_ok = 'acc_nobody' in pr or 'acc_body' in pr
    if rp and rp[2] in STD_R:
        body = STD_R[rp[2]]
    elif rp and rp[0] == 400 and rp[1] and 'body' in (rp[2] or '').lower() and 'unrecognized parameter' not in (rp[2] or ''):
        # The handler rejects a missing body with its own error.
        body = 'require:%s:%s' % (rp[1], rp[2])
    elif spec_body == 'none':
        if bp and bp[2] and 'does not support having a body' in bp[2]:
            body = 'none'
        elif bp and probed_ok and bp[1] == 'illegal_argument_exception' and 'body' in (bp[2] or ''):
            body = 'reject:' + bp[2]
        elif bp and probed_ok:
            body = 'optional'  # ES consumes the body despite the spec
        else:
            body = 'none'
    elif spec_body == 'required' and not rp:
        body = 'required'  # not probeable (security): ES handlers use requiredContent()
    else:
        body = 'optional'
    fallback = spec_params | FALLBACK_COMMON | set(u['parts'])
    if body != 'none':
        fallback.add('source')
    acc_nb = pr.get('acc_nobody')
    acc_b = pr.get('acc_body')
    if acc_nb is None and acc_b is None:
        acc_nb = acc_b = sorted(fallback)
    elif acc_nb is None:
        acc_nb = sorted(set(acc_b) | ({'source'} if body in ('required_or_source', 'optional') else set()))
    elif acc_b is None:
        acc_b = sorted(set(acc_nb) - {'source'})
    cand_nb = pr.get('cand_nobody') if pr.get('acc_nobody') is not None else None
    cand_b = pr.get('cand_body') if pr.get('acc_body') is not None else None
    q_only = []
    for v in ('nobody', 'body'):
        cond = pr.get('cond_' + v) or {}
        bad = {p: t for p, t in cond.items() if t != ['q']}
        if bad:
            raise SystemExit('unsupported conditional params %s %s: %s' % (key, v, bad))
        q_only.append(sorted(cond))
    return {
        'body': body, 'probed': probed_ok,
        'qcheck': ['qcheck_nobody' in pr, 'qcheck_body' in pr],
        'q_only': q_only,
        'accepted': [sorted(acc_nb), sorted(acc_b)],
        'candidates': [sorted(cand_nb if cand_nb is not None else acc_nb), sorted(cand_b if cand_b is not None else acc_b)],
    }


def head_rules(r, acc_head, get_acc):
    acc = sorted(set(acc_head) | set(get_acc))
    return {'probed': True, 'accepted': [acc, acc], 'candidates': [acc, acc]}


def build(api_dir, universe_path, p9, p9x, p8, p8x):
    specs = {}
    for f in sorted(glob.glob(os.path.join(api_dir, '*.json'))):
        if f.endswith('_common.json'):
            continue
        name, spec = next(iter(json.load(open(f)).items()))
        specs[name] = spec
    probes9 = load_probes(p9, p9x)
    probes8 = load_probes(p8, p8x)
    out = []
    for u in json.load(open(universe_path)):
        key = (u['method'], u['path'])
        spec_params, spec_body = set(), None
        for a in u['apis']:
            spec_params |= set((specs[a].get('params') or {}).keys())
            b = specs[a].get('body')
            rule = 'none' if b is None else ('required' if b.get('required') else 'optional')
            spec_body = rule if spec_body in (None, rule) else 'optional'
        spec_body = spec_body or 'optional'
        r8 = rules_for(u, spec_params, spec_body, probes8.get(key, {}))
        # A v8-only route never dispatches a native request: its rules are the v8 ones.
        r9 = r8 if u['compat'] else rules_for(u, spec_params, spec_body, probes9.get(key, {}))
        rec = {'method': u['method'], 'path': u['path'], 'api': (u['apis'] or ['?'])[0], 'v8_only': u['compat'], 'security': u['security']}
        rec.update(r9)
        if any(r8[k] != r9[k] for k in RULE_KEYS if k != 'probed'):
            rec['v8'] = r8
        out.append(rec)
    # HEAD routes (probed by status only) share their handler with the GET
    # route of the same path: add the params a bad probe value made 400.
    by_key = {(r['method'], r['path']): r for r in out}
    for r in out:
        g = by_key.get(('GET', r['path']))
        if r['method'] != 'HEAD' or g is None:
            continue
        p9 = probes9.get(('HEAD', r['path']), {})
        p8 = probes8.get(('HEAD', r['path']), {})
        if 'acc_head' not in p9 or 'acc_head' not in p8:
            continue
        g8 = g.get('v8', g)
        r9 = head_rules(r, p9['acc_head'], g['accepted'][0])
        r8 = head_rules(r, p8['acc_head'], g8['accepted'][0])
        r.update(r9)
        r.pop('v8', None)
        if r8 != r9:
            r['v8'] = dict({k: r[k] for k in RULE_KEYS}, **r8)
    write_routes(out)
    c = collections.Counter(r['body'].split(':')[0] for r in out)
    print(len(out), 'routes;', dict(c), 'probed', sum(r['probed'] for r in out), 'v8-only', sum(r['v8_only'] for r in out),
          'v8 rules differ', sum('v8' in r for r in out))


def write_routes(routes):
    """Compact form: deduplicated param sets, one route per line."""
    sets, idx = [], {}

    def sid(names):
        k = tuple(sorted(set(names)))
        if k not in idx:
            idx[k] = len(sets)
            sets.append(list(k))
        return idx[k]

    def compact(r):
        r = dict(r)
        for k in SET_KEYS:
            r[k] = [sid(x) for x in r[k]]
        return r

    rows = []
    for r in routes:
        r = compact(r)
        if 'v8' in r:
            r['v8'] = compact(r['v8'])
        rows.append(json.dumps(r, sort_keys=True))
    with open(ROUTES_JSON, 'w') as f:
        f.write('{"sets": [\n' + ',\n'.join(json.dumps(s) for s in sets) + '\n],\n"routes": [\n' + ',\n'.join(rows) + '\n]}\n')


def read_routes():
    d = json.load(open(ROUTES_JSON))

    def expand(r):
        for k in SET_KEYS:
            r[k] = [d['sets'][i] for i in r[k]]

    for r in d['routes']:
        expand(r)
        if 'v8' in r:
            expand(r['v8'])
    return d['routes']


def rust():
    routes = read_routes()
    sets, set_idx = [], {}

    def sid(names):
        k = tuple(sorted(set(names)))
        if k not in set_idx:
            set_idx[k] = len(sets)
            sets.append(k)
        return set_idx[k]

    def rules(r):
        acc = [sid(r['accepted'][0]), sid(r['accepted'][1])]
        cand = [sid(r['candidates'][0]), sid(r['candidates'][1])]
        q_only = [sid(r['q_only'][0]), sid(r['q_only'][1])]
        b = r['body']
        if (r.get('method'), r.get('path')) in BODY_OPTIONAL:
            b = 'optional'
        if b.startswith('reject:'):
            body = 'Body::Reject(%s)' % json.dumps(b[len('reject:'):])
        elif b.startswith('require:'):
            _, kind, msg = b.split(':', 2)
            body = 'Body::RequiredWith(%s, %s)' % (json.dumps(kind), json.dumps(msg))
        else:
            body = {'none': 'Body::None', 'optional': 'Body::Optional', 'required': 'Body::Required', 'required_or_source': 'Body::RequiredOrSource'}[b]
        return 'Rules { body: %s, accepted: [%d, %d], candidates: [%d, %d], q_only: [%d, %d], q_check: [%s, %s] }' % (
            body, acc[0], acc[1], cand[0], cand[1], q_only[0], q_only[1], str(r['qcheck'][0]).lower(), str(r['qcheck'][1]).lower())

    lines, v8_lines = [], []
    for r in routes:
        if 'v8' in r:
            v8 = str(len(v8_lines))
            v8_lines.append('    %s, // %s %s' % (rules(r['v8']), r['method'], r['path']))
        else:
            v8 = 'SAME'
        lines.append('    RouteDef { method: %s, path: %s, api: %s, v8_only: %s, security: %s, v8: %s, rules: %s },' % (
            json.dumps(r['method']), json.dumps(r['path']), json.dumps(r['api']), str(r['v8_only']).lower(), str(r['security']).lower(), v8, rules(r)))
    src = ['// @generated by spec/gen_routes.py from spec/routes.json. Do not edit by hand.',
           '//! Route table for the request-validation prologue (see `spec_validate`).',
           '',
           'use crate::spec_validate::{Body, RouteDef, Rules, SAME};',
           '',
           '/// Sorted parameter-name sets, referenced by index from [`ROUTES`].',
           'pub(crate) static SETS: &[&[&str]] = &[']
    for s in sets:
        src.append('    &[%s],' % ', '.join(json.dumps(x) for x in s))
    src.append('];')
    src.append('')
    src.append('/// REST-API-v8 (`compatible-with=8`) rules of the routes whose v8 rules differ (`RouteDef::v8`).')
    src.append('pub(crate) static V8_RULES: &[Rules] = &[')
    src.extend(v8_lines)
    src.append('];')
    src.append('')
    src.append('pub(crate) static ROUTES: &[RouteDef] = &[')
    src.extend(lines)
    src.append('];')
    open(OUT_RS, 'w').write('\n'.join(src) + '\n')
    print('wrote', OUT_RS, len(routes), 'routes', len(sets), 'sets', len(v8_lines), 'v8 rules')


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(*sys.argv[2:8])
    else:
        rust()
