#!/usr/bin/env python3
"""Extract (method, path, compat_only, class) route tuples from `javap -c -p -constants` output.

`compat` marks a route registered only for the previous REST API version
(`Route.builder(..).deprecatedForRemoval(msg, RestApiVersion.V_<previous>)`
or a `.replaces(..)` target): in ES 9.x that is `RestApiVersion.V_8`.

usage: routes_from_javap.py <javap.txt> <out.json> [previous major, default 8]
"""
import re, sys, json

PREV = 'RestApiVersion.V_%s' % (sys.argv[3] if len(sys.argv) > 3 else '8')


def route_path(s):
    """Route paths: `/x/{y}`, or relative ones like `{index}/_knn_search`."""
    if s.startswith('/'):
        return s
    if s[:1] in '{_' and '/' in s and not re.search(r'\s', s):
        return '/' + s
    return None


cls = None
in_routes = False
routes = []
pend_method = None
pend_path = None
builder = None  # dict(method, path, compat, replaced)
METH = re.compile(r'Field org/elasticsearch/rest/RestRequest\$Method\.(\w+):')
LDC = re.compile(r'// String (.*)$')
for line in open(sys.argv[1]):
    line = line.rstrip('\n')
    m = re.match(r'^(?:public |final |abstract |class )*.*class (\S+)', line)
    if m and ('class ' in line) and not line.startswith(' '):
        cls = m.group(1)
        continue
    if not line.startswith(' '):
        continue
    if re.match(r'^  \S.*\(.*\).*;$', line) or re.match(r'^  static \{\};$', line) or re.match(r'^  \S.*;$', line):
        # new member: any method may build routes (helpers, static init)
        in_routes = True
        pend_method = pend_path = None
        builder = None
        continue
    if not in_routes:
        continue
    mm = METH.search(line)
    if mm:
        if builder is not None and builder.get('path') and 'build' not in line:
            builder['rmethod'] = mm.group(1)
        else:
            pend_method = mm.group(1)
        continue
    ml = LDC.search(line)
    if ml and 'ldc' in line:
        s = route_path(ml.group(1))
        if s:
            if builder is not None and 'rmethod' in builder:
                builder['rpath'] = s
            else:
                pend_path = s
        continue
    if 'RestHandler$Route."<init>"' in line:
        if pend_method and pend_path:
            routes.append({'method': pend_method, 'path': pend_path, 'compat': False, 'cls': cls})
        pend_method = pend_path = None
        continue
    if 'RestHandler$Route.builder' in line:
        builder = {'method': pend_method, 'path': pend_path, 'compat': False}
        pend_method = pend_path = None
        continue
    if builder is not None and PREV in line:
        if 'rpath' in builder:
            builder['rcompat'] = True
        else:
            builder['compat'] = True
        continue
    if builder is not None and 'RouteBuilder.build' in line:
        if builder['method'] and builder['path']:
            routes.append({'method': builder['method'], 'path': builder['path'], 'compat': builder['compat'], 'cls': cls})
        if builder.get('rpath') and builder.get('rmethod'):
            routes.append({'method': builder['rmethod'], 'path': builder['rpath'], 'compat': True, 'cls': cls})
        builder = None
        continue

json.dump(routes, open(sys.argv[2], 'w'), indent=0)
print(len(routes), 'routes', sum(r['compat'] for r in routes), 'compat')
