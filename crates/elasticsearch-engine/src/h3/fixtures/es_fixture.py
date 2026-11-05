"""geohex fixtures from a running ES 9.5.5: geohex_grid buckets and
geo_grid geohex filter hits over points (on cell edges and vertices, near
pentagons and the poles, random) and shapes, for geo_point and geo_shape.

python3 es_fixture.py points.txt http://127.0.0.1:9299 out.json
"""
import json, struct, sys, urllib.request, random

pts_file, es, out = sys.argv[1], sys.argv[2], sys.argv[3]


def f(hexbits):
    return struct.unpack('>d', bytes.fromhex(hexbits.rjust(16, '0')))[0]


def call(method, path, body=None, nd=False):
    data = None
    if body is not None:
        data = (body if nd else json.dumps(body)).encode()
    req = urllib.request.Request(es + path, data=data, method=method, headers={'Content-Type': 'application/x-ndjson' if nd else 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(req).read() or b'{}')
    except urllib.error.HTTPError as e:
        raise SystemExit(f'{method} {path}: {e.read().decode()}')


lines = open(pts_file).read().split('\n')
lines = [l for l in lines if l]
# The random points first in the file, then the edge/vertex/pentagon/pole ones.
rnd = random.Random(7)
chosen = lines[:600] + [l for l in lines[1200:] if rnd.random() < 0.55]
points = []
for l in chosen:
    v = l.split(' ')
    lat, lon = f(v[0]), f(v[1])
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        points.append([lat, lon])

# Shapes: small polygons and lines across cell edges (from the edge points),
# and a few big ones (dateline, pole caps).
shapes = []
for i in range(0, min(len(points), 2400), 12):
    lat, lon = points[i]
    d = [0.5, 0.05, 0.002, 0.0001][i % 4]
    if abs(lat) > 85 or abs(lon) > 179:
        continue
    if i % 24 == 0:
        shapes.append({'type': 'LineString', 'coordinates': [[lon - d, lat - d], [lon + d, lat + d * 0.5]]})
    else:
        shapes.append({'type': 'Polygon', 'coordinates': [[[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]]})
shapes += [
    {'type': 'Polygon', 'coordinates': [[[170, -10], [179.9, -10], [179.9, 10], [170, 10], [170, -10]]]},
    {'type': 'Polygon', 'coordinates': [[[-180, 80], [180, 80], [180, 89], [-180, 89], [-180, 80]]]},
    {'type': 'Polygon', 'coordinates': [[[-30, -60], [30, -60], [30, -40], [-30, -40], [-30, -60]]]},
    {'type': 'LineString', 'coordinates': [[-170, 5], [170, 5]]},
    {'type': 'Point', 'coordinates': [0.0, 90.0]},
    {'type': 'Point', 'coordinates': [12.5, -90.0]},
]

try:
    urllib.request.urlopen(urllib.request.Request(es + '/h3parity', method='DELETE'))
except urllib.error.HTTPError:
    pass
call('PUT', '/h3parity', {'settings': {'number_of_shards': 1, 'number_of_replicas': 0}, 'mappings': {'properties': {'p': {'type': 'geo_point'}, 's': {'type': 'geo_shape'}, 'k': {'type': 'keyword'}}}})
docs = []
for i, (lat, lon) in enumerate(points):
    docs.append({'k': 'p', 'p': [lon, lat], 's': {'type': 'Point', 'coordinates': [lon, lat]}})
for s in shapes:
    docs.append({'k': 's', 's': s})
for start in range(0, len(docs), 1000):
    body = ''.join(json.dumps({'index': {'_id': str(start + j)}}) + '\n' + json.dumps(d) + '\n' for j, d in enumerate(docs[start:start + 1000]))
    r = call('POST', '/h3parity/_bulk?refresh=true', body, nd=True)
    assert not r.get('errors'), [it for it in r['items'] if 'error' in it['index']][:3]
call('POST', '/h3parity/_forcemerge?max_num_segments=1')

precisions = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
aggs = {}
for field in ['p', 's']:
    for prec in precisions:
        if field == 's' and prec > 5:
            continue
        q = {'size': 0, 'aggs': {'g': {'geohex_grid': {'field': field, 'precision': prec, 'size': 100000}}}}
        r = call('POST', '/h3parity/_search', q)
        aggs[f'{field}/{prec}'] = {b['key']: b['doc_count'] for b in r['aggregations']['g']['buckets']}

# Filter cells: each precision's busiest cells and some cells of edge points.
cells = set()
for key, b in aggs.items():
    field, prec = key.split('/')
    for c in sorted(b, key=lambda c: (-b[c], c))[:8]:
        cells.add(c)
for prec in [3, 7, 11, 15]:
    for c in list(aggs.get(f'p/{min(precisions, key=lambda x: abs(x - prec))}', {}))[:6]:
        cells.add(c)
filters = {}
for field in ['p', 's']:
    for c in sorted(cells):
        q = {'size': 10000, '_source': False, 'query': {'geo_grid': {field: {'geohex': c}}}, 'sort': [{'_doc': 'asc'}]}
        r = call('POST', '/h3parity/_search', q)
        filters[f'{field}/{c}'] = sorted(int(h['_id']) for h in r['hits']['hits'])

json.dump({'docs': docs, 'aggs': aggs, 'filters': filters}, open(out, 'w'), separators=(',', ':'))
print(len(docs), 'docs', len(aggs), 'aggs', len(filters), 'filters')
