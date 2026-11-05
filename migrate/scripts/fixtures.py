#!/usr/bin/env python3
"""Populate a real Elasticsearch 8.x with fixtures that exercise every format
path the importer reads, then save what ES itself returns as ground truth.

usage: fixtures.py <es-url> <expected.json>
"""
import json, random, sys, urllib.request

ES = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
random.seed(42)


def req(method, path, body=None, ndjson=False, ok=(200, 201)):
    data = None
    headers = {}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        headers["Content-Type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(ES + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        txt = e.read().decode()
        if e.code in ok:
            return json.loads(txt)
        raise SystemExit(f"{method} {path} -> {e.code}: {txt[:500]}")


def bulk(lines):
    body = "".join(json.dumps(l) + "\n" for l in lines)
    r = req("POST", "/_bulk", body, ndjson=True)
    if r["errors"]:
        bad = [i for i in r["items"] if list(i.values())[0].get("error")]
        raise SystemExit(f"bulk errors: {json.dumps(bad[:3])}")


def words(n):
    vocab = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
    return " ".join(random.choice(vocab) for _ in range(n))


# --- cluster metadata
req("PUT", "/_ingest/pipeline/add-tag", {"description": "adds a tag", "processors": [{"set": {"field": "tag", "value": "piped"}}, {"rename": {"field": "raw", "target_field": "renamed", "ignore_missing": True}}]})
req("PUT", "/_component_template/ct-common", {"template": {"mappings": {"properties": {"@timestamp": {"type": "date"}, "msg": {"type": "text"}}}}})
req("PUT", "/_index_template/events", {"index_patterns": ["events-*"], "data_stream": {}, "composed_of": ["ct-common"], "priority": 500})
req("PUT", "/_index_template/tpl-plain", {"index_patterns": ["tpl-*"], "template": {"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"k": {"type": "keyword"}}}}})
req("PUT", "/_scripts/add-one", {"script": {"lang": "painless", "source": "ctx._source.n += 1"}})
req("PUT", "/_ilm/policy/keep-30d", {"policy": {"phases": {"hot": {"actions": {}}, "delete": {"min_age": "30d", "actions": {"delete": {}}}}}})

# --- basic: numeric ids, several refreshes (segments), updates, deletes, routing
req("PUT", "/basic", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"n": {"type": "long"}, "body": {"type": "text"}, "k": {"type": "keyword"}}}, "aliases": {"basic-alias": {}, "basic-filtered": {"filter": {"term": {"k": "a"}}}}})
for batch in range(5):
    lines = []
    for i in range(batch * 600, (batch + 1) * 600):
        lines += [{"index": {"_index": "basic", "_id": str(i + 1)}}, {"n": i, "body": words(12), "k": random.choice("abc"), "f": i / 7.0}]
    bulk(lines)
    req("POST", "/basic/_refresh")
bulk(sum(([{"index": {"_index": "basic", "_id": str(i)}}, {"n": -i, "updated": True}] for i in range(1, 400, 2)), []))
req("POST", "/basic/_refresh")
bulk([{"delete": {"_index": "basic", "_id": str(i)}} for i in range(1000, 1100)])
req("POST", "/basic/_refresh")
bulk(sum(([{"index": {"_index": "basic", "_id": f"r{i}", "routing": f"user{i % 7}"}}, {"n": i, "routed": True}] for i in range(50)), []))

# --- ids: utf8, base64-looking, leading zero, auto ids, long, unicode source
req("PUT", "/ids", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
ids = ["héllo wörld", "007", "0", "12345678901234567890123", "kHyq3pIBAAAAhqtjXAKH", "AAAA", "a/b?c", "emoji-🦀", "-1", "1e5", " spaced "]
bulk(sum(([{"index": {"_index": "ids", "_id": i}}, {"id": i, "text": "naïve café ✓ 日本語"}] for i in ids), []))
for i in range(30):
    req("POST", "/ids/_doc", {"auto": i})

# --- best_compression codec, bigger docs
req("PUT", "/bc", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.codec": "best_compression"}})
for batch in range(3):
    bulk(sum(([{"index": {"_index": "bc", "_id": f"bc{batch}-{i}"}}, {"body": words(200), "n": i}] for i in range(500)), []))
    req("POST", "/bc/_refresh")

# --- nested docs (extra lucene docs without _id)
req("PUT", "/nested", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"comments": {"type": "nested", "properties": {"who": {"type": "keyword"}, "c": {"type": "text"}}}}}})
bulk(sum(([{"index": {"_index": "nested", "_id": str(i)}}, {"title": words(3), "comments": [{"who": f"u{j}", "c": words(4)} for j in range(i % 5)]}] for i in range(300)), []))

# --- large docs (sliced chunks in stored fields)
req("PUT", "/big", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {"properties": {"blob": {"type": "text", "index": False}}}})
for i in range(4):
    req("PUT", f"/big/_doc/{i}", {"blob": words(60000 * (i + 1)), "i": i})

# --- churn then force merge: soft deletes live inside the merged segment
req("PUT", "/merged", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
for rnd in range(4):
    bulk(sum(([{"index": {"_index": "merged", "_id": str(i)}}, {"v": rnd, "i": i}] for i in range(rnd * 100, 1000)), []))
    req("POST", "/merged/_refresh")
bulk([{"delete": {"_index": "merged", "_id": str(i)}} for i in range(0, 1000, 10)])
req("POST", "/merged/_forcemerge?max_num_segments=1")

# --- multi-shard + routing
req("PUT", "/sharded", {"settings": {"number_of_shards": 3, "number_of_replicas": 0}})
bulk(sum(([{"index": {"_index": "sharded", "_id": str(i), **({"routing": "x"} if i % 3 == 0 else {})}}, {"i": i}] for i in range(900)), []))

# --- default pipeline: stored _source is post-pipeline; must not be re-run
req("PUT", "/piped", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.default_pipeline": "add-tag"}})
bulk(sum(([{"index": {"_index": "piped", "_id": str(i)}}, {"raw": i}] for i in range(100)), []))

# --- closed index, template-created index
req("PUT", "/closed", {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
bulk(sum(([{"index": {"_index": "closed", "_id": str(i)}}, {"i": i}] for i in range(20)), []))
req("PUT", "/tpl-1/_doc/1?refresh=true", {"k": "v"})

# --- data stream with a rollover
for i in range(200):
    if i == 100:
        req("POST", "/events-app/_rollover")
    req("POST", "/events-app/_doc", {"@timestamp": f"2026-10-04T10:{i // 60:02d}:{i % 60:02d}Z", "msg": words(5), "i": i})

# --- synthetic source: importer must refuse it loudly
req("PUT", "/synthetic", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.mapping.source.mode": "synthetic"}}, ok=(200, 400))
req("PUT", "/synthetic/_doc/1", {"a": 1}, ok=(200, 201, 404))

req("POST", "/_refresh")

# --- ground truth from ES itself
expected = {}
for target in ["basic", "ids", "bc", "nested", "big", "merged", "sharded", "piped", "closed", "tpl-1", "events-app"]:
    docs = {}
    r = req("POST", f"/{target}/_search?scroll=1m&size=500", {"query": {"match_all": {}}, "sort": ["_doc"]})
    while r["hits"]["hits"]:
        for h in r["hits"]["hits"]:
            docs[h["_id"]] = {"routing": h.get("_routing"), "source": h["_source"]}
        r = req("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": r["_scroll_id"]})
    expected[target] = docs
req("POST", "/closed/_close")
json.dump(expected, open(OUT, "w"))
print({k: len(v) for k, v in expected.items()})
