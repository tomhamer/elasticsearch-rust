#!/usr/bin/env python3
"""Small deterministic POC vs actual ES comparison. Standard library only."""
import argparse
import collections
import hashlib
import http.client
import json
import math
import pathlib
import random
import statistics
import time

def request(conn, method, path, body=None, ndjson=False):
    payload = body if isinstance(body, (str, bytes)) else json.dumps(body) if body is not None else None
    conn.request(method, path, payload, {"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    r = conn.getresponse()
    raw = r.read()
    if r.status >= 300:
        raise RuntimeError(f"{method} {path}: {r.status} {raw[:1000]!r}")
    v = json.loads(raw)
    if v.get("errors") or v.get("_shards", {}).get("failed", 0):
        raise RuntimeError(f"failed bulk/shard: {str(v)[:1000]}")
    return v

def dsl(q):
    field = q["field"]
    if q["kind"] == "term":
        return {"term": {field: q["term"]}}
    if q["kind"] in ("and", "or"):
        return {"bool": {"must" if q["kind"] == "and" else "should": [{"term": {field: t}} for t in q["terms"]]}}
    if q["kind"] == "phrase":
        return {"match_phrase": {field: " ".join(q["terms"])}}
    if q["kind"] == "range":
        return {"range": {field: {"gte": q["lower"], "lte": q["upper"]}}}
    raise ValueError(q)

def percentile(xs, p):
    return sorted(xs)[max(0, math.ceil(len(xs) * p) - 1)]

def check(a, b, source):
    assert a["hits"]["total"] == b["hits"]["total"], (a["hits"]["total"], b["hits"]["total"])
    aa, bb = a["hits"]["hits"], b["hits"]["hits"]
    assert len(aa) == len(bb)
    # Scores and tied identities may differ across Lucene/Rustic; check scores
    # in rank order and membership whenever the boundary is not tied.
    for x, y in zip(aa, bb):
        assert math.isclose(x["_score"], y["_score"], rel_tol=2e-5, abs_tol=2e-5), (x, y)
    if aa and len({x["_score"] for x in aa}) == len(aa):
        assert [x["_id"] for x in aa] == [x["_id"] for x in bb]
    for x in aa:
        assert ("_source" in x) == source
    return len(aa)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--es-port", type=int, default=19211)
    ap.add_argument("--poc-port", type=int, default=19212)
    ap.add_argument("--our-port", type=int, help="optional small full-rewrite fidelity comparison")
    ap.add_argument("--before-port", type=int, help="optional unchanged full rewrite for matched before/after timing")
    ap.add_argument("--engine-port", type=int, help="optional unmodified engine/HTTP oracle")
    ap.add_argument("--fused-port", type=int, help="optional fused engine adapter")
    ap.add_argument("--load-our-only", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--no-force-merge", action="store_true", help="validate ordinary bulk + refresh lifecycle")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--warmup-rounds", type=int, default=20)
    ap.add_argument("--out", default="path-poc-results.json")
    args = ap.parse_args()
    es = http.client.HTTPConnection("127.0.0.1", args.es_port, timeout=120)
    poc = http.client.HTTPConnection("127.0.0.1", args.poc_port, timeout=120)
    our = http.client.HTTPConnection("127.0.0.1", args.our_port, timeout=120) if args.our_port else None
    audits = {name: http.client.HTTPConnection("127.0.0.1", port, timeout=120) for name, port in (("engine", args.engine_port), ("fused", args.fused_port), ("full_before", args.before_port)) if port}
    if our:
        audits["full"] = our
    version = request(es, "GET", "/")["version"]["number"]
    for load_conn in ([es] if args.load else []) + ([our] if our and (args.load or args.load_our_only) else []):
        # Dedicated disposable index; never touch existing benchmark datasets.
        load_conn.request("DELETE", "/path-poc"); load_conn.getresponse().read()
        request(load_conn, "PUT", "/path-poc", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1", "index.queries.cache.enabled": False}, "mappings": {"dynamic": "strict", "properties": {"id": {"type": "long"}, "title": {"type": "text"}, "body": {"type": "text"}, "price": {"type": "double"}, "ts": {"type": "long"}, "tag": {"type": "keyword"}}}})
        batch = []
        for line in open(args.corpus):
            d = json.loads(line)
            batch += [json.dumps({"index": {"_index": "path-poc", "_id": str(d["id"])}}), line.strip()]
            if len(batch) >= 1000:
                response = request(load_conn, "POST", "/_bulk", "\n".join(batch) + "\n", True)
                assert not response["errors"], response
                batch = []
        if batch:
            response = request(load_conn, "POST", "/_bulk", "\n".join(batch) + "\n", True)
            assert not response["errors"], response
        request(load_conn, "POST", "/path-poc/_refresh")
        if not args.no_force_merge:
            request(load_conn, "POST", "/path-poc/_forcemerge?max_num_segments=1")
    qs = []
    corpus_docs = {str(d["id"]): d for d in map(json.loads, open(args.corpus))}
    counts = collections.Counter()
    for line in open(args.queries):
        q = json.loads(line)
        if counts[q["class"]] < args.per_class:
            qs.append(q); counts[q["class"]] += 1
    samples = collections.defaultdict(lambda: collections.defaultdict(list))
    verified = 0
    for source in (False, True):
        for q in qs:
            body = {"query": dsl(q), "size": 10, "_source": source, "track_total_hits": True}
            a = request(poc, "POST", "/baseline/_search", body)
            b = request(poc, "POST", "/candidate/_search", body)
            c = request(es, "POST", "/path-poc/_search?request_cache=false", body)
            check(a, c, source); check(b, c, source)
            assert a["hits"] == b["hits"], "candidate changed result"
            audit_responses = []
            for conn in audits.values():
                response = request(conn, "POST", "/path-poc/_search?request_cache=false", body)
                check(response, c, source)
                check(a, response, source)
                audit_responses.append(response)
            if source:
                for response in [a, b, c] + audit_responses:
                    for hit in response["hits"]["hits"]:
                        assert hit["_source"] == corpus_docs[hit["_id"]], "incorrect fetched source"
            verified += 1
        # Rotate the order of systems on each round; warm all paths first.
        rng = random.Random(42)
        for r in range(args.rounds + args.warmup_rounds):
            for q in qs:
                modes = ["es", "baseline", "naive", "candidate", "core"]
                modes.extend(audits)
                rng.shuffle(modes)
                body = {"query": dsl(q), "size": 10, "_source": source, "track_total_hits": True}
                for mode in modes:
                    t = time.perf_counter_ns()
                    conn = es if mode == "es" else audits[mode] if mode in audits else poc
                    path = "/path-poc/_search?request_cache=false" if mode == "es" or mode in audits else f"/{mode}/_search"
                    v = request(conn, "POST", path, body)
                    wall = (time.perf_counter_ns() - t) / 1000
                    if r >= args.warmup_rounds:
                        key = f'{q["class"]}/source={source}/{mode}'
                        samples[key]["wall_us"].append(wall)
                        for k, val in v.get("_poc", {}).items():
                            if k.endswith("_us"):
                                samples[key][k].append(val)
    stats = {key: {metric: {"n": len(xs), "p50": statistics.median(xs), "p95": percentile(xs, .95)} for metric, xs in metrics.items()} for key, metrics in samples.items()}
    result = {"es_version": version, "verified_requests": verified, "score_parity": True, "score_tolerance": {"relative": 2e-5, "absolute": 2e-5}, "corpus_sha256": hashlib.sha256(pathlib.Path(args.corpus).read_bytes()).hexdigest(), "queries_sha256": hashlib.sha256(pathlib.Path(args.queries).read_bytes()).hexdigest(), "args": vars(args), "stats": stats}
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for source in (False, True):
        for cls in counts:
            vals = [stats[f"{cls}/source={source}/{m}"]["wall_us"]["p50"] for m in ("es", "baseline", "candidate", "core")]
            print(f'{cls:8} source={str(source):5} ES/baseline/candidate/core us: ' + " / ".join(f"{v:.1f}" for v in vals), flush=True)
            for mode in audits:
                print(f'         {mode}: {stats[f"{cls}/source={source}/{mode}"]["wall_us"]["p50"]:.1f} us', flush=True)
    print(f"verified {verified} requests; {args.out}")

if __name__ == "__main__":
    main()
