#!/usr/bin/env python3
"""Amortize transport/scheduling cost; report separately from single searches."""
import argparse
import collections
import hashlib
import http.client
import json
import pathlib
import random
import statistics
import time
from compare import check, dsl, percentile, request

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--es-port", type=int, default=19211)
ap.add_argument("--poc-port", type=int, default=19212)
ap.add_argument("--our-port", type=int)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--out", required=True)
a = ap.parse_args()
es = http.client.HTTPConnection("127.0.0.1", a.es_port, timeout=120)
poc = http.client.HTTPConnection("127.0.0.1", a.poc_port, timeout=120)
our = http.client.HTTPConnection("127.0.0.1", a.our_port, timeout=120) if a.our_port else None
classes = collections.defaultdict(list)
docs = {str(d["id"]): d for d in map(json.loads, open(a.corpus))}
for q in map(json.loads, open(a.queries)):
    if len(classes[q["class"]]) < 12:
        classes[q["class"]].append(q)
samples = collections.defaultdict(list)
verified = 0
rng = random.Random(42)
for source in (False, True):
    for cls, queries in classes.items():
        bodies = [{"query": dsl(queries[i % len(queries)]), "size": 10, "_source": source, "track_total_hits": True} for i in range(a.batch_size)]
        payload = "".join(json.dumps({"request_cache": False}) + "\n" + json.dumps(b) + "\n" for b in bodies)
        baseline = request(poc, "POST", "/baseline/_msearch", payload, True)["responses"]
        reference = request(es, "POST", "/path-poc/_msearch?max_concurrent_searches=1", payload, True)["responses"]
        candidate = request(poc, "POST", "/candidate/_msearch", payload, True)["responses"]
        other = request(our, "POST", "/path-poc/_msearch?max_concurrent_searches=1", payload, True)["responses"] if our else []
        assert len(baseline) == len(reference) == len(candidate) == a.batch_size
        if our:
            assert len(other) == a.batch_size
        for i, (b, c, r) in enumerate(zip(baseline, candidate, reference)):
            assert b["hits"] == c["hits"]
            check(b, r, source); check(c, r, source)
            if our:
                check(other[i], r, source)
            if source:
                for response in [b, c, r] + ([other[i]] if our else []):
                    for hit in response["hits"]["hits"]:
                        assert hit["_source"] == docs[hit["_id"]]
            verified += 1
        for round_number in range(a.rounds + 20):
            modes = ["es", "baseline", "candidate", "core"] + (["full"] if our else [])
            rng.shuffle(modes)
            for mode in modes:
                conn = es if mode == "es" else our if mode == "full" else poc
                path = "/path-poc/_msearch?max_concurrent_searches=1" if mode in ("es", "full") else f"/{mode}/_msearch"
                start = time.perf_counter_ns()
                response = request(conn, "POST", path, payload, True)
                elapsed = (time.perf_counter_ns() - start) / 1000 / a.batch_size
                assert len(response["responses"]) == a.batch_size
                if round_number >= 20:
                    samples[f"{cls}/source={source}/{mode}"].append(elapsed)
        vals = [statistics.median(samples[f"{cls}/source={source}/{m}"]) for m in ("es", "baseline", "candidate", "core")]
        print(f'{cls:8} source={source} amortized ES/baseline/candidate/core µs: ' + " / ".join(f"{v:.1f}" for v in vals), flush=True)
out = {"args": vars(a), "measurement": "HTTP batch wall time divided by batch size, not individual latency", "verified_responses": verified, "score_parity": True, "corpus_sha256": hashlib.sha256(pathlib.Path(a.corpus).read_bytes()).hexdigest(), "queries_sha256": hashlib.sha256(pathlib.Path(a.queries).read_bytes()).hexdigest(), "stats": {k: {"n": len(v), "p50_us": statistics.median(v), "p95_us": percentile(v, .95)} for k, v in samples.items()}}
pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
