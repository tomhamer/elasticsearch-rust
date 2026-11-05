#!/usr/bin/env python3
"""Qdrant vector benchmark: unfiltered + filtered, against a local server.

    ../tools/venv/bin/python bench.py --vectors ../corpus/vectors.f32 \
        --qdrant-bin ../tools/qdrant-1.19.1/qdrant --storage ../indexes/qdrant --results ../results

* Starts the prebuilt `qdrant` binary (no Docker) with storage under
  `--storage`, telemetry off, and stops it at the end (or use `--url` to talk
  to a server that is already running).
* Collection: 128-d, Distance.EUCLID, hnsw m=16 ef_construct=100, one
  segment (`default_segment_number=1`, large `max_segment_size`) so a query
  walks a single graph like the other systems, default `indexing_threshold`
  (10 MB < the 100 MB of vectors, so the HNSW is built), integer payload
  indexes on `tag` and `bucket` created before the upload.  All other
  settings default (incl. `full_scan_threshold` = 10 MB, the planner's exact
  scan cut-off, and `max_indexing_threads` = auto, so the build is
  multi-threaded).
* Build time = upload (gRPC `upload_collection`, batches of 1024, wait) +
  waiting until the collection is green with all vectors indexed.
* Search: single-threaded client loop of `POST /collections/c/points/query`
  over one keep-alive HTTP/1.1 connection (`httpx`), `params.hnsw_ef = ef`,
  limit 10.  Two timings are reported: client wall time per request
  (includes JSON encode/decode and the localhost round trip) and the
  server-side `time` field of each response (Qdrant's own handling time),
  which is the fairer comparison with the in-process engines.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import httpx  # noqa: E402
import numpy as np  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from pycommon import FILTERS, brute_force, latency_stats, load_vectors, payload, recall, timed_loop, write_result  # noqa: E402

COLL = "bench"


def start_server(binary, storage, http_port, grpc_port, keep=False):
    if os.path.exists(storage) and not keep:
        shutil.rmtree(storage)
    os.makedirs(storage, exist_ok=True)
    env = dict(os.environ,
               QDRANT__STORAGE__STORAGE_PATH=os.path.join(storage, "storage"),
               QDRANT__STORAGE__SNAPSHOTS_PATH=os.path.join(storage, "snapshots"),
               QDRANT__SERVICE__HTTP_PORT=str(http_port),
               QDRANT__SERVICE__GRPC_PORT=str(grpc_port),
               QDRANT__SERVICE__HOST="127.0.0.1",
               QDRANT__TELEMETRY_DISABLED="true",
               QDRANT__LOG_LEVEL="WARN")
    log = open(os.path.join(storage, "qdrant.log"), "a" if keep else "w")
    p = subprocess.Popen([os.path.abspath(binary)], cwd=storage, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{http_port}"
    for _ in range(300):
        try:
            if httpx.get(url + "/readyz", timeout=1).status_code == 200:
                return p, url
        except Exception:
            pass
        if p.poll() is not None:
            raise SystemExit(f"qdrant exited early, see {storage}/qdrant.log")
        time.sleep(0.1)
    p.kill()
    raise SystemExit("qdrant did not become ready")


def wait_indexed(client, n):
    while True:
        info = client.get_collection(COLL)
        if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n:
            return info
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--ef", default="50,100,200")
    ap.add_argument("--filter-ef", default="100")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup-rounds", type=int, default=5)
    ap.add_argument("--qdrant-bin")
    ap.add_argument("--storage", default="../indexes/qdrant")
    ap.add_argument("--url", help="use an already running server instead of --qdrant-bin")
    ap.add_argument("--http-port", type=int, default=6333)
    ap.add_argument("--grpc-port", type=int, default=6334)
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    k = a.k
    base, queries = load_vectors(a.vectors, a.dim, a.queries)
    n = len(base)
    pl = payload(n)

    proc = None
    if a.url:
        url = a.url
    else:
        proc, url = start_server(a.qdrant_bin, a.storage, a.http_port, a.grpc_port)
    try:
        version = httpx.get(url + "/").json().get("version", "?")
        print(f"[qdrant] server {version} at {url}", file=sys.stderr)
        client = QdrantClient(url=url, prefer_grpc=True, grpc_port=a.grpc_port, timeout=600)
        if client.collection_exists(COLL):
            client.delete_collection(COLL)
        client.create_collection(
            COLL,
            vectors_config=models.VectorParams(size=a.dim, distance=models.Distance.EUCLID),
            hnsw_config=models.HnswConfigDiff(m=a.m, ef_construct=a.ef_construction),
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=1, max_segment_size=10_000_000),
        )
        for f in ("tag", "bucket"):
            client.create_payload_index(COLL, f, models.PayloadSchemaType.INTEGER, wait=True)

        t0 = time.perf_counter()
        client.upload_collection(
            COLL, vectors=base, ids=range(n),
            payload=({"tag": int(pl["tag"][i]), "bucket": int(pl["bucket"][i])} for i in range(n)),
            batch_size=1024, parallel=1, wait=True)
        upload_s = time.perf_counter() - t0
        info = wait_indexed(client, n)
        build_s = time.perf_counter() - t0
        print(f"[qdrant] upload {upload_s:.2f}s, green+indexed after {build_s:.2f}s; "
              f"segments={info.segments_count} indexed={info.indexed_vectors_count}", file=sys.stderr)

        http = httpx.Client(base_url=url, timeout=60)
        qlists = [q.tolist() for q in queries]

        def query(qi, ef, flt=None):
            body = {"query": qlists[qi], "limit": k, "params": {"hnsw_ef": ef}}
            if flt is not None:
                body["filter"] = flt
            r = http.post(f"/collections/{COLL}/points/query", json=body)
            r.raise_for_status()
            j = r.json()
            return [p["id"] for p in j["result"]["points"]], j["time"]

        def measure(ef, truth, flt_of=None):
            fn = lambda qi: query(qi, ef, flt_of(qi) if flt_of else None)  # noqa: E731
            lat, total, first = timed_loop(fn, queries, a.rounds, a.warmup_rounds)
            # Server-side times of the timed rounds: rerun-free, collected from the first round only is
            # too few samples, so recompute from all timed calls via a second pass of the same loop.
            srv = []
            for _ in range(a.rounds):
                for qi in range(len(queries)):
                    srv.append(fn(qi)[1])
            rec = sum(recall(first[qi][0], truth[qi]) for qi in range(len(queries))) / len(queries)
            st = latency_stats(lat)
            sst = latency_stats(srv)
            return {"ef": ef, "qps": len(lat) / total, "recall_at_k": rec, "latency": st,
                    "server_p50_us": sst["p50_us"], "server_mean_us": sst["mean_us"],
                    "server_qps_equiv": 1e6 / sst["mean_us"] if sst["mean_us"] else None}

        t = time.perf_counter()
        truth = [brute_force(base, q, k) for q in queries]
        brute_qps = len(queries) / (time.perf_counter() - t)
        runs = []
        for ef in [int(x) for x in a.ef.split(",")]:
            r = measure(ef, truth)
            print(f"[qdrant] ef={ef}: client {r['qps']:.0f} QPS p50 {r['latency']['p50_us']:.0f}us, "
                  f"server p50 {r['server_p50_us']:.0f}us, recall {r['recall_at_k']:.4f}", file=sys.stderr)
            runs.append(r)

        params = {"vectors": a.vectors, "dim": a.dim, "n": n, "queries": len(queries), "k": k, "m": a.m,
                  "ef_construction": a.ef_construction, "similarity": "l2", "rounds": a.rounds,
                  "warmup_rounds": a.warmup_rounds, "qdrant_version": version,
                  "transport": "HTTP/1.1 keep-alive JSON, localhost, single-threaded client",
                  "segments": info.segments_count,
                  "note": "qps/latency include client<->server round trips; server_* = Qdrant's response 'time' field"}
        write_result(a.results, "qdrant", "vectors", params, {
            "build_s": build_s, "upload_s": upload_s, "build_vectors_per_sec": n / build_s,
            "build_threads": "auto (max_indexing_threads=0)", "graph_bytes": None,
            "brute_force_qps": brute_qps, "runs": runs})

        fruns = []
        for name, sel, expr, field, lo_f, hi_f in FILTERS:
            col = pl[field]
            masks = [(col >= lo_f(qi)) & (col <= hi_f(qi)) for qi in range(len(queries))]
            ftruth = [brute_force(base, queries[qi], k, masks[qi]) for qi in range(len(queries))]

            def flt_of(qi, field=field, lo_f=lo_f, hi_f=hi_f):
                lo, hi = lo_f(qi), hi_f(qi)
                if lo == hi:
                    return {"must": [{"key": field, "match": {"value": lo}}]}
                return {"must": [{"key": field, "range": {"gte": lo, "lte": hi}}]}

            for ef in [int(x) for x in a.filter_ef.split(",")]:
                r = measure(ef, ftruth, flt_of)
                r.update({"filter": name, "expr": expr, "selectivity": sel})
                print(f"[qdrant] filter {name} ef={ef}: client {r['qps']:.0f} QPS p50 {r['latency']['p50_us']:.0f}us, "
                      f"server p50 {r['server_p50_us']:.0f}us, recall {r['recall_at_k']:.4f}", file=sys.stderr)
                fruns.append(r)
        fparams = dict(params, path="POST /points/query with filter (match / range) on integer payload indexes",
                       payload="tag = i % 100, bucket = i % 10 (integer payload indexes)")
        write_result(a.results, "qdrant", "vectors-filtered", fparams, {"build_s": build_s, "runs": fruns})
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
