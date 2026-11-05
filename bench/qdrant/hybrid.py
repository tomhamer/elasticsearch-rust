#!/usr/bin/env python3
"""Qdrant hybrid (BM25 sparse + dense) benchmark; same workload as `compare hybrid`.

    ../tools/venv/bin/python hybrid.py --corpus ../corpus/corpus.jsonl --queries ../corpus/queries.jsonl \
        --vectors ../corpus/vectors-q100.f32 --vec-queries 100 \
        --qdrant-bin ../tools/qdrant-1.19.1/qdrant --storage ../indexes/qdrant-hybrid --results ../results

Collection: named vectors `dense` (128-d EUCLID, hnsw m=16 ef_construct=100) and
sparse `bm25` with `modifier: idf`; float payload index on `price`; one
segment requested (`default_segment_number=1`).

BM25 is built **client-side** (not fastembed's `Qdrant/bm25`): same tokenizer as
the other systems (lowercase, split on non-alphanumerics), doc value per term =
`tf / (tf + k1 * (1 - b + b * dl / avgdl))` (k1 = 1.2, b = 0.75, i.e. Lucene's
BM25 without the legacy `(k1 + 1)` factor), query value 1.0 per unique term, and
Qdrant's server-side IDF modifier supplies `ln(1 + (N - n + 0.5) / (n + 0.5))`,
so a sparse score is Lucene's BM25 (up to Lucene's lossy length norms).

Modes (single-threaded client, one keep-alive HTTP/1.1 connection, httpx):

* `rrf`: `POST /points/query` with `prefetch` [sparse bm25 limit 100, dense
  limit 100 hnsw_ef 100] and `query: {"rrf": {"k": 60}}` (parametrised RRF),
  limit 10.  Falls back to `{"fusion": "rrf"}` (Qdrant's default k) if the
  server rejects the parametrised form; the JSON records which was used.
* `sum`: ES-style score sum emulated with a `formula` query over the same two
  prefetches: `$score[0] + 1 / (1 + $score[1]^2)` (Qdrant returns the Euclid
  *distance* for dense hits; ES/Lucene/rustic score L2 as 1/(1+d^2)), with
  defaults `$score[0] = 0`, `$score[1] = 1e9` (absent from the k-NN top-10
  contributes ~0).  Dense prefetch limit = k = 10 (hnsw_ef 100), text prefetch
  limit `--sum-text-limit` (100): a k-NN-only doc whose BM25 rank is below
  that window gets text score 0, unlike ES (approximation, recorded).
* Filters `price <= q10` / `price <= q50` go into both prefetches (filterable
  HNSW / sparse index with payload filter).
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import httpx  # noqa: E402
import numpy as np  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from pycommon import latency_stats, load_vectors, write_result  # noqa: E402
from bench import start_server  # noqa: E402

COLL = "hybrid"
TOKEN = re.compile(r"[a-z0-9]+")


def pair_text(queries, n):
    or2 = [q for q in queries if q["class"] == "or2"]
    term = [q for q in queries if q["class"] == "term"]
    return [or2[(i // 2) % len(or2)] if i % 2 == 0 else term[(i // 2) % len(term)] for i in range(n)]


def pct_nearest_rank(sorted_vals, p):
    r = math.ceil(p / 100 * len(sorted_vals))
    return sorted_vals[max(1, min(len(sorted_vals), r)) - 1]


def wait_indexed(client, n):
    while True:
        info = client.get_collection(COLL)
        if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= n:
            return info
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--vec-queries", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num-candidates", type=int, default=100)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--sum-text-limit", type=int, default=100)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup-rounds", type=int, default=3)
    ap.add_argument("--qdrant-bin")
    ap.add_argument("--storage", default="../indexes/qdrant-hybrid")
    ap.add_argument("--reuse", action="store_true", help="keep existing storage/collection if complete")
    ap.add_argument("--url")
    ap.add_argument("--http-port", type=int, default=6333)
    ap.add_argument("--grpc-port", type=int, default=6334)
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    k, nc, window = a.k, a.num_candidates, a.window
    k1, b = 1.2, 0.75

    base, qvecs = load_vectors(a.vectors, a.dim, a.vec_queries)
    n = len(base)
    queries = [json.loads(line) for line in open(a.queries) if line.strip()]
    texts = pair_text(queries, a.n)

    t = time.perf_counter()
    vocab, docs_tf, dls, prices, ids = {}, [], [], [], []
    with open(a.corpus) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            toks = TOKEN.findall(d["body"].lower())
            c = Counter(toks)
            docs_tf.append({vocab.setdefault(w, len(vocab)): tf for w, tf in c.items()})
            dls.append(len(toks))
            prices.append(d["price"])
            ids.append(d["id"])
    assert len(docs_tf) == n, f"corpus docs {len(docs_tf)} != base vectors {n}"
    assert ids == list(range(n))
    avgdl = sum(dls) / n
    print(f"[qdrant-hybrid] tokenized {n} docs, vocab {len(vocab)}, avgdl {avgdl:.1f} in {time.perf_counter() - t:.1f}s",
          file=sys.stderr)

    def sparse_doc(i):
        tfs = docs_tf[i]
        norm = k1 * (1 - b + b * dls[i] / avgdl)
        idx = list(tfs.keys())
        return models.SparseVector(indices=idx, values=[tfs[j] / (tfs[j] + norm) for j in idx])

    def sparse_query(q):
        terms = [q["term"]] if q.get("term") else q["terms"]
        idx = sorted({vocab[w] for w in terms if w in vocab})
        return {"indices": idx, "values": [1.0] * len(idx)}

    sorted_prices = sorted(prices)
    q10, q50 = pct_nearest_rank(sorted_prices, 10), pct_nearest_rank(sorted_prices, 50)
    parr = np.array(prices)
    filters = [("none", None), ("price_p10", q10), ("price_p50", q50)]

    proc = None
    if a.url:
        url = a.url
    elif a.reuse and os.path.exists(os.path.join(a.storage, "hybrid.done")):
        proc, url = start_server(a.qdrant_bin, a.storage, a.http_port, a.grpc_port, keep=True)
    else:
        proc, url = start_server(a.qdrant_bin, a.storage, a.http_port, a.grpc_port)
    try:
        version = httpx.get(url + "/").json().get("version", "?")
        client = QdrantClient(url=url, prefer_grpc=True, grpc_port=a.grpc_port, timeout=600)
        build_s = upload_s = None
        if client.collection_exists(COLL) and client.count(COLL).count == n and a.reuse:
            info = wait_indexed(client, n)
            print(f"[qdrant-hybrid] reusing collection ({info.segments_count} segments)", file=sys.stderr)
        else:
            if client.collection_exists(COLL):
                client.delete_collection(COLL)
            client.create_collection(
                COLL,
                vectors_config={"dense": models.VectorParams(size=a.dim, distance=models.Distance.EUCLID)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                hnsw_config=models.HnswConfigDiff(m=a.m, ef_construct=a.ef_construction),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=1, max_segment_size=10_000_000),
            )
            client.create_payload_index(COLL, "price", models.PayloadSchemaType.FLOAT, wait=True)
            t0 = time.perf_counter()
            client.upload_points(
                COLL,
                points=(models.PointStruct(id=i, vector={"dense": base[i].tolist(), "bm25": sparse_doc(i)},
                                           payload={"price": prices[i]}) for i in range(n)),
                batch_size=512, parallel=1, wait=True)
            upload_s = time.perf_counter() - t0
            info = wait_indexed(client, n)
            build_s = time.perf_counter() - t0
            if a.storage:
                open(os.path.join(a.storage, "hybrid.done"), "w").write("ok")
            print(f"[qdrant-hybrid] upload {upload_s:.1f}s, green after {build_s:.1f}s, segments={info.segments_count}",
                  file=sys.stderr)
        del docs_tf

        http = httpx.Client(base_url=url, timeout=60)
        qlists = [q.tolist() for q in qvecs[:a.n]]
        sq = [sparse_query(q) for q in texts]

        def flt(hi):
            return None if hi is None else {"must": [{"key": "price", "range": {"lte": hi}}]}

        def post(body):
            r = http.post(f"/collections/{COLL}/points/query", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
            j = r.json()
            return [p["id"] for p in j["result"]["points"]], j["time"]

        def prefetches(qi, hi, text_limit, dense_limit):
            f = flt(hi)
            pt = {"query": sq[qi], "using": "bm25", "limit": text_limit}
            pd = {"query": qlists[qi], "using": "dense", "limit": dense_limit, "params": {"hnsw_ef": nc}}
            if f is not None:
                pt["filter"] = f
                pd["filter"] = f
            return [pt, pd]

        rrf_query = {"rrf": {"k": int(60)}}
        try:
            post({"prefetch": prefetches(0, None, window, window), "query": rrf_query, "limit": k})
            rrf_form = "query: {rrf: {k: 60}} (parametrised RRF)"
        except RuntimeError as e:
            print(f"[qdrant-hybrid] parametrised rrf rejected ({e}); using fusion: rrf", file=sys.stderr)
            rrf_query = {"fusion": "rrf"}
            rrf_form = "query: {fusion: rrf} (server default k)"

        sum_formula = {"formula": {"sum": ["$score[0]", {"div": {"left": 1.0, "right": {"sum": [1.0, {"pow": {"base": "$score[1]", "exponent": 2.0}}]}}}]},
                       "defaults": {"$score[0]": 0.0, "$score[1]": 1e9}}
        sum_ok = True
        try:
            post({"prefetch": prefetches(0, None, a.sum_text_limit, k), "query": sum_formula, "limit": k})
        except RuntimeError as e:
            print(f"[qdrant-hybrid] formula sum rejected: {e}", file=sys.stderr)
            sum_ok = False

        def exec_fn(mode, qi, hi):
            if mode == "rrf":
                return post({"prefetch": prefetches(qi, hi, window, window), "query": rrf_query, "limit": k})
            return post({"prefetch": prefetches(qi, hi, a.sum_text_limit, k), "query": sum_formula, "limit": k})

        runs, filter_meta = [], []
        for fname, hi in filters:
            sel = float((parr <= hi).mean()) if hi is not None else 1.0
            filter_meta.append({"name": fname, "price_lte": hi, "selectivity": sel})
            # kNN leg recall vs brute force
            rec = 0.0
            for qi in range(a.n):
                d = ((base - qvecs[qi]) ** 2).sum(axis=1)
                idsx = np.arange(n)
                if hi is not None:
                    msk = parr <= hi
                    d, idsx = d[msk], idsx[msk]
                truth = set(int(x) for x in idsx[np.lexsort((idsx, d))[:k]])
                body = {"query": qlists[qi], "using": "dense", "limit": k, "params": {"hnsw_ef": nc}}
                if hi is not None:
                    body["filter"] = flt(hi)
                found, _ = post(body)
                rec += len(set(found) & truth) / len(truth) if truth else 1.0
            knn_recall = rec / a.n
            for mode in ("sum", "rrf"):
                if mode == "sum" and not sum_ok:
                    continue
                for _ in range(a.warmup_rounds):
                    for qi in range(a.n):
                        exec_fn(mode, qi, hi)
                lat, srv, first = [], [], []
                t0 = time.perf_counter()
                for r in range(a.rounds):
                    for qi in range(a.n):
                        s = time.perf_counter()
                        out, st = exec_fn(mode, qi, hi)
                        lat.append(time.perf_counter() - s)
                        srv.append(st)
                        if r == 0:
                            first.append(out)
                total = time.perf_counter() - t0
                ls, ss = latency_stats(lat), latency_stats(srv)
                run = {"mode": mode, "filter": fname, "selectivity": sel, "qps": len(lat) / total, "latency": ls,
                       "server_p50_us": ss["p50_us"], "server_p99_us": ss["p99_us"], "server_mean_us": ss["mean_us"],
                       "knn_recall_at_k": knn_recall, "top_ids": first}
                if mode == "sum":
                    run["note"] = f"formula query; text prefetch limit {a.sum_text_limit} (approximate ES sum)"
                runs.append(run)
                print(f"[qdrant-hybrid] {mode} filter={fname} ({sel * 100:.0f} %): client {run['qps']:.0f} QPS "
                      f"p50 {ls['p50_us']:.0f}us p99 {ls['p99_us']:.0f}us, server p50 {ss['p50_us']:.0f}us; "
                      f"knn recall@{k} {knn_recall:.4f}", file=sys.stderr)

        params = {"corpus": a.corpus, "vectors": a.vectors, "dim": a.dim, "n_docs": n, "n_queries": a.n, "k": k,
                  "num_candidates": nc, "rrf_window": window, "rank_constant": 60, "m": a.m,
                  "ef_construction": a.ef_construction, "rounds": a.rounds, "warmup_rounds": a.warmup_rounds,
                  "qdrant_version": version, "segments": info.segments_count,
                  "bm25": "client-side tf/(tf+k1*(1-b+b*dl/avgdl)) (k1=1.2, b=0.75) + sparse modifier=idf (server IDF)",
                  "rrf_path": "prefetch [bm25 sparse limit 100, dense limit 100 hnsw_ef 100] + " + rrf_form,
                  "sum_path": ("formula $score[0] + 1/(1+$score[1]^2), prefetch [bm25 limit %d, dense limit %d hnsw_ef %d]"
                               % (a.sum_text_limit, k, nc)) if sum_ok else "unsupported",
                  "filter_path": "payload range filter on float index, inside both prefetches",
                  "transport": "HTTP/1.1 keep-alive JSON, localhost, single-threaded Python client"}
        write_result(a.results, "qdrant", "hybrid", params,
                     {"build_s": build_s, "upload_s": upload_s, "filters": filter_meta, "runs": runs})
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    main()
