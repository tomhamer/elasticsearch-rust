#!/usr/bin/env python3
"""hnswlib vector benchmark: unfiltered + filtered.

    ../tools/venv/bin/python bench.py --vectors ../corpus/vectors.f32 --results ../results

* Build: `hnswlib.Index('l2', dim)`, `init_index(M=16, ef_construction=100,
  random_seed=100)`, `add_items` once with 1 thread and once with all cores
  (the 1-thread index is the one searched; build_s = 1 thread).
* Search: single-threaded loop of `knn_query(q, k=10)` (`set_num_threads(1)`),
  `set_ef(ef)` for ef in 50/100/200; 5 untimed warm-up + 20 timed rounds;
  recall@10 vs numpy brute force.  Latency includes the Python -> C++ call
  (numpy array in/out), a few microseconds per query.
* Filtered (`results/hnswlib-vectors-filtered.json`): `knn_query(q, k,
  filter=callable)` with ef = 100.  hnswlib's filter is a Python callback
  invoked for every candidate the graph search visits (holding the GIL), so
  the filtered numbers are dominated by Python call overhead and are NOT a
  measure of hnswlib's C++ filtered search; hnswlib has no exact-scan
  fallback, so low-selectivity filters also lose recall.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hnswlib  # noqa: E402
import numpy as np  # noqa: E402

from pycommon import FILTERS, brute_force, latency_stats, load_vectors, payload, recall, timed_loop, write_result  # noqa: E402


def build(base, dim, m, efc, threads):
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(base), M=m, ef_construction=efc, random_seed=100)
    idx.set_num_threads(threads)
    t0 = time.perf_counter()
    idx.add_items(base, np.arange(len(base)), num_threads=threads)
    return idx, time.perf_counter() - t0


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
    ap.add_argument("--filter-rounds", type=int, default=5)
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    k = a.k
    efs = [int(x) for x in a.ef.split(",")]
    base, queries = load_vectors(a.vectors, a.dim, a.queries)
    n = len(base)
    ncpu = os.cpu_count() or 1
    print(f"[hnswlib] {n} base, {len(queries)} queries, dim {a.dim}", file=sys.stderr)

    _, build_all = build(base, a.dim, a.m, a.ef_construction, ncpu)
    print(f"[hnswlib] build {ncpu} threads: {build_all:.2f}s", file=sys.stderr)
    idx, build_1 = build(base, a.dim, a.m, a.ef_construction, 1)
    print(f"[hnswlib] build 1 thread: {build_1:.2f}s", file=sys.stderr)
    idx.set_num_threads(1)

    t0 = time.perf_counter()
    truth = [brute_force(base, q, k) for q in queries]
    brute_qps = len(queries) / (time.perf_counter() - t0)

    runs = []
    for ef in efs:
        idx.set_ef(max(ef, k))
        fn = lambda qi: idx.knn_query(queries[qi], k=k, num_threads=1)[0][0]  # noqa: E731
        lat, total, first = timed_loop(fn, queries, a.rounds, a.warmup_rounds)
        rec = sum(recall(first[qi].tolist(), truth[qi]) for qi in range(len(queries))) / len(queries)
        st = latency_stats(lat)
        qps = len(lat) / total
        print(f"[hnswlib] ef={ef}: {qps:.0f} QPS p50 {st['p50_us']:.0f}us recall {rec:.4f}", file=sys.stderr)
        runs.append({"ef": ef, "qps": qps, "recall_at_k": rec, "latency": st})

    params = {"vectors": a.vectors, "dim": a.dim, "n": n, "queries": len(queries), "k": k, "m": a.m,
              "ef_construction": a.ef_construction, "similarity": "l2", "rounds": a.rounds,
              "warmup_rounds": a.warmup_rounds, "threads": 1, "hnswlib_version": getattr(hnswlib, "__version__", "0.8.0"),
              "note": "python binding; per-query latency includes the Python->C++ call"}
    write_result(a.results, "hnswlib", "vectors", params, {
        "build_s": build_1, "build_vectors_per_sec": n / build_1, "build_s_all_threads": build_all,
        "build_threads_all": ncpu, "graph_bytes": None, "brute_force_qps": brute_qps, "runs": runs})

    # ---- filtered ----
    pl = payload(n)
    fruns = []
    for name, sel, expr, field, lo_f, hi_f in FILTERS:
        col = pl[field]
        masks = [(col >= lo_f(qi)) & (col <= hi_f(qi)) for qi in range(len(queries))]
        ftruth = [brute_force(base, queries[qi], k, masks[qi]) for qi in range(len(queries))]
        # Python predicates over the label (= index i); plain arithmetic, no numpy in the callback.
        mod = 100 if field == "tag" else 10
        preds = []
        for qi in range(len(queries)):
            lo, hi = lo_f(qi), hi_f(qi)
            preds.append(lambda label, lo=lo, hi=hi, mod=mod: lo <= label % mod <= hi)
        for ef in [int(x) for x in a.filter_ef.split(",")]:
            idx.set_ef(max(ef, k))

            def fn(qi):
                try:
                    return idx.knn_query(queries[qi], k=k, num_threads=1, filter=preds[qi])[0][0]
                except RuntimeError:  # fewer than k results found
                    return np.array([], dtype=np.int64)

            lat, total, first = timed_loop(fn, queries, a.filter_rounds, 1)
            rec = sum(recall(first[qi].tolist(), ftruth[qi]) for qi in range(len(queries))) / len(queries)
            st = latency_stats(lat)
            qps = len(lat) / total
            print(f"[hnswlib] filter {name} ef={ef}: {qps:.0f} QPS p50 {st['p50_us']:.0f}us recall {rec:.4f}", file=sys.stderr)
            fruns.append({"filter": name, "expr": expr, "selectivity": sel, "ef": ef, "qps": qps,
                          "recall_at_k": rec, "latency": st})
    fparams = dict(params, rounds=a.filter_rounds, warmup_rounds=1,
                   path="knn_query(q, k, filter=python_callable)",
                   payload="tag = i % 100, bucket = i % 10 (computed from the label in the Python callback)",
                   note="filter is a Python callback per visited node (GIL, ~100 ns+/call): slow by construction")
    write_result(a.results, "hnswlib", "vectors-filtered", fparams, {"build_s": build_1, "runs": fruns})


if __name__ == "__main__":
    main()
