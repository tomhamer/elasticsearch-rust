"""Shared helpers for the Python vector harnesses (hnswlib/, qdrant/).

Same data, filters, ground truth, latency statistics and JSON result
envelope as `compare` (Rust) and `lucene/` (Java):

* vectors.f32: n base + `queries` query vectors, little-endian f32, no header.
* payload: tag = i % 100, bucket = i % 10.
* filters per query qi: tag == qi % 100 (1 %), bucket == qi % 10 (10 %), bucket < 5 (50 %).
* latency: nearest-rank percentiles in microseconds.
"""
import json
import math
import os
import platform
import subprocess
import time

import numpy as np

FILTERS = [
    # name, selectivity, expr, field, lo(qi), hi(qi)  (inclusive)
    ("tag_eq", 0.01, "tag == qi % 100", "tag", lambda qi: qi % 100, lambda qi: qi % 100),
    ("bucket_eq", 0.10, "bucket == qi % 10", "bucket", lambda qi: qi % 10, lambda qi: qi % 10),
    ("bucket_lt5", 0.50, "bucket < 5", "bucket", lambda qi: 0, lambda qi: 4),
]


def load_vectors(path, dim, n_queries):
    a = np.fromfile(path, dtype="<f4")
    assert a.size % dim == 0
    a = a.reshape(-1, dim)
    return np.ascontiguousarray(a[:-n_queries]), np.ascontiguousarray(a[-n_queries:])


def payload(n):
    i = np.arange(n)
    return {"tag": i % 100, "bucket": i % 10}


def brute_force(base, q, k, mask=None):
    d = ((base - q) ** 2).sum(axis=1)
    ids = np.arange(len(base))
    if mask is not None:
        d, ids = d[mask], ids[mask]
    # ties by lower id, like the other harnesses
    order = np.lexsort((ids, d))[:k]
    return set(int(x) for x in ids[order])


def recall(found, truth):
    if not truth:
        return 1.0
    return len(set(found) & truth) / len(truth)


def latency_stats(seconds):
    us = sorted(s * 1e6 for s in seconds)
    if not us:
        return {"n": 0, "p50_us": 0.0, "p90_us": 0.0, "p99_us": 0.0, "mean_us": 0.0, "min_us": 0.0, "max_us": 0.0}

    def pct(p):
        r = math.ceil(p / 100 * len(us))
        return us[max(1, min(len(us), r)) - 1]

    return {"n": len(us), "p50_us": pct(50), "p90_us": pct(90), "p99_us": pct(99),
            "mean_us": sum(us) / len(us), "min_us": us[0], "max_us": us[-1]}


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _proc(path, key):
    try:
        for line in open(path):
            k, _, v = line.partition(":")
            if k.strip() == key:
                return v.strip()
    except OSError:
        pass
    return None


def machine():
    label = os.environ.get("BENCH_MACHINE", "")
    if _proc("/proc/cpuinfo", "model name"):
        osname = "Linux"
        try:
            for line in open("/etc/os-release"):
                if line.startswith("PRETTY_NAME="):
                    osname = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        cores = {l.strip() for l in open("/proc/cpuinfo") if l.startswith("core id")}
        m = {"cpu": _proc("/proc/cpuinfo", "model name"), "logical_cores": os.cpu_count() or 0,
             "physical_cores": len(cores), "ram_bytes": int(_proc("/proc/meminfo", "MemTotal").split()[0]) * 1024,
             "os": osname, "arch": platform.machine(), "python": platform.python_version()}
        if label:
            m["label"] = label
        return m
    m = {
        "cpu": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "logical_cores": int(_sh("sysctl", "-n", "hw.ncpu") or 0),
        "physical_cores": int(_sh("sysctl", "-n", "hw.physicalcpu") or 0),
        "ram_bytes": int(_sh("sysctl", "-n", "hw.memsize") or 0),
        "os": f"{_sh('sw_vers', '-productName')} {_sh('sw_vers', '-productVersion')}",
        "arch": "aarch64" if platform.machine() == "arm64" else platform.machine(),
        "python": platform.python_version(),
    }
    if label:
        m["label"] = label
    return m


def write_result(results_dir, system, subcommand, params, metrics):
    os.makedirs(results_dir, exist_ok=True)
    env = {"system": system, "subcommand": subcommand, "timestamp_unix": int(time.time()),
           "machine": machine(), "params": params, "metrics": metrics}
    p = os.path.join(results_dir, f"{system}-{subcommand}.json")
    with open(p, "w") as f:
        json.dump(env, f, indent=2)
    print(f"wrote {p}")
    return p


def timed_loop(fn, queries, rounds, warmup_rounds):
    """Run fn(qi) over all queries: `warmup_rounds` untimed, then `rounds` timed.

    Returns (per-call latencies, total seconds, results of the first timed round).
    """
    for _ in range(warmup_rounds):
        for qi in range(len(queries)):
            fn(qi)
    lat, first = [], []
    t0 = time.perf_counter()
    for r in range(rounds):
        for qi in range(len(queries)):
            s = time.perf_counter()
            out = fn(qi)
            lat.append(time.perf_counter() - s)
            if r == 0:
                first.append(out)
    return lat, time.perf_counter() - t0, first
