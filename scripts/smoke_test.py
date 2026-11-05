#!/usr/bin/env python3
"""Single-node runtime smoke test, portable across Linux, macOS and Windows.

    python scripts/smoke_test.py path/to/elasticsearch[.exe]

Starts one node in a fresh data directory and exercises the storage paths
that differ between operating systems: segment writes and directory fsync,
merges that retire files a point-in-time still has memory-mapped, deletes
of documents and of a whole index, the translog, and crash recovery (the
node is killed, not stopped, and restarted twice). It also checks that the
OS / filesystem stats are populated. Exits non-zero on the first failure,
printing the node log's tail.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = 19250
BASE = f"http://127.0.0.1:{PORT}"
DOCS = 3000


def call(method, path, body=None, ndjson=False, ok=(200, 201)):
    data = None
    headers = {}
    if body is not None:
        if ndjson:
            data = ("\n".join(json.dumps(x) for x in body) + "\n").encode()
            headers["content-type"] = "application/x-ndjson"
        else:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    if status not in ok:
        raise AssertionError(f"{method} {path} -> {status}: {text[:2000]}")
    return json.loads(text) if text else {}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"ok: {msg}", flush=True)


class Node:
    def __init__(self, binary, data, log):
        self.binary, self.data, self.log_path = binary, data, log
        self.proc = None

    def start(self):
        args = [self.binary, "-E", f"path.data={self.data}", "-E", f"http.port={PORT}",
                "-E", f"transport.port={PORT + 100}", "-E", "network.host=127.0.0.1",
                "-E", "node.name=smoke-0", "-E", "cluster.name=smoke",
                "-E", "xpack.security.enabled=false"]
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        deadline = time.time() + 240
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"node exited with {self.proc.returncode} during startup")
            try:
                h = call("GET", "/_cluster/health?wait_for_status=yellow&timeout=1s", ok=(200, 408))
                if h.get("status") in ("green", "yellow"):
                    print(f"node up (pid {self.proc.pid})", flush=True)
                    return
            except (urllib.error.URLError, ConnectionError, OSError, AssertionError):
                pass
            time.sleep(1)
        raise AssertionError("node did not become ready within 240s")

    def kill(self):
        # A crash, not a shutdown: SIGKILL / TerminateProcess.
        self.proc.kill()
        self.proc.wait(timeout=60)
        print("node killed", flush=True)

    def tail(self, n=80):
        try:
            with open(self.log_path, "rb") as f:
                return b"\n".join(f.read().splitlines()[-n:]).decode(errors="replace")
        except OSError:
            return "(no log)"


def bulk(index, start, end):
    ops = []
    for i in range(start, end):
        ops.append({"index": {"_index": index, "_id": str(i)}})
        ops.append({"title": f"document number {i}", "tag": f"t{i % 7}", "n": i,
                    "vec": [((i * 31 + k) % 17) / 17.0 for k in range(8)]})
    r = call("POST", "/_bulk", ops, ndjson=True)
    check(not r.get("errors"), f"bulk [{start}, {end}) into {index} has no item errors")


def count(index):
    return call("GET", f"/{index}/_count")["count"]


def main():
    binary = os.path.abspath(sys.argv[1])
    tmp = tempfile.mkdtemp(prefix="rustic-smoke-")
    node = Node(binary, os.path.join(tmp, "data"), os.path.join(tmp, "node.log"))
    try:
        node.start()
        mapping = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                   "mappings": {"properties": {"title": {"type": "text"}, "tag": {"type": "keyword"},
                                               "n": {"type": "long"},
                                               "vec": {"type": "dense_vector", "dims": 8, "similarity": "l2_norm"}}}}
        call("PUT", "/smoke", mapping)
        call("PUT", "/doomed", mapping)

        # Several segments: refresh between batches.
        for b in range(0, DOCS, 500):
            bulk("smoke", b, b + 500)
            call("POST", "/smoke/_refresh")
        bulk("doomed", 0, 500)
        call("POST", "/doomed/_refresh")
        check(count("smoke") == DOCS, f"smoke holds {DOCS} docs")

        r = call("POST", "/smoke/_search", {"query": {"match": {"title": "number"}}, "size": 3})
        check(r["hits"]["total"]["value"] >= 3, "full-text search returns hits")
        r = call("POST", "/smoke/_search", {"knn": {"field": "vec", "query_vector": [0.5] * 8, "k": 5, "num_candidates": 50}})
        check(len(r["hits"]["hits"]) == 5, "kNN search returns 5 hits")

        # A point-in-time keeps the pre-merge segments open (mapped) while the
        # force merge retires them; their deletion must not fail or break it.
        pit = call("POST", "/smoke/_pit?keep_alive=2m")["id"]
        call("POST", "/smoke/_forcemerge?max_num_segments=1")
        r = call("POST", "/_search", {"pit": {"id": pit}, "size": 0, "track_total_hits": True})
        check(r["hits"]["total"]["value"] == DOCS, "PIT taken before the force merge still sees every doc")
        call("DELETE", "/_pit", {"id": pit})
        segs = call("GET", "/smoke/_segments")
        n_segs = sum(len(c["segments"]) for s in segs["indices"]["smoke"]["shards"].values() for c in s)
        check(n_segs >= 1, f"segments after force merge: {n_segs}")

        # Deletes: by query, then a whole index.
        call("POST", "/smoke/_delete_by_query?refresh=true", {"query": {"term": {"tag": "t0"}}})
        deleted = len([i for i in range(DOCS) if i % 7 == 0])
        expected = DOCS - deleted
        check(count("smoke") == expected, f"delete_by_query leaves {expected} docs")
        call("DELETE", "/doomed")
        call("HEAD", "/doomed", ok=(404,))
        print("ok: deleted index is gone", flush=True)

        # Crash after a flush: everything committed comes back.
        call("POST", "/smoke/_flush")
        node.kill()
        node.start()
        check(count("smoke") == expected, "flushed docs survive a crash")

        # Crash with acked-but-unflushed writes: the translog replays them.
        bulk("smoke", DOCS, DOCS + 200)
        node.kill()
        node.start()
        call("POST", "/smoke/_refresh")
        check(count("smoke") == expected + 200, "acked unflushed docs replay from the translog after a crash")
        call("HEAD", "/doomed", ok=(404,))
        print("ok: deleted index stays deleted after restart", flush=True)

        # OS / filesystem stats (the per-platform probes).
        st = call("GET", "/_nodes/stats/os,fs,process")
        n = next(iter(st["nodes"].values()))
        check(n["fs"]["total"]["total_in_bytes"] > 0, "fs total bytes reported")
        check(n["os"]["mem"]["total_in_bytes"] > 0, "os memory total reported")
        check(n["process"]["mem"]["total_virtual_in_bytes"] >= 0, "process stats reported")
        print("SMOKE TEST PASSED", flush=True)
    except Exception as e:
        print(f"SMOKE TEST FAILED: {e}", flush=True)
        print("---- node log tail ----\n" + node.tail(), flush=True)
        return 1
    finally:
        if node.proc and node.proc.poll() is None:
            node.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
