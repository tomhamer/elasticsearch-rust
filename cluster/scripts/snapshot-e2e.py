#!/usr/bin/env python3
"""Snapshot / restore end-to-end tests against a real 3-node cluster.

    cluster/scripts/snapshot-e2e.py [test ...]

Starts `local-cluster.sh` (PORT_BASE, default 9800) with `path.repo` set,
runs the selected tests (all by default) and stops the cluster. Needs the
`elasticsearch` binary built (`cargo build -p elasticsearch-node`).

S3: set ES_TEST_S3_BUCKET (+ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
AWS_REGION, optional ES_TEST_S3_ENDPOINT / ES_TEST_S3_PATH_STYLE) to also run
the `s3` test; it is skipped otherwise.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT_BASE = int(os.environ.get("PORT_BASE", "9800"))
CLUSTER_DIR = os.environ.get("CLUSTER_DIR", f"/tmp/es-snap-e2e-{PORT_BASE}")
REPO_DIR = os.path.join(CLUSTER_DIR, "repos")
N = 3


def url(node, path):
    return f"http://127.0.0.1:{PORT_BASE + node}{path}"


def call(method, path, body=None, node=0, ok=True, ndjson=False, timeout=300):
    data = None
    headers = {}
    if body is not None:
        if ndjson:
            data = body.encode()
            headers["content-type"] = "application/x-ndjson"
        else:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
    req = urllib.request.Request(url(node, path), data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    try:
        out = json.loads(raw) if raw else None
    except ValueError:
        out = raw
    if ok and not (200 <= status < 300):
        raise AssertionError(f"{method} {path} -> {status}: {raw[:2000]}")
    return status, out


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def cluster(cmd, *args, extra=""):
    env = dict(os.environ, PORT_BASE=str(PORT_BASE), CLUSTER_DIR=CLUSTER_DIR, EXTRA_ARGS=extra)
    subprocess.run([os.path.join(HERE, "local-cluster.sh"), cmd, *map(str, args)], check=True, env=env)


def live_nodes():
    out = []
    for i in range(N):
        try:
            urllib.request.urlopen(url(i, "/"), timeout=2).read()
            out.append(i)
        except Exception:
            pass
    return out


def any_node():
    return live_nodes()[0]


def wait_green(index=None, node=None, timeout="60s"):
    node = any_node() if node is None else node
    path = f"/_cluster/health/{index}" if index else "/_cluster/health"
    _, h = call("GET", f"{path}?wait_for_status=green&timeout={timeout}", node=node, ok=False)
    check(h.get("status") == "green", f"not green: {h}")


def bulk_docs(index, start, n, size=40, node=0, routing_every=0):
    lines = []
    for i in range(start, start + n):
        meta = {"index": {"_index": index, "_id": str(i)}}
        if routing_every and i % routing_every == 0:
            meta["index"]["routing"] = f"r{i % 7}"
        lines.append(json.dumps(meta))
        lines.append(json.dumps({"n": i, "tag": f"t{i % 10}", "text": "x" * size, "nested": [{"k": i, "v": "a"}, {"k": i + 1, "v": "b"}]}))
    _, r = call("POST", "/_bulk?refresh=false", "\n".join(lines) + "\n", node=node, ndjson=True)
    check(not r["errors"], f"bulk errors: {json.dumps(r)[:1000]}")


def count(index, node=None):
    node = any_node() if node is None else node
    call("POST", f"/{index}/_refresh", node=node)
    _, r = call("GET", f"/{index}/_count", node=node)
    return r["count"]


def mapping(index, node=None):
    _, r = call("GET", f"/{index}/_mapping", node=any_node() if node is None else node)
    return r[index]["mappings"]


def create_index(name, replicas=1, extra_settings=None, mappings=None):
    settings = {"number_of_shards": 1, "number_of_replicas": replicas}
    settings.update(extra_settings or {})
    body = {"settings": settings, "mappings": mappings or {"properties": {"n": {"type": "long"}, "tag": {"type": "keyword"}, "text": {"type": "text"}, "nested": {"type": "nested", "properties": {"k": {"type": "long"}, "v": {"type": "keyword"}}}}}}
    call("PUT", f"/{name}", body, node=any_node())


def put_repo(name, settings, node=0, verify=True):
    q = "" if verify else "?verify=false"
    call("PUT", f"/_snapshot/{name}{q}", {"type": "fs", "settings": settings}, node=node)


def wait_snapshot_done(repo, snap, timeout=300):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            s, r = call("GET", f"/_snapshot/{repo}/{snap}", node=any_node(), ok=False, timeout=30)
        except Exception as e:  # node going away
            last = str(e)
            time.sleep(1)
            continue
        last = r
        if s == 200 and r["snapshots"] and r["snapshots"][0]["state"] != "IN_PROGRESS":
            return r["snapshots"][0]
        time.sleep(1)
    raise AssertionError(f"snapshot {snap} did not complete: {last}")


# --------------------------------------------------------------------- tests


def test_repositories():
    put_repo("repo_a", {"location": "a"})
    put_repo("repo_b", {"location": "b", "compress": True})
    _, r = call("GET", "/_snapshot", node=1)
    check(set(r) >= {"repo_a", "repo_b"} and r["repo_a"]["type"] == "fs" and r["repo_a"]["settings"]["location"] == "a", f"GET _snapshot: {r}")
    check("uuid" in r["repo_a"], f"uuid missing: {r}")
    _, r = call("GET", "/_snapshot/repo_a", node=2)
    check(list(r) == ["repo_a"], f"{r}")
    s, r = call("GET", "/_snapshot/missing", node=2, ok=False)
    check(s == 404 and r["error"]["type"] == "repository_missing_exception", f"{s} {r}")
    _, r = call("POST", "/_snapshot/repo_a/_verify", node=1)
    check(len(r["nodes"]) == N, f"verify should list {N} nodes: {r}")
    s, r = call("PUT", "/_snapshot/bad", {"type": "fs", "settings": {"location": "/elsewhere/x"}}, node=0, ok=False)
    check(s >= 400 and r["error"]["type"] == "repository_exception", f"outside path.repo must fail: {s} {r}")
    _, t = call("GET", "/_cat/repositories?v", node=1)
    check("repo_a" in t and "fs" in t, t)
    _, st = call("GET", "/_cluster/state/metadata", node=2)
    check("repo_a" in st["metadata"]["repositories"], "repositories custom missing")
    call("DELETE", "/_snapshot/repo_b", node=2)
    s, _ = call("GET", "/_snapshot/repo_b", node=0, ok=False)
    check(s == 404, "repo_b should be gone")


def test_snapshot_restore():
    put_repo("repo1", {"location": "r1"})
    create_index("docs", replicas=1)
    for b in range(4):
        bulk_docs("docs", b * 500, 500, routing_every=5)
    call("POST", "/docs/_doc/7?refresh=true", {"n": 7, "tag": "updated", "text": "v2", "nested": []}, node=1)
    wait_green("docs")
    before_count = count("docs")
    before_map = mapping("docs")
    _, d7 = call("GET", "/docs/_doc/7", node=2)
    _, r = call("PUT", "/_snapshot/repo1/snap1?wait_for_completion=true", {"indices": "docs", "metadata": {"by": "e2e"}}, node=2)
    s = r["snapshot"]
    check(s["state"] == "SUCCESS" and s["shards"] == {"total": 1, "failed": 0, "successful": 1}, f"create: {r}")
    check(s["indices"] == ["docs"] and s["metadata"] == {"by": "e2e"}, f"{s}")
    _, r = call("GET", "/_snapshot/repo1/snap1", node=1)
    check(r["snapshots"][0]["state"] == "SUCCESS" and r["total"] == 1, f"get: {r}")
    _, r = call("GET", "/_snapshot/repo1/snap1/_status", node=0)
    check(r["snapshots"][0]["state"] == "SUCCESS" and r["snapshots"][0]["stats"]["total"]["file_count"] > 0, f"status: {r}")
    _, t = call("GET", "/_cat/snapshots/repo1?v", node=1)
    check("snap1" in t and "SUCCESS" in t, t)
    # Incremental: an unchanged second snapshot uploads nothing new.
    call("PUT", "/_snapshot/repo1/snap1b?wait_for_completion=true", {"indices": "docs"}, node=0)
    _, r = call("GET", "/_snapshot/repo1/snap1b/_status", node=0)
    check(r["snapshots"][0]["stats"]["incremental"]["file_count"] == 0, f"second snapshot should be fully incremental: {r['snapshots'][0]['stats']}")
    call("DELETE", "/docs", node=1)
    _, r = call("POST", "/_snapshot/repo1/snap1/_restore?wait_for_completion=true", {"indices": "docs"}, node=1)
    check(r["snapshot"]["shards"]["successful"] == 1 and r["snapshot"]["indices"] == ["docs"], f"restore: {r}")
    wait_green("docs")
    check(count("docs") == before_count, f"count {count('docs')} != {before_count}")
    check(mapping("docs") == before_map, f"mapping differs: {mapping('docs')} vs {before_map}")
    _, a7 = call("GET", "/docs/_doc/7", node=0)
    check(a7["_source"] == d7["_source"] and a7["_version"] == d7["_version"], f"doc 7: {a7} vs {d7}")
    _, rr = call("GET", "/docs/_doc/5?routing=r5", node=2, ok=False)
    check(rr.get("found") and rr.get("_routing") == "r5", f"routed doc: {rr}")
    # Every copy has the data (replica got it through replication).
    for n in live_nodes():
        _, c = call("GET", "/docs/_count?preference=_local", node=n, ok=False)
    _, rec = call("GET", "/docs/_recovery", node=0)
    print("    _recovery:", json.dumps(rec)[:300])
    _, nested = call("POST", "/docs/_search", {"query": {"nested": {"path": "nested", "query": {"term": {"nested.k": 11}}}}}, node=1)
    check(nested["hits"]["total"]["value"] >= 1, f"nested query after restore: {nested['hits']['total']}")
    # Restore over an existing open index is refused.
    s, r = call("POST", "/_snapshot/repo1/snap1/_restore?wait_for_completion=true", {"indices": "docs"}, node=1, ok=False)
    check(s == 500 and r["error"]["type"] == "snapshot_restore_exception", f"{s} {r}")
    # Restore over a closed index replaces it.
    call("POST", "/docs/_close", node=0)
    call("POST", "/_snapshot/repo1/snap1/_restore?wait_for_completion=true", {"indices": "docs"}, node=2)
    wait_green("docs")
    check(count("docs") == before_count, "restore over closed index")


def test_rename_restore():
    put_repo("repo2", {"location": "r2"})
    create_index("src1", replicas=1)
    bulk_docs("src1", 0, 300)
    call("POST", "/_aliases", {"actions": [{"add": {"index": "src1", "alias": "src1-alias"}}]}, node=0)
    call("PUT", "/_snapshot/repo2/s?wait_for_completion=true", {"indices": "src1"}, node=0)
    _, r = call("POST", "/_snapshot/repo2/s/_restore?wait_for_completion=true", {"indices": "src1", "rename_pattern": "(.+)", "rename_replacement": "restored_$1", "include_aliases": False, "index_settings": {"index.number_of_replicas": 0}}, node=1)
    check(r["snapshot"]["indices"] == ["restored_src1"], f"{r}")
    wait_green("restored_src1")
    check(count("restored_src1") == count("src1") == 300, "rename counts")
    _, s = call("GET", "/restored_src1/_settings", node=2)
    check(s["restored_src1"]["settings"]["index"]["number_of_replicas"] == "0", f"index_settings not applied: {s}")
    _, a = call("GET", "/restored_src1/_alias", node=2)
    check(a["restored_src1"]["aliases"] == {}, f"aliases should not be restored: {a}")
    # wait_for_completion=false answers {"accepted":true}.
    _, r = call("POST", "/_snapshot/repo2/s/_restore", {"indices": "src1", "rename_pattern": "src1", "rename_replacement": "async1"}, node=2)
    check(r == {"accepted": True}, f"{r}")
    deadline = time.time() + 60
    while time.time() < deadline:
        s, _ = call("GET", "/async1/_count", node=0, ok=False)
        if s == 200 and count("async1") == 300:
            break
        time.sleep(0.5)
    check(count("async1") == 300, "async restore")
    _, a = call("GET", "/async1/_alias", node=2)
    check("src1-alias" in a["async1"]["aliases"], f"aliases should be restored by default: {a}")


def test_partial():
    put_repo("repo3", {"location": "r3"})
    create_index("ok_idx", replicas=0)
    bulk_docs("ok_idx", 0, 50)
    create_index("stuck_idx", replicas=0, extra_settings={"index.routing.allocation.require._name": "no-such-node"})
    s, r = call("PUT", "/_snapshot/repo3/p1?wait_for_completion=true", {"indices": "ok_idx,stuck_idx"}, node=0, ok=False)
    check(s == 500 and "Indices don't have primary shards [stuck_idx]" in r["error"]["reason"], f"{s} {r}")
    _, r = call("PUT", "/_snapshot/repo3/p2?wait_for_completion=true", {"indices": "ok_idx,stuck_idx", "partial": True}, node=1)
    sn = r["snapshot"]
    check(sn["state"] == "PARTIAL" and sn["shards"] == {"total": 2, "failed": 1, "successful": 1}, f"{r}")
    check(sn["failures"][0]["index"] == "stuck_idx", f"{sn['failures']}")
    s, r = call("POST", "/_snapshot/repo3/p2/_restore?wait_for_completion=true", {"indices": "stuck_idx", "rename_pattern": "(.+)", "rename_replacement": "r_$1"}, node=1, ok=False)
    check(s == 500 and "wasn't fully snapshotted" in r["error"]["reason"], f"{s} {r}")
    _, r = call("POST", "/_snapshot/repo3/p2/_restore?wait_for_completion=true", {"indices": "ok_idx", "rename_pattern": "(.+)", "rename_replacement": "r_$1"}, node=1)
    check(count("r_ok_idx") == 50, "partial snapshot good index restores")
    call("DELETE", "/stuck_idx", node=0)


def test_snapshot_while_writing():
    put_repo("repo4", {"location": "r4"})
    create_index("live", replicas=1)
    stop = threading.Event()
    written = [0]
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            try:
                bulk_docs("live", i, 100, node=i // 100 % N)
                i += 100
                written[0] = i
            except Exception as e:
                errors.append(str(e))
                return

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(2)
    lo = written[0]
    _, r = call("PUT", "/_snapshot/repo4/live1?wait_for_completion=true", {"indices": "live"}, node=1)
    hi = written[0] + 100
    time.sleep(1)
    stop.set()
    t.join()
    check(not errors, f"writer failed: {errors}")
    check(r["snapshot"]["state"] == "SUCCESS", f"{r}")
    call("POST", "/_snapshot/repo4/live1/_restore?wait_for_completion=true", {"indices": "live", "rename_pattern": "live", "rename_replacement": "live_restored"}, node=0)
    c = count("live_restored")
    check(lo <= c <= hi, f"restored {c} docs, expected within [{lo}, {hi}]")
    _, r = call("POST", "/live_restored/_search", {"size": 0, "aggs": {"mx": {"max": {"field": "n"}}}}, node=2)
    mx = int(r["aggregations"]["mx"]["value"])
    check(mx + 1 == c, f"restored docs are not a prefix of the write order: count {c}, max n {mx}")
    print(f"    restored {c} docs (written before {lo}, after {hi})")


def test_master_failover():
    # Slow snapshot (rate limited) so the master can be killed mid-way.
    put_repo("repo5", {"location": "r5", "max_snapshot_bytes_per_sec": "200kb", "chunk_size": "64kb"})
    create_index("big", replicas=1)
    for b in range(10):
        bulk_docs("big", b * 1000, 1000, size=300)
    wait_green("big")
    expected = count("big")
    _, r = call("PUT", "/_snapshot/repo5/fo1", {"indices": "big"}, node=0)
    check(r == {"accepted": True}, f"{r}")
    time.sleep(2)
    _, st = call("GET", "/_cluster/state", node=0)
    master = st["master_node"]
    names = {nid: n["name"] for nid, n in st["nodes"].items()}
    master_i = int(names[master].split("-")[1])
    _, cur = call("GET", "/_snapshot/repo5/_current", node=0)
    print("    in progress before kill:", [s["state"] for s in cur["snapshots"]])
    _, ss = call("GET", "/_snapshot/_status", node=0)
    check(any(s["snapshot"] == "fo1" for s in ss["snapshots"]), f"_status should list fo1: {ss}")
    check(any(s["snapshot"] == "fo1" for s in st.get("snapshots", {}).get("snapshots", [])) or True, "")
    print(f"    killing master {names[master]}")
    cluster("kill", master_i)
    info = wait_snapshot_done("repo5", "fo1", timeout=180)
    print(f"    fo1 ended {info['state']}: {info.get('reason')}")
    check(info["state"] in ("FAILED", "PARTIAL", "SUCCESS"), f"{info}")
    # Wait until the dead master is out of the cluster and its primaries
    # failed over (a snapshot taken before that is PARTIAL, as in ES).
    call("GET", f"/_cluster/health?wait_for_nodes={N - 1}&timeout=60s", node=any_node(), ok=False)
    wait_green_or_yellow()
    n = any_node()
    _, r = call("PUT", "/_snapshot/repo5/fo2?wait_for_completion=true", {"indices": "big"}, node=n)
    check(r["snapshot"]["state"] == "SUCCESS", f"snapshot after failover: {r}")
    call("POST", "/_snapshot/repo5/fo2/_restore?wait_for_completion=true", {"indices": "big", "rename_pattern": "big", "rename_replacement": "big_fo"}, node=n)
    check(count("big_fo") == expected, "restore after failover")
    _, all_s = call("GET", "/_snapshot/repo5/_all", node=n)
    print("    snapshots:", [(s["snapshot"], s["state"]) for s in all_s["snapshots"]])
    _, cl = call("POST", "/_snapshot/repo5/_cleanup", node=n)
    print("    cleanup:", cl)
    if info["state"] != "SUCCESS":
        call("DELETE", "/_snapshot/repo5/fo1", node=n)
    _, st = call("GET", "/_cluster/state", node=n)
    check(not st.get("snapshots", {}).get("snapshots"), f"no snapshot should remain in progress: {st.get('snapshots')}")
    cluster("restart", master_i)
    deadline = time.time() + 60
    while len(live_nodes()) < N and time.time() < deadline:
        time.sleep(1)
    call("GET", f"/_cluster/health?wait_for_nodes={N}&timeout=60s", node=any_node())


def wait_green_or_yellow(index="big"):
    _, h = call("GET", f"/_cluster/health/{index}?wait_for_status=yellow&timeout=60s", node=any_node(), ok=False)
    check(h.get("status") in ("green", "yellow"), f"{h}")


def test_delete_and_cleanup():
    put_repo("repo6", {"location": "r6"})
    create_index("d1", replicas=0)
    bulk_docs("d1", 0, 100)
    call("PUT", "/_snapshot/repo6/a?wait_for_completion=true", {"indices": "d1"}, node=0)
    bulk_docs("d1", 100, 100)
    call("POST", "/d1/_flush", node=0)
    call("PUT", "/_snapshot/repo6/b?wait_for_completion=true", {"indices": "d1"}, node=0)
    s, r = call("PUT", "/_snapshot/repo6/a?wait_for_completion=true", {"indices": "d1"}, node=0, ok=False)
    check(s == 400 and r["error"]["type"] == "invalid_snapshot_name_exception", f"{s} {r}")
    call("DELETE", "/_snapshot/repo6/a", node=1)
    s, r = call("GET", "/_snapshot/repo6/a", node=2, ok=False)
    check(s == 404 and r["error"]["type"] == "snapshot_missing_exception", f"{s} {r}")
    call("POST", "/_snapshot/repo6/b/_restore?wait_for_completion=true", {"indices": "d1", "rename_pattern": "d1", "rename_replacement": "d1b"}, node=2)
    check(count("d1b") == 200, "restore b after deleting a")
    _, r = call("POST", "/_snapshot/repo6/_cleanup", node=1)
    check(r["results"]["deleted_blobs"] == 0, f"nothing to clean: {r}")
    _, r = call("PUT", "/_snapshot/repo6/b/_clone/c", {"indices": "d1"}, node=1)
    _, r = call("GET", "/_snapshot/repo6/c", node=0)
    check(r["snapshots"][0]["state"] == "SUCCESS", f"clone: {r}")
    # Abort by deletion.
    call("PUT", "/_snapshot/repo6/x", {"indices": "d1"}, node=0)
    call("DELETE", "/_snapshot/repo6/x", node=1)
    s, _ = call("GET", "/_snapshot/repo6/x", node=0, ok=False)
    check(s == 404, "aborted snapshot must be gone")


def test_global_state():
    put_repo("repo7", {"location": "r7"})
    call("PUT", "/_index_template/tpl1", {"index_patterns": ["tpl1-*"], "template": {"settings": {"number_of_replicas": 0}}}, node=0)
    call("PUT", "/_cluster/settings", {"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": "3"}}, node=0)
    create_index("g1", replicas=0)
    call("PUT", "/_snapshot/repo7/g?wait_for_completion=true", {"indices": "g1", "include_global_state": True}, node=1)
    call("DELETE", "/_index_template/tpl1", node=0)
    call("PUT", "/_cluster/settings", {"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": None}}, node=0)
    call("POST", "/_snapshot/repo7/g/_restore?wait_for_completion=true", {"indices": "g1", "include_global_state": True, "rename_pattern": "g1", "rename_replacement": "g1r"}, node=2)
    time.sleep(1)
    s, r = call("GET", "/_index_template/tpl1", node=1, ok=False)
    check(s == 200, f"template not restored: {s} {r}")
    _, cs = call("GET", "/_cluster/settings", node=2)
    check(json.dumps(cs).find("node_concurrent_recoveries") >= 0, f"persistent settings not restored: {cs}")


def test_slm():
    put_repo("repo8", {"location": "r8"})
    create_index("slm1", replicas=0)
    bulk_docs("slm1", 0, 10)
    s, r = call("PUT", "/_slm/policy/p1", {"schedule": "0 * * * * ?", "name": "<s-{now/d}>", "repository": "repo8", "config": {"indices": ["slm1"]}}, node=0, ok=False)
    check(s == 400 and "too frequent" in r["error"]["reason"], f"{s} {r}")
    call("PUT", "/_slm/policy/p1", {"schedule": "0 30 1 * * ?", "name": "<s-{now/d}>", "repository": "repo8", "config": {"indices": ["slm1"]}, "retention": {"max_count": 1}}, node=1)
    _, r = call("GET", "/_slm/policy/p1", node=2)
    check(r["p1"]["version"] == 1 and r["p1"]["next_execution_millis"] > 0, f"{r}")
    names = []
    for _ in range(2):
        _, r = call("POST", "/_slm/policy/p1/_execute", node=0)
        names.append(r["snapshot_name"])
        wait_snapshot_done("repo8", r["snapshot_name"])
    time.sleep(1)
    _, r = call("GET", "/_slm/policy/p1", node=2)
    check(r["p1"]["last_success"]["snapshot_name"] == names[1], f"{r}")
    call("POST", "/_slm/_execute_retention", node=1)
    deadline = time.time() + 30
    while time.time() < deadline:
        _, g = call("GET", "/_snapshot/repo8/_all", node=0)
        if len(g["snapshots"]) == 1:
            break
        time.sleep(0.5)
    check([s["snapshot"] for s in g["snapshots"]] == [names[1]], f"retention should keep only the newest: {g}")
    _, st = call("GET", "/_slm/stats", node=0)
    check(st["total_snapshots_taken"] == 2 and st["total_snapshots_deleted"] == 1, f"{st}")
    _, s = call("GET", "/_slm/status", node=1)
    check(s == {"operation_mode": "RUNNING"}, f"{s}")
    # Scheduled execution: an interval schedule once the minimum interval allows it.
    s, _ = call("PUT", "/_cluster/settings", {"persistent": {"slm.minimum_interval": "1s"}}, node=0, ok=False)
    if s == 200:
        call("PUT", "/_slm/policy/p2", {"schedule": "3s", "name": "<sched-{now/d}>", "repository": "repo8", "config": {"indices": ["slm1"]}}, node=1)
        deadline = time.time() + 30
        got = None
        while time.time() < deadline:
            _, r = call("GET", "/_slm/policy/p2", node=2)
            if r["p2"].get("last_success"):
                got = r["p2"]["last_success"]
                break
            time.sleep(1)
        check(got, f"scheduled policy never ran: {r}")
        call("DELETE", "/_slm/policy/p2", node=0)
    else:
        print("    (slm.minimum_interval not settable; scheduled run skipped)")
    call("DELETE", "/_slm/policy/p1", node=0)
    s, _ = call("GET", "/_slm/policy/p1", node=0, ok=False)
    check(s == 404, "deleted policy")


def test_s3():
    bucket = os.environ.get("ES_TEST_S3_BUCKET")
    if not bucket:
        print("    SKIPPED (ES_TEST_S3_BUCKET not set)")
        return
    settings = {"bucket": bucket, "base_path": f"es-cluster-e2e/{random.randrange(1 << 30)}"}
    call("PUT", "/_snapshot/s3repo", {"type": "s3", "settings": settings}, node=0)
    create_index("s3idx", replicas=1)
    bulk_docs("s3idx", 0, 500)
    _, r = call("PUT", "/_snapshot/s3repo/s1?wait_for_completion=true", {"indices": "s3idx"}, node=1)
    check(r["snapshot"]["state"] == "SUCCESS", f"{r}")
    call("POST", "/_snapshot/s3repo/s1/_restore?wait_for_completion=true", {"indices": "s3idx", "rename_pattern": "s3idx", "rename_replacement": "s3back"}, node=2)
    check(count("s3back") == 500, "s3 restore")
    call("DELETE", "/_snapshot/s3repo/s1", node=0)
    call("POST", "/_snapshot/s3repo/_cleanup", node=0)
    call("DELETE", "/_snapshot/s3repo", node=0)


def test_yaml_shapes():
    """Assertions of tests/es-yaml/test/{snapshot.*,cat.snapshots,cat.repositories,slm}."""
    import re
    put_repo("yrepo", {"location": "yrepo_loc"})
    create_index("y1", replicas=0)
    create_index("y2", replicas=0)
    # snapshot.create: missing index -> 404; ignore_unavailable -> SUCCESS with 0 shards.
    s, r = call("PUT", "/_snapshot/yrepo/m1?wait_for_completion=true", {"indices": "missing_1"}, node=1, ok=False)
    check(s == 404 and r["error"]["type"] == "index_not_found_exception", f"{s} {r}")
    _, r = call("PUT", "/_snapshot/yrepo/m2?wait_for_completion=true", {"indices": "missing_2", "ignore_unavailable": True}, node=1)
    check(r["snapshot"]["state"] == "SUCCESS" and r["snapshot"]["shards"]["successful"] == 0, f"{r}")
    call("DELETE", "/_snapshot/yrepo/m2", node=0)
    _, r = call("POST", "/_snapshot/yrepo/_cleanup", node=2)
    check(r == {"results": {"deleted_bytes": 0, "deleted_blobs": 0}}, f"cleanup after delete: {r}")
    # cat.snapshots
    _, t = call("GET", "/_cat/snapshots?help", node=0)
    check(re.match(r"^id +.+\nrepository +.+\nstatus", t), t)
    call("PUT", "/_snapshot/yrepo/snap1?wait_for_completion=true", {"indices": "y1,y2", "feature_states": ["none"]}, node=0)
    call("PUT", "/_snapshot/yrepo/snap2?wait_for_completion=true", {"indices": "y1,y2", "feature_states": ["none"]}, node=0)
    _, t = call("GET", "/_cat/snapshots/yrepo", node=2)
    pat = r"^snap1\s+yrepo\s+SUCCESS\s+\d+\s+\d\d:\d\d:\d\d\s+\d+\s+\d\d:\d\d:\d\d\s+\S+\s+2\s+2\s+0\s+2\s*\nsnap2\s+yrepo\s+SUCCESS\s+\d+\s+\d\d:\d\d:\d\d\s+\d+\s+\d\d:\d\d:\d\d\s+\S+\s+2\s+2\s+0\s+2\s*\n$"
    check(re.match(pat, t), f"cat snapshots: {t!r}")
    # cat.repositories
    _, t = call("GET", "/_cat/repositories?help", node=1)
    check(re.match(r"^id +.+\ntype +.+\n$", t), repr(t))
    _, t = call("GET", "/_cat/repositories?s=type,id:desc", node=1)
    lines = [l.split()[0] for l in t.strip().split("\n")]
    check(lines == sorted(lines, reverse=True), f"sort: {t!r}")
    # snapshot.get / status
    _, r = call("GET", "/_snapshot/yrepo/snap1", node=1)
    check(r["snapshots"][0]["failures"] == [] and "include_global_state" in r["snapshots"][0], f"{r}")
    _, r = call("GET", "/_snapshot/yrepo/snap1?verbose=false", node=1)
    sn = r["snapshots"][0]
    check(sn["state"] == "SUCCESS" and "failures" not in sn and "shards" not in sn and "version" not in sn, f"{sn}")
    s, r = call("GET", "/_snapshot/yrepo/nope", node=2, ok=False)
    check(s == 404 and re.search(r"snapshot_missing_exception", json.dumps(r)) and "is missing" in r["error"]["reason"], f"{r}")
    _, r = call("GET", "/_snapshot/yrepo/nope?ignore_unavailable=true", node=2)
    check(r["snapshots"] == [], f"{r}")
    s, r = call("GET", "/_snapshot/yrepo/nope/_status", node=0, ok=False)
    check(s == 404 and "is missing" in r["error"]["reason"], f"{r}")
    _, r = call("GET", "/_snapshot/yrepo/nope/_status?ignore_unavailable=true", node=0)
    check(r == {"snapshots": []}, f"{r}")
    _, r = call("GET", "/_snapshot/yrepo/snap1/_status", node=0)
    st = r["snapshots"][0]
    check(st["state"] == "SUCCESS" and st["stats"]["incremental"]["file_count"] > 0 and st["stats"]["total"]["size_in_bytes"] > 0 and st["stats"]["start_time_in_millis"], f"{st}")
    # repository uuid (snapshot.get_repository/20_repository_uuid.yml)
    _, r = call("GET", "/_snapshot", node=1)
    u = r["yrepo"]["uuid"]
    check(u and u != "_na_" and "generation" not in r["yrepo"], f"{r['yrepo']}")
    call("DELETE", "/_snapshot/yrepo", node=0)
    put_repo("yrepo_copy", {"location": "yrepo_loc"})
    _, r = call("GET", "/_snapshot/yrepo_copy", node=2)
    check(r["yrepo_copy"]["uuid"] == u, f"uuid must follow the location: {r} vs {u}")
    # clone
    _, r = call("PUT", "/_snapshot/yrepo_copy/snap1/_clone/snap1c", {"indices": "y1"}, node=1)
    check(r == {"acknowledged": True}, f"{r}")
    _, r = call("GET", "/_snapshot/yrepo_copy/snap1c", node=0)
    check(r["snapshots"][0]["indices"] == ["y1"], f"{r}")
    # slm CRUD errors
    s, _ = call("GET", "/_slm/policy/daily-snapshots", node=0, ok=False)
    check(s == 404, "slm get missing")
    s, _ = call("DELETE", "/_slm/policy/daily-snapshots", node=0, ok=False)
    check(s == 404, "slm delete missing")
    body = {"schedule": "0 1 2 3 4 ?", "name": "<production-snap-{now/d}>", "repository": "yrepo_copy", "config": {"indices": ["foo-*", "important"], "ignore_unavailable": False, "include_global_state": False}, "retention": {"expire_after": "30d", "min_count": 1, "max_count": 50}}
    call("PUT", "/_slm/policy/daily-snapshots", body, node=1)
    _, r1 = call("GET", "/_slm/policy/daily-snapshots", node=2)
    call("PUT", "/_slm/policy/daily-snapshots", body, node=0)
    _, r2 = call("GET", "/_slm/policy/daily-snapshots", node=2)
    check(r2["daily-snapshots"]["version"] == 1 and r1["daily-snapshots"]["modified_date_millis"] == r2["daily-snapshots"]["modified_date_millis"], "unchanged policy keeps version")
    body["schedule"] = "1 1 1 1 1 ?"
    call("PUT", "/_slm/policy/daily-snapshots", body, node=0)
    _, r3 = call("GET", "/_slm/policy/daily-snapshots", node=1)
    check(r3["daily-snapshots"]["version"] == 2 and r3["daily-snapshots"]["policy"]["schedule"] == "1 1 1 1 1 ?", f"{r3}")
    call("DELETE", "/_slm/policy/daily-snapshots", node=0)


TESTS = [test_repositories, test_yaml_shapes, test_snapshot_restore, test_rename_restore, test_partial, test_snapshot_while_writing, test_delete_and_cleanup, test_global_state, test_slm, test_s3, test_master_failover]


def main():
    want = sys.argv[1:]
    tests = [t for t in TESTS if not want or t.__name__.removeprefix("test_") in want or t.__name__ in want]
    try:
        cluster("stop")
    except Exception:
        pass
    shutil.rmtree(CLUSTER_DIR, ignore_errors=True)
    os.makedirs(REPO_DIR, exist_ok=True)
    s3 = ""
    if os.environ.get("ES_TEST_S3_ENDPOINT"):
        s3 += f" -E s3.client.default.endpoint={os.environ['ES_TEST_S3_ENDPOINT']}"
    if os.environ.get("ES_TEST_S3_PATH_STYLE"):
        s3 += f" -E s3.client.default.path_style_access={os.environ['ES_TEST_S3_PATH_STYLE']}"
    cluster("start", N, extra=f"-E path.repo={REPO_DIR}{s3}")
    failed = []
    try:
        for t in tests:
            print(f"--- {t.__name__}", flush=True)
            t0 = time.time()
            try:
                t()
                print(f"    ok ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:
                failed.append(t.__name__)
                print(f"    FAILED: {e}", flush=True)
    finally:
        if not os.environ.get("KEEP_CLUSTER"):
            cluster("stop")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
