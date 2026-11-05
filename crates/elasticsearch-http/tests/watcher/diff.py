#!/usr/bin/env python3
"""Differential watcher test: runs one request sequence against a reference
Elasticsearch and the Rust server, normalizes volatile values and diffs the
JSON responses (key order insensitive).

usage: diff.py [REF=http://localhost:9259] [RS=http://localhost:19259]
"""
import json, re, sys, urllib.request, urllib.error, time, random

RUN = "-r%d" % random.randint(1000, 999999)
IDS = ["w1", "w2", "w3", "s1", "t1", "nope"]

REF = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9259"
RS = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:19259"

def sub(text):
    for i in IDS:
        text = re.sub(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_\-])' % i, i + RUN, text)
    return text

def call(base, method, path, body=None):
    path = sub(path)
    data = None if body is None else sub(body if isinstance(body, str) else json.dumps(body)).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            t = r.read().decode(); st = r.status
    except urllib.error.HTTPError as e:
        t = e.read().decode(); st = e.code
    try:
        return st, json.loads(t) if t else None
    except Exception:
        return st, t

TS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
VOLATILE = {"headers", "_headers", "took", "execution_duration", "node", "node_id", "index_uuid", "_seq_no", "cluster_name", "max_size",
            "total_time_in_ms", "max_score", "_score", "_primary_term", "cluster_uuid", "shard"}

def norm(v, key=None):
    if isinstance(v, dict):
        return {k: norm(x, k) for k, x in v.items() if not (k in VOLATILE)}
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, str):
        v = v.replace(RUN, "")
        v = re.sub(r"seqNo \[\d+\]", "seqNo [<n>]", v)
        v = re.sub(r"elapsed since last execution is \[[^\]]*\]", "elapsed since last execution is [<t>]", v)
        v = UUID.sub("<uuid>", v)
        v = TS.sub("<ts>", v)
        return v
    return v

def diff(a, b, path=""):
    out = []
    if type(a) != type(b):
        return [f"{path}: ref={json.dumps(a)[:200]} rs={json.dumps(b)[:200]}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: out.append(f"{path}/{k}: only in rs = {json.dumps(b[k])[:160]}")
            elif k not in b: out.append(f"{path}/{k}: only in ref = {json.dumps(a[k])[:160]}")
            else: out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b): out.append(f"{path}: len ref={len(a)} rs={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)): out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: ref={json.dumps(a)[:200]} rs={json.dumps(b)[:200]}")
    return out

def W(i): return {"trigger": {"schedule": {"interval": "1h"}}, **i}

import threading, http.server

class _H(http.server.BaseHTTPRequestHandler):
    def _do(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        out = json.dumps({"ok": True, "n": 5, "echo": body, "path": self.path, "method": self.command}).encode()
        code = 500 if self.path.startswith("/fail") else 200
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    do_GET = do_POST = do_PUT = _do
    def log_message(self, *a): pass

def serve_helper():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 19301), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

HTTP_STEPS = [
    ("POST", "/_watcher/watch/_execute", {"watch": {"trigger": {"schedule": {"interval": "1h"}},
        "input": {"http": {"request": {"host": "127.0.0.1", "port": 19301, "path": "/data", "params": {"q": "x"}}, "extract": ["n", "path"]}},
        "condition": {"compare": {"ctx.payload.n": {"eq": 5}}},
        "actions": {"w": {"webhook": {"method": "POST", "host": "127.0.0.1", "port": 19301, "path": "/hook/{{ctx.watch_id}}", "body": "n={{ctx.payload.n}}"}}}}}),
    ("POST", "/_watcher/watch/_execute", {"watch": {"trigger": {"schedule": {"interval": "1h"}},
        "input": {"http": {"request": {"url": "http://127.0.0.1:19301/data2"}, "response_content_type": "json"}},
        "actions": {"w": {"webhook": {"method": "POST", "host": "127.0.0.1", "port": 19301, "path": "/fail", "body": "x"}}}}}),
    ("POST", "/_watcher/watch/_execute", {"watch": {"trigger": {"schedule": {"interval": "1h"}},
        "input": {"http": {"request": {"host": "127.0.0.1", "port": 1, "path": "/"}}},
        "actions": {"l": {"logging": {"text": "x"}}}}}),
]

STEPS = [
    ("PUT", "/_watcher/watch/w3", W({})),
    ("DELETE", "/_watcher/watch/w3", None),
    ("DELETE", "/_watcher/watch/nope", None),
    ("PUT", "/src/_doc/1?refresh=true", {"msg": "a", "n": 5}),
    ("PUT", "/src/_doc/2?refresh=true", {"msg": "b", "n": 9}),
    ("PUT", "/_watcher/watch/w1", W({"input": {"simple": {"a": 1}}, "condition": {"always": {}}, "actions": {"l": {"logging": {"text": "hi {{ctx.payload.a}}"}}}})),
    ("GET", "/_watcher/watch/w1", None),
    ("GET", "/_watcher/watch/nope", None),
    ("PUT", "/_watcher/watch/w1?if_seq_no=99&if_primary_term=1", W({})),
    ("PUT", "/_watcher/watch/w1?version=3", W({})),
    ("PUT", "/_watcher/watch/w3", {"input": {"simple": {}}}),
    ("PUT", "/_watcher/watch/w3", W({"input": {"foo": {}}})),
    ("PUT", "/_watcher/watch/w3", W({"condition": {"foo": {}}})),
    ("PUT", "/_watcher/watch/w3", W({"actions": {"a": {"foo": {}}}})),
    ("PUT", "/_watcher/watch/w3", {"trigger": {"schedule": {"interval": "0s"}}}, "noroot"),
    ("PUT", "/_watcher/watch/w3", {"trigger": {"foo": {}}}),
    ("PUT", "/_watcher/watch/w3", W({"bogus": 1})),
    ("PUT", "/_watcher/watch/w3", W({"actions": {"s": {"slack": {"message": {"text": "t"}}}}})),
    ("POST", "/_watcher/watch/w1/_execute", {}),
    ("POST", "/_watcher/watch/w1/_execute", {"record_execution": True}),
    ("POST", "/_watcher/watch/w1/_execute", {"ignore_condition": True, "action_modes": {"l": "force_simulate"}}),
    ("POST", "/_watcher/watch/w1/_execute", {"alternative_input": {"a": 42}}),
    ("POST", "/_watcher/watch/w1/_execute", {"action_modes": {"_all": "skip"}}),
    ("POST", "/_watcher/watch/w1/_execute", {"debug": True}),
    ("POST", "/_watcher/watch/nope/_execute", {}),
    ("GET", "/_watcher/watch/w1", None),
    ("POST", "/_watcher/watch/w1/_ack", None),
    ("POST", "/_watcher/watch/w1/_ack/zzz", None),
    ("POST", "/_watcher/watch/nope/_ack", None),
    ("PUT", "/_watcher/watch/nope/_activate", None),
    ("PUT", "/_watcher/watch/w1/_deactivate", None),
    ("PUT", "/_watcher/watch/w1/_deactivate", None),
    ("PUT", "/_watcher/watch/w1/_activate", None),
    ("PUT", "/_watcher/watch/s1", {"metadata": {"k": "v"}, "trigger": {"schedule": {"interval": "1h"}},
        "input": {"search": {"request": {"indices": ["src"], "body": {"query": {"match_all": {}}}}}},
        "condition": {"compare": {"ctx.payload.hits.total": {"gte": 2}}},
        "transform": {"script": {"source": "return [\"c\": ctx.payload.hits.total]"}},
        "actions": {"idx": {"index": {"index": "out", "doc_id": "x1", "execution_time_field": "at"}},
                    "lg": {"logging": {"text": "n={{ctx.payload.c}} {{ctx.metadata.k}}"}},
                    "wh": {"webhook": {"method": "POST", "host": "127.0.0.1", "port": 1, "path": "/x/{{ctx.watch_id}}", "body": "b"}}}}),
    ("GET", "/_watcher/watch/s1", None),
    ("POST", "/_watcher/watch/s1/_execute", {"record_execution": True}),
    ("POST", "/_watcher/watch/s1/_execute", {"action_modes": {"_all": "force_simulate"}, "ignore_condition": True}),
    ("POST", "/_watcher/watch/s1/_execute", {"record_execution": True, "action_modes": {"lg": "skip", "idx": "simulate"}}),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"v": [1, 2, 3]}}, "condition": {"script": "return ctx.payload.v.size() > 2"},
        "actions": {"a": {"foreach": "ctx.payload.v", "max_iterations": 2, "logging": {"text": "v={{ctx.payload}}"}}}}),
        "alternative_input": {"v": [4, 5, 6]}, "trigger_data": {"triggered_time": "2020-01-01T00:00:00Z", "scheduled_time": "2020-01-01T00:00:00Z"}}),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"n": 1}}, "condition": {"compare": {"ctx.payload.n": {"gt": 5}}}})}),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"n": 1}}, "condition": {"never": {}}})}),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"arr": [{"v": 1}, {"v": 5}]}}, "condition": {"array_compare": {"ctx.payload.arr": {"path": "v", "gte": {"value": 5, "quantifier": "some"}}}}})}),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"n": 1}}, "transform": {"chain": [{"script": "return ['x': ctx.payload.n + 1]"}, {"script": {"source": "return ['y': ctx.payload.x * 2]"}}]}})}),
    ("PUT", "/_watcher/watch/t1", W({"throttle_period": "1h", "input": {"simple": {"a": 1}}, "actions": {"l": {"logging": {"text": "x"}}}})),
    ("POST", "/_watcher/watch/t1/_execute", {"record_execution": True}),
    ("POST", "/_watcher/watch/t1/_execute", {"record_execution": True}),
    ("POST", "/_watcher/watch/t1/_ack", None),
    ("POST", "/_watcher/watch/t1/_execute", {"record_execution": True}),
    ("POST", "/_watcher/watch/t1/_execute", {"record_execution": True, "action_modes": {"l": "force_execute"}}),
    ("PUT", "/_watcher/watch/w2?active=false", W({})),
    ("GET", "/_watcher/watch/w2", None),
    ("POST", "/_watcher/_query/watches", {}, "unordered"),
    ("POST", "/_watcher/_query/watches", {"query": {"term": {"_id": "s1"}}}),
    ("GET", "/_watcher/stats", None),
    ("GET", "/_watcher/stats/_all", None),
    ("GET", "/_watcher/stats/queued_watches", None),
    ("GET", "/_watcher/stats/current_watches", None),
    ("GET", "/_watcher/stats/bogus", None),
    ("GET", "/_watcher/settings", None),
    ("PUT", "/_watcher/settings", {"index.auto_expand_replicas": "0-2"}),
    ("GET", "/_watcher/settings", None),
    ("PUT", "/_watcher/settings", {"index.bogus": 1}),
    ("POST", "/_watcher/_stop", None),
    ("GET", "/_watcher/stats", None),
    ("POST", "/_watcher/_start", None),
    ("GET", "/_watcher/stats", None, "nostats"),
    *HTTP_STEPS,
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"simple": {"a": 1}}, "condition": {"script": "return foo.bar"}})}, "errtype"),
    ("PUT", "/_watcher/watch/w3", W({"transform": {"script": "return foo.bar"}}), "errtype"),
    ("POST", "/_watcher/watch/_execute", {"watch": W({"input": {"search": {"request": {"indices": ["nonexist"], "body": {}}}}, "actions": {"l": {"logging": {"text": "x"}}}})}),
    ("GET", "/_xpack/usage?filter_path=watcher.watch,watcher.count,watcher.available,watcher.enabled", None),
    ("POST", "/.watcher-history-17*/_refresh", None, "nobody"),
    ("POST", "/.watcher-history-17*/_search?size=100&sort=@timestamp:asc&filter_path=hits.hits._source", {"query": {"terms": {"watch_id": ["w1", "s1", "t1", "_inlined_"]}}}),
    ("DELETE", "/_watcher/watch/w2", None, "nover"),
    ("DELETE", "/_watcher/watch/w2", None, "nover"),
]

# steps whose differences are known/inherent (transient state, JVM-specific values)

def main():
    serve_helper()
    for b in (REF, RS):
        call(b, "DELETE", "/src,out"); call(b, "PUT", "/_watcher/settings", {"index.auto_expand_replicas": "0-1"})
        st, q = call(b, "POST", "/_watcher/_query/watches", {"size": 100})
        for w in (q or {}).get("watches", []) if isinstance(q, dict) else []:
            call(b, "DELETE", "/_watcher/watch/" + w["_id"])
    bad = 0
    for i, step in enumerate(STEPS):
        m, p, b = step[:3]
        flags = step[3] if len(step) > 3 else ""
        r = call(REF, m, p, b); s = call(RS, m, p, b)
        d = [] if r[0] == s[0] else [f"status ref={r[0]} rs={s[0]}"]
        nr, ns = norm(r[1]), norm(s[1])
        if flags == "unordered":
            for x in (nr, ns):
                if isinstance(x, dict) and isinstance(x.get("watches"), list):
                    x["watches"].sort(key=lambda w: w.get("_id", ""))
        if flags == "errtype":
            # painless diagnostics (symbol/offset wording) are owned by the script module
            for x in (nr, ns):
                if isinstance(x, dict) and isinstance(x.get("error"), dict):
                    x["error"] = {"type": x["error"].get("type"), "reason": x["error"].get("reason")}
        if flags == "nobody":
            nr = ns = None
        for flag, key in (("nostats", "stats"), ("nover", "_version")):
            if flag == flags:
                for x in (nr, ns):
                    if isinstance(x, dict): x.pop(key, None)
        if flags == "noroot":
            for x in (nr, ns):
                if isinstance(x, dict) and isinstance(x.get("error"), dict): x["error"].pop("root_cause", None)
        d += diff(nr, ns)
        if d:
            bad += 1
            print(f"#{i} {m} {p} {json.dumps(b)[:120] if b is not None else ''}")
            for x in d[:12]: print("    " + x)
    print(f"{len(STEPS) - bad}/{len(STEPS)} steps identical")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
