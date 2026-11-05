#!/usr/bin/env python3
"""P14 Painless runtime-library probe: diffs corpus.py through
POST /_scripts/painless/_execute on a reference ES 8.19 and on this server
(same comparison as ../painless_diff/run.py).

  ES_REF=http://localhost:9420 ES_OURS=http://localhost:9421 python3 run.py [-v] [-k substr] [--json out.json]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("painless_diff_run", os.path.join(HERE, "..", "painless_diff", "run.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
from corpus import CASES  # noqa: E402

base.REF = os.environ.get("ES_REF", "http://localhost:9420")
base.OURS = os.environ.get("ES_OURS", "http://localhost:9421")


def main():
    verbose = "-v" in sys.argv
    sub = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    for b in (base.REF, base.OURS):
        base.setup(b)
    ok, bad, rows = 0, [], {}
    for case in CASES:
        if sub and sub not in case[0]:
            continue
        body = base.request(case)
        r = base.http(base.REF, "POST", "/_scripts/painless/_execute", body)
        o = base.http(base.OURS, "POST", "/_scripts/painless/_execute", body)
        same = r[0] == o[0] and base.norm(r[1]) == base.norm(o[1])
        rows[case[0]] = {"src": case[2], "ok": same, "ref": [r[0], r[1]], "ours": [o[0], o[1]]}
        if same:
            ok += 1
        else:
            bad.append((case[0], case[2], r, o))
    print(f"PASS {ok}/{ok + len(bad)}  FAIL {len(bad)}")
    for name, src, r, o in bad:
        print(f"--- {name}: {src!r}")
        lim = 700 if verbose else 220
        print("   REF ", r[0], r[1][:lim])
        print("   OURS", o[0], o[1][:lim])
    if out:
        json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
