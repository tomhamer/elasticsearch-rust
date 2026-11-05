#!/usr/bin/env python3
"""Summarize matched references, POC stages and optional full-path results."""
import json
import math
import pathlib
import sys

root = pathlib.Path(__file__).parent
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/table-cache.json"
poc = json.loads(path.read_text())
native = {s: json.loads((root / f"results/native/{s}-query.json").read_text())["metrics"] for s in ("rustic", "lucene")}
classes = ["term", "and2", "or2", "and3", "phrase", "tag"]
def wall(cls, source, mode):
    return poc["stats"][f"{cls}/source={source}/{mode}"]["wall_us"]["p50"]
def lib(cls, system):
    return native[system]["classes"][cls]["latency"]["p50_us"]
ratios = [lib(c, "lucene") / lib(c, "rustic") for c in classes]
geomean = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("| Class | Rustic / Lucene query µs | Library gain | ES / baseline / candidate HTTP µs, source=false | Candidate gain |")
print("| --- | ---: | ---: | ---: | ---: |")
for c, ratio in zip(classes, ratios):
    print(f"| {c} | {lib(c, 'rustic'):.1f} / {lib(c, 'lucene'):.1f} | {ratio:.2f}× | {wall(c, False, 'es'):.1f} / {wall(c, False, 'baseline'):.1f} / {wall(c, False, 'candidate'):.1f} | {wall(c, False, 'es')/wall(c, False, 'candidate'):.2f}× |")
print(f"\nLibrary gain, geometric mean excluding ranges: {geomean(ratios):.2f}×.")
for source in (False, True):
    gain = geomean([wall(c, source, "es") / wall(c, source, "candidate") for c in classes])
    print(f"Candidate gain with source={source}: {gain:.2f}×; {gain/geomean(ratios):.1%} of library ratio.")
    if f"term/source={source}/full" in poc["stats"]:
        full_gain = geomean([wall(c, source, "es") / wall(c, source, "full") for c in classes])
        overhead = geomean([wall(c, source, "full") / wall(c, source, "baseline") for c in classes])
        print(f"Current-code audit path gain (port {poc['args']['our_port']}): {full_gain:.2f}×; audit/baseline POC wall ratio: {overhead:.2f}×.")
    for mode in ("engine", "fused"):
        if f"term/source={source}/{mode}" in poc["stats"]:
            gain = geomean([wall(c, source, "es") / wall(c, source, mode) for c in classes])
            print(f"{mode} gain: {gain:.2f}×.")
print(f"\nVerified requests: {poc['verified_requests']}; raw score parity: {poc['score_parity']}.")
if not poc["score_parity"]:
    print("This intermediate result used explicit legacy BM25 normalization and does not pass the score-parity gate.")
print("HTTP includes fixed request overhead and fetch; native query-only ratios are a reference, not identical timed operations. Per-class recovery matters: the OR library ratio is not fully recovered by HTTP.")
if "term/source=False/fused" in poc["stats"]:
    print("\n| Class | Actual ES | Full cluster path | Engine HTTP | Fused engine adapter | POC candidate |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for c in classes:
        vals = [wall(c, False, m) for m in ("es", "full", "engine", "fused", "candidate")]
        print(f"| {c} | " + " | ".join(f"{v:.1f} µs" for v in vals) + " |")
