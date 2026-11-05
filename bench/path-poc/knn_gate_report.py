#!/usr/bin/env python3
"""Report per-workload native query ratios and ES-relative recall, never HTTP/native mixtures."""
import argparse
import json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--rustic", required=True)
ap.add_argument("--reference", required=True)
ap.add_argument("--speedup", type=float, default=2.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
v = json.loads(Path(a.input).read_text())
rows = []
classes = sorted({key.rsplit("/", 1)[0] for key in v["recall"]})
assert classes, "Recall measurements required"
for cls in classes:
    r = v["recall"][f"{cls}/{a.rustic}"]
    e = v["recall"][f"{cls}/{a.reference}"]
    rk = f"{cls}/{a.rustic}/query_us"
    ek = f"{cls}/{a.reference}/query_us"
    assert rk in v["stats"] and ek in v["stats"], "Both native query timers required"
    rt = v["stats"][rk]["p50_us"]
    et = v["stats"][ek]["p50_us"]
    ratio = et / rt
    quality = all(r[m] + 1e-12 >= e[m] for m in ("mean", "min"))
    rows.append({"class": cls, "rustic_us": rt, "reference_us": et, "native_speedup": ratio,
                 "rustic_recall": r, "reference_recall": e,
                 "quality_passed": quality, "speed_passed": ratio >= a.speedup})
result = {"source": a.input, "target_native_speedup": a.speedup,
          "scope": "Native query timers only; reference is the current ES vector backend",
          "quality_passed": all(r["quality_passed"] for r in rows),
          "performance_passed": all(r["speed_passed"] for r in rows), "workloads": rows}
Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
for r in rows:
    print(f'{r["class"]}: {r["native_speedup"]:.2f}x, recall {r["rustic_recall"]["mean"]:.3f}/{r["reference_recall"]["mean"]:.3f}, speed gate {r["speed_passed"]}')
