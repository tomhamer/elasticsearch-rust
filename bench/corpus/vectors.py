#!/usr/bin/env python3
"""Emit vectors.f32: little-endian f32 vectors, ``--dim`` (default 128) wide.

Layout (no header):  ``n`` base vectors followed by ``--queries`` (default 20)
query vectors, all contiguous.  Consumers derive ``n`` from the file size:
``n = len(file) / (4 * dim) - queries``.

The data is mildly clustered (``--clusters`` Gaussian centers, unit-ish noise)
so that HNSW recall numbers are meaningful; pure i.i.d. Gaussian data in 128-d
has nearly uniform pairwise distances and makes every ANN index look bad.

Standard library only; deterministic for a given ``--seed``.

Usage:
    python3 vectors.py --n 200000 --dim 128 --queries 20 --seed 42 --out vectors.f32
"""
import argparse
import array
import random
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--clusters", type=int, default=1_000)
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="vectors.f32")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    gauss = rng.gauss
    centers = [[gauss(0.0, 1.0) for _ in range(args.dim)] for _ in range(args.clusters)]

    def sample() -> list:
        c = centers[rng.randrange(args.clusters)]
        noise = args.noise
        return [c[i] + gauss(0.0, noise) for i in range(args.dim)]

    buf = array.array("f")
    if sys.byteorder != "little":
        raise SystemExit("this script assumes a little-endian host")
    with open(args.out, "wb") as f:
        for i in range(args.n + args.queries):
            buf.extend(sample())
            if len(buf) >= 1 << 20:
                buf.tofile(f)
                del buf[:]
            if (i + 1) % 50_000 == 0:
                print(f"[vectors] {i + 1} vectors, {time.time() - t0:.1f}s", file=sys.stderr)
        if buf:
            buf.tofile(f)
    print(f"[vectors] wrote {args.n} base + {args.queries} query vectors (dim {args.dim}) "
          f"to {args.out} in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
