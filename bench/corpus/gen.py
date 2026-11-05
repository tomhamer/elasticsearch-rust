#!/usr/bin/env python3
"""Deterministic synthetic corpus generator (standard library only).

Writes JSONL documents of the form

    {"id": int, "title": str, "body": str, "price": float, "ts": int, "tag": str}

Shape of the data:

* Vocabulary: ``--vocab`` (default 200_000) distinct purely-alphabetic lowercase
  words.  Words are alphabetic only so that every analyzer under test
  (rustic ``standard``, tantivy ``default``, Lucene letter-or-digit + lowercase)
  tokenizes them identically.
* Term distribution: Zipf with exponent ``--zipf-s`` (default 1.0) over the
  vocabulary ranks.
* Body length: lognormal with mean ``--body-mean`` words (default 120),
  sigma 0.5, clamped to [5, 2000].  Title: uniform 5..10 words.
* Hot band: ``--hot-frac`` (default 10%) of docs draw ``--hot-mix`` (default
  40%) of their body tokens uniformly from a band of ``--hot-band`` (default 64)
  terms taken from the middle of the Zipf ranking, so those terms end up with
  dense postings (df ~ 5-10% of the corpus) that would otherwise not exist.
* price: lognormal around ~50, 2 decimals.  ts: epoch millis uniformly within
  a 3-year window.  tag: one of 50 tags, Zipf-ish.

Everything is driven by ``random.Random(seed)`` so the output is reproducible
for a given (``--docs``, ``--seed``, ...) tuple.

Usage:
    python3 gen.py --docs 200000 --seed 42 --out corpus.jsonl
"""
import argparse
import json
import math
import random
import sys
import time

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def make_vocab(n: int, rng: random.Random) -> list:
    """``n`` distinct lowercase alphabetic words, 3..10 letters, in rank order.

    Shorter words are placed at lower (more frequent) ranks, roughly mimicking
    natural language where frequent words are short.
    """
    seen = set()
    words = []
    while len(words) < n:
        # Length grows slowly with rank: ranks < 1000 get 3-5 letters, etc.
        r = len(words)
        lo = 3 if r < 1_000 else 4 if r < 20_000 else 5
        hi = 5 if r < 1_000 else 7 if r < 20_000 else 10
        length = rng.randint(lo, hi)
        w = "".join(rng.choice(ALPHABET) for _ in range(length))
        if w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def zipf_cum_weights(n: int, s: float) -> list:
    """Cumulative weights of a Zipf(s) distribution over ranks 1..n."""
    cum = []
    acc = 0.0
    for r in range(1, n + 1):
        acc += 1.0 / (r ** s)
        cum.append(acc)
    return cum


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--vocab", type=int, default=200_000)
    ap.add_argument("--zipf-s", type=float, default=1.0)
    ap.add_argument("--body-mean", type=float, default=120.0)
    ap.add_argument("--body-sigma", type=float, default=0.5)
    ap.add_argument("--hot-frac", type=float, default=0.10)
    ap.add_argument("--hot-mix", type=float, default=0.40)
    ap.add_argument("--hot-band", type=int, default=64)
    ap.add_argument("--hot-band-start", type=int, default=2_000,
                    help="rank at which the hot band starts (terms there are otherwise medium-frequency)")
    ap.add_argument("--tags", type=int, default=50)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"[gen] building vocabulary of {args.vocab} words ...", file=sys.stderr)
    vocab = make_vocab(args.vocab, rng)
    cum = zipf_cum_weights(args.vocab, args.zipf_s)
    hot_band = vocab[args.hot_band_start:args.hot_band_start + args.hot_band]
    tags = [f"tag{i:02d}" for i in range(args.tags)]
    tag_cum = zipf_cum_weights(args.tags, 0.8)

    # Lognormal parameters so that E[len] == body_mean.
    sigma = args.body_sigma
    mu = math.log(args.body_mean) - sigma * sigma / 2.0

    ts_base = 1_600_000_000_000  # 2020-09-13T12:26:40Z
    ts_span = 3 * 365 * 24 * 3600 * 1000

    print(f"[gen] writing {args.docs} docs to {args.out} ...", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.docs):
            n_body = int(rng.lognormvariate(mu, sigma))
            n_body = max(5, min(2000, n_body))
            n_title = rng.randint(5, 10)
            body = rng.choices(vocab, cum_weights=cum, k=n_body)
            if rng.random() < args.hot_frac:
                n_hot = int(n_body * args.hot_mix)
                hot = rng.choices(hot_band, k=n_hot)
                # Scatter hot tokens across the body (keep positions mixed so
                # phrase queries still see realistic adjacency).
                for j, w in enumerate(hot):
                    body[(j * 7919) % n_body] = w
            title = rng.choices(vocab, cum_weights=cum, k=n_title)
            price = round(rng.lognormvariate(math.log(50.0), 0.8), 2)
            ts = ts_base + rng.randrange(ts_span)
            tag = rng.choices(tags, cum_weights=tag_cum, k=1)[0]
            doc = {
                "id": i,
                "title": " ".join(title),
                "body": " ".join(body),
                "price": price,
                "ts": ts,
                "tag": tag,
            }
            f.write(json.dumps(doc, separators=(",", ":")))
            f.write("\n")
            if (i + 1) % 100_000 == 0:
                print(f"[gen]   {i + 1} docs, {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"[gen] done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
