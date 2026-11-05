#!/usr/bin/env python3
"""Generate queries.jsonl from a corpus produced by gen.py.

Query classes (all on the `body` field unless noted):

* ``term``      200 single-term queries: 66 rare (df ~ 1-5), 66 medium
                (df ~ 1% of docs), 68 frequent (df ~ 5-15% of docs).
* ``and2``      200 two-term AND queries (medium x frequent).
* ``or2``       200 two-term OR queries (medium x frequent).
* ``and3``      100 three-term AND queries (medium x frequent x frequent).
* ``phrase``    100 two-word adjacent bigrams that actually occur in the
                corpus (collected from every k-th doc; 50 frequent, 50 rare).
* ``range``     100 numeric ranges on ``price``: 50 at ~1% selectivity,
                50 at ~10% selectivity (computed from the price histogram).
* ``tag``       100 keyword term queries on ``tag`` (cycling the ~50 tags).

Each line is a JSON object:

    {"id": int, "class": str, "field": str, "kind": str, ...}

where ``kind`` is one of ``term`` / ``and`` / ``or`` / ``phrase`` / ``range``
and carries the class-specific payload (``terms``, ``lower``/``upper``).

Usage:
    python3 queries.py --corpus corpus.jsonl --out queries.jsonl --seed 42
"""
import argparse
import json
import random
import sys
import time


def tokenize(text: str) -> list:
    """Lowercase, split on non-alphanumerics (matches all three analyzers)."""
    out = []
    cur = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--out", default="queries.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bigram-every", type=int, default=20,
                    help="collect candidate phrase bigrams from every k-th doc")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    # ---- Pass 1: exact df over the whole corpus; bigrams from every k-th doc. ----
    df = {}
    prices = []
    tags = {}
    bigrams = {}
    n_docs = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            n_docs += 1
            toks = tokenize(doc["body"])
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
            if n_docs % args.bigram_every == 0:
                for a, b in zip(toks, toks[1:]):
                    bg = (a, b)
                    bigrams[bg] = bigrams.get(bg, 0) + 1
            prices.append(doc["price"])
            tags[doc["tag"]] = tags.get(doc["tag"], 0) + 1

    def df_bucket(lo: float, hi: float) -> list:
        return [t for t, c in df.items() if lo * n_docs <= c < hi * n_docs]

    rare = df_bucket(0.0, 5.0 / n_docs)          # df < 5
    medium = df_bucket(0.005, 0.02)              # 0.5%..2% of docs
    frequent = df_bucket(0.05, 0.15)             # 5%..15% of docs
    print(f"[queries] n_docs={n_docs} rare={len(rare)} medium={len(medium)} "
          f"frequent={len(frequent)} bigrams={len(bigrams)}", file=sys.stderr)

    def pick(pool: list, k: int) -> list:
        return rng.sample(pool, min(k, len(pool)))

    queries = []
    qid = 0

    def add(q: dict) -> None:
        nonlocal qid
        q["id"] = qid
        qid += 1
        queries.append(q)

    # ---- term ----
    for t in pick(rare, 66):
        add({"class": "term", "field": "body", "kind": "term", "term": t, "df": df[t]})
    for t in pick(medium, 66):
        add({"class": "term", "field": "body", "kind": "term", "term": t, "df": df[t]})
    for t in pick(frequent, 68):
        add({"class": "term", "field": "body", "kind": "term", "term": t, "df": df[t]})

    # ---- and2 / or2 / and3 ----
    # AND pairs mix medium x frequent so intersections are non-empty
    # (~1% x ~10% of N docs); OR pairs mix medium x frequent; AND3 adds a
    # second frequent term.
    med_p, freq_p = pick(medium, 200), pick(frequent, 200)
    for i in range(200):
        a, b = med_p[i % len(med_p)], freq_p[i % len(freq_p)]
        add({"class": "and2", "field": "body", "kind": "and", "terms": [a, b]})
    for i in range(200):
        a, b = med_p[(i + 100) % len(med_p)], freq_p[(i + 100) % len(freq_p)]
        add({"class": "or2", "field": "body", "kind": "or", "terms": [a, b]})
    for i in range(100):
        a = med_p[(i + 50) % len(med_p)]
        b = freq_p[(i + 50) % len(freq_p)]
        c = freq_p[(i + 83) % len(freq_p)]
        terms = list(dict.fromkeys([a, b, c]))
        add({"class": "and3", "field": "body", "kind": "and", "terms": terms})

    # ---- phrase: bigrams that occur in at least one sampled doc (so they are
    # guaranteed to be present); half from the frequent end, half from the tail.
    bg_items = sorted(bigrams.items(), key=lambda kv: -kv[1])
    head = [bg for bg, _ in bg_items[:2_000] if bg[0] != bg[1]]
    tail = [bg for bg, c in bg_items[2_000:] if c == 1 and bg[0] != bg[1]]
    for bg in pick(head, 50) + pick(tail, 50):
        add({"class": "phrase", "field": "body", "kind": "phrase", "terms": list(bg)})

    # ---- range on price ----
    prices.sort()

    def quantile(q: float) -> float:
        return prices[min(len(prices) - 1, int(q * len(prices)))]

    # 50 windows each covering ~1% of the price distribution, then 50 windows
    # each covering ~10%, at seeded random offsets.
    for sel in (0.01, 0.10):
        for _ in range(50):
            start = rng.uniform(0.0, 1.0 - sel)
            lo = quantile(start)
            hi = quantile(start + sel)
            add({"class": "range", "field": "price", "kind": "range",
                 "lower": lo, "upper": hi, "selectivity": sel})

    # ---- tag ----
    # Only ~50 distinct tags exist; cycle through them (most frequent first).
    tag_items = sorted(tags.items(), key=lambda kv: -kv[1])
    for i in range(100):
        tag, c = tag_items[i % len(tag_items)]
        add({"class": "tag", "field": "tag", "kind": "term", "term": tag, "df": c})

    with open(args.out, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, separators=(",", ":")))
            f.write("\n")
    print(f"[queries] wrote {len(queries)} queries to {args.out} in {time.time() - t0:.1f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
