#!/usr/bin/env python3
"""Generate a synthetic percolator corpus: N stored match queries over a `body`
text field, mimicking the Rally percolator track (2M stored news queries).

Usage: gen_percolate_corpus.py OUT.jsonl N
"""
import json
import random
import sys

# A pool of words to build stored queries from. The Rally percolator corpus is
# news headlines; we use a similar vocabulary so term frequencies are realistic.
WORDS = (
    "president bush saddam hussein hurricane katrina google israeli prime "
    "minister ariel sharon suffers massive stroke replaced acting ehud "
    "olmert ignore me white house congress senate election campaign "
    "democrat republican economy market stock oil gas price tax budget "
    "defense military army navy air force base soldier general admiral "
    "attack bomb missile weapon nuclear weapon program sanction trade "
    "deal agreement treaty border immigration visa refugee crisis "
    "hospital doctor patient drug disease cancer virus infection treatment "
    "research study science technology computer internet software hardware "
    "phone tablet device chip processor memory storage data cloud server "
    "network security breach hack virus malware phishing password login "
    "account user profile social media post share comment like follow "
    "video image photo camera lens sensor pixel resolution display screen "
    "monitor keyboard mouse printer scanner document file folder directory "
    "path name value type class method function variable constant loop "
    "condition branch return yield call invoke execute run stop start "
    "begin end first last next previous current past future time date "
    "year month week day hour minute second clock calendar schedule "
    "meeting appointment interview call phone email message letter note "
    "memo report article story news headline press media broadcast "
    "television radio station channel program show series episode season "
    "film movie theater cinema actor actress director producer writer "
    "screenplay script dialogue scene shot cut edit color sound music "
    "song band artist album track beat rhythm melody harmony chord "
    "guitar piano violin drum bass trumpet saxophone flute oboe clarinet"
).split()


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
    rng = random.Random(42)
    with open(out, "w") as f:
        for i in range(n):
            # Most stored queries are 1-3 word match queries on body, like the
            # Rally corpus (news search queries).
            k = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
            terms = [rng.choice(WORDS) for _ in range(k)]
            q = {"match": {"body": {"query": " ".join(terms)}}}
            f.write(json.dumps({"id": f"q{i}", "query": q}) + "\n")
    print(f"wrote {n} stored queries -> {out}")


if __name__ == "__main__":
    main()
