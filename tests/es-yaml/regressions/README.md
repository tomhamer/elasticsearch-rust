# Supplemental ES9 CAT coverage

These four tests supplement upstream v9.5.5 cases whose prerequisites prevent
execution. They do not replace or edit the original upstream fixtures.

- The current closed-index case requires gte_v7.2.0, instead of skipping it.
- Three template cases express default_shards/no_xpack as separate feature names.
- Sorting and multi-pattern output require exactly one newline per row, matching
  stock ES9.5.5. Upstream dormant assertions expect extra blank lines.
- The exact stock legacy-template deprecation warning is allowed in setup calls.

Run via bench/path-poc/es9-parity/run_cat_regressions.py on EC2, with an explicit
pinned node binary, YAML runner, upstream OSS API directory, and fresh output
path. The helper starts two real nodes using the cat.templates fixture export
(disabled built-in registries), uses separate ports, checks all four cases ran
and passed, and stops only its own node processes. Do not target an active full
matrix's fixture or results directory. Skips are not accepted by this gate.
