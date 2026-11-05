# bench/path-poc

Comparison and correctness tooling for running elasticsearch-rust against
Elasticsearch 9.5.5.

| Path | What it is |
|---|---|
| `es9-parity/` | YAML REST test matrix and module fixture launchers (`parity_all.py`, `start_module_fixture.py`) |
| `pr3-gates/` | Fault matrix, replicated differential checks and correctness gate launchers |
| `pr3-vector-oracle/` | Encoded-score oracle for vector scoring parity |
| `esql-spatial/` | ES\|QL spatial comparison harness |
| `src/`, `Cargo.toml` | Diagnostic binaries over `rustic-core` and the engine |
| `*.py` | Comparison and smoke-test scripts |

Benchmark results are summarised in the [top-level README](../../README.md#performance).
