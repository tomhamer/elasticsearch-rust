# Verification of the current revision

This file records the correctness gates the current `master` revision passed, with the raw outputs alongside it. The gates run on dedicated EC2 machines, not GitHub Actions, so GitHub shows no checks for these commits; this file and the files under [`verification/`](verification/) are the record.

**Revision:** `f3ee3dc95940cf57c031f5a349c18b96d2fc3bd1`, run 12–13 November 2026. Outputs: [`verification/c33f-f3ee3dc9/`](verification/c33f-f3ee3dc9/).

## Summary

| Gate | What it runs | Result | Raw output |
|---|---|---|---|
| Unit and integration tests | `cargo test` for every crate and test target in the workspace | **5,093 passed, 0 failed** | [`unit-tests.txt`](verification/c33f-f3ee3dc9/unit-tests.txt) |
| Elasticsearch 9.5.5 REST YAML suites | the ES 9.5.5 `rest-api-spec` (oss), x-pack (main), security and plugin YAML suites, each on a fixture with that module's topology | oss **2,657 / 0 failed**, main 1,729 / 5 failed, sec **195 / 0**, plugins **1,692 / 0** | [`yaml-matrix.md`](verification/c33f-f3ee3dc9/yaml-matrix.md) |
| Fault injection | a Jepsen-style harness on 5-node clusters: kill, pause, partition, bridge, flapping, rolling and full restarts, kill during recovery, and chaos, with seeds 1–3 | **33 / 33 scenarios passed; 0 acknowledged writes lost** | [`fault-matrix.md`](verification/c33f-f3ee3dc9/fault-matrix.md) |
| Replicated differential | replicated search and aggregation results compared against a static oracle | 4 / 4 passed | [`differential-summary.json`](verification/c33f-f3ee3dc9/differential-summary.json) |

## Known failures and limits in this revision

- **YAML:** the 5 main failures are `rollup` job-execution tests (start, stop and delete of a running job). Rollup job execution is not implemented and is listed in the README's unsupported features. The licence-gated suites and the GPU suites are skipped.
- **Follow-ups after these gates:** two unit tests flaked in reruns on a faster host, and two commits fix them. `501cd1a6` makes `cancelled_task_stops_enhanced_loops_without_a_deadline` deterministic and changes the test only. `0e15b370` adds a short backoff between connector version-conflict retries, so `concurrent_updates_to_different_fields_all_survive` can't starve one writer. Rerun on `0e15b370`: the `elasticsearch-engine` library tests passed 3 out of 3 times (1,723 passed, 0 failed each), and the connector concurrency tests passed 50 out of 50 times.
- **Differential:** this gate is a small, fixed oracle of 4 queries. Broader parity evidence comes from the YAML suites.

## Machine

The correctness gates run on a c7g.4xlarge (aarch64).
