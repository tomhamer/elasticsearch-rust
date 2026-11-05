# scripts

| Path | What it does |
|---|---|
| `smoke_test.py` | Single-node runtime smoke test of a built `elasticsearch` binary (`python scripts/smoke_test.py path/to/elasticsearch`) |
| `check-windows.sh` | Type-checks the workspace for Windows from the Linux build host |
| `pr3/` | PR3 test and gate launchers, below. Run them on EC2 only (`bt2_checked.sh` refuses non-Linux hosts; the Python launchers check for an EC2 identity) |

## scripts/pr3

| File | What it does |
|---|---|
| `bt2_checked.sh SOURCE RESULTS TARGET` | The full suite: every workspace library and integration-test target, plus the real old/new rolling-upgrade test (needs `PR3_OLD_SOURCE` / `PR3_OLD_REVISION`). Fails closed and never overwrites a results directory |
| `supplemental_coverage.py` | The supplemental ignored tests listed in [docs/pr3/REQUIRED_COVERAGE_COMMANDS.md](../docs/pr3/REQUIRED_COVERAGE_COMMANDS.md) |
| `consolidated_correctness.py` | YAML matrix, fault matrix and ES 9.5.5 replicated differential for one immutable candidate. Refuses each stage while other workloads run |
| `materialize_correctness_inventories.py` | Prepares the fixture inventories the correctness launcher checks |
| `gpu_gate.py` | Real-hardware GPU correctness gate |
| `refresh_after_ingest.py URL DOCS [CLIENTS] [BULK]` | Refresh-disabled bulk load into a fresh one-shard index, then times the first refresh with searches running beside it; prints PASS when the refresh takes at most 10 s, no search during it takes over 1 s, the count is complete and the first match query takes at most 5 s |
| `consolidated11_correctness_waiter.py` | One-off waiter that started the correctness gate after a suite run. Historical |
| `*.pending.json`, `gpu-ec2-proposed.json` | Bindings prepared for candidates 7–11 and a proposed GPU instance. Unresolved hashes are `null` on purpose. Historical: they are not evidence of a run |
