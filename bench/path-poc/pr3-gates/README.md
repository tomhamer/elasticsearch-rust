# PR3 gate scripts

These scripts do not build binaries. Run them on the scheduled EC2 host once any existing batch finishes. Keep YAML fixture ports, fault ports and differential fixtures disjoint; no overlapping wiping runners. Preserve all generated directories.

## Fault matrix

The existing fault harness requires 3–15 processes, and `bridge` requires five. Its native workload is writes, CAS, delete, bulk and GET; it does **not** generate search or aggregation traffic. `fault_matrix.py` runs five real processes, all eleven scenarios, three seeds and long mode, with concurrent search/statistics-aggregation load. It records every supplemental response. Fault-related unavailability is recorded separately; successful responses must have consistent total/count and no partial shards. The native history checker is authoritative for write correctness.

Run as the account with the harness's required cgroup/iptables privileges (typically root), passing absolute pinned binaries:

```sh
sudo python3 fault_matrix.py --node /absolute/pinned/elasticsearch-node --node-sha256 EXPECTED_SHA256 --harness /absolute/pinned/elasticsearch-faults --harness-sha256 EXPECTED_SHA256 --source-revision EXACT_REVISION --out /home/ubuntu/pr3-gates --port-base 20000 --seeds 3
```

The launcher checks hashes, holds cooperative per-port locks, rejects existing ownership, fault cgroups and matching cluster processes, then checks free listeners before native cleanup. Ownership markers persist under `/var/lock/es-pr3-faults` for explicit inspection. Its unique output path is printed first; the exact announced native invocation path is recorded (older native layout directly under `matrix/summary.json` is also supported). Exact unique scenario/seed pairs and five nodes are mandatory. Existing scenarios are not removed or skipped. Long mode is about 300 seconds per scenario before setup/recovery; do not schedule a short window.

## ES9.5.5 replicated differential

`replicated_correctness.rs` compares split/reference **Rust** indices; that is not an ES oracle. `replicated_differential.py` supplements it with two distinct real nodes per engine, two primary shards and one replica, fixed documents, identical queries/aggregations, and `_only_local` checks on both copies. It requires the ES reference to report exactly 9.5.5 and the two clusters to have different UUIDs. It creates a unique index and never deletes data.

Parent must launch both dedicated clusters from pinned distributions and supply a provenance JSON containing `source_revision`, `node_sha256`, and `fixtures` (PIDs, binary/distribution hashes and exact node configs). No fixture is inferred or started implicitly:

```sh
python3 replicated_differential.py --rust http://127.0.0.1:24200,http://127.0.0.1:24201 --es http://127.0.0.1:24400,http://127.0.0.1:24401 --provenance /absolute/provenance.json --out /home/ubuntu/pr3-gates
```

No `summary.json` / `exit.code=0` means incomplete or failed, never a pass. All requests and responses are retained. This fixed corpus is a correctness fixture, not a Rally benchmark.

## Upstream YAML preparation

```sh
python3 prepare_yaml.py --source /home/ubuntu/essrc-9.5.5 --destination /home/ubuntu/es955-yaml
```

This materializes the established OSS/main/security/plugin layout from source, including all plugin modules and the verified QA module aliases, without editing tests. It refuses an existing destination, checks differing path collisions, preserves failed staging directories, and writes every source/target hash plus upstream test roots outside this established matrix. Main/security splitting uses the baseline's twelve security directories. The adjacent `yaml18-files.json` pins file paths observed in the retained YAML18 reports; the preparer checks every suite against it before publishing. OSS is core plus data-streams, ILM, SLM and two main-suite directories, not all built-in modules (whose paths collide). Inference QA/plugin fixture aliases follow the baseline. No forced gates are set.

## Security downsample replay

`security_replay.py` consumes a completed Round25 candidate and a separately
pinned YAML runner built from **the identical source manifest**. It performs no
build and refuses an active prerequisite service/compiler, source or binary
hash mismatch, wrong module topology, forced gates, occupied/owned ports, or
existing targets. It runs exactly `--suites sec --filter
security/authz/80_downsample.yml --parallel 1` (the path is in the pinned fixture
inventory). Both nodes use the security module's exported fixture configuration.

Supply `--root`, `--binary`, `--binary-sha256`, `--runner`, `--runner-sha256`,
`--manifest` (the tracked-path to SHA256 JSON used to stage Round25),
`--manifest-sha256`, `--runner-manifest-sha256`, `--revision`,
`--completed-results`, and `--out`. The revision/manifest must describe the
actual built snapshot, not a newer worktree: Round25 includes diagnostic
`a7e63b22`, but not the later identity invariant test `45f81c8e`.
`--completed-service` defaults to `pr3-round25`; `--port-base` defaults to 28000.

Each invocation prints its unique output directory and records source/binary/
fixture hashes, exported topology, exact command, case counts and explicit exit.
The unchanged full report remains at the recorded `reports` path, and process
logs remain at `fixture`. Per-port locks use the existing fault-gate lock root;
fixture ownership markers are retained even on success until an operator checks
that all fixture processes exited. No prior evidence or data is deleted by the
launcher.
