# Consolidated PR3 validation

These drivers orchestrate the consolidated PR3 validation. Nothing is run on import. `run.py`, native adapters and workload drivers reject non-EC2 execution. `prepare.py` only resolves and hashes a reviewed binding manifest. Run binary builds and all workloads on EC2, never on a workstation.

## Freeze and bind

Reconcile the candidate with current master before producing one immutable source archive. Record the full 40-character revisions, archive hash and a JSON manifest mapping every source-relative file name to SHA256. Keep build targets outside the source tree. Independently build current master in a separate target, and retain its binary hash. Obtain actual ES 9.5.5 distribution files, bundled JDK and a complete SHA256 inventory, not only a launcher hash. The independent Lucene generator uses the same distribution.

A bindings JSON contains `repository`, `candidate_revision`, `master_revision`, `source_root`, pinned `source_manifest`, `hosts`, `artifact_paths`, and ordered `stages`. Pins are `{ "path": "/absolute/path", "sha256": "64 lowercase hex characters" }`. Each host contains `instance_id`, `purpose` (`correctness`, `performance`, or `pressure`), `local`, and, for remote hosts, `ssh_target` and `ssh_key`. Credentials are never included. Host processes must be launched by a separately reviewed fixture setup under a finite cgroup and retained for the appropriate stages.

Each stage supplies `kind`, `binary_sha256`, absolute `adapter_path`, argv array whose first item is that same absolute adapter, a fresh absolute `report` destination, and `fixture_paths`. Every adapter config, imported helper, native script, executable, distribution inventory, rendered track, corpus and fixture proof belongs in `artifact_paths` or its config's pinned artifacts. Commands are argv arrays, never shell fragments. Stage configs use concrete EC2 paths and new output directories; no generic placeholder command is executable evidence.

On EC2 after review:

```
python3 consolidated/prepare.py --bindings bindings.json --out frozen-plan.json
python3 consolidated/run.py --plan frozen-plan.json --out retained-validation-result
```

`run.py` checks remote master has not moved and obtains one advisory host lock per participating host. The native Rust harness also owns its target lock. All stages execute serially. Matching timings require dedicated hosts, no compilers or profiling, and process provenance before/after. Every stage keeps its raw outputs and must pass for acceptance; independent subsequent stages collect their own evidence, while their own prerequisites must pass. The next run uses a new result directory. Profile collection is the final stage and cannot substitute for timings.

## Required stage matrix

| Kind | Native command or adapter input | Required evidence |
|---|---|---|
| unit-integration | `scripts/pr3/bt2_checked.sh` with explicit `PR3_OLD_REVISION`, candidate source/target and separate old source/target | Enumerated default-feature targets, every library/integration result, feature-required targets run against their real fixtures. |
| yaml | `parity_all.py --root /home/ubuntu/es955-yaml --binary CANDIDATE --runner MATCHING_RUNNER --suites oss,main,sec,plug --parallel 3 --port-base PORT --run-name UNIQUE` | Full matching two-node run, all four native summaries, actual module fixtures, users/roles/license/registry/services, no filters or forced capability gates. Separate CCS/CCR clusters are required; no self-alias substitute. |
| fault | `fault_matrix.py` with pinned candidate and matching harness | All 11 native scenarios × three seeds × five processes; unique native summary, histories, monitor/DIAG and zero violations; additional search/aggregation proof. |
| differential | `replicated_differential.py` | Two real Rust nodes, two real ES nodes, distinct cluster UUIDs, 2,000 documents, two shards/four active copies, all four comparisons. |
| rolling | Existing `rolling_capabilities` result from the unit/integration stage | Both actual executable hashes, old-master and new-master load, known-mapping upgrade writes, refused unsupported downgrade. No repeated run is needed if the exact complete result is already retained. |
| pressure | `pressure.py` | Three actual distinct EC2 instance/machine IDs; live executable/cgroup proof; sustained source/target RSS, successful admitted relocation publication correlated to exact target allocation STARTED, preserved count/cooldown, and concurrent 429/ack preservation. |
| ivf-128-candidate / ivf-128-es9 / ivf-768-candidate / ivf-768-es9 | `ivf_compare.py` | Each full 1,000,000-vector corpus, pinned queries, all/filter/delete/mixed phases, exact full-float truth, fixed candidate count, repeated measurements and update readback. |
| ivf-comparison | `ivf_summary.py` | Both dimensions, pinned raw candidate/ES reports, identical corpus/search/mixed settings and host conditions, explicit recall-loss and p99-ratio thresholds for all nine groups. |
| mixed | Existing `pr3-performance-gates/run.sh mixed`, normalized by `compare.py` | Candidate/current-master/ES under matching conditions, at least three repetitions, actual replica 1→0→1 active-copy checks, write throughput and query p99. |
| rally | `rally_full.py`, normalized by `compare.py` | Every original rendered track and full corpus, offline benchmark-only pipeline, no test mode or document sampling, exact final index counts, zero operation errors and matched repeated engine comparison. |
| idle | Existing performance `run.sh idle`, normalized by `compare.py` | Before/idle/after activity and timings; report the whole-product baseline comparison. |
| segments | Existing performance `run.sh segments`, normalized by `compare.py` | Observably different segment counts at both two and sixteen shards, matching repeated engine comparisons. |
| profile | `compare.py` profile config | All completed timing reports pinned first; short diagnostic `perf.data` retained on EC2 and textual report collected afterward. |

All pass/fail/skip/excluded counts remain visible. The consolidated acceptance function requires exact reviewed coverage accounting for every skipped/excluded case. Required cases need actual execution evidence; historical experiments and unrelated non-test targets remain explicitly outside scope and receive no pass credit.

## External fixture inputs that must be supplied

* AWS task instance IDs and current addresses, live Linux PID, actual `/proc/PID/exe` hash, machine ID, finite cgroup memory cap, effective engine memory budget, node ID, URL and audit-log path. ES executable hash is the bundled Java executable; the complete ES/JDK distribution inventory must also be pinned. Neither a configured instance ID nor a launcher hash proves the live process.
* Full original Rally track checkout/rendered track JSON, every track asset and corpus file, original document counts/compressed and uncompressed byte counts, Rally Python/config pins, exact final index counts, and original track/challenge inventory. Existing historical reduced-data results are not inputs to acceptance.
* Full 1M×128 and 1M×768 float32 vector files and independently pinned query vectors (at least 100), fixed `k`/`num_candidates`, reviewed recall limits and mixed writer schedule. Exact ground truth scans all million source vectors; filtering/deletion never changes the source corpus into a sample.
* Explicit reviewed performance thresholds against both current master and ES for every metric; no default tolerance or invented speedup target. At least three repetitions with the same instance/CPU/driver affinity/memory/storage/kernel/workload within each comparison scope.
* Pressure placement fixture: one retained real shard on the selected source, source-only allocation filter, exact document count, resident source pressure above 85% and target at most 60% of actual engine budget for at least 65 seconds. Pressure must arise from that engine's retained workload, not a side-process RSS balloon or forged telemetry. All three hosts must have comparable observed domains. The driver removes only its fixture allocation filter; it never issues reroute to fake a balancer move.
* Full security/x-pack/module fixtures plus remote CCS/CCR clusters, GPU-required deployment where required, licenses, users/roles and external test services.
* The Wikipedia mixed-workload comparison uses the bench-box harness `~/wiki/tools/wiki_mixed.py` and its pinned corpus; the synthetic 100k mixed driver is not relabeled Wikipedia or Rally.

These inputs are runtime fixture prerequisites. Source revision, binary and distribution hashes, raw request/response histories, measurement boundaries and complete native reports must accompany every result.

## Retained fixture discovery (read-only before provisioning)

The repository already identifies concrete retained sources; inspect these before requesting new inputs:

* `prepare_full_vector_corpus.sh`: `~/perf-validation/results/full-vector-corpus-preparation/{manifest.json,sha256.txt}` and `~/.rally/benchmarks/data/dense_vector_floats/documents.json{,.bz2}`. Original 10,000,000 records, compressed 7,689,453,829 bytes, uncompressed 20,905,226,599 bytes; this full Rally corpus is separate from the requested 1M IVF fixture.
* `prepare_full_pmc_corpus_v3.sh`: `~/perf-validation/results/full-pmc-corpus-preparation-v3/{manifest.json,sha256.txt}` and PMC cache. Original 574,199 records, compressed 5,931,724,449 bytes, uncompressed 23,256,051,757 bytes.
* `run_full_corpus.sh`: `~/perf-validation/full-corpus-tracks`, retained upstream track revision and full-corpus result manifests. Existing default set includes metricbeat, percolator, nested, geonames and geopoint; reconcile with the original requested matrix and other retained new-track records, rather than treating this default subset as the whole assignment.
* `run_integrated_rally_v184.py` and retained rally-master/newtracks records identify additional workload setup and prior full-corpus runs. Read their manifests but never automatically restart the historical jobs.

A hash inventory must be refreshed from the retained actual EC2 files after the final snapshot is chosen. Byte counts above locate original data; they do not substitute for SHA256.

## Coverage ledger

Unit/integration ignored and feature-required target records retain their original counts. A pinned `coverage_ledger` identifies every ignored `log-name::test-name` and every omitted `feature:package::target`, binds the candidate source revision, and supplies a reason plus original scope reference. Required cases need a pinned explicit execution report with exact case ID and `executed=1, passed=1, failed=0, skipped=0`. Truly historical experiments/non-test benchmarks/outside-requested-scope entries are classified explicitly and receive no pass credit. Root reviews this ledger against A–I.

## Source review

Implementation commits `57a47389` and `e91431bb` were independently source-reviewed by the HTTP owner for orchestration/native adapters/metrics/Rally/live provenance and by the recovery owner for full-million IVF truth/filter/delete/mixed checks and distinct-host pressure/ack/cooldown ownership. All reported HIGH findings were repaired. Runtime fixture bindings and the consolidated run are root-owned.
