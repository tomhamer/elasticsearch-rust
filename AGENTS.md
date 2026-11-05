# Working rules for agents

Repository overview and doc map: [README.md](README.md).

## Build and test: EC2 only

- Never build Rust (or Java) or run binary tests on the local Mac. Edit source
  locally, sync it to an EC2 build host and build/test there. Binaries move
  only box-to-box over SSH or through S3, never via the Mac.
- Build hosts change between PRs and their public IPs change on every
  restart. Use the host named in the current PR's hand-off, with key
  `~/.ssh/rustic_es_build`. Reuse an existing build host before launching
  new capacity. Give new build/bench instances 500 GB–1 TB of EBS, and never
  delete datasets to make room.
- Finish a whole change (code and tests) before starting a box. Then do one
  build/test pass, fix every failure, and do one more pass. Don't build in
  small increments. Stop (don't terminate) instances as soon as they are idle.
- Disable unattended-upgrades timers on an instance before long runs.
- The full suite is `scripts/pr3/bt2_checked.sh SOURCE RESULTS TARGET`. It
  runs every workspace library and every integration-test target, plus the
  real old/new rolling-upgrade test. Then run the supplemental ignored tests
  in [docs/pr3/REQUIRED_COVERAGE_COMMANDS.md](docs/pr3/REQUIRED_COVERAGE_COMMANDS.md).
  Libraries alone are not the full suite.
- The YAML, fault-matrix and ES-differential correctness gates
  (`scripts/pr3/consolidated_correctness.py`) refuse to start a stage while any
  cargo, rustc, test binary or engine process is running on the box. Don't run
  ad-hoc cargo or tests on the gate box between the end of the suite and the
  gate's exit record. Run diagnostics on another box.
- Never overlap builds with timings, and never share a target directory
  between two concurrent builds.


## Licensing: always enterprise

- Rustic is a from-scratch, open-source rewrite of Elasticsearch whose goal is to make Elasticsearch better: to show that open-source developers can beat the Elastic team's own engine, feature for feature and benchmark for benchmark. Licensing is not the point, so rustic always runs with the full enterprise (trial) feature set and is always compared with ES at its best: every licence check passes as enterprise, and `GET _license` reports an active trial whatever licence was requested.
- Never make the engine default to, fall back to, or gate any feature behind a basic, expired or other non-enterprise licence, in code, config, tests or benchmarks, without asking Tom Hamer first. That includes ES's basic-licence fallbacks, such as synthetic source falling back to stored `_source`.
- Licence-transition tests are ignored (`#[ignore = "licence hardwired to trial"]`), and licence YAML suites are on the correctness skip list. Don't re-enable them, or add tests that assume a non-enterprise licence, without asking Tom Hamer.

## Branches and PRs

- Combine in-flight work into one integration branch on the latest master,
  and ship it as one PR with one build, one suite and one benchmark. Don't
  split a PR for size, only for a real code dependency. Flaky-test and
  post-merge fixes go into the next in-flight PR.
- Fix agents work in their own worktrees. Their commits are cherry-picked
  (`-x`) onto the integration branch after inspection.
- Start the review, the EC2 build/suite and the benchmark on the same head at
  the same time, rather than one after another.
- Preserve unrelated branches and workspaces.

## Correctness (ES 9.5.5 YAML)

- Use the default YAML prerequisite policy. Required ES 9.5.5
  capability/feature gaps execute and count as failures, including missing
  discovery when assertions pass. Unclassified required capability gates must
  not silently become skipped coverage. Don't use `ES_YAML_CLAIM_ALL` or other
  forced gates to establish correctness.
- A skipped or excluded case has not passed. Report pass/fail/skip counts,
  reference/deployment exclusions, the selected suite scope, source revision,
  topology and whether the reports are complete. An assertion pass rate is not
  test coverage or full ES9 parity. Don't claim full parity from filtered,
  partial or in-process-only runs.
- Use two real nodes by default for core/security/x-pack fixtures so replicas
  can allocate. Match module-specific users, roles, licences, registries, test
  services and explicit topology exceptions. Real CCS requires separate
  clusters and networking; a self-alias test is not enough. CCR is not
  supported (requests are rejected with `connect_transport_exception`).
- Fix related functionality in batches, then build and run the affected
  groups on EC2. Start from the failure batches of the existing baseline; don't
  wait for supplemental coverage runs before fixing known defects.
- Never run two wiping YAML runners against the same fixture concurrently.
  Preserve baseline reports and source/binary provenance when starting a
  corrected run.

## Mixed-load gate

Every PR that changes scheduling, cluster coordination, locking, memory
accounting/breakers or the search path must include the wikipedia text
mixed-load benchmark at 1 and 0 replicas, against the ES 9.5.5 baseline and
current master. On the bench box (harness in `~/wiki/tools/`, not in this
repo):

```sh
python3 ~/wiki/tools/wiki_mixed.py <rust-binary> mixed-<tag> --preload 5000000 --secs 600 --qps 100 --rate 4000
```

Setup and reproduction details are in the results
[README](bench/path-poc/perf-validation/results/mixed-load-wikipedia-20261022/README.md).

- Benchmark the final PR head. Re-run if commits land after the run, and never
  overlap a build on the bench box.
- The ES 9.5.5 baseline was run once (22 Oct 2026) and must not be re-run.
  Compare against it and against the latest master run, and put the 0/1-replica
  tables in the PR body. Results live under
  `bench/path-poc/perf-validation/results/mixed-load-wikipedia-20261022/` (and
  S3 `benchmarks/rustic/mixed-load-wikipedia-20261022/`).
- PRs touching engine, search or vector paths also run the KNN + text
  mixed-load (1M real vectors indexed and searched alongside the text load) at
  r1 and r0, against its one-time ES 9.5.5 baseline and master. Record per-node
  memory and disk per phase alongside latency.

## Rally benchmark policy

- Run Rally with the full original corpus. A fast or "1%" run means 1% of
  query volume (query repetitions or duration), never 1% of documents. Don't
  run reduced-data Rally at all: no Rally test mode, no sampled or downsized
  corpus, including for exploratory runs.
- Label historical reduced-data results as reduced-data evidence; don't use
  them to claim full-corpus performance. Preserve their original raw records.
- Run rustic and ES 9.5.5 one after the other on the same box with the same
  shard/replica settings.

## Performance measurement and diagnostics

- Keep profiling/tracing disabled during authoritative timings. Tracing can
  change latency and throughput.
- Collect a short diagnostic CPU profile afterwards on EC2, preferably against
  the same retained index/process so expensive indexing isn't repeated. Use
  judgment about scope; a separate full rebuild is not the default.
- Record instrumentation, source/binary hashes and measurement boundaries.
  Never mix traced diagnostic timings with unprofiled acceptance results;
  compare engines under matching conditions.
- Preserve raw profiles on EC2 and sync text reports and measurements locally.

## Credentials

AWS credentials stay in the protected session location. Never print or
commit them.
