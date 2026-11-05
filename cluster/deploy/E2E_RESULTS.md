# Deployment end-to-end results

## Re-run 2026-10-04 ~18:30-19:10 UTC (failover + rollover/by-query/data-stream changes)

Same builder and toolchain as the run below. The images (`elasticsearch-rs:e2e2`,
`elasticsearch-operator:e2e2`) were built from the current working tree of
`cluster/` plus the engine crates at `9e46c2f`, synced with
`E2E_REMOTE_DIR=repo-k8s E2E_ENGINE_REF=9e46c2f cluster/scripts/ec2-e2e.sh sync`.
The build includes the master-failover change (disconnect notification and a
2 s bounded dial for coordination messages). No builder-side patches.

| Scenario | 02:03 run | This run |
|---|---|---|
| Image build with the documented command (`--locked`) | built with `--build-arg CARGO_LOCKED=` | PASS |
| kind/helm: 3 masters + 2 data form a cluster, green | PASS | PASS |
| Kill the elected master with writes in flight | PASS, 4013/4013 acknowledged, 0 lost | PASS, 3599 attempted, 3599 acknowledged (201), 0 lost |
| Kill a data pod with writes in flight | PASS, 2599/2601, 0 lost | PASS, 3002 attempted, 3002 acknowledged, 0 lost |
| Scale data 2 -> 3, 3 -> 2 with drain | PASS | PASS |
| Rolling restart of both StatefulSets | PASS | PASS (16 consistent counts during the rollouts) |
| Rolling upgrade, new tag + JSON logs, writes in flight | PASS, 3024/3036, 0 lost | PASS, 7933 attempted, 7933 acknowledged (7931 x 201, 2 x 200), 0 lost |
| Full restart (all pods deleted), PVC persistence | PASS | PASS, same 6 PVCs, same cluster UUID, all 14534 acknowledged writes present |
| `helm test` | PASS | PASS |
| docker compose: kill the elected master | PASS | PASS, 0.2-0.4 s to a new master, 3.2-3.4 s to removal (table below) |
| Kustomize overlays, operator, docker single node | PASS | covered by the 02:03 run |

kind/helm totals: 14534 writes attempted, 14534 acknowledged,
**0 acknowledged writes lost**, primary and replica doc counts equal for every
shard, identical counts through all 5 pods after every phase. `E2E PASSED`.

### docker compose master failover

Measured with `cluster/scripts/compose-failover-timing.sh` (3 runs of
3 trials; each trial is a fresh `down -v` / `up -d` of
`deploy/docker/docker-compose.yml`, index `fo` with 3 shards x 2 replicas,
`docker compose kill` of the elected master, then both survivors polled every
200 ms over their container IPs with one write every 200 ms, alternating
survivors). Times are seconds after the kill.

| Run/trial | Killed | New master | Node removed | Acknowledged lost |
|---|---|---|---|---|
| 1/1 | es02 | 0.2 | 3.3 | not checked |
| 1/2 | es01 | 0.3 | 3.3 | not checked |
| 1/3 | es01 | 0.3 | 3.3 | not checked |
| 2/1 | es02 | 0.2 | 3.3 | 0 of 437 |
| 2/2 | es01 | 0.3 | 3.4 | 0 of 437 |
| 3/1 | es01 | 0.4 | 3.2 | 0 of 438 |
| 3/2 | es02 | 0.3 | 3.4 | 0 of 436 |
| 3/3 | es01 | 0.4 | 3.2 | 0 of 437 |

The killed container stays `exited` with 0 restarts (`docker compose kill` is
not undone by `restart: unless-stopped`).

A new master is elected 0.2-0.4 s after the kill and the dead node is removed
after 3.2-3.4 s, in line with the loopback numbers.

---

Run on 2026-10-04 02:03-02:27 UTC on the EC2 builder (Ubuntu 24.04 arm64,
32 vCPU, 64 GB, Docker 29.1.3 + compose 2.40.3, kind 0.24 / kindest/node
v1.31.0, kubectl 1.31, helm 3).

The images were built from the repository as synced by
`E2E_REMOTE_DIR=repo-k8s cluster/scripts/ec2-e2e.sh sync` to `~/repo-k8s`: the
working tree of `cluster/` plus the engine crates at git HEAD `9e46c2f`, with
no builder-side patches. Every scenario below forms its cluster through
headless-service / compose DNS discovery.

The source changes made for this run are in the repo tree and were part of
the image that was tested:

- `cluster/crates/elasticsearch-node/src/bin/elasticsearch.rs`: logging
  (no ANSI unless stdout is a terminal; JSON format switch), see
  [Logging](#logging).
- `cluster/deploy/k8s/e2e.sh`: the test itself (acknowledged-write checks,
  rolling upgrade, full restart, and a term-based master-kill assertion), see
  [Master-kill assertion](#master-kill-assertion).

## Summary

| Scenario | Result |
|---|---|
| Image build from the repo tree | PASS (`--build-arg CARGO_LOCKED=`) |
| Helm chart in kind: 3 masters + 2 data form a cluster through the headless service | PASS (all 5 pods Ready 64s after `helm install`, green) |
| Kill the elected master with writes in flight | PASS: 4013 writes attempted, 4013 acknowledged, 0 acknowledged writes lost, identical counts on all 5 pods |
| Kill a data pod (holding the primary) with writes in flight | PASS: 2601 attempted, 2599 acknowledged, 0 acknowledged lost, identical counts on all 5 pods |
| Scale data 2 -> 3 | PASS |
| Scale data 3 -> 2 with allocation-exclude drain | PASS |
| Rolling restart of data and master StatefulSets | PASS (15 searches during the rollouts all returned the full count) |
| Rolling upgrade to a new image tag with writes in flight | PASS: 3036 attempted, 3024 acknowledged, 0 acknowledged lost |
| Full restart (all pods deleted at once), PVC persistence | PASS: same 6 PVCs, same cluster UUID, all data and all 9636 acknowledged writes present |
| `helm test` | PASS |
| `helm lint` / `helm template` (defaults and `values-production.yaml`), server-side dry run | PASS |
| Kustomize overlays `three-node` and `single-node` applied in kind | PASS (green after 36s / 26s) |
| Operator: quickstart, status, Services/PDB/ConfigMap/PVCs | PASS (green / 3 nodes / Ready 32s after create) |
| Operator: scale 3 -> 5 -> 3 with writes in flight | PASS: 0 acknowledged writes lost, identical counts on all pods |
| Operator: config change -> rolling restart with writes in flight | PASS: 0 acknowledged writes lost |
| Operator: force-delete the master pod | PASS (green / 3 nodes again within about 15s, count unchanged) |
| docker single node, non-TTY logs, JSON logs, SIGTERM | PASS |
| docker compose 3 nodes, kill master, rejoin | PASS (failover timing in the re-run above) |

## 1. Image

```
docker build -f cluster/deploy/docker/Dockerfile --build-arg CARGO_LOCKED= -t elasticsearch-rs:repo .
docker build -f cluster/deploy/operator/Dockerfile --build-arg CARGO_LOCKED= -t elasticsearch-operator:repo .
```

Resulting images: `elasticsearch-rs:repo` (`07a829157144`),
`elasticsearch-operator:repo` (`fbcef0c22d5e`). `:0.1.0` was re-tagged from
`:repo` for the kustomize and operator manifests, which reference that tag.

## 2. kind: `cluster/deploy/k8s/e2e.sh` (Helm chart)

```
IMAGE=elasticsearch-rs:repo BUILD_IMAGE=0 KEEP_CLUSTER=1 cluster/deploy/k8s/e2e.sh
```

kind cluster with 1 control-plane and 3 workers; chart installed with 3
master pods and 2 data pods, zone awareness attributes, PVCs. Wall time
about 12 minutes; all 13 steps passed (`E2E PASSED`).

A writer pod issues single-document `PUT e2e-w/_doc/<id>` requests in a loop
through the HTTP Service and records the HTTP status of each one. After each
disruptive step the script refreshes, fetches every acknowledged id (201)
with `_mget`, compares primary and replica doc counts in `_cat/shards`, and
runs `_count` for every index through every pod individually.

| Step | Measured |
|---|---|
| 4. install | 5 pods Running after 64s, green, bootstrapped via `cluster.initial_master_nodes`, master `e2e-master-2`, `node.attr.zone` set on all nodes, node logs plain text without ANSI escapes |
| 5. index | 3 indices x 300 docs, `_count` 900; multi-index `_count` over `e2e-a,e2e-b,e2e-c` = 900 |
| 6. master kill | `e2e-master-2` force-deleted in term 1; a master was elected in term 2 (the replacement pod with the same name and node id won). Writer: 4013 attempts, 4013 x 201. All 4013 acknowledged docs found, index holds 4013, primary = replica, identical counts through all 5 pods |
| 7. data pod kill | `e2e-data-1` (primary of `e2e-w`) force-deleted. Writer: 2601 attempts, 2599 x 201, 2 unacknowledged while the pod was going away. Cumulative: all 6612 acknowledged docs found (index holds 6613 of 6614 attempted), copies equal, identical through all 5 pods |
| 8. scale 2 -> 3 | green, 3 data nodes, count 900 |
| 9. scale 3 -> 2 | `cluster.routing.allocation.exclude._name=e2e-data-2`, no shard left on it, scaled, exclusion cleared, green, 2 data nodes |
| 10. rolling restart | `rollout restart` of `e2e-data` then `e2e-master`: 6 + 9 searches during the rollouts all returned 900; green; all 6612 acknowledged docs found; identical through all pods |
| 11. rolling upgrade | `helm upgrade` to `elasticsearch-rs:repo-v2` (re-tag) plus `ES_LOG_FORMAT=json`, writer running: 10 searches during the rollout all returned 900. Writer: 3036 attempts, 3024 x 201, 12 unacknowledged while pods were being replaced. Every pod runs the new tag and logs JSON. All 9636 acknowledged docs found (index holds 9641 of 9650 attempted), identical through all pods |
| 12. full restart | all 5 pods deleted at once; all replaced, green, same 6 PVCs bound to the same volumes, cluster UUID unchanged (`fBwEIKcRcNFdC-JRJCTTeg`), count 900, `e2e-w` still 9641, all 9636 acknowledged docs found, copies equal, identical through all pods |
| 13. `helm test` | succeeded |

Across the whole run: 9650 writes attempted, 9636 acknowledged, 0
acknowledged writes lost. The 14 unacknowledged requests happened while a pod
was being killed or replaced; 5 of them were nevertheless persisted, which is
allowed.

### Master-kill assertion

The StatefulSet recreates a force-deleted pod within about a second with the
same name and the same persistent node id, and that node can win the next
election, so the master *name* does not have to change. The master-kill step
in `cluster/deploy/k8s/e2e.sh` (helpers `cluster_term` and
`elected_after_term`) therefore records the coordination term and the pod UID
before the kill, then requires a master elected in a *higher term* and a
*different pod UID*.

### Static checks

- `helm lint cluster/deploy/helm/elasticsearch` with defaults and with
  `values-production.yaml`: 0 failed (only the "icon is recommended" info).
- `helm template`: 489 lines (defaults), 1235 lines (production values).
- `helm template ... | kubectl apply --dry-run=server -f -` (defaults):
  accepted.

### Kustomize overlays

`kubectl apply -k cluster/deploy/k8s/overlays/<overlay>` in namespace
`elasticsearch`:

| Overlay | Result |
|---|---|
| `three-node` | StatefulSet rolled out in 36s, `{"status":"green","number_of_nodes":3}`, 0 log lines with ANSI escapes |
| `single-node` | rolled out in 26s, `{"status":"green","number_of_nodes":1}`, 0 log lines with ANSI escapes |

## 3. Operator in kind

`kubectl apply -k cluster/deploy/operator`, then
`cluster/deploy/operator/examples/quickstart.yaml` (3 nodes,
master/data/ingest). Same writer and acknowledged-write check as above,
against `quickstart-es-http`; two indices with 500 docs each plus the writer
index, all with one replica.

| Step | Measured |
|---|---|
| quickstart | `green 3 Ready` 32s after create. Created: StatefulSet `quickstart-es-default` (3/3), Services `quickstart-es-http`, `quickstart-es-transport`, `quickstart-es-default`, PDB, ConfigMap, 3 PVCs. Operator logged `cluster formed; dropping cluster.initial_master_nodes` |
| scale 3 -> 5 | `green 5 Ready` after 21s. Writer: 1325 attempts, 1325 x 201, all found, identical counts through all 5 pods (500 / 500 / 1325) |
| scale 5 -> 3 | `green 3 Ready` after 18s. Writer: 2140 attempts, 2134 x 201, 6 x connection failure; all 2134 acknowledged found, index holds 2134; identical through all 3 pods; voting configuration back to 3 nodes, no leftover voting exclusions or allocation settings |
| config change (`node.attr.rack: r1`) | all 3 pods replaced after 33s, `green 3 Ready`. Writer: 4513 attempts, 4505 x 201, 8 x connection failure; all 4505 acknowledged found, index holds 4505; identical through all pods; `_cat/nodeattrs` shows `rack r1` on all three; `_nodes/shutdown` empty afterwards |
| master pod force-deleted (no writer) | `green 3` and pod Ready again within about 15s, count 5505 unchanged |
| delete the CR | StatefulSet, Services, PDB and PVCs all removed |

Final status: `phase: Ready`, `health: green`, `availableNodes: 3`,
`observedGeneration: 4`, conditions `ReconciliationComplete`,
`ElasticsearchIsReachable`, `RunningDesiredVersion` all `True`. The operator
logged no errors. The operator logs JSON; node pods log plain text with 0
ANSI lines.

## 4. docker

Single node (`docker run -d -p 30920:9200 -e discovery.type=single-node elasticsearch-rs:repo`):
green in about 1s, `docker stop` exits 0 in 178 ms.

docker compose (`ES_IMAGE=elasticsearch-rs:repo ES_PORT=30921 docker compose up -d --no-build`):
3 nodes green in about 2s; 200 docs in an index with 2 replicas; elected
master killed with `docker compose kill`; a new master was elected and the
count through a surviving node was 200; the killed node restarted and
rejoined (green, 3 nodes) in about 1s and also returned 200.

## Logging

Logging behaviour in `cluster/crates/elasticsearch-node/src/bin/elasticsearch.rs`:

- ANSI colour is only emitted when stdout is a terminal and `NO_COLOR` is
  unset. Measured: 0 lines with escape sequences in `docker logs` and
  `kubectl logs` (helm, kustomize and operator pods); 4 of 4 lines coloured
  with `docker run -t`.
- `ES_LOG_FORMAT=json` (also `LOG_FORMAT=json`, the variable the operator
  manifest uses, or `--log-format=json`) switches to one JSON object per
  line. Measured: 4 of 4 lines valid JSON under `docker run`; in the helm
  rolling-upgrade step every pod logged JSON.

## Teardown

`kind delete cluster --name es-e2e`, `docker compose down -v` and removal of
the single-node container were run; no kind clusters, containers or volumes
from this run remain on the builder. Host ports used: 30920 and 30921 only.
Logs on the builder: `~/k8s-build.log`, `~/k8s-e2e.log`, `~/k8s-op.log`,
`~/k8s-dk.log`.
