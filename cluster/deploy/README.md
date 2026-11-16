# Deploying elasticsearch

This directory deploys the Rust `elasticsearch` node: a container image, a
3-node Docker Compose cluster, a Helm chart, plain Kubernetes manifests, and
an ECK-compatible Kubernetes operator. Operators who have run Elasticsearch
before should find what they expect: the same paths, ports, settings,
environment-variable conventions, APIs and procedures.

```
deploy/
├── docker/
│   ├── Dockerfile                  node image (multi-stage: rust builder -> debian slim)
│   ├── Dockerfile.dockerignore     build-context excludes (picked up automatically by BuildKit)
│   ├── Dockerfile.operator         operator image (distroless)
│   ├── elasticsearch.yml           image default config (cluster.name: docker-cluster, network.host: 0.0.0.0)
│   ├── docker-entrypoint.sh        tini-supervised entrypoint (settings.d -> -E key=value)
│   ├── docker-compose.yml          3-node cluster es01..es03, security on (setup service: CA, certificates, password)
│   ├── docker-compose.insecure.yml the same without TLS or authentication: DEVELOPMENT ONLY
│   ├── LICENSE.txt                 /usr/share/elasticsearch/LICENSE.txt (Apache-2.0 + the line stock ECK checks for)
│   └── bin/                        helpers baked into the image:
│       ├── es-readiness            readiness probe (official chart semantics)
│       ├── es-prestop              graceful stop: node shutdown API, voting exclusions on scale-down
│       ├── es-zone-init            init container: node label -> node.attr.zone
│       ├── es-cluster-cleanup      post-upgrade: clear voting exclusions, stale shutdown records
│       ├── es-healthcheck          container healthcheck (verified TLS + credentials when configured)
│       ├── es-certgen              CA / certificate generator (OpenSSL), like elasticsearch-certutil
│       └── es-http-lib             shared by the scripts above: HTTPS, CA trust, credentials
├── helm/elasticsearch/             Helm chart (values.yaml, values-production.yaml, values.schema.json)
├── k8s/
│   ├── base/, overlays/{single-node,three-node}/   kustomize manifests
│   ├── create-secrets.sh           certificates + elastic password Secrets for the overlays
│   ├── kind-config.yaml            kind cluster: 3 workers in zones zone-a/b/c
│   └── e2e.sh                      end-to-end test on kind
└── operator/
    ├── crd.yaml, operator.yaml, kustomization.yaml
    └── examples/{quickstart,production}.yaml
```

---

## 1. The image

Build from the **repository root**, because the node depends on `crates/` (engine) and
`cluster/` (cluster layer):

```sh
docker build -f cluster/deploy/docker/Dockerfile -t elasticsearch-rs:dev .
# operator
docker build -f cluster/deploy/docker/Dockerfile.operator -t elasticsearch-operator:dev .
```

**Build context.** The Dockerfiles use BuildKit features (`RUN --mount=type=cache`,
per-Dockerfile ignore files), so build with BuildKit: Docker 23+ with the buildx
plugin (`docker buildx version`; on Ubuntu `apt install docker-buildx`). BuildKit
reads `cluster/deploy/docker/Dockerfile.dockerignore` automatically when you build
with `-f cluster/deploy/docker/Dockerfile`. It is an allow-list: only the root
`Cargo.toml`/`Cargo.lock`/`src/` (the engine crates inherit `*.workspace = true`
keys from the root manifest, which cargo can only load if the root package's
`src/main.rs` exists), `crates/` and `cluster/` are sent, minus `target/`
directories and the deploy manifests (about 10 MB).

The build uses `cargo build --locked` against `cluster/Cargo.lock`. If a
`Cargo.toml` change has not been locked yet, build with
`--build-arg CARGO_LOCKED=` (empty) to let cargo resolve.

A cold release build takes about 1.5 minutes on a 32-vCPU builder; the image is
about 216 MB on disk (51 MB compressed), the operator image about 61 MB.

Image layout, following the official image:

| | |
|---|---|
| binary | `/usr/share/elasticsearch/bin/elasticsearch` |
| config | `/usr/share/elasticsearch/config/elasticsearch.yml` (`ES_PATH_CONF`) |
| data / logs | `/usr/share/elasticsearch/data`, `/usr/share/elasticsearch/logs` |
| user | `elasticsearch`, uid 1000, gid 0 (group-writable, so arbitrary UIDs work) |
| PID 1 | `tini -g` forwards SIGTERM to the node |
| ports | 9200 (HTTP), 9300 (transport) |
| healthcheck | `es-healthcheck`: `/_cluster/health?local=true`, verified TLS + credentials when `ES_CA_CERT` / `ELASTIC_PASSWORD[_FILE]` are set |
| security | on by default, as in ES 8/9 (see below) |

Ways to set settings, from lowest to highest precedence:
1. `elasticsearch.yml`, with `${VAR}` substitution.
2. Environment variables named after the setting, for example `-e discovery.type=single-node`.
3. Files in `settings.d`. In each directory listed in `ES_SETTINGS_DIRS`
   (default `/usr/share/elasticsearch/settings.d`, searched two levels deep), each
   file becomes `-E <filename>=<content>`.
4. `-E key=value` arguments, for example `docker run IMAGE -E node.attr.rack=r1`.

`ELASTIC_PASSWORD` (or `ELASTIC_PASSWORD_FILE`, a file holding it, as in
ES's image) sets the `elastic` user's bootstrap password; `KEYSTORE_PASSWORD`
/ `KEYSTORE_PASSWORD_FILE` unlock a password-protected keystore.

## 2. Single node in Docker

Security is on, as in ES 8/9: on its first start a single node generates a
CA and certificates under `config/certs/` and serves HTTPS (the node's
security auto-configuration).

```sh
docker run -d --name es -p 9200:9200 \
  -e discovery.type=single-node \
  -e ELASTIC_PASSWORD=changeme \
  -v esdata:/usr/share/elasticsearch/data \
  --ulimit nofile=65535:65535 \
  elasticsearch-rs:dev
docker cp es:/usr/share/elasticsearch/config/certs/http_ca.crt .
curl --cacert http_ca.crt -u elastic:changeme https://localhost:9200/_cluster/health?pretty
docker stop es     # SIGTERM -> graceful shutdown (default 10 s; use -t 120 for big nodes)
```

## 3. Three-node cluster with Docker Compose

```sh
sudo sysctl -w vm.max_map_count=262144   # bootstrap check; see below
cd cluster/deploy/docker
docker compose up -d --build             # ELASTIC_PASSWORD=... (env or .env) to choose the password
docker compose cp es01:/usr/share/elasticsearch/config/certs/ca/ca.crt .
PASSWORD=$(docker compose exec es01 cat config/certs/elastic-password)
curl --cacert ca.crt -u "elastic:$PASSWORD" 'https://localhost:9200/_cat/nodes?v'
docker compose down        # keeps the named volumes; `down -v` wipes data
```

This is Elastic's documented compose example without Kibana: a one-shot
`setup` service creates a CA and one certificate per node with `es-certgen`
(the image's equivalent of `elasticsearch-certutil`), and the `elastic`
password (`ELASTIC_PASSWORD` if set, else random), in the `certs` volume.
es01 to es03 then start with authentication, HTTPS and transport TLS
(`verification_mode: certificate`), `discovery.seed_hosts` pointing at the
peers, and `cluster.initial_master_nodes=es01,es02,es03`. Each node has a
named volume, and only es01 publishes port 9200. The healthchecks make
verified, authenticated requests. After the cluster has formed once, you may
remove `cluster.initial_master_nodes`. A formed cluster ignores it.

`docker-compose.insecure.yml` is the same cluster with
`xpack.security.enabled=false`: no TLS and no authentication, for local
development only (`docker compose -f docker-compose.insecure.yml up -d`; a
separate compose project, so it never shares volumes with the secure one).

## 4. Helm

```sh
# 3 master+data nodes, security on (the default)
helm install es cluster/deploy/helm/elasticsearch -n search --create-namespace \
  --set image.repository=elasticsearch-rs --set image.tag=dev
PASSWORD=$(kubectl -n search get secret elasticsearch-credentials -o go-template='{{.data.password | base64decode}}')
kubectl -n search get secret elasticsearch-certs -o go-template='{{index .data "ca.crt" | base64decode}}' > ca.crt
kubectl -n search port-forward svc/elasticsearch 9200 &
curl --cacert ca.crt -u "elastic:$PASSWORD" https://localhost:9200/_cluster/health?pretty
helm test es -n search

# production: 3 dedicated masters + 6 data + 2 coordinating nodes over 3 zones
helm install prod cluster/deploy/helm/elasticsearch -n search --create-namespace \
  -f cluster/deploy/helm/elasticsearch/values-production.yaml
```

What the chart creates:

* **Security (on by default, like Elastic's chart and ECK).** Authentication
  is required, HTTP is served over TLS and transport uses TLS with
  `verification_mode: certificate`. The `elastic` password comes from
  `security.elasticPassword.existingSecret`, or the chart generates it in the
  Secret `<clusterName>-credentials`; it reaches the nodes as their bootstrap
  password (`ELASTIC_PASSWORD`). Certificates come from
  `security.tls.http.secretName` / `security.tls.transport.secretName`
  (Secrets with `tls.crt`, `tls.key`, `ca.crt`), or from
  `security.tls.provider`: `generate` (default; the chart creates a CA and
  certificates in `<clusterName>-certs`, kept across upgrades through
  `lookup`) or `cert-manager` (Certificates from your issuer, or from a
  per-cluster CA the chart sets up; renewed automatically, and the nodes
  reload renewed files without a restart). The probes, the preStop hook, the
  cleanup Job, `helm test` and the metrics exporter use HTTPS, verify the
  certificate against the client Service name and authenticate.
  `security.enabled: false` turns all of it off, for development only.
  Changing the security mode of a running cluster needs a full restart
  (delete all pods at once): nodes with and without transport TLS cannot form
  one cluster. With offline rendering (`helm template`, Argo CD, Flux) use
  `existingSecret` and `cert-manager` or `secretName`: `lookup` cannot see
  existing Secrets there, so generated ones would change on every render.
* **One StatefulSet per entry of `nodeGroups`**, named `<clusterName>-<group>`.
  Each group sets its own `replicas`, `roles`, `resources`, `persistence`
  (size/class), `nodeSelector`, `tolerations`, `affinity`, `config`
  (`elasticsearch.yml`), `extraEnv`, extra volumes and containers. Defaults
  come from `nodeGroupDefaults`, and keys set on a group win, including `false`
  and `[]`.
* **Services.** Each group gets a headless `<clusterName>-<group>-headless` Service
  (`publishNotReadyAddresses: true`). The headless Services of master-eligible
  groups form `discovery.seed_hosts`. A client Service `<clusterName>` (HTTP 9200)
  fronts the groups with `clientService: true`.
* **Node identity.** `node.name` is the pod name (downward API). `network.publish_host`
  is the pod IP. With `zoneAwareness.enabled`, `node.attr.zone` comes from the Kubernetes
  node's `topology.kubernetes.io/zone` label: the `es-zone-init` init container reads it,
  using a ClusterRole with `get nodes`, because the downward API cannot expose node labels.
* **Bootstrap.** `cluster.initial_master_nodes` holds the pod names of all master
  groups. It is passed through the `<clusterName>-bootstrap` ConfigMap, which is
  mounted into `settings.d` and is not part of the pod template. On `helm upgrade`,
  the chart finds the existing master StatefulSet with `lookup` and empties that
  ConfigMap, so later restarts never re-bootstrap and no rolling restart happens.
  As with the official chart:
  * `helm template`, Argo CD and Flux render offline, where `lookup` returns nothing.
    In that case, set `clusterBootstrapped: true` once the cluster has formed.
  * Don't change master groups in the release that first creates the cluster.
  * If every master volume is lost, the cluster has to be bootstrapped again.
* **Probes.** The startup and liveness probes are TCP checks on 9200.
  The readiness probe (`es-readiness`) first waits for
  `clusterHealthCheckParams` (default `wait_for_status=yellow&timeout=1s`), once per
  container start. After that it only checks `/_cluster/health?local=true`. This
  matches the official chart's script, and it is why a `RollingUpdate` waits
  for the cluster to recover before it restarts the next pod.
* **Graceful stop.** With `drainOnShutdown.enabled`, the default, `es-prestop`
  registers `PUT /_nodes/<id>/shutdown` and waits, within the grace period, until the
  record is `COMPLETE`. A restart registers `type: restart` with an
  `allocation_delay`. A pod that is being *removed* (its ordinal ≥ the StatefulSet's
  replicas) registers `type: remove`. Once the node is back, the readiness probe
  deletes the record. `terminationGracePeriodSeconds` defaults to 120.
* **Master scale-down.** When a master-eligible pod is removed by a scale-down, preStop
  adds a voting config exclusion for it. It never does this on a restart. The
  `post-upgrade` hook Job (`es-cluster-cleanup`) clears the exclusions once the node
  has left, and deletes stale shutdown records.
* **Disruption and placement.** Each group gets a PodDisruptionBudget (`maxUnavailable: 1`).
  Pod anti-affinity on the hostname is soft by default (`antiAffinity: hard` makes it
  required). Each group is also spread across zones with `topologySpreadConstraints`.
* **Security context.** `fsGroup: 1000`, `runAsNonRoot`, all capabilities dropped,
  and seccomp `RuntimeDefault`.
* **Optional extras.** These are a privileged `vm.max_map_count` init container (off),
  a Prometheus exporter sidecar plus ServiceMonitor, and an Ingress.
  `values.schema.json` validates values.

**Day-2 procedures with the chart**

* *Rolling restart / config change:* `helm upgrade` handles it. Pods restart one at a time, highest ordinal first, and each restart waits for the cluster to return to yellow/green.
* *Add data nodes:* increase `replicas` and run `helm upgrade`.
* *Remove data nodes:* drain the leaving pods (the highest ordinals) first, then shrink:
  ```sh
  es() { curl --cacert ca.crt -u "elastic:$PASSWORD" "$@"; }   # see "What the chart creates: Security"
  es -XPUT https://localhost:9200/_cluster/settings -H 'Content-Type: application/json' \
    -d '{"persistent":{"cluster.routing.allocation.exclude._name":"prod-data-5"}}'
  es 'https://localhost:9200/_cat/shards?h=node' | grep -c prod-data-5   # wait for 0
  helm upgrade ... --set ...replicas=5
  es -XPUT https://localhost:9200/_cluster/settings -H 'Content-Type: application/json' \
    -d '{"persistent":{"cluster.routing.allocation.exclude._name":null}}'
  ```
  The preStop `remove` shutdown is a safety net bounded by the grace period, not a substitute for draining.
* *Remove master-eligible nodes:* one per `helm upgrade`, and never below a majority of the current voters. The preStop hook and the cleanup Job handle the voting exclusions.
* *Upgrades:* change `image.tag` and run `helm upgrade`. This is a rolling restart in reverse ordinal order.

## 5. Plain manifests (kustomize)

```sh
cluster/deploy/k8s/create-secrets.sh elasticsearch          # once: certificates + elastic password
kubectl apply -k cluster/deploy/k8s/overlays/single-node   # 1 node, discovery.type=single-node
kubectl apply -k cluster/deploy/k8s/overlays/three-node    # 3 master+data nodes
```

The base is the same design as the chart with one group: StatefulSet
`elasticsearch`, Services `elasticsearch` and `elasticsearch-headless`, a PDB,
a ServiceAccount with a Role, and the same probes and hooks. Security is on:
the pods need the Secrets `elasticsearch-certs` (`tls.crt`, `tls.key`,
`ca.crt`; one certificate for HTTP and transport that names the
`elasticsearch` Service) and `elasticsearch-credentials` (`password`), which
`create-secrets.sh` creates. The three-node
overlay ships the `elasticsearch-bootstrap` ConfigMap. Delete it and drop it
from the overlay once the cluster has formed. Set the image with
`kustomize edit set image elasticsearch-rs=<repo>:<tag>`.

## 6. Operator

The operator implements the ECK
`elasticsearch.k8s.elastic.co/v1` `Elasticsearch` resource. The source is in
`cluster/crates/elasticsearch-operator`.

```sh
kubectl apply -k cluster/deploy/operator                    # CRD + operator in elastic-system
kubectl apply -f cluster/deploy/operator/examples/quickstart.yaml
kubectl get es -w                                           # HEALTH NODES VERSION PHASE AGE
PASSWORD=$(kubectl get secret quickstart-es-elastic-user -o go-template='{{.data.elastic | base64decode}}')
kubectl get secret quickstart-es-http-certs-public -o go-template='{{index .data "ca.crt" | base64decode}}' > ca.crt
kubectl port-forward svc/quickstart-es-http 9200 &
curl --cacert ca.crt -u "elastic:$PASSWORD" https://localhost:9200/_cluster/health?pretty
```

`elasticsearch-operator crd` prints the CRD; `cluster/deploy/operator/crd.yaml` must match it.
Operator flags (with their environment equivalents): `--namespace` (`WATCH_NAMESPACE`),
`--default-image-repository` (`DEFAULT_IMAGE_REPOSITORY`, default `elasticsearch-rs`; the
image is `<repo>:<spec.version>` unless `spec.image` is set), `--es-timeout-secs`, and
`--health-addr`.

Supported spec fields are `version` and `image`. Each entry of `nodeSets[]` takes
`name`, `count`, `config`, `podTemplate` and `volumeClaimTemplates`. Top-level
`updateStrategy.changeBudget` takes `maxSurge` and `maxUnavailable`, there is a
`podDisruptionBudget` field, and `http.tls` takes ECK's
`selfSignedCertificate.subjectAltNames` and `certificate.secretName`. The status reports `health`, `phase`, `availableNodes`,
`version`, `observedGeneration` and `conditions`.

What the operator manages, using ECK naming:

* **Security, as ECK.** Every cluster runs with authentication, HTTPS and
  transport TLS; plain-text HTTP is not offered
  (`http.tls.selfSignedCertificate.disabled: true` is rejected as invalid,
  and `xpack.security.*` TLS settings in `nodeSets[].config` are ignored).
  The operator generates the `elastic` password in `<name>-es-elastic-user`
  (key `elastic`; the nodes get it as `ELASTIC_PASSWORD`), an HTTP CA
  (`<name>-es-http-ca-internal`) and certificate (`<name>-es-http-certs-internal`,
  naming `<name>-es-http[.<ns>[.svc]]`, every pod through its headless Service,
  localhost and your `subjectAltNames`), the client trust bundle
  `<name>-es-http-certs-public`, and a transport CA
  (`<name>-es-transport-ca-internal`) and certificate (`<name>-es-transport-certs`,
  one shared by all nodes; `verification_mode: certificate`). With
  `http.tls.certificate.secretName` your `tls.crt`/`tls.key`/`ca.crt` replace
  the generated HTTP certificate; it must name `<name>-es-http.<ns>.svc`.
  Certificates last a year and are re-issued 30 days before expiry, CAs last
  five years and roll over with the previous CA kept in the trust bundle; the
  nodes reload changed files without a restart. The operator talks to the
  cluster only over HTTPS, trusting only the cluster CA (the platform roots
  for a user certificate without `ca.crt`), as `elastic`; the readiness probe
  does the same against the local node. Clusters created by operator versions
  before this one ran without security: after upgrading the operator, delete
  all of their pods at once (a one-by-one restart cannot mix TLS and plain
  transport), or recreate them.
* **Objects.** Each nodeSet gets a StatefulSet `<name>-es-<nodeSet>` with
  `updateStrategy: OnDelete` and `podManagementPolicy: Parallel`, a headless
  Service of the same name, and a ConfigMap `<name>-es-<nodeSet>-es-config`
  holding the generated `elasticsearch.yml`. The cluster as a whole gets a
  headless `<name>-es-transport` Service (the discovery seed), a
  `<name>-es-http` Service and a `<name>-es-default` PDB.
* **Bootstrap.** `cluster.initial_master_nodes` is rendered only until the
  cluster first answers. The operator then records
  `elasticsearch.k8s.elastic.co/cluster-uuid` on the resource and drops the
  setting. The config hash excludes it, so dropping it causes no restart.
* **Scale-down.** Leaving data nodes are first added to
  `cluster.routing.allocation.exclude._name` (other names you put there are
  kept). A pod is removed only after `_cat/shards` shows nothing on it.
  Master-eligible nodes leave one at a time: the operator adds a voting
  exclusion, shrinks the StatefulSet, then clears the exclusions once the node
  has gone. A nodeSet removed from the spec is drained and then deleted.
  Upscales respect `maxSurge`.
* **Rolling changes.** Data and master pods restart one at a time: the operator
  calls `PUT /_nodes/{id}/shutdown` with type restart, waits for `COMPLETE`,
  deletes the pod, waits until it is back and the cluster is green again, then
  deletes the record. Non-master pods go before masters, highest ordinal first.
  Coordinating-only pods restart in parallel up to `maxUnavailable`. If the
  cluster is unreachable and no pod is ready, for example after a bad config,
  all outdated pods are replaced at once.
* **PDB.** The default allows one disruption while the cluster is green and none
  otherwise. Setting `podDisruptionBudget: {}` disables it.

The operator runs as a single Deployment replica with the `Recreate` strategy.
The zone attribute comes from the init container in
`examples/production.yaml`.

## 7. End-to-end test on kind

```sh
cluster/deploy/k8s/e2e.sh                 # needs docker, kind, kubectl, helm, jq
KEEP_CLUSTER=1 IMAGE=elasticsearch-rs:dev BUILD_IMAGE=0 cluster/deploy/k8s/e2e.sh
```

The script runs these steps:

1. Creates a kind cluster with 3 workers labelled `zone-a`, `zone-b` and `zone-c`.
2. Loads the image and installs the chart (security on, chart-generated certificates and password) with 3 dedicated masters, 2 data nodes and zone awareness.
3. Waits for green and checks that every node has `node.attr.zone`, then checks security: the client speaks HTTPS with the chart's CA and the generated password; anonymous requests, wrong passwords and plain HTTP are refused; transport requires TLS.
4. Indexes 3×300 documents.
5. Force-kills the elected master. It checks that a different master is elected and that the cluster returns to green with an unchanged document count.
6. Scales data nodes 2→3→2, draining before the shrink.
7. Runs `kubectl rollout restart` on every StatefulSet, asserting the exact count whenever all shards answer.
8. Runs `helm test --logs` (the test pod is kept until the next `helm test` so its logs can be read).

Results of the last run are in `E2E_RESULTS.md`.

On failure it dumps pods, events, logs, `_cat/nodes` and `_cat/shards`. On exit it deletes
the kind cluster, unless you set `KEEP_CLUSTER=1`.

---

## Production checklist

**Security.** Keep the defaults: authentication, HTTPS and transport TLS
(Helm `security.*`, the operator always). Prefer cert-manager or your own
certificate Secrets for production and GitOps, and a pre-created password
Secret. Change the `elastic` password after installation with
`POST _security/user/elastic/_password` (or `elasticsearch-reset-password`)
and then update the Secret the probes read, or create dedicated users and
roles for applications. Only `docker-compose.insecure.yml` and
`security.enabled: false` run without security, for development.

The `curl` examples below omit the TLS and credential options for brevity:
add `--cacert ca.crt -u "elastic:$PASSWORD"` and use `https://`.

**Memory sizing (there is no JVM heap).** Ignore `-Xms`/`-Xmx` and the
"50% of RAM for heap" rule. The node is a native process. Its memory has two parts:
* its working set: indexing buffers, caches, aggregation state, per-segment metadata, vector graphs;
* the **OS page cache** for memory-mapped segment files. The engine reads segment files through `mmap`.

The page cache is what makes search fast. Size the container memory **limit**
as working set plus the hot portion of the on-disk index, and leave the rest
of the host's RAM to the page cache. In Kubernetes, page cache pages from
files that a container touched are charged to its cgroup. Under memory
pressure the kernel reclaims them before it OOM-kills the process, so a
generous limit improves performance, while a tight one shows up as page faults
and disk reads, not crashes. Set requests equal to limits for memory to get
the Guaranteed QoS class on data nodes. Dedicated masters need little: 1–2 GiB.
Monitor RSS and major page faults per node.

**Bootstrap checks.** As in Elasticsearch, a node that binds or publishes its
transport on a non-loopback address (every Docker, Compose, Kubernetes and Helm
setup here, which bind `0.0.0.0`) refuses to start, with exit code 78 and
`bootstrap check failure [i] of [n]: ...` lines, when any of these fail:
`ulimit -n` at least 65535 (the node raises its soft limit to the hard one
first), max user threads (`ulimit -u`) at least 4096, unlimited virtual memory
(`ulimit -v`) and file size (`ulimit -f`), `vm.max_map_count` at least 262144
(Linux, unless `node.store.allow_mmap: false`), memory locked when
`bootstrap.memory_lock: true` (`ulimit -l unlimited` / `memlock` in
`limits.conf`, `LimitMEMLOCK=infinity` under systemd, `--ulimit memlock=-1:-1`
in Docker), one of `discovery.seed_hosts`, `discovery.seed_providers` or
`cluster.initial_master_nodes` set, and transport TLS while security is
enabled. `discovery.type: single-node` skips enforcement (the failures are only
logged), as does binding loopback only; `ES_JAVA_OPTS=-Des.enforce.bootstrap.checks=true`
enforces them anywhere.

`node.store.allow_mmap: false` skips the `vm.max_map_count` check as in
Elasticsearch, and the node then maps no store file. Unlike Elasticsearch's
`niofs`, which reads on demand through the page cache, this engine reads each
segment file into process memory when it opens it, so use it only where the
shards fit in memory. Those buffers are counted against the node's memory (on
the circuit breakers: with the default `indices.breaker.total.use_real_memory`,
the process's memory plus the file must stay under
`indices.breaker.total.limit`). A file that would go over is refused: the shard
copy fails with an error naming the file, its size and the limit (a merge whose
output would not fit fails the same way), and the node stays up. Once there is
room, retry with `POST _cluster/reroute?retry_failed`. The node also refuses
`index.store.type: mmapfs` and `hybridfs`, as Elasticsearch does.

`bootstrap.memory_lock` calls `mlockall(MCL_CURRENT)` once at start, as
Elasticsearch does, but there is no pre-sized JVM heap to lock. What it pins is what is mapped at that moment,
mostly the allocator's first arena (about 1 GiB, faulted in and resident from
the start, so allow for it in container memory limits). Memory allocated beyond
that arena, and the segment files mapped later, stay swappable. Disable swap on
the host instead where you can, which is also Elasticsearch's first
recommendation.

**`vm.max_map_count`.** Every memory-mapped segment file uses map areas, so
many shards and segments can exceed the kernel default of 65530. Set
`vm.max_map_count=262144` on every host (the bootstrap check refuses to start a
node below it): in the node pool configuration, with
a DaemonSet, or with the chart's `sysctlInitContainer.enabled=true`, which
runs a privileged init container. For Docker, run
`sysctl -w vm.max_map_count=262144` on the host.

**File descriptors.** Allow at least 65535 (`ulimit -n`). The compose file sets
this. Container runtimes usually default to 1048576, but check yours.

**Zone awareness.** Label the nodes with `topology.kubernetes.io/zone` (managed
Kubernetes does this). Then:
1. Enable `zoneAwareness` in the chart.
2. Set `cluster.routing.allocation.awareness.attributes: zone`.
3. Keep `topologySpreadConstraints` with `DoNotSchedule` for data and masters.
4. Run 3 dedicated masters, one per zone.
5. Use a StorageClass with `volumeBindingMode: WaitForFirstConsumer`.
6. Optionally set forced awareness (`cluster.routing.allocation.awareness.force.zone.values`)
   so a lost zone doesn't overload the surviving ones.

Every index needs `number_of_replicas ≥ 1` to survive a zone loss.

**PodDisruptionBudgets.** Keep them enabled (`maxUnavailable: 1` per group).
Node drains and cluster autoscaler evictions then go one pod at a time, and
the readiness gate makes each step wait for the cluster to recover. Check
that your cluster autoscaler and node-upgrade tooling respect PDBs.

**Graceful termination.** `terminationGracePeriodSeconds` must exceed
`drainOnShutdown.timeoutSeconds` plus the node's own shutdown time (flush). The
default is 120 s against a 100 s drain. Raise both for large nodes.

**Snapshots to S3.** Register a repository and a snapshot lifecycle policy:
```sh
curl -XPUT localhost:9200/_snapshot/s3-backups -H 'Content-Type: application/json' -d '{
  "type": "s3",
  "settings": { "bucket": "my-es-snapshots", "region": "us-east-1", "base_path": "prod" }
}'
curl -XPUT localhost:9200/_slm/policy/nightly -H 'Content-Type: application/json' -d '{
  "schedule": "0 30 1 * * ?", "name": "<nightly-{now/d}>", "repository": "s3-backups",
  "config": { "indices": ["*"] }, "retention": { "expire_after": "30d", "min_count": 5, "max_count": 50 }
}'
```
Give the pods AWS credentials through IRSA or EKS Pod Identity: set the
ServiceAccount annotation with `serviceAccount.annotations` in the chart.
Static keys work too, through `extraEnv` (`AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`) from a Secret. Test restores regularly.

**Storage.** Use SSD-backed, `ReadWriteOnce`, expandable StorageClasses
(`allowVolumeExpansion: true`). The chart retains PVCs when a StatefulSet is
scaled down or deleted, so delete the PVCs of permanently removed pods yourself.
Watch the disk watermarks (85/90/95%).

**Monitoring.** Enable `metrics.exporter` and `metrics.serviceMonitor`. Alert on
cluster status ≠ green, unassigned shards, disk usage, page-fault rate and
pod restarts.

**Network exposure.** Keep the client Service `ClusterIP` and restrict access
with NetworkPolicies. Expose the Ingress or a `LoadBalancer` only with security
(TLS and authentication) enabled.

**Remote clusters (cross-cluster search and replication).** A cluster reaches
another one over the remote cluster's *transport* port (9300), so that port
must be reachable from every node of the querying (or following) cluster with
the `remote_cluster_client` role (on by default; keep it on the coordinating
and data nodes). Configure the remote with dynamic cluster settings, as in
Elasticsearch:

```sh
curl -XPUT -H 'content-type: application/json' https://es:9200/_cluster/settings -d '{
  "persistent": {"cluster.remote.leader": {"seeds": ["leader-transport.example:9300"], "skip_unavailable": false}}
}'
# or behind a load balancer / proxy:
#   {"cluster.remote.leader": {"mode": "proxy", "proxy_address": "leader-lb.example:9300"}}
curl https://es:9200/_remote/info
```

In sniff mode the gateway nodes the seed reports (every node that is not a
dedicated master, or those with `node.attr.<cluster.remote.node.attr>: true`)
must be reachable at their published transport addresses; behind NAT or a
single load balancer use proxy mode. Security follows Elasticsearch's
certificate-based remote cluster model: both clusters run transport TLS
and trust each other's transport CA (`xpack.security.transport.ssl.certificate_authorities`
lists both CAs; narrow it with `trust_restrictions`); the remote cluster
authorizes each request as the calling user, by its own definitions of the
user's role names, so define the roles (for example `read` and
`read_cross_cluster` on the indices searched, `read_ccr` and `monitor` +
`read` on leader indices) on the remote cluster under the same names. API key
based remote clusters (`cluster.remote.<alias>.credentials`, the remote
cluster server port 9443) are not supported: an alias with credentials does
not connect.

---

## Contract the `elasticsearch` binary must honour

Everything above assumes the following of the node binary (crate
`elasticsearch-node`, binary target `elasticsearch`):

1. **Location and arguments.** The binary is `/usr/share/elasticsearch/bin/elasticsearch`.
   It accepts repeated `-E key=value` (also `-Ekey=value`). A value may be a
   comma-separated list for list settings, as in
   `-E cluster.initial_master_nodes=a,b,c` or `node.roles=master,data`.
   An empty value for `node.roles` means no roles (coordinating-only).
2. **Configuration sources.** It reads `$ES_PATH_CONF/elasticsearch.yml` with `${VAR}`
   substitution, honours dotted environment variables as settings, and gives
   `-E` arguments the highest precedence. An empty or comment-only file is valid.
3. **Settings it must accept.**
   * Applied by the operator and the chart: `cluster.name`, `node.name`,
     `node.roles`, `node.attr.*`, `network.host`, `network.publish_host`,
     `http.port`, `transport.port`, `path.data`, `path.logs`,
     `discovery.seed_hosts`, `discovery.type: single-node`,
     `cluster.initial_master_nodes`,
     `cluster.routing.allocation.awareness.attributes` (and `.force.*`),
     `cluster.routing.allocation.exclude._name`, `action.destructive_requires_name`.
   * In the compose file: `xpack.security.enabled: false`, which turns security
     off as in ES.
   * Unknown settings: reject them as ES does, but those listed here must be known.
4. **Discovery.**
   * A `discovery.seed_hosts` entry may be a DNS name that resolves to **several**
     A records, such as a headless Service with `publishNotReadyAddresses`. The node
     must probe every address, defaulting to port 9300, and must re-resolve on every
     discovery round, because pod IPs change.
   * The node must bind HTTP on 9200 and transport on 9300 when `network.host: 0.0.0.0`.
     It must publish `network.publish_host`, which is set to the pod IP.
5. **Signals.** On SIGTERM, shut down gracefully: stop accepting requests, flush,
   leave the cluster, and exit 0 within the grace period. The process runs as a
   child of `tini`.
6. **Filesystem.** Run as uid 1000 with gid 0, and write only under `path.data`,
   `path.logs` and `/tmp`. The config directory may be read-only: the operator
   mounts a ConfigMap there, and Helm mounts `elasticsearch.yml` with `subPath`.
   **The node ID must persist in `path.data` across restarts.** Shutdown records
   and the operator's restart tracking are keyed by node ID.
7. **REST endpoints and their shapes.** All must be ES-compatible.
   * `GET /`: includes `cluster_uuid`, which is `"_na_"` until the cluster has formed.
   * `GET /_cluster/health`:
     * supports `wait_for_status` and `timeout`. If the wait times out, it returns HTTP
       408 with `timed_out: true`;
     * supports `local=true`, which answers HTTP 200 from local state whenever the HTTP
       layer is up (it is used for liveness and readiness);
     * without a master and without `local`, it returns 503;
     * the body includes `status`, `cluster_name`, `number_of_nodes`, `relocating_shards`,
       `initializing_shards`, `unassigned_shards` and `timed_out`.
   * `GET /_nodes/_local`: `{"nodes":{"<id>":{"name":…, "roles":[…]}}}`.
   * `GET /_cat/nodes`, with `format=json`, `full_id=true` and
     `h=id,name,node.role,master` (`master` is `*` for the elected master);
     text output with `h=name` and `h=id`.
   * `GET /_cat/master?h=node`, `GET /_cat/nodeattrs?h=node,attr,value`,
     `GET /_cat/allocation?v`.
   * `GET /_cat/shards`, with `format=json` and `h=index,shard,prirep,state,node`
     (`node` is null when unassigned; a relocating shard prints `src -> ip id dst`);
     text output with `h=node`.
   * `GET /_cluster/settings?flat_settings=true` and `PUT /_cluster/settings`, with
     `persistent` and `transient` sections. `null` resets a setting.
   * `GET /_cluster/state/metadata?filter_path=metadata.cluster_coordination.voting_config_exclusions`.
     It returns entries of the form `{node_id, node_name}`.
   * `POST /_cluster/voting_config_exclusions?node_names=a,b[&timeout=]`.
   * `DELETE /_cluster/voting_config_exclusions?wait_for_removal=false`.
   * `PUT /_nodes/{node_id}/shutdown` with the body
     `{"type":"restart"|"remove","reason":…,"allocation_delay":"5m"}` (`allocation_delay`
     applies to `restart` only).
   * `GET /_nodes/{node_id}/shutdown` and `GET /_nodes/shutdown`. Both return
     `{"nodes":[{"node_id","type":"RESTART"|"REMOVE","reason","status":"NOT_STARTED"|"IN_PROGRESS"|"STALLED"|"COMPLETE"}]}`.
     A `restart` must reach `COMPLETE` promptly. A `remove` must reach it once the
     node holds no shards.
   * `DELETE /_nodes/{node_id}/shutdown`, which returns 404 when there is no record.
   * The e2e test also uses index create, `_bulk?refresh=true`, `_count` (with
     `_shards.failed`) and `_cluster/allocation`-related `_cat` APIs. The metrics
     exporter, if enabled, uses `_nodes/_local/stats` and `_cluster/health`.
8. **Allocation behaviour.** The operator and the procedures rely on:
   * `cluster.routing.allocation.exclude._name` moving shards off the named nodes;
   * a `restart` shutdown record delaying replica reallocation of that node's shards
     for `allocation_delay`;
   * voting exclusions removing the node from the voting configuration before the
     call returns.
