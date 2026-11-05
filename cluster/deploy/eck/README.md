# Running the Rust engine under Elastic Cloud on Kubernetes (ECK)

The node image built from `cluster/deploy/docker/Dockerfile` runs under the
**stock** ECK operator (`elastic/cloud-on-k8s` 3.x, unmodified, installed from
Elastic's published manifests). You write a normal `Elasticsearch` resource
and set `spec.image` to this image. There is no rustic operator, Helm chart
or sidecar involved: ECK runs its own init containers, generates TLS
certificates and the `elastic` password, maintains the seed hosts file,
probes readiness, and performs scaling and rolling changes through the
node's ES APIs.

Tested with ECK 3.5.0 and `spec.version: 9.5.5` (see `e2e.sh`).

## 1. Install ECK (Elastic's manifests, unchanged)

```sh
kubectl create -f https://download.elastic.co/downloads/eck/3.5.0/crds.yaml
kubectl apply  -f https://download.elastic.co/downloads/eck/3.5.0/operator.yaml
kubectl -n elastic-system rollout status statefulset/elastic-operator
```

Elastic's Helm chart (`elastic/eck-operator`) works the same way.

## 2. Build and publish the image

```sh
# From the repository root (BuildKit; Debian bookworm runtime):
docker build -f cluster/deploy/docker/Dockerfile -t <registry>/elasticsearch-rs:<tag> .
docker push <registry>/elasticsearch-rs:<tag>
```

If the three binaries are already built (`elasticsearch`,
`elasticsearch-reset-password`, `elasticsearch-keystore`,
`elasticsearch-create-enrollment-token`), copy them to
`cluster/deploy/docker/.prebuilt/` and add
`--build-arg BINARIES=prebuilt --build-arg RUNTIME_IMAGE=<base with the glibc they were linked against>`
(e.g. `ubuntu:24.04` for binaries built on Ubuntu 24.04).

On kind: `kind load docker-image <image> --name <cluster>`.

## 3. Apply an Elasticsearch resource

[`elasticsearch.yaml`](elasticsearch.yaml) is a minimal example: 3 nodes,
2 GiB each. Change `spec.image`, then:

```sh
kubectl apply -f elasticsearch.yaml
kubectl get elasticsearch rustic -w          # HEALTH green, PHASE Ready
PASSWORD=$(kubectl get secret rustic-es-elastic-user -o go-template='{{.data.elastic | base64decode}}')
kubectl get secret rustic-es-http-certs-public -o go-template='{{index .data "ca.crt" | base64decode}}' > ca.crt
kubectl port-forward svc/rustic-es-http 9200 &
curl --cacert ca.crt --connect-to rustic-es-http.default.svc:9200:127.0.0.1:9200 \
     -u "elastic:$PASSWORD" https://rustic-es-http.default.svc:9200/
```

Everything in ECK's `Elasticsearch` spec applies as documented by Elastic:
`nodeSets` (`count`, `config`, `podTemplate`, `volumeClaimTemplates`),
`http.tls`, `secureSettings`, `updateStrategy`, `podDisruptionBudget`, and
Kibana resources with `elasticsearchRef`.

`spec.version` must be an ES version ECK supports (8.18+ or 9.x). Use the
version the engine implements, `9.5.5`; ECK chooses its behaviour from it
(readiness port, node shutdown API, desired nodes, file-based settings).

## What the image and node provide for ECK

| ECK expects | Provided by |
|---|---|
| `/usr/share/elasticsearch/{bin,config,plugins,data,logs}`, uid 1000 gid 0, `bash`, coreutils | the image |
| `LICENSE.txt` with one of Elastic's distribution license headings (else the init container exits 42, `unsupported_distribution`) | not in the image yet (pending a licensing decision): without it ECK cannot start the pods |
| `nc` for the readiness probe (`nc -z 127.0.0.1 8080`) | `netcat-openbsd` in the image |
| `curl`, `date --iso-8601`, `mktemp`, `grep`, `cut` for the pre-stop hook | the image |
| `bin/elasticsearch-keystore create` / `add-file` (with `secureSettings`) | the keystore CLI |
| `bin/elasticsearch-plugin` (user init containers) | a CLI with no plugin runtime: `list`; `install` succeeds for built-in features and ES-9 modules, fails with ES's exit codes otherwise |
| `readiness.port: 8080`: a TCP port open only while a master is elected and the node is not shutting down | the node's readiness service (ES `ReadinessService`) |
| `discovery.seed_providers: file` + `config/unicast_hosts.txt` rewritten as pods come and go | the node re-reads the file every discovery round |
| `network.host: "0"`, `${POD_IP}` / `${POD_NAME}` substitution, `node.attr.k8s_node_name`, allocation awareness | node settings |
| file realm `users` (bcrypt) / `users_roles` / `roles.yml` / `service_tokens` (`{PBKDF2_STRETCH}`), reloaded on change; `reserved_realm.enabled: false` with `elastic` in the file realm; realms named `file1` / `native1` | the security layer |
| HTTP and transport TLS from ECK's PEM files (PKCS#1 RSA keys, `verification_mode: certificate`) | TLS settings |
| certificates ECK rotates (re-issued before expiry, new SANs) used without a restart | key/trust files re-read when they change (ES `SSLConfigurationReloader`, 5s) |
| `xpack.security.transport.ssl.trust_restrictions.path` (`trust.yml`, set by users, e.g. for remote clusters) | trust restrictions on the `otherName` CN / DNS SANs (`*`/`?` patterns) |
| APIs: `GET /`, `_cluster/health` (incl. `wait_for_events=languid&timeout=0s` -> 408 with the health body), `_nodes/_all/no-metrics`, `_nodes/_all/stats/os` (cgroup), `_nodes/_master`, `_cat/nodes`, `_cat/shards`, `_cluster/settings`, `_cluster/voting_config_exclusions`, `_nodes/{id}/shutdown`, `_internal/desired_nodes`, `_license` (+ `start_basic` / `start_trial`), `_security/_authenticate` | the REST layer |

## Not supported / differences

* Elasticsearch plugins cannot be installed (no plugin runtime). Built-in
  equivalents: `mapper-size`, `mapper-murmur3`, `mapper-annotated-text`, and
  the repository/ingest modules.
* File-based settings (`config/operator/settings.json`, used by ECK's
  enterprise `StackConfigPolicy`) are not applied; ECK's default empty
  file is harmless. Readiness therefore does not wait for it.
* Remote clusters through ECK (`remoteClusters`, `remoteClusterServer`) and
  CCR are not covered by the end-to-end test.
* Kibana from ECK (`Kibana` with `elasticsearchRef`, the `elastic/kibana`
  service account token ECK writes to `service_tokens`) reaches `green`;
  its `kibana_system` privileges follow ES 9.5.5.
* `GET _nodes/stats/os` reports the host's memory as
  `os.mem.adjusted_total_in_bytes`; ES reports the container limit there.
  The node's own memory budget and circuit breakers do follow the
  container limit (`process.mem.native_budget_in_bytes`).
* The licence is always the engine's own; ECK's licence management
  (`start_basic` on a non-basic licence, uploading an enterprise licence
  Secret) is accepted but does not change behaviour.

## End-to-end test

[`e2e.sh`](e2e.sh) creates a kind cluster, loads the image, installs stock
ECK, applies a 3-node `Elasticsearch`, checks green, HTTPS with ECK's CA and
the ECK-generated `elastic` password, indexes and searches, scales 3 -> 5 -> 3,
rolls a config change and an image change through ECK, force-kills a pod,
and optionally (`KIBANA=1`) runs a stock `Kibana` against it:

```sh
# On a Linux box with docker, kind, kubectl, jq, curl:
cluster/deploy/eck/e2e.sh                               # builds the image in docker
PREBUILT_DIR=$CARGO_TARGET_DIR/fast cluster/deploy/eck/e2e.sh   # reuse built binaries
KIBANA=1 KEEP_CLUSTER=1 cluster/deploy/eck/e2e.sh
```
