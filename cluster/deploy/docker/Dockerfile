# syntax=docker/dockerfile:1.7
#
# elasticsearch (Rust) node image.
#
# Build from the REPOSITORY ROOT (the node depends on crates/ and cluster/):
#
#   docker build -f cluster/deploy/docker/Dockerfile -t elasticsearch-rs:dev .
#
# BuildKit picks up cluster/deploy/docker/Dockerfile.dockerignore automatically
# (it sits next to this Dockerfile), so target/ dirs and bench corpora are not
# sent as build context.
#
# Layout matches the official image:
#   /usr/share/elasticsearch/{bin,config,data,logs,plugins}, uid 1000 `elasticsearch`,
#   ES_PATH_CONF=/usr/share/elasticsearch/config, ports 9200 (HTTP) / 9300 (transport).

ARG RUST_VERSION=1.99
ARG DEBIAN_RELEASE=bookworm
# `builder` compiles the binaries in the image build. `prebuilt` takes them
# from cluster/deploy/docker/.prebuilt/ (elasticsearch,
# elasticsearch-reset-password, elasticsearch-keystore,
# elasticsearch-create-enrollment-token built on the same
# architecture and glibc as the runtime image; git-ignored), so a host
# that already built them does not compile twice.
ARG BINARIES=builder
# The runtime base. With `BINARIES=prebuilt` it must have at least the glibc
# the binaries were linked against (e.g. `RUNTIME_IMAGE=ubuntu:24.04` for
# binaries built on Ubuntu 24.04).
ARG RUNTIME_IMAGE=debian:${DEBIAN_RELEASE}-slim

############################ builder ############################
FROM rust:${RUST_VERSION}-slim-${DEBIAN_RELEASE} AS builder
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*
# Flaky HTTP/2 to crates.io (seen on EC2) -> HTTP/1.1 + more retries.
ENV CARGO_HTTP_MULTIPLEXING=false CARGO_NET_RETRY=10
WORKDIR /src
COPY . .
# `--locked` builds exactly the root Cargo.lock. Pass `--build-arg CARGO_LOCKED=`
# to build while the lock file is behind a Cargo.toml change.
ARG CARGO_LOCKED=--locked
ARG CARGO_PROFILE=release
# Cache mounts keep the registry and target dir across builds without
# bloating layers.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build ${CARGO_LOCKED} --profile "${CARGO_PROFILE}" \
        -p elasticsearch --bin elasticsearch --bin elasticsearch-reset-password --bin elasticsearch-keystore --bin elasticsearch-create-enrollment-token \
 && mkdir -p /out \
 && for b in elasticsearch elasticsearch-reset-password elasticsearch-keystore elasticsearch-create-enrollment-token; do \
      cp "target/$( [ "${CARGO_PROFILE}" = dev ] && echo debug || echo "${CARGO_PROFILE}" )/$b" "/out/$b" \
      && { strip "/out/$b" || true; }; \
    done

FROM scratch AS prebuilt
COPY cluster/deploy/docker/.prebuilt/ /out/

FROM ${BINARIES} AS binaries

############################ runtime ############################
FROM ${RUNTIME_IMAGE}
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="elasticsearch" \
      org.opencontainers.image.description="Elasticsearch-compatible search engine written in Rust" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"

# tini: PID 1, reaps zombies and forwards SIGTERM to the node (with -g, to
# its whole process group). curl + jq: healthcheck, probes and the k8s
# helper scripts. netcat-openbsd: `nc`, which ECK's readiness probe runs
# (`nc -z -v -w5 127.0.0.1 8080`, the `readiness.port` service); ECK's
# pre-stop hook also needs curl and GNU coreutils (`date --iso-8601`).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini curl jq ca-certificates netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

ENV ES_HOME=/usr/share/elasticsearch \
    ES_PATH_CONF=/usr/share/elasticsearch/config \
    ES_SETTINGS_DIRS=/usr/share/elasticsearch/settings.d \
    PATH=/usr/share/elasticsearch/bin:$PATH

# uid 1000 / gid 0 like the official image: files are group-writable by
# root's group so arbitrary-UID platforms (OpenShift) work too.
RUN { ! id -u ubuntu >/dev/null 2>&1 || userdel ubuntu; } \
 && { ! getent group ubuntu >/dev/null || groupdel ubuntu; } \
 && groupadd -g 1000 elasticsearch \
 && useradd -u 1000 -g 1000 -G 0 -d /usr/share/elasticsearch -M -s /bin/bash elasticsearch \
 && mkdir -p /usr/share/elasticsearch/bin /usr/share/elasticsearch/config \
             /usr/share/elasticsearch/data /usr/share/elasticsearch/logs \
             /usr/share/elasticsearch/plugins /usr/share/elasticsearch/settings.d

COPY --from=binaries /out/elasticsearch /usr/share/elasticsearch/bin/elasticsearch
COPY --from=binaries /out/elasticsearch-reset-password /usr/share/elasticsearch/bin/elasticsearch-reset-password
COPY --from=binaries /out/elasticsearch-keystore /usr/share/elasticsearch/bin/elasticsearch-keystore
COPY --from=binaries /out/elasticsearch-create-enrollment-token /usr/share/elasticsearch/bin/elasticsearch-create-enrollment-token
COPY cluster/deploy/docker/elasticsearch.yml /usr/share/elasticsearch/config/elasticsearch.yml
COPY cluster/deploy/docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY cluster/deploy/docker/bin/ /usr/share/elasticsearch/bin/

RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh /usr/share/elasticsearch/bin/* \
 && chown -R 1000:0 /usr/share/elasticsearch \
 && chmod -R g=u /usr/share/elasticsearch

WORKDIR /usr/share/elasticsearch
USER 1000:0
EXPOSE 9200 9300
VOLUME ["/usr/share/elasticsearch/data"]

# Security is on by default (as in ES 8/9): on first start the node
# auto-configures HTTPS, so the check accepts an HTTPS or HTTP answer, and
# 401 (authentication required) counts as up.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=6 \
  CMD code="$(curl -sk -o /dev/null -w '%{http_code}' --max-time 2 "https://127.0.0.1:9200/_cluster/health?local=true" || true)"; \
      if [ "$code" = 000 ]; then code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:9200/_cluster/health?local=true" || true)"; fi; \
      case "$code" in 200|401) exit 0 ;; *) exit 1 ;; esac

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["eswrapper"]
