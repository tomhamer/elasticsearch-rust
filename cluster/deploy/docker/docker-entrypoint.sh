#!/bin/bash
# Entrypoint of the elasticsearch image. Runs under tini (PID 1), which
# forwards SIGTERM to the node; we `exec` so the node is tini's direct child.
#
#   docker run IMAGE                      -> elasticsearch
#   docker run IMAGE -E key=value ...     -> elasticsearch -E key=value ...
#   docker run IMAGE eswrapper [args]     -> elasticsearch [args]
#   docker run IMAGE <other command>      -> runs that command
#
# Settings sources, lowest to highest precedence (all handled by the node
# itself except the settings directories):
#   1. $ES_PATH_CONF/elasticsearch.yml (with ${ENV} substitution)
#   2. environment variables named like settings (`discovery.type=single-node`)
#   3. files in $ES_SETTINGS_DIRS (colon-separated; default
#      /usr/share/elasticsearch/settings.d): every regular file (searched two
#      levels deep, hidden paths skipped) is a setting whose key is the file
#      name and whose value is the file content (trimmed). Passed as
#      `-E key=value`. Used by Kubernetes for values only known at runtime
#      (e.g. `node.attr.zone` from the node's topology label, written by an
#      init container) and for `cluster.initial_master_nodes` from a
#      ConfigMap that is emptied after bootstrap.
#   4. explicit `-E` arguments.
set -euo pipefail

ES_BIN="${ES_HOME:-/usr/share/elasticsearch}/bin/elasticsearch"

if [[ $# -eq 0 || "$1" == "eswrapper" || "$1" == -* ]]; then
  if [[ $# -gt 0 && "$1" == "eswrapper" ]]; then
    shift
  fi

  if [[ "$(id -u)" == "0" && "${ES_ALLOW_ROOT:-false}" != "true" ]]; then
    echo "elasticsearch: refusing to run as root; run as uid 1000 (the image default)" >&2
    exit 1
  fi

  extra=()
  IFS=':' read -r -a dirs <<< "${ES_SETTINGS_DIRS:-/usr/share/elasticsearch/settings.d}"
  for dir in "${dirs[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' f; do
      key="$(basename "$f")"
      value="$(tr -d '\r' < "$f" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      [[ -n "$value" ]] || continue
      extra+=("-E" "${key}=${value}")
    done < <(find -L "$dir" -mindepth 1 -maxdepth 2 -type f ! -path '*/.*' -print0 | sort -z)
  done

  if [[ ${#extra[@]} -gt 0 ]]; then
    echo "elasticsearch: settings from ${ES_SETTINGS_DIRS:-settings.d}: ${extra[*]}" >&2
  fi

  # As the official image: ENROLLMENT_TOKEN becomes --enrollment-token (not
  # echoed: the token carries an API key).
  if [[ -n "${ENROLLMENT_TOKEN:-}" ]]; then
    extra+=("--enrollment-token" "${ENROLLMENT_TOKEN}")
  fi

  # As ES's image: `ELASTIC_PASSWORD` becomes the keystore's
  # `bootstrap.password` (a secure setting: never in elasticsearch.yml), and a
  # password-protected keystore's password is fed to the node from
  # `KEYSTORE_PASSWORD` on standard input. A read-only config directory only
  # warns (the node also reads `ELASTIC_PASSWORD` itself).
  if [[ -n "${ELASTIC_PASSWORD:-}" ]]; then
    ks="${ES_HOME:-/usr/share/elasticsearch}/bin/elasticsearch-keystore"
    {
      [[ -f "${ES_PATH_CONF}/elasticsearch.keystore" ]] || "$ks" create >/dev/null
      if ! "$ks" has-passwd --silent; then
        "$ks" list | grep -q '^bootstrap.password$' \
          || printf '%s\n' "$ELASTIC_PASSWORD" | "$ks" add -x 'bootstrap.password'
      else
        printf '%s\n' "${KEYSTORE_PASSWORD:-}" | "$ks" list | grep -q '^bootstrap.password$' \
          || printf '%s\n%s\n' "${KEYSTORE_PASSWORD:-}" "$ELASTIC_PASSWORD" | "$ks" add -x 'bootstrap.password'
      fi
    } || echo "elasticsearch: could not store ELASTIC_PASSWORD in the keystore" >&2
  fi
  exec "$ES_BIN" "${extra[@]}" "$@" <<<"${KEYSTORE_PASSWORD:-}"
fi

exec "$@"
