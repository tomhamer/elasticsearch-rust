{{/* vim: set filetype=mustache: */}}

{{- define "es.clusterName" -}}
{{- .Values.clusterName | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{- define "es.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "es.selectorLabels" -}}
app.kubernetes.io/name: elasticsearch
app.kubernetes.io/instance: {{ .Release.Name }}
elasticsearch/cluster: {{ include "es.clusterName" . }}
{{- end -}}

{{- define "es.labels" -}}
{{ include "es.selectorLabels" . }}
helm.sh/chart: {{ include "es.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- with .Values.labels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
A node group with nodeGroupDefaults applied. Usage:
  {{- $g := include "es.group" (dict "root" $ "group" .) | fromYaml }}
Keys set on the group win, including false / empty values (sprig's
mergeOverwrite would ignore those). Map values (config, resources,
persistence, topologySpread, nodeSelector, ...) are merged one level deep.
*/}}
{{- define "es.group" -}}
{{- $d := .root.Values.nodeGroupDefaults | default dict -}}
{{- $g := .group -}}
{{- $out := dict -}}
{{- range $k, $v := $d -}}
{{- if hasKey $g $k -}}
{{- $gv := get $g $k -}}
{{- if and (kindIs "map" $v) (kindIs "map" $gv) -}}
{{- $m := dict -}}
{{- range $k2, $v2 := $v }}{{ $_ := set $m $k2 $v2 }}{{ end -}}
{{- range $k2, $v2 := $gv }}{{ $_ := set $m $k2 $v2 }}{{ end -}}
{{- $_ := set $out $k $m -}}
{{- else -}}
{{- $_ := set $out $k $gv -}}
{{- end -}}
{{- else -}}
{{- $_ := set $out $k $v -}}
{{- end -}}
{{- end -}}
{{- range $k, $v := $g }}{{ if not (hasKey $out $k) }}{{ $_ := set $out $k $v }}{{ end }}{{ end -}}
{{- toYaml $out -}}
{{- end -}}

{{/* elasticsearch.yml settings of a group: values.config overlaid with the group's config (top-level keys). */}}
{{- define "es.groupConfig" -}}
{{- $out := dict -}}
{{- range $k, $v := (.root.Values.config | default dict) }}{{ $_ := set $out $k $v }}{{ end -}}
{{- range $k, $v := (.group.config | default dict) }}{{ $_ := set $out $k $v }}{{ end -}}
{{- if $out }}{{ toYaml $out }}{{ end -}}
{{- end -}}

{{/* StatefulSet / pod-name prefix of a group: <clusterName>-<group> */}}
{{- define "es.groupFullname" -}}
{{- printf "%s-%s" (include "es.clusterName" .root) .group.name | trunc 52 | trimSuffix "-" -}}
{{- end -}}

{{- define "es.groupSelectorLabels" -}}
{{ include "es.selectorLabels" .root }}
app.kubernetes.io/component: {{ .group.name }}
{{- end -}}

{{- define "es.isMaster" -}}
{{- if has "master" .roles }}true{{ end -}}
{{- end -}}

{{- define "es.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "es.clusterName" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "es.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}

{{/* Headless services of master-eligible groups, comma-separated: discovery.seed_hosts */}}
{{- define "es.seedHosts" -}}
{{- $root := . -}}
{{- $hosts := list -}}
{{- range .Values.nodeGroups -}}
{{- $g := include "es.group" (dict "root" $root "group" .) | fromYaml -}}
{{- if and (has "master" $g.roles) (gt (int $g.replicas) 0) -}}
{{- $hosts = append $hosts (printf "%s-headless" (include "es.groupFullname" (dict "root" $root "group" $g))) -}}
{{- end -}}
{{- end -}}
{{- join "," $hosts -}}
{{- end -}}

{{/* Pod names of every master-eligible node: cluster.initial_master_nodes */}}
{{- define "es.initialMasterNodes" -}}
{{- $root := . -}}
{{- $nodes := list -}}
{{- range .Values.nodeGroups -}}
{{- $g := include "es.group" (dict "root" $root "group" .) | fromYaml -}}
{{- if has "master" $g.roles -}}
{{- $name := include "es.groupFullname" (dict "root" $root "group" $g) -}}
{{- range $i := until (int $g.replicas) -}}
{{- $nodes = append $nodes (printf "%s-%d" $name $i) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- join "," $nodes -}}
{{- end -}}

{{/*
"true" once the cluster is known to have formed: clusterBootstrapped is set;
or the release's bootstrap ConfigMap already records formation (the
`elasticsearch/cluster-formed` annotation, written below, so the decision is
sticky); or a master-eligible StatefulSet of this release reports a ready
replica. A pod first turns ready only once /_cluster/health answers
(es-readiness), which needs an elected master, so a ready master pod proves
the cluster formed. A StatefulSet merely existing does not: a first install
whose masters never started (bad image, bad config) must keep
cluster.initial_master_nodes through the corrective upgrade (production
review #416). `lookup` returns nothing under `helm template` / --dry-run, so
offline rendering always includes the bootstrap list (harmless: a formed
cluster ignores it).
*/}}
{{- define "es.bootstrapped" -}}
{{- $root := . -}}
{{- $sts := list -}}
{{- range .Values.nodeGroups -}}
{{- $g := include "es.group" (dict "root" $root "group" .) | fromYaml -}}
{{- if has "master" $g.roles -}}
{{- $sts = append $sts (lookup "apps/v1" "StatefulSet" $root.Release.Namespace (include "es.groupFullname" (dict "root" $root "group" $g))) -}}
{{- end -}}
{{- end -}}
{{- $cm := lookup "v1" "ConfigMap" .Release.Namespace (printf "%s-bootstrap" (include "es.clusterName" .)) -}}
{{- include "es.formed" (dict "clusterBootstrapped" .Values.clusterBootstrapped "configMap" $cm "masterStatefulSets" $sts) -}}
{{- end -}}

{{/*
The formation decision over looked-up objects (empty dicts when absent):
dict "clusterBootstrapped" bool "configMap" <bootstrap ConfigMap>
"masterStatefulSets" (list <StatefulSet>...). Separate from the lookups so
deploy/helm/bootstrap-test.sh can render it with fixtures.
*/}}
{{- define "es.formed" -}}
{{- $found := .clusterBootstrapped -}}
{{- if and .configMap (eq (dig "metadata" "annotations" "elasticsearch/cluster-formed" "" .configMap) "true") -}}
{{- $found = true -}}
{{- end -}}
{{- range .masterStatefulSets -}}
{{- if and . (gt (int (dig "status" "readyReplicas" 0 .)) 0) -}}
{{- $found = true -}}
{{- end -}}
{{- end -}}
{{- if $found }}true{{ end -}}
{{- end -}}

{{/* ---------------------------------------------------------------- security */}}

{{/* "true" when HTTPS is served (security on and HTTP TLS on). */}}
{{- define "es.httpsEnabled" -}}
{{- if and .Values.security.enabled .Values.security.tls.http.enabled }}true{{ end -}}
{{- end -}}

{{- define "es.scheme" -}}
{{- if include "es.httpsEnabled" . }}https{{ else }}http{{ end -}}
{{- end -}}

{{/* Client Service DNS name (in every generated HTTP certificate). */}}
{{- define "es.serviceHost" -}}
{{- printf "%s.%s.svc" (include "es.clusterName" .) .Release.Namespace -}}
{{- end -}}

{{- define "es.clientUrl" -}}
{{- printf "%s://%s:%d" (include "es.scheme" .) (include "es.serviceHost" .) (int .Values.service.httpPort) -}}
{{- end -}}

{{/* Secret + key of the elastic password. */}}
{{- define "es.credentialsSecret" -}}
{{- .Values.security.elasticPassword.existingSecret | default (printf "%s-credentials" (include "es.clusterName" .)) -}}
{{- end -}}
{{- define "es.credentialsKey" -}}
{{- if .Values.security.elasticPassword.existingSecret }}{{ .Values.security.elasticPassword.key | default "password" }}{{ else }}password{{ end -}}
{{- end -}}

{{/* "true" when the chart generates certificates itself (Helm genCA). */}}
{{- define "es.certsGenerated" -}}
{{- $t := .Values.security.tls -}}
{{- if and .Values.security.enabled (eq $t.provider "generate") (or (and $t.http.enabled (not $t.http.secretName)) (not $t.transport.secretName)) }}true{{ end -}}
{{- end -}}

{{- define "es.certManagerIssuerName" -}}
{{- .Values.security.tls.certManager.issuerRef.name | default (printf "%s-ca" (include "es.clusterName" .)) -}}
{{- end -}}

{{/*
Volume source of the HTTP / transport certificates (`kind`: http | transport):
always presented to the node as tls.crt, tls.key and ca.crt.
*/}}
{{- define "es.certVolume" -}}
{{- $root := .root -}}
{{- $t := $root.Values.security.tls -}}
{{- $cluster := include "es.clusterName" $root -}}
{{- $own := ternary $t.http.secretName $t.transport.secretName (eq .kind "http") -}}
{{- if $own }}
secret:
  secretName: {{ $own }}
  defaultMode: 0440
  items:
    - { key: tls.crt, path: tls.crt }
    - { key: tls.key, path: tls.key }
    - { key: ca.crt, path: ca.crt }
{{- else if eq $t.provider "cert-manager" }}
secret:
  secretName: {{ printf "%s-%s-certs" $cluster .kind }}
  defaultMode: 0440
  items:
    - { key: tls.crt, path: tls.crt }
    - { key: tls.key, path: tls.key }
    - { key: ca.crt, path: ca.crt }
{{- else }}
secret:
  secretName: {{ printf "%s-certs" $cluster }}
  defaultMode: 0440
  items:
    - { key: {{ printf "%s.crt" .kind }}, path: tls.crt }
    - { key: {{ printf "%s.key" .kind }}, path: tls.key }
    - { key: ca.crt, path: ca.crt }
{{- end }}
{{- end -}}

{{/* DNS names of the HTTP certificate. */}}
{{- define "es.httpDnsNames" -}}
{{- $root := . -}}
{{- $cluster := include "es.clusterName" . -}}
{{- $ns := .Release.Namespace -}}
{{- $domain := .Values.security.tls.clusterDomain -}}
{{- $names := list $cluster (printf "%s.%s" $cluster $ns) (printf "%s.%s.svc" $cluster $ns) (printf "%s.%s.svc.%s" $cluster $ns $domain) "localhost" -}}
{{- range .Values.nodeGroups -}}
{{- $g := include "es.group" (dict "root" $root "group" .) | fromYaml -}}
{{- $h := printf "%s-headless" (include "es.groupFullname" (dict "root" $root "group" $g)) -}}
{{- $names = concat $names (list $h (printf "%s.%s.svc" $h $ns) (printf "*.%s" $h) (printf "*.%s.%s.svc" $h $ns) (printf "*.%s.%s.svc.%s" $h $ns $domain)) -}}
{{- end -}}
{{- $names = concat $names (.Values.security.tls.extraDnsNames | default list) -}}
{{- toJson (uniq $names) -}}
{{- end -}}

{{- define "es.httpIPs" -}}
{{- toJson (uniq (concat (list "127.0.0.1" "::1") (.Values.security.tls.extraIPs | default list))) -}}
{{- end -}}

{{/*
Environment of every Elasticsearch container: the security settings (as
dotted setting names, which override elasticsearch.yml) and the connection
settings of the image's probe/hook scripts (es-http-lib). Context: root.
*/}}
{{- define "es.securityEnv" -}}
{{- if .Values.security.enabled }}
- name: xpack.security.enabled
  value: "true"
- name: ELASTIC_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "es.credentialsSecret" . }}
      key: {{ include "es.credentialsKey" . }}
- name: xpack.security.transport.ssl.enabled
  value: "true"
- name: xpack.security.transport.ssl.verification_mode
  value: {{ .Values.security.tls.transport.verificationMode | quote }}
- name: xpack.security.transport.ssl.key
  value: certs/transport/tls.key
- name: xpack.security.transport.ssl.certificate
  value: certs/transport/tls.crt
- name: xpack.security.transport.ssl.certificate_authorities
  value: certs/transport/ca.crt
{{- if .Values.security.tls.http.enabled }}
- name: xpack.security.http.ssl.enabled
  value: "true"
- name: xpack.security.http.ssl.key
  value: certs/http/tls.key
- name: xpack.security.http.ssl.certificate
  value: certs/http/tls.crt
- name: xpack.security.http.ssl.certificate_authorities
  value: certs/http/ca.crt
# Probes and hooks: verify the certificate against the client Service name
# while connecting to the local node.
- name: ES_URL
  value: {{ printf "https://%s:9200" (include "es.serviceHost" .) | quote }}
- name: ES_CONNECT_TO
  value: "::127.0.0.1:"
- name: ES_CA_CERT
  value: /usr/share/elasticsearch/config/certs/http/ca.crt
{{- else }}
- name: xpack.security.http.ssl.enabled
  value: "false"
- name: ES_URL
  value: "http://127.0.0.1:9200"
{{- end }}
{{- else }}
# security.enabled=false: DEVELOPMENT ONLY (no authentication, no TLS).
- name: xpack.security.enabled
  value: "false"
- name: ES_URL
  value: "http://127.0.0.1:9200"
{{- end }}
{{- end -}}

{{/*
Environment of the in-cluster clients (cleanup Job, helm test): the client
Service URL, the CA and the elastic password. Context: root.
*/}}
{{- define "es.clientEnv" -}}
- name: ES_URL
  value: {{ include "es.clientUrl" . | quote }}
{{- if .Values.security.enabled }}
- name: ES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "es.credentialsSecret" . }}
      key: {{ include "es.credentialsKey" . }}
{{- if .Values.security.tls.http.enabled }}
- name: ES_CA_CERT
  value: /usr/share/elasticsearch/config/certs/http/ca.crt
{{- end }}
{{- end }}
{{- end -}}

{{/* Volume (named http-certs) holding only the HTTP CA, for clients. */}}
{{- define "es.clientCaVolume" -}}
{{- if include "es.httpsEnabled" . }}
- name: http-certs
  {{- $v := include "es.certVolume" (dict "root" . "kind" "http") | fromYaml }}
  secret:
    secretName: {{ $v.secret.secretName }}
    defaultMode: 0444
    items:
      - { key: ca.crt, path: ca.crt }
{{- end }}
{{- end -}}

{{- define "es.clientCaMount" -}}
{{- if include "es.httpsEnabled" . }}
- name: http-certs
  mountPath: /usr/share/elasticsearch/config/certs/http
  readOnly: true
{{- end }}
{{- end -}}
