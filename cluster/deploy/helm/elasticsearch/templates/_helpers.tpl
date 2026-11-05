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
"true" once the cluster is known to have formed: either clusterBootstrapped
is set, or (on helm upgrade) a master-eligible StatefulSet of this release
already exists. `lookup` returns nothing under `helm template` / --dry-run,
so offline rendering always includes the bootstrap list (harmless: a formed
cluster ignores it).
*/}}
{{- define "es.bootstrapped" -}}
{{- $root := . -}}
{{- $found := .Values.clusterBootstrapped -}}
{{- range .Values.nodeGroups -}}
{{- $g := include "es.group" (dict "root" $root "group" .) | fromYaml -}}
{{- if has "master" $g.roles -}}
{{- if lookup "apps/v1" "StatefulSet" $root.Release.Namespace (include "es.groupFullname" (dict "root" $root "group" $g)) -}}
{{- $found = true -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $found }}true{{ end -}}
{{- end -}}

{{- define "es.clientUrl" -}}
{{- printf "http://%s.%s.svc:%d" (include "es.clusterName" .) .Release.Namespace (int .Values.service.httpPort) -}}
{{- end -}}
