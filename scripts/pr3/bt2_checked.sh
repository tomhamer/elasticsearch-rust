#!/usr/bin/env bash
# EC2 only. Usage: bt2_checked.sh SOURCE_DIR NEW_RESULT_DIR TARGET_DIR
# Enumerate workspace lib/integration targets instead of silently omitting test files.
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'EC2 Linux execution required' >&2; exit 2; }
[[ $# == 3 ]] || exit 2
source_dir=$(realpath "$1")
result_dir=$(realpath -m "$2")
target_dir=$(realpath -m "$3")
[[ ! -e "$result_dir" ]] || { echo 'Refusing to overwrite results' >&2; exit 2; }
mkdir -p "$result_dir" "$target_dir"
finish() { rc=$?; trap - EXIT; echo "$rc" > "$result_dir/exit.code"; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec 9>"$target_dir/.pr3-build.lock"
flock -n 9 || { echo 'Shared target busy' >&2; exit 2; }
cd "$source_dir"
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$target_dir"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(nproc)}" RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
cargo +stable metadata --locked --no-deps --format-version 1 > "$result_dir/metadata.json"
python3 - "$result_dir" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);m=json.loads((out/'metadata.json').read_text());members=set(m['workspace_members']);targets=[]
for p in m['packages']:
 if p['id'] not in members: continue
 for t in p['targets']:
  if t.get('required-features'):
   # Explicitly retain feature-required targets as uncovered, never a pass.
   continue
  if 'lib' in t['kind']: targets.append([p['name'],'--lib','-'])
  elif 'test' in t['kind']: targets.append([p['name'],'--test',t['name']])
(out/'suites.tsv').write_text(''.join('\t'.join(t)+'\n' for t in sorted(targets)))
(out/'feature-required-targets.json').write_text(json.dumps([{'package':p['name'],'target':t} for p in m['packages'] if p['id'] in members for t in p['targets'] if t.get('required-features')],indent=2))
assert targets,'No suites selected'
PY
# A real rolling test needs a separately built old executable, not a renamed
# copy of this candidate. Verify the baseline identity before compiling it.
if [[ -f "$source_dir/cluster/crates/elasticsearch-node/tests/rolling_capabilities.rs" ]]; then
 old_source=${PR3_OLD_SOURCE:?PR3_OLD_SOURCE is required for the real rolling gate}
 old_revision=${PR3_OLD_REVISION:?Pin the current master revision explicitly}
 [[ "$old_revision" =~ ^[0-9a-f]{40}$ ]]
 [[ $(cat "$old_source/.pr3-source-revision") == "$old_revision" ]] || { echo 'Wrong rolling baseline source' >&2; exit 2; }
 (cd "$old_source" && CARGO_TARGET_DIR="$target_dir/pr3-baseline" cargo +stable build --locked --profile fast -p elasticsearch-node) > "$result_dir/old-build.log" 2>&1
 cp "$target_dir/pr3-baseline/fast/elasticsearch-node" "$result_dir/elasticsearch-node-old"
 sha256sum "$result_dir/elasticsearch-node-old" > "$result_dir/old-binary.sha256"
 export PR3_OLD_BINARY="$result_dir/elasticsearch-node-old"
fi
# Source snapshots can share a target directory across rounds. Force Cargo to
# observe every candidate Rust source after any baseline work. The baseline
# also has an independent target: path-switch fingerprints cannot select it.
find "$source_dir" -name '*.rs' -exec touch {} +
# Never copy a stale executable after a failed build.
cargo +stable build --locked --profile fast -p elasticsearch-node > "$result_dir/build.log" 2>&1
# The YAML fixtures write secure settings (bootstrap.password) with the
# build's keystore tool beside the node executable; without it they fall
# back to elasticsearch.yml, which the node refuses.
cargo +stable build --locked --profile fast -p elasticsearch --bin elasticsearch-keystore >> "$result_dir/build.log" 2>&1
cp "$target_dir/fast/elasticsearch-keystore" "$result_dir/elasticsearch-keystore"
cp "$target_dir/fast/elasticsearch-node" "$result_dir/elasticsearch-node"
sha256sum "$result_dir/elasticsearch-node" > "$result_dir/binary.sha256"
if [[ -f "$result_dir/elasticsearch-node-old" ]] && cmp -s "$result_dir/elasticsearch-node-old" "$result_dir/elasticsearch-node"; then
 echo "Old and candidate executables are identical; refusing rolling validation" >&2
 exit 2
fi
failed=0
while IFS=$'\t' read -r package kind target; do
 args=(-p "$package" "$kind")
 [[ "$kind" == --lib ]] || args+=("$target")
 slug="$package-$kind-$target"
 test_env=()
 if [[ "$package" == elasticsearch-node && "$kind" == --test && "$target" == dynamic_mappings ]]; then
  # This target reuses fixed listener ports across cases (38000/38020).
  # Serialize independent cases; each case retains its own real two-node and
  # multi-threaded Tokio workload. No assertion, retry or fixture is changed.
  test_env+=(RUST_TEST_THREADS=1)
  printf '%s\t%s\n' "$slug" RUST_TEST_THREADS=1 >> "$result_dir/runner-env.tsv"
 fi
 set +e
 env "${test_env[@]}" timeout --signal=TERM --kill-after=30 1800 cargo +stable test --locked --profile fast "${args[@]}" > "$result_dir/$slug.log" 2>&1
 rc=$?
 set -e
 printf '%s\t%s\t%s\t%s\n' "$package" "$kind" "$target" "$rc" >> "$result_dir/suites-exit.tsv"
 if [[ "$rc" != 0 ]]; then
  failed=1
  [[ -f "$result_dir/first-failure.txt" ]] || printf '%s exit=%s\n' "$slug" "$rc" > "$result_dir/first-failure.txt"
 fi
done < "$result_dir/suites.tsv"
touch "$result_dir/run.done"
exit "$failed"
