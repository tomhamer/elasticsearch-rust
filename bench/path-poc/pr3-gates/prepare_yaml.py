#!/usr/bin/env python3
"""Materialize the established four-suite layout; never replace an existing corpus."""
import argparse, hashlib, json, pathlib, shutil, tempfile, os
p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--destination',required=True); a=p.parse_args()
s=pathlib.Path(a.source).resolve(); dst=pathlib.Path(a.destination).resolve()
if dst.exists(): raise SystemExit(f'Refusing existing destination {dst}')
security={'api_key','authenticate','change_password','privileges','role_mapping','roles','security','service_accounts','set_security_user','token','user_profile','users'}
manifest=[]; selected=set()
tmp=pathlib.Path(tempfile.mkdtemp(prefix=dst.name+'.preparing-',dir=dst.parent))
def copy_tree(src, dest):
    if not src.is_dir(): raise RuntimeError(f'Missing required source {src}')
    selected.add(src)
    for f in sorted(src.rglob('*')):
        if not f.is_file(): continue
        target=dest/f.relative_to(src); data=f.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        if target.exists() and target.read_bytes()!=data: raise RuntimeError(f'Conflicting fixture {f} -> {target}')
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        manifest.append({'source':str(f.relative_to(s)), 'target':str(target.relative_to(tmp)), 'sha256':digest})
api=s/'rest-api-spec/src/main/resources/rest-api-spec/api'
for suite in ('oss','main','sec','plug'): copy_tree(api,tmp/suite/'api')
copy_tree(s/'rest-api-spec/src/yamlRestTest/resources/rest-api-spec/test',tmp/'oss/test')
# Exact YAML18 additions to core: do not merge unrelated modules with colliding paths.
for path in ('modules/data-streams', 'x-pack/plugin/ilm', 'x-pack/plugin/slm'):
    copy_tree(s/path/'src/yamlRestTest/resources/rest-api-spec/test',tmp/'oss/test')
for name in ('data_streams','dlm'):
    copy_tree(s/'x-pack/plugin/src/yamlRestTest/resources/rest-api-spec/test'/name,tmp/'oss/test'/name)
main=s/'x-pack/plugin/src/yamlRestTest/resources/rest-api-spec/test'
for src in sorted(main.iterdir()):
    if src.is_dir(): copy_tree(src,tmp/('sec' if src.name in security else 'main')/'test'/src.name)
plugins=s/'x-pack/plugin'
for src in sorted(plugins.glob('*/src/yamlRestTest/resources/rest-api-spec/test')):
    name=src.relative_to(plugins).parts[0]
    # Historical fixture names: secured inference QA is inference; plugin suite is inference-2.
    if name == 'inference': name = 'inference-2'
    copy_tree(src,tmp/'plug/test'/name)
extra={'downsample':'downsample/qa/mixed-cluster','downsample-rest':'downsample/qa/rest','eql':'eql/qa/rest','searchable-snapshots':'searchable-snapshots/qa/rest','snapshot-repo-test-kit':'snapshot-repo-test-kit/qa/rest','inference':'inference/qa/inference-with-security','inference-multi-node':'inference/qa/multi-node'}
for name, path in extra.items(): copy_tree(plugins/path/'src/yamlRestTest/resources/rest-api-spec/test',tmp/'plug/test'/name)
# Custom core API definitions are consumed by plugin cases as well.
custom=plugins/'core/src/yamlRestTest/resources/rest-api-spec/api'
copy_tree(custom,tmp/'plug/api')
inventory=sorted(str(x.relative_to(s)) for x in s.rglob('rest-api-spec/test') if x.is_dir())
not_selected=[x for x in inventory if not any((s/x)==y or (s/x) in y.parents for y in selected)]
(tmp/'provenance.json').write_text(json.dumps({'source':str(s),'files':manifest,'other_upstream_test_roots_not_in_established_matrix':not_selected,'note':'These source roots are not silent prerequisite skips. This reproduces the established four-suite matrix, not every upstream deployment.'},indent=2)+'\n')
for suite in ('oss','main','sec','plug'):
    if not any((tmp/suite/'test').rglob('*.yml')): raise RuntimeError(f'Empty suite {suite}')
expected=json.loads(pathlib.Path(__file__).with_name('yaml18-files.json').read_text())
for suite, files in expected.items():
    actual={str(f.relative_to(tmp/suite/'test')) for f in (tmp/suite/'test').rglob('*.yml')}
    # parity_all's watcher fixture selectors do not execute these retained script suites.
    known_unselected={f for f in actual if suite=='plug' and (f.startswith('watcher/mustache/') or f.startswith('watcher/painless/'))}
    if actual - known_unselected != set(files):
        raise RuntimeError(f'{suite} differs from YAML18 file inventory: missing={sorted(set(files)-actual)}, extra={sorted(actual-set(files))}')
(tmp/'baseline-inventory-check.json').write_text(json.dumps({'baseline':'pr3-yaml18','expected_files':expected,'retained_not_selected_by_existing_watcher_fixture_filters':sorted(known_unselected)},indent=2)+'\n')
os.rename(tmp,dst)
print(dst)
