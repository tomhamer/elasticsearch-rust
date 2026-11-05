#!/usr/bin/env python3
"""AWS-only fast YAML check against a persistent two-node fixture.

The fixture for a module stays up between checks and is restarted only when
target/fast/elasticsearch changed (or with --restart). By default the run
stops at the first failing case and prints it with its error, so a fix can
start right away; --all runs the whole selection.

    fastcheck.py --suite main --module main esql/60_enrich.yml
    fastcheck.py --suite oss data_stream/ --all
"""
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time

p = argparse.ArgumentParser()
p.add_argument('filters', nargs='+')
p.add_argument('--root', default='/home/ubuntu/es9-feature-work')
p.add_argument('--suite', default='main', choices=['oss', 'main', 'plug', 'sec'])
p.add_argument('--module', help='fixture module (default: oss for core, else the suite, or the plugin directory)')
p.add_argument('--all', action='store_true', help='do not stop at the first failure')
p.add_argument('--restart', action='store_true')
p.add_argument('--http-base', type=int, default=25000)
p.add_argument('--tag', default='', help='separate fixtures/runner for a parallel workstream (use with its own --root and --http-base)')
a = p.parse_args()

root = pathlib.Path(a.root)
# Plugin subdirectories ES runs on their own test cluster (see parity_all.py).
SUB_FIXTURES = {'watcher/security': 'watcher-security', 'rank-rrf/license': 'rank-rrf-license'}


def plug_module(path):
    for prefix, fixture in SUB_FIXTURES.items():
        if path.startswith(prefix):
            return fixture
    return path.split('/')[0]


module = a.module or ('oss' if a.suite == 'oss' else plug_module(a.filters[0]) if a.suite == 'plug' else a.suite)
slot = {'oss': 0, 'main': 1, 'sec': 2}.get(module, 3 + sum(map(ord, module)) % 20)
http, transport = a.http_base + slot * 10, a.http_base + 500 + slot * 10
fixture = pathlib.Path('/tmp') / f'fastcheck{a.tag}-{module}'
binary = root / 'target/fast/elasticsearch'
runner = pathlib.Path(f'/tmp/fastcheck{a.tag}-es-cluster-yaml')


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def alive(info):
    return all(pathlib.Path(f'/proc/{pid}').exists() for pid in info['pids'])


def stop(info):
    for pid in info.get('pids', []):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(30):
        if not any(pathlib.Path(f'/proc/{pid}').exists() for pid in info.get('pids', [])):
            return
        time.sleep(0.5)


# The runner is copied so a concurrent build cannot replace it mid-run.
src_runner = root / 'target/fast/es-cluster-yaml'
if not runner.exists() or digest(runner) != digest(src_runner):
    subprocess.run(['cp', str(src_runner), str(runner)], check=True)

info = json.load(open(fixture / 'fixture.json')) if (fixture / 'fixture.json').exists() else None
dirty = fixture / 'dirty'
if info and (a.restart or dirty.exists() or not alive(info) or info['binary_sha256'] != digest(binary)):
    print('restarting fixture (binary changed, nodes down, or a stopped run skipped its teardown)', flush=True)
    stop(info)
    info = None
if info is None:
    subprocess.run(['rm', '-rf', str(fixture)], check=True)
    spec = json.loads(subprocess.check_output([str(runner), '--export-fixture', module], text=True))
    subprocess.run(['python3', str(root / 'bench/path-poc/es9-parity/start_module_fixture.py'), '--root', str(root), '--runner', str(runner),
                    '--module', module, '--directory', str(fixture), '--nodes', str(spec.get('nodes', 2)),
                    '--http-base', str(http), '--transport-base', str(transport)], check=True, stdout=subprocess.DEVNULL)
    info = json.load(open(fixture / 'fixture.json'))

out = fixture / 'last.json'
args = [str(runner), '--suite', str(pathlib.Path.home() / 'es955-yaml' / a.suite), '--url', info['urls'],
        '--module-security-fixtures', '--json', str(out)]
for f in a.filters:
    args += ['--filter', f]
if info['security']:
    args.append('--security')
# Verbose: each FAIL line is followed by its indented error.
args.append('-v')
started = time.time()
proc = subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
counts = {'PASS': 0, 'FAIL': 0, 'SKIP': 0}
first = None
detail = []
for line in proc.stdout:
    tag = line.split(' ', 1)[0]
    if tag in counts:
        counts[tag] += 1
    if first and len(detail) < 5 and line.startswith((' ', '\t')) and line.strip():
        detail.append(line.strip())
        if not a.all and len(detail) == 1:
            os.killpg(proc.pid, signal.SIGTERM)
            # The stopped case skipped its teardown: start clean next time.
            dirty.touch()
            break
        continue
    if tag == 'FAIL' and first is None:
        first = line.strip()
        continue
    if first and not a.all and tag in counts:
        # No indented error after the FAIL line: stop at the next case.
        os.killpg(proc.pid, signal.SIGTERM)
        dirty.touch()
        break
proc.wait()
print(f"{counts['PASS']} pass, {counts['FAIL']} fail, {counts['SKIP']} skip in {time.time() - started:.0f}s", flush=True)
if first:
    print(first)
    print('\n'.join(detail))
    sys.exit(1)
