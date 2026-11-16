#!/usr/bin/env python3
"""AWS-only YAML run of ES 9.5.5's cross-cluster suites against real
separate fixture clusters (no self-alias substitute for CCS):

* `mcss`: x-pack/qa/multi-cluster-search-security (`BasicLicenseIT`,
  `FullLicenseIT`). A fulfilling (remote) cluster and a querying cluster,
  two nodes each, security on, `test_user` superuser. The fulfilling
  cluster's files run first (`YamlFileOrder`), then the querying
  cluster's, against a querying cluster whose `my_remote_cluster` points
  at the fulfilling cluster's first transport address. Every suite runs in
  both connection modes (`sniff`: `seeds`; `proxy`: `proxy_address`); in
  proxy mode the tests ES's `BasicLicenseIT` skips there are skipped.
  Indices persist across files (`preserveIndicesUponCompletion`).
* `ccr`: x-pack/plugin/ccr's YAML suite (`CcrRestIT`) on the module's
  fixture. Those tests follow the cluster's own indices through a remote
  cluster alias seeded with the master's transport address, as ES's suite
  does: a real remote-cluster connection to the same cluster.
* `oss-ccs`: rest-api-spec `cluster.stats/30_ccs_stats.yml` (remotes
  seeded with the cluster's own address, as written by ES) and
  `cluster.remote_info/`.

The licence is hardwired to trial (AGENTS.md): tests asserting basic-licence
errors are reported as skipped with that reason, never as passes.

`--corpus` holds `{mcss,plug,oss}/{api,test}` assembled from the ES v9.5.5
tag (`assemble_ccs_ccr_corpus.sh`). Writes `<out>/<job>.json` per runner
invocation, `summary.json` and `REPORT.md`. No binaries are built here.
"""
import argparse, collections, json, os, pathlib, signal, subprocess, sys, tempfile, time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, help='repository checkout (fixture launcher)')
p.add_argument('--runner', required=True, help='a copy of es-cluster-yaml')
p.add_argument('--binary', required=True, help='pinned candidate server binary')
p.add_argument('--corpus', required=True)
p.add_argument('--out', required=True)
p.add_argument('--port-base', type=int, default=26000)
p.add_argument('--jobs', default='mcss,ccr,oss-ccs')
p.add_argument('--modes', default='sniff,proxy')
p.add_argument('--licenses', default='basic_license,full_license')
p.add_argument('--keep', action='store_true', help='leave the mcss fixtures running (debugging; prints their pids)')
a = p.parse_args()
if not hasattr(os, 'pidfd_open'):
    p.error('Linux pidfd support is required for identity-safe fixture cleanup')
root, corpus, out = pathlib.Path(a.root), pathlib.Path(a.corpus), pathlib.Path(a.out)
if out.exists() and any(out.iterdir()):
    raise RuntimeError(f'Refusing to reuse nonempty results directory: {out}')
out.mkdir(parents=True, exist_ok=True)
launcher = root / 'bench/path-poc/es9-parity/start_module_fixture.py'

# ES BasicLicenseIT.SKIPPED_TESTS (skipped in proxy mode there).
PROXY_SKIPS = (
    'querying_cluster/10_basic.yml::Add persistent remote cluster based on the preset cluster',
    'querying_cluster/20_info.yml::Add persistent remote cluster based on the preset cluster and check remote info',
    'querying_cluster/20_info.yml::Fetch remote cluster info for existing cluster',
    'querying_cluster/70_connection_mode_configuration.yml::',
)
# Basic-licence assertions; the licence is hardwired to trial.
LICENCE_SKIPS = ('fails with basic license',)


def stop(processes):
    owned = []
    for process in processes:
        try:
            fd = os.pidfd_open(process['pid'])
        except ProcessLookupError:
            continue
        stat = pathlib.Path(f"/proc/{process['pid']}/stat")
        if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[19] != process['start_ticks']:
            os.close(fd)
            continue
        owned.append(fd)
    for fd in owned:
        try: signal.pidfd_send_signal(fd, signal.SIGTERM)
        except ProcessLookupError: pass
    deadline = time.monotonic() + 30
    import select
    remaining = set(owned)
    poll = select.poll()
    for fd in owned: poll.register(fd, select.POLLIN)
    while remaining and time.monotonic() < deadline:
        for fd, _ in poll.poll(1000):
            remaining.discard(fd); poll.unregister(fd)
    for fd in remaining:
        try: signal.pidfd_send_signal(fd, signal.SIGKILL)
        except ProcessLookupError: pass
    for fd in owned: os.close(fd)


def start_fixture(module, slot, settings=()):
    http, transport = a.port_base + slot * 200, a.port_base + slot * 200 + 100
    directory = pathlib.Path(tempfile.mkdtemp(prefix=f'ccs-ccr-{module}-'))
    spec = json.loads(subprocess.check_output([a.runner, '--export-fixture', module], text=True))
    args = ['python3', str(launcher), '--root', str(root), '--runner', a.runner, '--module', module, '--directory', str(directory),
            '--nodes', str(spec.get('nodes', 2)), '--http-base', str(http), '--transport-base', str(transport), '--binary', a.binary]
    for s in settings:
        args += ['--setting', s]
    run = subprocess.run(args, capture_output=True, text=True)
    processes = json.load(open(directory / 'processes.json')) if (directory / 'processes.json').exists() else []
    if run.returncode != 0:
        stop(processes)
        raise RuntimeError(f'{module} fixture failed to start ({directory}): {(run.stderr or run.stdout)[-600:]}')
    fixture = json.load(open(directory / 'fixture.json'))
    return {'dir': directory, 'processes': processes, 'urls': fixture['urls'], 'transport': transport, 'security': spec['security']}


def run_runner(name, suite, filters, fixture, wipe=True, wipe_settings=False):
    args = [a.runner, '--suite', str(corpus / suite), '--url', fixture['urls'], '--module-security-fixtures', '--json', str(out / f'{name}.json')]
    for f in filters:
        args += ['--filter', f]
    if fixture['security']:
        args.append('--security')
    if not wipe:
        args.append('--no-wipe')
        # ES's suites that keep their indices still reset cluster settings
        # after every test (`ESRestTestCase.wipeClusterSettings`).
        if wipe_settings:
            args.append('--wipe-settings')
    with open(out / f'{name}.log', 'w') as log:
        code = subprocess.call(args, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    path = out / f'{name}.json'
    return json.load(open(path)) if path.exists() else [{'file': '<runner>', 'name': name, 'status': 'fail', 'msg': f'runner exit {code} without results', 'dir': ''}]


cases = {}
status = {}


def record(job, results, proxy=False):
    for c in results:
        key = (job, c['file'], c['name'])
        if c['status'] == 'fail' and any(s in c['name'] for s in LICENCE_SKIPS):
            c['status'], c['msg'] = 'skip', 'licence hardwired to trial'
        if proxy and any(f"{c['file']}::{c['name']}".endswith(s) or (s.endswith('::') and c['file'].endswith(s[:-2])) for s in PROXY_SKIPS):
            c['status'], c['msg'] = 'skip', 'skipped in proxy mode by ES BasicLicenseIT'
        cases[key] = c


jobs = a.jobs.split(',')
if 'mcss' in jobs:
    for lic in a.licenses.split(','):
        for mode in a.modes.split(','):
            job = f'mcss-{lic}-{mode}'
            fulfilling = querying = None
            try:
                fulfilling = start_fixture('mcss-fulfilling', 0)
                record(job, run_runner(f'{job}-fulfilling', 'mcss', [f'{lic}/fulfilling_cluster'], fulfilling, wipe=False))
                address = f"127.0.0.1:{fulfilling['transport']}"
                remote = ['cluster.remote.my_remote_cluster.mode=sniff', f'cluster.remote.my_remote_cluster.seeds=[{address}]'] if mode == 'sniff' else \
                    ['cluster.remote.my_remote_cluster.mode=proxy', f'cluster.remote.my_remote_cluster.proxy_address={address}']
                querying = start_fixture('mcss-querying', 1, remote)
                record(job, run_runner(f'{job}-querying', 'mcss', [f'{lic}/querying_cluster'], querying, wipe=False, wipe_settings=True), proxy=mode == 'proxy')
                status[job] = 'ran'
            except Exception as error:
                status[job] = f'error: {error}'
            finally:
                for f in (querying, fulfilling):
                    if f and a.keep:
                        print(f"kept {f['dir']} {f['urls']}", flush=True)
                    elif f:
                        stop(f['processes'])
if 'ccr' in jobs:
    fixture = None
    try:
        fixture = start_fixture('ccr', 2)
        record('ccr', run_runner('ccr', 'plug', ['ccr/'], fixture))
        status['ccr'] = 'ran'
    except Exception as error:
        status['ccr'] = f'error: {error}'
    finally:
        if fixture:
            stop(fixture['processes'])
if 'oss-ccs' in jobs:
    fixture = None
    try:
        fixture = start_fixture('oss', 3)
        record('oss-ccs', run_runner('oss-ccs', 'oss', ['cluster.stats/30_ccs_stats.yml', 'cluster.remote_info/'], fixture))
        status['oss-ccs'] = 'ran'
    except Exception as error:
        status['oss-ccs'] = f'error: {error}'
    finally:
        if fixture:
            stop(fixture['processes'])

by_job = collections.defaultdict(collections.Counter)
for (job, _, _), c in cases.items():
    by_job[job][c['status']] += 1
lines = ['# ES 9.5.5 cross-cluster YAML suites', '', '| Job | Pass | Fail | Skip |', '| --- | ---: | ---: | ---: |']
for job in sorted(by_job):
    n = by_job[job]
    lines.append(f"| {job} | {n['pass']} | {n['fail']} | {n['skip']} |")
lines += ['', '## Jobs', ''] + [f'- {k}: {v}' for k, v in sorted(status.items())]
lines += ['', '## Failures', ''] + [f"- {j} `{f}` :: {n}: {(c.get('msg') or '').splitlines()[0][:200] if c.get('msg') else ''}" for (j, f, n), c in sorted(cases.items()) if c['status'] == 'fail']
lines += ['', '## Skipped (not passes)', ''] + [f"- {j} `{f}` :: {n}: {c.get('msg', '')}" for (j, f, n), c in sorted(cases.items()) if c['status'] == 'skip']
(out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
summary = {'jobs': status, 'counts': {j: dict(n) for j, n in by_job.items()},
           'execution_passed': bool(cases) and all(v == 'ran' for v in status.values()) and not any(c['status'] == 'fail' for c in cases.values())}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print('\n'.join(lines[:4 + len(by_job)]))
sys.exit(0 if summary['execution_passed'] else 1)
