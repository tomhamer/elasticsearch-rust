#!/usr/bin/env python3
"""Run supplemental CAT coverage on an isolated, two-node no-registry fixture.

Never modifies upstream YAML or any active matrix fixture. Execute only on EC2.
"""
import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=pathlib.Path, required=True)
p.add_argument('--runner', type=pathlib.Path, required=True)
p.add_argument('--binary', type=pathlib.Path, required=True)
p.add_argument('--api', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--http-base', type=int, default=24080)
p.add_argument('--transport-base', type=int, default=24180)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
suite = a.output / 'suite'
shutil.copytree(a.root / 'tests/es-yaml/regressions', suite / 'test')
(suite / 'api').symlink_to(a.api.resolve())
fixture = a.output / 'fixture'
code = 1
try:
    # This directory-specific export has no built-in template registries.
    subprocess.run(['python3', str(a.root / 'bench/path-poc/es9-parity/start_module_fixture.py'),
                    '--root', str(a.root), '--runner', str(a.runner), '--binary', str(a.binary),
                    '--module', 'cat.templates', '--directory', str(fixture), '--nodes', '2',
                    '--http-base', str(a.http_base), '--transport-base', str(a.transport_base)],
                   check=True, stdout=(a.output / 'fixture.log').open('w'), stderr=subprocess.STDOUT)
    urls = json.loads((fixture / 'fixture.json').read_text())['urls']
    with (a.output / 'runner.log').open('w') as log:
        code = subprocess.call([str(a.runner), '--suite', str(suite), '--url', urls,
                                '--no-xpack', '--json', str(a.output / 'cases.json')],
                               stdout=log, stderr=subprocess.STDOUT)
    cases = json.loads((a.output / 'cases.json').read_text())
    # An empty selection or a prerequisite skip cannot satisfy this regression gate.
    passed = code == 0 and len(cases) == 4 and all(c['status'] == 'pass' for c in cases)
    code = 0 if passed else 1
    (a.output / 'summary.json').write_text(json.dumps({
        'complete': True, 'expected_cases': 4, 'pass': sum(c['status'] == 'pass' for c in cases),
        'fail': sum(c['status'] == 'fail' for c in cases),
        'skip': sum(c['status'] == 'skip' for c in cases), 'execution_passed': passed,
        'scope': 'supplemental CAT regressions, not full ES9 parity', 'nodes': 2}, indent=2))
finally:
    if (fixture / 'pids.json').exists():
        for pid in json.loads((fixture / 'pids.json').read_text()):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    (a.output / 'exit.code').write_text(str(code) + '\n')
raise SystemExit(code)
