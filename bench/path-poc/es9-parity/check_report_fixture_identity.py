"""EC2 regression for reporting identical YAML cases in different fixtures."""
import collections
import json
import pathlib
import tempfile
import types

source = pathlib.Path(__file__).with_name('parity_all.py').read_text()
report = source[source.index('cases = {}'):source.index('# A completed report is not a passing run.')]
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    out = root / 'parity-results/candidate'
    out.mkdir(parents=True)
    jobs = [('plug', 'gpu-single', []), ('plug', 'gpu-multi', [])]
    status = {f'{s}-{m}': {'exit': 0 if m == 'gpu-single' else 1} for s, m, _ in jobs}
    for s, m, _ in jobs:
        case = {'file': 'gpu/supported/10_hnsw.yml', 'dir': 'gpu', 'name': 'same test',
                'status': 'pass' if m == 'gpu-single' else 'fail', 'msg': 'missing GPU' if m == 'gpu-multi' else ''}
        (out / f'{s}-{m}.json').write_text(json.dumps([case]))
    env = dict(collections=collections, json=json, out=out, root=root, jobs=jobs, status=status,
               a=types.SimpleNamespace(run_name='candidate', suites='plug', modules='gpu', filters=None, baseline=None))
    exec(compile(report, 'parity-report', 'exec'), env)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['suites']['plug'] == {'pass': 1, 'fail': 1}, summary
    assert summary['complete'] and not summary['execution_passed']
    assert 'plug/gpu-multi' in (out / 'REPORT.md').read_text()
    # A repeated case in the SAME fixture is ambiguous and must fail loudly.
    p = out / 'plug-gpu-single.json'
    cases = json.loads(p.read_text())
    p.write_text(json.dumps(cases + cases))
    try:
        exec(compile(report, 'parity-report', 'exec'), env)
    except RuntimeError as error:
        assert 'Duplicate test identity' in str(error)
    else:
        raise AssertionError('Duplicate fixture case was silently overwritten')
print('Fixture identity reporting and duplicate detection passed')
