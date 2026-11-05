"""Pure report-location parsing shared with the gate's regression tests."""
import pathlib

def native_report(log, matrix):
    matrix=pathlib.Path(matrix).resolve()
    announced={pathlib.Path(line.removeprefix('artifacts: ').strip()).resolve()
               for line in log.splitlines() if line.startswith('artifacts: ')}
    if not announced:
        return matrix/'summary.json'
    if len(announced)!=1:
        raise ValueError('Ambiguous native artifact paths')
    artifact=announced.pop()
    if not artifact.is_relative_to(matrix):
        raise ValueError('Native artifact path escaped invocation')
    return artifact/'summary.json'
