"""Run on EC2 with: python3 -m unittest discover -s <gate script directory>."""
import pathlib
import tempfile
import unittest
from gate_reports import native_report

class NativeReportTests(unittest.TestCase):
    def test_repeated_normalized_announcements_are_one_invocation(self):
        with tempfile.TemporaryDirectory() as d:
            matrix=pathlib.Path(d)/'matrix'
            log=f'artifacts: {matrix}/run-1\nfinished\nartifacts: {matrix}/./run-1\n'
            self.assertEqual(native_report(log,matrix),matrix/'run-1/summary.json')

    def test_distinct_invocations_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                native_report(f'artifacts: {d}/a\nartifacts: {d}/b\n',d)

    def test_escaped_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'escaped'):
                native_report(f'artifacts: {d}/../elsewhere\n',d)

    def test_old_native_layout_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(native_report('finished\n',d),pathlib.Path(d)/'summary.json')
