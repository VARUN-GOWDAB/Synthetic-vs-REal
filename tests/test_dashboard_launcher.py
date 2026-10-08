"""Verify interpreter selection without installing or modifying environments."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dashboard_launcher', Path(__file__).resolve().parents[1] / 'run_dashboard.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_existing_environment_fallback_and_dedicated_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
            existing = root / '.venv' / relative
            dedicated = root / '.venv-inference' / relative
            self.assertEqual(launcher.select_python(root), dedicated)
            existing.parent.mkdir(parents=True)
            existing.touch()
            self.assertEqual(launcher.select_python(root), existing)
            self.assertEqual(launcher.select_python(root, setup=True), dedicated)
            dedicated.parent.mkdir(parents=True)
            dedicated.touch()
            self.assertEqual(launcher.select_python(root), dedicated)

    def test_explicit_current_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            for setup in (False, True):
                self.assertEqual(launcher.select_python(Path(tmp), current_env=True, setup=setup), Path(sys.executable))


if __name__ == '__main__':
    unittest.main()
