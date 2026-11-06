import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from Kosmulator_main.run_lock import run_lock

CODE = """
from Kosmulator_main.run_lock import run_lock
import sys
with run_lock(sys.argv[1]):
    print('locked', flush=True)
    if len(sys.argv) > 2:
        sys.stdin.read()
"""


class RunLockTests(unittest.TestCase):
    def test_competing_process_and_release(self):
        with tempfile.TemporaryDirectory() as folder:
            with run_lock(folder):
                blocked = subprocess.run([sys.executable, "-c", CODE, folder], capture_output=True)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn(b"Another process", blocked.stderr)
            subprocess.run([sys.executable, "-c", CODE, folder], check=True, capture_output=True)

    def test_process_exit_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            child = subprocess.Popen([sys.executable, "-c", CODE, folder, "hold"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                self.assertEqual(child.stdout.readline().strip(), b"locked")
                child.terminate()
                child.wait(timeout=10)
                with run_lock(folder):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdin.close()
                child.stdout.close()

    def test_default_directory_matches_generic_parser(self):
        # Catch divergence between the pre-parser and the actual CLI default.
        import ast
        root = Path(__file__).resolve().parents[1]
        defaults = []
        for filename in ("Kosmulator.py", "Kosmulator_main/utils.py"):
            tree = ast.parse((root/filename).read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and n.args and isinstance(n.args[0], ast.Constant)
                     and n.args[0].value == "--output_suffix"]
            defaults.append(next(k.value.value for k in calls[0].keywords if k.arg == "default"))
        self.assertEqual(defaults[0], defaults[1])
