import tempfile
import unittest
from pathlib import Path

from wildfire_mvp.reproducibility import collect_environment_info, command_log, save_manifest, utc_timestamp_slug


class ReproducibilityTests(unittest.TestCase):
    def test_timestamp_slug_format(self):
        slug = utc_timestamp_slug()
        self.assertIn("T", slug)
        self.assertTrue(slug.endswith("Z"))

    def test_command_log_has_expected_fields(self):
        item = command_log()
        self.assertIn("argv", item)
        self.assertIn("command", item)
        self.assertIn("python_executable", item)

    def test_save_manifest_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(path, {"ok": True})
            self.assertTrue(path.exists())
            self.assertIn('"ok": true', path.read_text(encoding="utf-8"))

    def test_environment_info_contains_python_and_packages(self):
        info = collect_environment_info()
        self.assertIn("python_version", info)
        self.assertIn("platform", info)
        self.assertIn("packages", info)
        self.assertIn("numpy", info["packages"])


if __name__ == "__main__":
    unittest.main()
