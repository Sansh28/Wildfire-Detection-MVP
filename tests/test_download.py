import unittest
from pathlib import Path

from wildfire_mvp.download import extract_sentinel_patches


class DownloadPipelineTests(unittest.TestCase):
    def test_extract_sentinel_rejects_unsupported_provider(self):
        with self.assertRaises(ValueError):
            extract_sentinel_patches(
                events_csv=Path("does_not_matter.csv"),
                output_dir=Path("tmp"),
                provider="mock_provider",
                collection="unused",
            )


if __name__ == "__main__":
    unittest.main()
