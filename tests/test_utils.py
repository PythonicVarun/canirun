import unittest

from canirun.enum import COMPATIBILITY
from canirun.human_readable import get_human_readable_size, get_human_readable_status


class TestHumanReadable(unittest.TestCase):
    """Test suite for human-readable helper functions."""

    def test_get_human_readable_size(self) -> None:
        """Tests bytes to human-readable string conversion."""
        self.assertEqual(get_human_readable_size(0), "0 B")
        self.assertEqual(get_human_readable_size(100), "100.00 B")
        self.assertEqual(get_human_readable_size(1024), "1.00 KB")
        self.assertEqual(get_human_readable_size(1024**2), "1.00 MB")
        self.assertEqual(get_human_readable_size(1024**3), "1.00 GB")
        self.assertEqual(get_human_readable_size(1.5 * 1024**3), "1.50 GB")
        self.assertEqual(get_human_readable_size(1024**4), "1.00 TB")
        self.assertEqual(get_human_readable_size(1024**5), "1.00 PB")

    def test_get_human_readable_status(self) -> None:
        """Tests compatibility enum to string conversion."""
        self.assertEqual(get_human_readable_status(COMPATIBILITY.FULL), "✅ GPU")
        self.assertEqual(
            get_human_readable_status(COMPATIBILITY.PARTIAL), "⚠️ CPU/RAM only (Slow)"
        )
        self.assertEqual(get_human_readable_status(COMPATIBILITY.NONE), "❌ Impossible")


if __name__ == "__main__":
    unittest.main()
