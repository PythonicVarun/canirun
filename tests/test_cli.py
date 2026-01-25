import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from canirun.cli import main
from canirun.enum import COMPATIBILITY


class TestCli(unittest.TestCase):
    """Test suite for the CLI module."""

    def setUp(self) -> None:
        """Sets up the test environment."""
        self.runner = CliRunner()

    @patch("canirun.cli.ModelAnalyzer")
    def test_cli_success(self, MockAnalyzer: MagicMock) -> None:
        """Tests the CLI with a successful model analysis."""
        # Setup mock
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = {"some": "data"}
        instance.specs = {
            "ram": 16 * 1024**3,
            "vram": 8 * 1024**3,
            "name": "Test GPU",
            "is_mac": False,
        }
        instance.calculate.return_value = [
            {
                "quant": "FP16",
                "total_ram": 14 * 1024**3,
                "kv_cache": 1 * 1024**3,
                "status": COMPATIBILITY.FULL,
            }
        ]

        # Run CLI
        result = self.runner.invoke(main, ["test-model"])

        # Verify
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ANALYSIS REPORT: test-model", result.output)
        self.assertIn("FP16", result.output)
        self.assertIn("✅ GPU", result.output)

    @patch("canirun.cli.ModelAnalyzer")
    def test_cli_fetch_error(self, MockAnalyzer: MagicMock) -> None:
        """Tests the CLI when model data fetch fails."""
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = None

        result = self.runner.invoke(main, ["bad-model"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error: Could not fetch data", result.output)

    @patch("canirun.cli.ModelAnalyzer")
    def test_cli_verbose(self, MockAnalyzer: MagicMock) -> None:
        """Tests the CLI with verbose flag."""
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = {"some": "data"}
        instance.specs = {"ram": 0, "vram": 0, "name": "Test", "is_mac": False}
        instance.calculate.return_value = [
            {
                "quant": "FP16",
                "total_ram": 100,
                "kv_cache": 10,
                "status": COMPATIBILITY.FULL,
            }
        ]

        result = self.runner.invoke(main, ["test-model", "--verbose"])

        self.assertEqual(result.exit_code, 0)
        # Verify ModelAnalyzer was initialized with verbose=True
        MockAnalyzer.assert_called_with("test-model", verbose=True, hf_token=None)

    @patch("canirun.cli.ModelAnalyzer")
    def test_cli_with_token(self, MockAnalyzer: MagicMock) -> None:
        """Tests the CLI with an HF token."""
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = {"some": "data"}
        instance.specs = {"ram": 0, "vram": 0, "name": "Test", "is_mac": False}
        instance.calculate.return_value = [
            {
                "quant": "FP16",
                "total_ram": 100,
                "kv_cache": 10,
                "status": COMPATIBILITY.FULL,
            }
        ]

        result = self.runner.invoke(main, ["test-model", "--hf-token", "hf_123"])

        self.assertEqual(result.exit_code, 0)
        MockAnalyzer.assert_called_with("test-model", verbose=False, hf_token="hf_123")


if __name__ == "__main__":
    unittest.main()
