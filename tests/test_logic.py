import unittest
from typing import Any
from unittest.mock import patch

from canirun.enum import COMPATIBILITY
from canirun.logic import ModelAnalyzer


class TestModelAnalyzer(unittest.TestCase):
    """Test suite for the ModelAnalyzer class."""

    def setUp(self) -> None:
        """Sets up the test environment."""
        self.analyzer = ModelAnalyzer("test-model", verbose=False)
        # Mock specs with bytes
        gb = 1024**3
        self.analyzer.specs = {
            "ram": 32 * gb,
            "vram": 24 * gb,
            "name": "Test GPU",
            "is_mac": False,
        }

    def test_calculate_returns_bytes_as_int(self) -> None:
        """Tests that calculate returns memory values as integers."""
        data = {
            "params_billions": 7.0,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
        }
        results = self.analyzer.calculate(data, ctx=4096)

        self.assertTrue(len(results) > 0)
        first_result = results[0]

        # Check types
        self.assertIsInstance(first_result["total_ram"], int)
        self.assertIsInstance(first_result["kv_cache"], int)
        self.assertIsInstance(first_result["status"], COMPATIBILITY)

        # 7B params in FP16 (2 bytes/param) is 14GB
        # 14GB = 14 * 1024^3 bytes
        gb = 1024**3
        self.assertGreater(first_result["total_ram"], 14 * gb)

    @patch("canirun.logic.model_info")
    @patch("canirun.logic.hf_hub_download")
    def test_fetch_model_data_auth_error(
        self, mock_download: Any, mock_info: Any
    ) -> None:
        """Tests proper error handling when authentication fails.

        Args:
            mock_download: Mock for hf_hub_download.
            mock_info: Mock for model_info.
        """
        # Setup mock to raise 401 error
        mock_download.side_effect = Exception("401 Client Error: Unauthorized for url")
        mock_info.side_effect = Exception("Some error")

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 16 * 1024**3

            with self.assertLogs("canirun.logic", level="ERROR") as cm:
                self.analyzer.fetch_model_data()

            # Check if the tip message is in the logs
            found_tip = any(
                "Tip: This model might be gated or private" in log for log in cm.output
            )
            self.assertTrue(found_tip, f"Tip not found in logs: {cm.output}")

    @patch("canirun.logic.GPUAnalyzer")
    @patch("psutil.virtual_memory")
    def test_get_specs_with_gpu(self, mock_vm: Any, MockGPU: Any) -> None:
        """Tests that _get_specs correctly prioritizes GPU over CPU/Mac."""
        # Setup RAM
        mock_vm.return_value.total = 32 * 1024**3

        # Setup GPU
        mock_gpu_instance = MockGPU.return_value
        mock_gpu_instance.is_gpu_available.return_value = True
        mock_gpu_instance.vram = 24 * 1024**3
        mock_gpu_instance.device_name = "NVIDIA RTX 3090"

        # Initialize analyzer
        analyzer = ModelAnalyzer("test-model", verbose=False)

        self.assertEqual(analyzer.specs["vram"], 24 * 1024**3)
        self.assertEqual(analyzer.specs["name"], "NVIDIA RTX 3090")
        self.assertFalse(analyzer.specs["is_mac"])

    @patch("canirun.logic.GPUAnalyzer")
    @patch("platform.machine")
    @patch("platform.system")
    @patch("psutil.virtual_memory")
    def test_get_specs_mac_silicon(
        self, mock_vm: Any, mock_system: Any, mock_machine: Any, MockGPU: Any
    ) -> None:
        """Tests that _get_specs correctly detects Apple Silicon."""
        # Setup RAM
        mock_vm.return_value.total = 16 * 1024**3

        # Setup Mac Environment
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        # Setup GPU (No discrete GPU)
        mock_gpu_instance = MockGPU.return_value
        mock_gpu_instance.is_gpu_available.return_value = False

        analyzer = ModelAnalyzer("test-model", verbose=False)

        self.assertTrue(analyzer.specs["is_mac"])
        self.assertEqual(analyzer.specs["name"], "Apple Silicon (Unified Memory)")
        # Check VRAM calculation (75% of RAM)
        expected_vram = 16 * 1024**3 * 0.75
        self.assertEqual(analyzer.specs["vram"], expected_vram)

    def test_calculate_fallback_params(self) -> None:
        """Tests calculation fallback when params_billions is 0."""
        # Data with 0 params but architecture details
        data = {
            "params_billions": 0,
            "hidden_size": 1024,
            "num_hidden_layers": 10,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 1000,
        }

        results = self.analyzer.calculate(data, ctx=1024)
        self.assertTrue(len(results) > 0)

        # Verify params were calculated
        # block = 12 * 10 * 1024^2 = 120 * 1,048,576 = 125M
        # embed = 1000 * 1024 = 1M
        # Total = 126M params -> 0.126B
        # FP16 size = 0.25 GB

        res_fp16 = results[0]
        self.assertEqual(res_fp16["quant"], "FP16")
        self.assertGreater(res_fp16["total_ram"], 0)


if __name__ == "__main__":
    unittest.main()
