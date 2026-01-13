import unittest

from canirun.enum import COMPATIBILITY
from canirun.logic import ModelAnalyzer


class TestModelAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ModelAnalyzer("test-model", verbose=False)
        # Mock specs with bytes
        gb = 1024**3
        self.analyzer.specs = {
            "ram": 32 * gb,
            "vram": 24 * gb,
            "name": "Test GPU",
            "is_mac": False,
        }

    def test_calculate_returns_bytes_as_int(self):
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


if __name__ == "__main__":
    unittest.main()
