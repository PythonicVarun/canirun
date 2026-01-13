import unittest
from unittest.mock import patch

from canirun import COMPATIBILITY, canirun


class TestApi(unittest.TestCase):
    @patch("canirun.api.ModelAnalyzer")
    def test_canirun_supported(self, MockAnalyzer):
        # Setup mock for a supported model
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = {"some": "data"}
        instance.calculate.return_value = [
            {
                "quant": "FP16",
                "total_ram": 100,
                "kv_cache": 10,
                "status": COMPATIBILITY.FULL,
            },
            {
                "quant": "INT8",
                "total_ram": 50,
                "kv_cache": 10,
                "status": COMPATIBILITY.FULL,
            },
        ]
        instance.specs = {"ram": 1000, "vram": 1000, "name": "Test", "is_mac": False}

        # Call API
        result = canirun("test-model")

        # Verify
        self.assertIsNotNone(result)
        self.assertTrue(result.issupported)
        self.assertTrue(result.is_supported)

        report = result.report()
        self.assertEqual(len(report), 2)
        self.assertEqual(report[0]["quant"], "FP16")

    @patch("canirun.api.ModelAnalyzer")
    def test_canirun_not_supported(self, MockAnalyzer):
        # Setup mock for an unsupported model
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = {"some": "data"}
        instance.calculate.return_value = [
            {
                "quant": "FP16",
                "total_ram": 2000,
                "kv_cache": 10,
                "status": COMPATIBILITY.NONE,
            },
        ]
        instance.specs = {"ram": 1000, "vram": 1000, "name": "Test", "is_mac": False}

        result = canirun("big-model")

        self.assertIsNotNone(result)
        self.assertFalse(result.issupported)

    @patch("canirun.api.ModelAnalyzer")
    def test_canirun_fetch_failure(self, MockAnalyzer):
        instance = MockAnalyzer.return_value
        instance.fetch_model_data.return_value = None  # Failure

        result = canirun("bad-model")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
