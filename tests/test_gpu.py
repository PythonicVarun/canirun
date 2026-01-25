import sys
import unittest
from unittest.mock import MagicMock, patch

from canirun.gpu import GPUAnalyzer


class TestGPUAnalyzer(unittest.TestCase):
    """Test suite for the GPUAnalyzer class."""

    @patch("canirun.gpu.shutil.which")
    @patch("subprocess.run")
    def test_gpu_detection_linux_success(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """Tests successful GPU detection on Linux."""
        # Simulate Linux environment
        with patch("sys.platform", "linux"):
            mock_which.return_value = "/usr/bin/nvidia-smi"

            # Mock successful subprocess output
            mock_run.return_value.stdout = "0, NVIDIA GeForce RTX 3080, 10240"
            mock_run.return_value.returncode = 0

            analyzer = GPUAnalyzer(verbose=True)

            self.assertTrue(analyzer.is_gpu_available())
            self.assertEqual(analyzer.gpu_count, 1)
            self.assertEqual(analyzer.device_name, "NVIDIA GeForce RTX 3080")
            self.assertEqual(analyzer.get_device_name(0), "NVIDIA GeForce RTX 3080")

            # 10240 MiB -> bytes
            expected_vram = 10240 * 1024 * 1024
            self.assertEqual(analyzer.vram, expected_vram)
            self.assertEqual(analyzer.get_vram(0), expected_vram)

            # Check subprocess call
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            self.assertEqual(args[0][0], "nvidia-smi")

    @patch("canirun.gpu.shutil.which")
    @patch("subprocess.run")
    def test_gpu_detection_windows_success(
        self,
        mock_run: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        """Tests successful GPU detection on Windows."""
        # Simulate Windows environment
        with patch("sys.platform", "win32"):
            nvidia_path = "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
            mock_which.return_value = nvidia_path

            mock_run.return_value.stdout = "0, NVIDIA GeForce RTX 4090, 24576"
            mock_run.return_value.returncode = 0

            analyzer = GPUAnalyzer(verbose=False)

            self.assertTrue(analyzer.is_gpu_available())
            self.assertEqual(analyzer.device_name, "NVIDIA GeForce RTX 4090")

            # 24576 MiB -> bytes
            expected_vram = 24576 * 1024 * 1024
            self.assertEqual(analyzer.vram, expected_vram)
            self.assertEqual(analyzer.get_vram(0), expected_vram)

            # Check subprocess call used full path or found executable
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            self.assertEqual(args[0][0], nvidia_path)

    @patch("canirun.gpu.shutil.which")
    @patch("subprocess.run")
    def test_gpu_not_found(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Tests behavior when no GPU is detected (nvidia-smi fails or missing)."""
        # if nvidia-smi missing
        mock_which.return_value = None
        # On windows, it tries default path even if which returns None,
        # so we also need subprocess to fail or return empty
        mock_run.side_effect = FileNotFoundError("No file")

        analyzer = GPUAnalyzer()
        self.assertFalse(analyzer.is_gpu_available())
        self.assertEqual(analyzer.gpu_count, 0)
        self.assertEqual(analyzer.device_name, "CPU")
        self.assertEqual(analyzer.vram, 0)
        self.assertEqual(analyzer.get_device_name(0), "CPU")
        self.assertEqual(analyzer.get_vram(0), 0)

    @patch("canirun.gpu.shutil.which")
    @patch("subprocess.run")
    def test_multiple_gpus(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Tests detection of multiple GPUs."""
        mock_which.return_value = "nvidia-smi"
        output = "0, GPU A, 8000\n" "1, GPU B, 12000"
        mock_run.return_value.stdout = output
        mock_run.return_value.returncode = 0

        analyzer = GPUAnalyzer()

        self.assertEqual(analyzer.gpu_count, 2)
        self.assertEqual(analyzer.get_device_name(0), "GPU A")
        self.assertEqual(analyzer.get_device_name(1), "GPU B")

        # Test out of bounds
        self.assertEqual(
            analyzer.get_device_name(99), "GPU A"
        )  # Should warn and return first
        self.assertEqual(analyzer.get_vram(99), 8000 * 1024 * 1024)

    @patch("subprocess.run")
    def test_windows_default_path(self, mock_run: MagicMock) -> None:
        """Tests fallback to default Windows path if shutil.which fails."""
        with (
            patch("sys.platform", "win32"),
            patch("canirun.gpu.shutil.which", return_value=None),
            patch("os.environ.get", return_value="C:"),
        ):
            mock_run.return_value.stdout = "0, GPU Default, 4000"
            mock_run.return_value.returncode = 0

            analyzer = GPUAnalyzer()

            self.assertTrue(analyzer.is_gpu_available())
            self.assertEqual(analyzer.device_name, "GPU Default")

            # Verify it tried the hardcoded path
            args, _ = mock_run.call_args
            cmd = args[0]
            self.assertTrue("Program Files" in cmd[0])
            self.assertTrue("nvidia-smi.exe" in cmd[0])


if __name__ == "__main__":
    unittest.main()
