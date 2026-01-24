"""GPU hardware analysis using nvidia-smi."""

import logging
import os
import shutil
import subprocess
import sys

from canirun.human_readable import get_human_readable_size

logger = logging.getLogger(__name__)


class GPUAnalyzer:
    """Analyzes GPU capabilities and VRAM."""

    def __init__(self, verbose: bool = True) -> None:
        """Initializes the GPUAnalyzer.

        Args:
            verbose: Whether to enable verbose logging. Defaults to True.
        """
        self.verbose = verbose

        # Adjust log level based on verbosity
        if not self.verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        self.gpu_info = self._get_gpu_info()
        if self.gpu_info:
            logger.info(
                f"Detected GPU: {self.gpu_info[0]['name']} | "
                f"VRAM: {get_human_readable_size(self.gpu_info[0]['vram'])}"
            )
        else:
            logger.info("No compatible GPU detected.")

    def is_gpu_available(self) -> bool:
        """Checks if a compatible GPU is available.

        Returns:
            bool: True if a compatible GPU is available, False otherwise.
        """
        return self.gpu_info is not None

    @property
    def gpu_count(self) -> int:
        """Returns the number of detected GPUs.

        Returns:
            int: Number of GPUs detected.
        """
        if self.gpu_info:
            return len(self.gpu_info)
        return 0

    @property
    def device_name(self) -> str:
        """Returns the name of the detected GPU or 'CPU' if none is found.

        Returns:
            str: GPU name or 'CPU'.
        """
        if self.gpu_info:
            return self.gpu_info[0]["name"]
        return "CPU"

    def get_device_name(self, index: int = 0) -> str:
        """Returns the name of the detected GPU or 'CPU' if none is found.

        Args:
            index (int): The index of the GPU to query. Defaults to 0.

        Returns:
            str: GPU name or 'CPU'.
        """
        if self.gpu_info:
            if index < self.gpu_count:
                return self.gpu_info[index]["name"]
            else:
                logger.warning(
                    f"GPU index {index} out of range. Returning first GPU name."
                )
                return self.gpu_info[0]["name"]
        return "CPU"

    @property
    def vram(self) -> int:
        """Returns the VRAM of the detected GPU in bytes.

        Returns:
            int: VRAM in bytes, or 0 if no GPU is found.
        """
        if self.gpu_info:
            return self.gpu_info[0]["vram"]
        return 0

    def get_vram(self, index: int = 0) -> int:
        """Returns the VRAM of the detected GPU in bytes.

        Args:
            index (int): The index of the GPU to query. Defaults to 0.

        Returns:
            int: VRAM in bytes, or 0 if no GPU is found.
        """
        if self.gpu_info:
            if index < self.gpu_count:
                return self.gpu_info[index]["vram"]
            else:
                logger.warning(
                    f"GPU index {index} out of range. Returning first GPU VRAM."
                )
                return self.gpu_info[0]["vram"]
        return 0

    def _get_gpu_info(self) -> list[dict] | None:
        """Detects GPU information using nvidia-smi.

        Returns:
            list[dict] | None: A list of dictionaries containing GPU info, or None if no GPU is found.
        """
        if sys.platform == "win32":
            nvidia_smi_path = shutil.which("nvidia-smi.exe")
            if nvidia_smi_path is None:
                # Try the default installation path
                system_drive = os.environ.get("SystemDrive", "C:")
                nvidia_smi_path = os.path.join(
                    system_drive,
                    "Program Files",
                    "NVIDIA Corporation",
                    "NVSMI",
                    "nvidia-smi.exe",
                )
        else:
            nvidia_smi_path = "nvidia-smi"

        try:
            result = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip()
            if output:
                gpu_list = []
                for line in output.splitlines():
                    index_str, name, vram_str = line.split(", ")
                    index = int(index_str)
                    vram = int(vram_str) * 1024 * 1024
                    gpu_list.append({"index": index, "name": name, "vram": vram})

                return gpu_list
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"nvidia-smi not found or error occurred: {e}")

        return None
