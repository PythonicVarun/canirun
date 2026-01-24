"""Core logic for model memory analysis and hardware detection."""

import json
import logging
import platform
from typing import Any

import psutil
from huggingface_hub import hf_hub_download, login, model_info

from canirun.enum import COMPATIBILITY
from canirun.gpu import GPUAnalyzer
from canirun.human_readable import get_human_readable_size

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyzes model compatibility with local hardware."""

    def __init__(
        self,
        model_id: str,
        verbose: bool = True,
        hf_token: str | None = None,
    ) -> None:
        """Initializes the ModelAnalyzer.

        Args:
            model_id: The ID of the model to analyze.
            verbose: Whether to enable verbose logging. Defaults to True.
            hf_token: The Hugging Face API token. Defaults to None.
        """
        self.model_id = model_id
        self.verbose = verbose
        self.hf_token = hf_token

        # Adjust log level based on verbosity
        if not self.verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Login to Hugging Face if token is provided
        self._login()

        # Initialize GPU Analyzer
        self._gpu_analyzer = GPUAnalyzer(verbose=self.verbose)

        self.specs = self._get_specs()
        logger.info(
            f"Detected Hardware: {self.specs['name']} | "
            f"VRAM: {get_human_readable_size(self.specs['vram'])} | "
            f"RAM: {get_human_readable_size(self.specs['ram'])}"
        )

    def _login(self) -> None:
        """Logs in to Hugging Face using the token (if provided)."""
        if self.hf_token:
            login(self.hf_token)

    def _get_specs(self) -> dict[str, Any]:
        """Detects System RAM and VRAM (CUDA or Apple Silicon).

        Returns:
            dict[str, Any]: A dictionary containing RAM, VRAM, device name, and Mac status.
        """
        ram = psutil.virtual_memory().total
        vram = 0
        device_name = "CPU Only"

        is_mac = platform.system() == "Darwin" and platform.machine() == "arm64"

        if self._gpu_analyzer.is_gpu_available():
            vram = self._gpu_analyzer.vram
            device_name = self._gpu_analyzer.device_name
        elif is_mac:
            vram = ram * 0.75
            device_name = "Apple Silicon (Unified Memory)"

        return {"ram": ram, "vram": vram, "name": device_name, "is_mac": is_mac}

    def fetch_model_data(self) -> dict[str, Any] | None:
        """Fetches architecture details for accurate GQA and Parameter calculation.

        Returns:
            dict[str, Any] | None: A dictionary containing model architecture details, or None if fetching fails.
        """
        logger.info(f"Fetching config for: {self.model_id}...")
        try:
            total_params_billions = 0
            try:
                info = model_info(self.model_id)
                if hasattr(info, "safetensors") and info.safetensors.get("total"):  # type: ignore
                    total_params_billions = info.safetensors["total"] / 1e9  # type: ignore
            except Exception as e:
                logger.debug(f"Could not fetch safetensor info: {e}")

            path = hf_hub_download(self.model_id, "config.json")
            with open(path) as f:
                config = json.load(f)

            return {
                "params_billions": total_params_billions,
                "hidden_size": config.get("hidden_size") or config.get("d_model", 4096),
                "num_hidden_layers": (
                    config.get("num_hidden_layers") or config.get("n_layer", 32)
                ),
                "num_attention_heads": (
                    config.get("num_attention_heads") or config.get("n_head", 32)
                ),
                "num_key_value_heads": (
                    config.get("num_key_value_heads") or config.get("n_head", 32)
                ),
                "vocab_size": config.get("vocab_size", 32000),
            }
        except Exception as e:
            logger.error(f"Error fetching config for {self.model_id}: {e}")
            if not self.hf_token and (
                "401" in str(e) or "403" in str(e) or "gated" in str(e).lower()
            ):
                logger.error(
                    "Tip: This model might be gated or private. "
                    "Please provide a Hugging Face token using --hf-token or set the HF_TOKEN environment variable."
                )
            return None

    def calculate(
        self,
        data: dict[str, Any],
        ctx: int = 4096,
    ) -> list[dict[str, Any]]:
        """Calculates memory requirements and compatibility for different quantization levels.

        Args:
            data: Model architecture details.
            ctx: Context length to simulate.

        Returns:
            list[dict[str, Any]]: Analysis results for each quantization level.
        """
        if not data:
            return []
        # Model Weights Calculation
        params = data["params_billions"]
        if params == 0:
            block_params = 12 * data["num_hidden_layers"] * (data["hidden_size"] ** 2)
            embed_params = data["vocab_size"] * data["hidden_size"]
            params = (block_params + embed_params) / 1e9

        # KV Cache Formula: 2 * Layers * KV_Heads * Head_Dim * Context
        head_dim = data["hidden_size"] // data["num_attention_heads"]
        kv_elements = (
            2 * data["num_hidden_layers"] * data["num_key_value_heads"] * head_dim * ctx
        )

        quants = {
            "FP16": {"w_bits": 16, "kv_bytes": 2},
            "INT8": {"w_bits": 8, "kv_bytes": 2},
            "4-bit": {"w_bits": 4.6, "kv_bytes": 2},
            "2-bit": {"w_bits": 2.5, "kv_bytes": 2},
        }

        results = []
        overhead_bytes = 1.5 * (1024**3)

        # Summary Log
        is_gqa = data["num_key_value_heads"] != data["num_attention_heads"]
        logger.info(f"Analysis for {ctx} context: {params:.2f}B Params | GQA: {is_gqa}")

        for name, conf in quants.items():
            weight_mem_bytes = (params * 1e9 * conf["w_bits"]) / 8
            kv_cache_bytes = kv_elements * conf["kv_bytes"]
            req_bytes = weight_mem_bytes + kv_cache_bytes + overhead_bytes

            if req_bytes <= self.specs["vram"]:
                status = COMPATIBILITY.FULL
            elif req_bytes <= self.specs["ram"]:
                status = COMPATIBILITY.PARTIAL
            else:
                status = COMPATIBILITY.NONE

            results.append(
                {
                    "quant": name,
                    "total_ram": int(req_bytes),
                    "kv_cache": int(kv_cache_bytes),
                    "status": status,
                }
            )

        return results
