from typing import Any, Dict, List, Optional

from .enum import COMPATIBILITY
from .logic import ModelAnalyzer


class AnalysisResult:
    def __init__(
        self, model_id: str, hardware: Dict[str, Any], analysis: List[Dict[str, Any]]
    ):
        self.model_id = model_id
        self.hardware = hardware
        self.analysis = analysis

    @property
    def issupported(self) -> bool:
        """
        Returns True if the model can run (at least partially) on the current hardware
        with any of the checked quantization levels.
        """
        if not self.analysis:
            return False
        return any(r["status"] != COMPATIBILITY.NONE for r in self.analysis)

    @property
    def is_supported(self) -> bool:
        """Alias for issupported following Python naming conventions."""
        return self.issupported

    def report(self) -> List[Dict[str, Any]]:
        """Returns the complete analysis list containing all quantization details."""
        return self.analysis

    def __repr__(self) -> str:
        return f"<AnalysisResult model='{self.model_id}' supported={self.issupported}>"


def canirun(
    model_id: str,
    context_length: int = 2048,
    verbose: bool = False,
    hf_token: Optional[str] = None,
) -> Optional[AnalysisResult]:
    """
    Analyzes memory usage for a given model on the current hardware.

    Args:
        model_id (str): Hugging Face model ID.
        context_length (int): Context window size (default: 2048).
        verbose (bool): Enable detailed logs (default: False).
        hf_token (Optional[str]): Hugging Face API token for gated or private models (default: None).

    Returns:
        AnalysisResult: An object containing hardware specs and compatibility results.
                        Returns None if model data fetch fails.
    """
    analyzer = ModelAnalyzer(model_id, verbose=verbose, hf_token=hf_token)
    model_data = analyzer.fetch_model_data()

    if not model_data:
        return None

    results = analyzer.calculate(model_data, ctx=context_length)

    return AnalysisResult(model_id, analyzer.specs, results)
