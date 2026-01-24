"""API for checking model compatibility with local hardware."""

from typing import Any

from .enum import COMPATIBILITY
from .logic import ModelAnalyzer


class AnalysisResult:
    """Stores the result of the memory analysis."""

    def __init__(
        self,
        model_id: str,
        hardware: dict[str, Any],
        analysis: list[dict[str, Any]],
    ) -> None:
        """Initializes the AnalysisResult.

        Args:
            model_id: The ID of the analyzed model.
            hardware: A dictionary containing hardware specifications.
            analysis: A list of dictionaries containing analysis results for different quantization levels.
        """
        self.model_id = model_id
        self.hardware = hardware
        self.analysis = analysis

    @property
    def issupported(self) -> bool:
        """Checks if the model is supported on the current hardware.

        Returns True if the model can run (at least partially) on the current hardware
        with any of the checked quantization levels.

        Returns:
            bool: True if supported, False otherwise.
        """
        if not self.analysis:
            return False
        return any(r["status"] != COMPATIBILITY.NONE for r in self.analysis)

    @property
    def is_supported(self) -> bool:
        """Alias for issupported following Python naming conventions.

        Returns:
            bool: True if supported, False otherwise.
        """
        return self.issupported

    def report(self) -> list[dict[str, Any]]:
        """Returns the complete analysis list.

        Returns:
            list[dict[str, Any]]: The complete analysis list containing all quantization details.
        """
        return self.analysis

    def __repr__(self) -> str:
        """Returns a string representation of the AnalysisResult."""
        return f"<AnalysisResult model='{self.model_id}' supported={self.issupported}>"


def canirun(
    model_id: str,
    context_length: int = 2048,
    verbose: bool = False,
    hf_token: str | None = None,
) -> AnalysisResult | None:
    """Analyzes memory usage for a given model on the current hardware.

    Args:
        model_id: Hugging Face model ID.
        context_length: Context window size (default: 2048).
        verbose: Enable detailed logs (default: False).
        hf_token: Hugging Face API token for gated or private models (default: None).

    Returns:
        AnalysisResult | None: An object containing hardware specs and compatibility results.
            Returns None if model data fetch fails.
    """
    analyzer = ModelAnalyzer(model_id, verbose=verbose, hf_token=hf_token)
    model_data = analyzer.fetch_model_data()

    if not model_data:
        return None

    results = analyzer.calculate(model_data, ctx=context_length)

    return AnalysisResult(model_id, analyzer.specs, results)
