"""Canirun: A CLI tool to check if you can run a Hugging Face model locally."""

from .api import AnalysisResult, canirun
from .enum import COMPATIBILITY

__version__ = "1.0.1"


__all__ = ["COMPATIBILITY", "canirun", "AnalysisResult"]
