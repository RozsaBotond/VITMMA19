"""Baseline model package.

Contains:
- MLPBaseline: Simple MLP baseline (neural network)
- StatisticalBaseline: Trend/volatility-based rule detection (non-neural)
"""
from .model import MLPBaseline
from .statistical import StatisticalBaseline

__all__ = ["MLPBaseline", "StatisticalBaseline"]
