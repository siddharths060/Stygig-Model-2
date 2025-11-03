"""
StyGig Fashion Recommendation System

Enterprise-grade fashion recommendation engine using computer vision,
color theory, and intelligent matching algorithms.
"""

__version__ = "1.0.0"
__author__ = "StyGig Team"

from .core.recommendation_engine import FashionEngine
from .core.color_logic import ColorProcessor
from .core.gender_logic import GenderClassifier

__all__ = [
    'FashionEngine',
    'ColorProcessor',
    'GenderClassifier'
]
