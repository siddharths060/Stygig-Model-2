"""StyGig Core Module"""

from .recommendation_engine import FashionEngine
from .color_logic import ColorProcessor
from .gender_logic import GenderClassifier

__all__ = [
    'FashionEngine',
    'ColorProcessor',
    'GenderClassifier'
]
