"""
Gender Classification Logic for Fashion Recommendations

This module handles gender prediction and compatibility calculations
for fashion items based on category and visual features.
"""

import logging
from typing import Tuple, List
import random

logger = logging.getLogger(__name__)


class GenderClassifier:
    """
    Professional gender classifier for fashion items.
    
    Uses rule-based logic and category analysis to predict gender
    and ensure gender-appropriate recommendations.
    """
    
    # Gender compatibility matrix
    GENDER_COMPATIBILITY = {
        'male': ['male', 'unisex'],
        'female': ['female', 'unisex'],
        'unisex': ['male', 'female', 'unisex']
    }
    
    def __init__(self):
        """Initialize gender classifier."""
        logger.info("GenderClassifier initialized")
    
    def predict_gender(self, image_path: str, category: str) -> Tuple[str, float]:
        """
        Predict gender from image and category information.
        
        Uses rule-based logic based on category names and fashion conventions.
        
        Args:
            image_path: Path to the fashion image
            category: Category name (e.g., 'upperwear_shirt', 'bottomwear_skirt')
            
        Returns:
            Tuple of (gender, confidence) where gender is 'male', 'female', or 'unisex'
        """
        if not category:
            return ("unisex", 0.5)
        
        category_lower = category.lower()
        
        # Female-specific items
        if any(keyword in category_lower for keyword in ['dress', 'skirt', 'heels']):
            return ("female", 0.85)
        
        # Male-leaning items (but can be unisex)
        if any(keyword in category_lower for keyword in ['suit', 'tie']):
            return ("male", 0.75)
        
        # Typically unisex items
        if any(keyword in category_lower for keyword in ['jacket', 'coat', 'sweater', 
                                                          'jeans', 'shorts', 'sneakers',
                                                          'bag', 'hat', 'watch']):
            return ("unisex", 0.70)
        
        # Shirts and t-shirts - lean unisex but check for specific styles
        if 'shirt' in category_lower or 'tshirt' in category_lower:
            return ("unisex", 0.65)
        
        # Pants - generally unisex
        if 'pants' in category_lower or 'trouser' in category_lower:
            return ("unisex", 0.70)
        
        # Default to unisex for unknown categories
        return ("unisex", 0.60)
    
    def get_compatible_genders(self, user_gender: str) -> List[str]:
        """
        Get list of item genders compatible with user gender.
        
        Hard filtering rules:
        - Male users: male and unisex items only
        - Female users: female and unisex items only
        - Unisex: all items allowed
        
        Args:
            user_gender: User's gender ('male', 'female', or 'unisex')
            
        Returns:
            List of compatible gender labels
        """
        if user_gender not in self.GENDER_COMPATIBILITY:
            logger.warning(f"Unknown gender '{user_gender}', defaulting to unisex")
            user_gender = 'unisex'
        
        return self.GENDER_COMPATIBILITY[user_gender]
    
    def calculate_gender_score(self, user_gender: str, item_gender: str) -> float:
        """
        Calculate gender compatibility score.
        
        Scoring:
        - Exact match: 1.0
        - Unisex compatibility: 0.75
        - Incompatible: 0.0
        
        Args:
            user_gender: User's gender
            item_gender: Item's gender
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Exact match
        if user_gender == item_gender:
            return 1.0
        
        # Check compatibility
        compatible_genders = self.get_compatible_genders(user_gender)
        
        if item_gender in compatible_genders:
            # Unisex items with specific gender users
            if item_gender == 'unisex':
                return 0.75
            # Specific gender items with unisex users
            elif user_gender == 'unisex':
                return 0.75
            else:
                return 0.75
        
        # Incompatible
        return 0.0
