"""
Professional Fashion Recommendation Engine

This is the main recommendation system that provides intelligent outfit suggestions
based on color harmony, gender compatibility, and category matching.

Key Features:
- Excludes input item from recommendations (prevents self-matching)
- Advanced color harmony using CIELAB color space and Itten color theory
- Hard gender filtering (male items only for male users, etc.)
- Category compatibility rules (no same-category recommendations)
- Configurable items per category for diversity (default: 2)
- Professional weighted scoring: color (45%), category (25%), gender (30%)
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import json

# Import core logic modules
try:
    from .color_logic import ColorProcessor
    from .gender_logic import GenderClassifier
    from .rules.category_compatibility import CATEGORY_COMPATIBILITY
except ImportError:
    # Fallback for different import contexts
    from color_logic import ColorProcessor
    from gender_logic import GenderClassifier
    from rules.category_compatibility import CATEGORY_COMPATIBILITY

logger = logging.getLogger(__name__)


class FashionEngine:
    """
    Professional fashion recommendation engine with intelligent matching logic.
    
    This engine solves key recommendation challenges:
    - Prevents recommending the same item as input
    - Uses advanced color harmony (not just exact color matches)  
    - Implements hard gender filtering rules
    - Ensures category diversity (no same-category recommendations)
    - Returns configurable items per category for better diversity
    """
    
    def __init__(self, 
                 dataset_path: str = None,
                 cache_dir: str = 'stygig_cache',
                 items_per_category: int = 2,
                 color_weight: float = 0.45,
                 category_weight: float = 0.25,
                 gender_weight: float = 0.30):
        """
        Initialize the fashion recommendation engine.
        
        Args:
            dataset_path: Path to the fashion dataset
            cache_dir: Directory for caching processed data
            items_per_category: Number of items to return per category (default 2)
            color_weight: Weight for color harmony in final score (default 0.45)
            category_weight: Weight for category compatibility (default 0.25) 
            gender_weight: Weight for gender compatibility (default 0.30)
        """
        self.dataset_path = dataset_path
        self.cache_dir = Path(cache_dir)
        self.items_per_category = items_per_category
        
        # Scoring weights (must sum to 1.0)
        total_weight = color_weight + category_weight + gender_weight
        self.weights = {
            'color': color_weight / total_weight,
            'category': category_weight / total_weight,
            'gender': gender_weight / total_weight
        }
        
        # Initialize core processors
        self.color_processor = ColorProcessor(n_clusters=3)
        self.gender_classifier = GenderClassifier()
        
        # Data storage
        self.items_by_category = {}  # category -> list of items
        self.all_items = []  # flat list of all items
        self.index_built = False
        
        logger.info(f"FashionEngine initialized")
        logger.info(f"Weights - Color: {self.weights['color']:.2f}, "
                   f"Category: {self.weights['category']:.2f}, "
                   f"Gender: {self.weights['gender']:.2f}")
    
    def build_index(self, dataset_path: str = None, force_rebuild: bool = False):
        """
        Build the recommendation index from the dataset.
        
        Args:
            dataset_path: Optional override for dataset path
            force_rebuild: Force rebuilding even if cache exists
        """
        dataset_path = dataset_path or self.dataset_path
        if not dataset_path:
            raise ValueError("Dataset path must be provided")
        
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Look for train directory (common structure)
        train_dir = dataset_dir / 'train'
        if train_dir.exists():
            dataset_dir = train_dir
        
        logger.info(f"Building index from: {dataset_dir}")
        
        # Process each category directory
        self.items_by_category = {}
        self.all_items = []
        
        for category_dir in dataset_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            logger.info(f"Processing category: {category_name}")
            
            category_items = []
            
            # Process images in category
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
            for image_path in category_dir.iterdir():
                if image_path.suffix.lower() not in image_extensions:
                    continue
                
                try:
                    # Extract color information
                    colors = self.color_processor.extract_dominant_colors(str(image_path))
                    dominant_color = colors[0][0] if colors else 'unknown'
                    
                    # Predict gender
                    gender, gender_confidence = self.gender_classifier.predict_gender(
                        str(image_path), category_name
                    )
                    
                    # Create item entry
                    item = {
                        'id': image_path.stem,
                        'path': str(image_path),
                        'category': category_name,
                        'color': dominant_color,
                        'gender': gender,
                        'gender_confidence': gender_confidence
                    }
                    
                    category_items.append(item)
                    self.all_items.append(item)
                
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
                    continue
            
            self.items_by_category[category_name] = category_items
            logger.info(f"Processed {len(category_items)} items in {category_name}")
        
        self.index_built = True
        logger.info(f"Index built successfully. Total items: {len(self.all_items)} "
                   f"across {len(self.items_by_category)} categories")
    
    def get_recommendations(self, 
                          image_path: str, 
                          n_recommendations: Optional[int] = None,
                          user_gender: Optional[str] = None,
                          items_per_category: Optional[int] = None) -> Dict:
        """
        Get fashion recommendations for an input image.
        
        This is the main method that solves all the key issues:
        1. Excludes the input item from recommendations (no self-matching)
        2. Uses advanced color harmony scoring
        3. Applies hard gender filtering
        4. Ensures category diversity
        5. Returns specified items per category
        
        Args:
            image_path: Path to the input fashion image
            n_recommendations: Total number of recommendations (unused if items_per_category specified)
            user_gender: User's gender ('male', 'female', 'unisex'). If None, inferred from input
            items_per_category: Items per category override
            
        Returns:
            Dictionary with query item info and recommendations
        """
        if not self.index_built:
            return self._error_response("Index not built. Call build_index() first.")
        
        items_per_category = items_per_category or self.items_per_category
        
        try:
            # Extract input item characteristics
            input_colors = self.color_processor.extract_dominant_colors(image_path)
            input_color = input_colors[0][0] if input_colors else 'unknown'
            
            # Infer input category from path (if possible)
            input_category = self._infer_category_from_path(image_path)
            
            # Determine user gender (use provided or infer from input)
            if user_gender is None:
                user_gender, _ = self.gender_classifier.predict_gender(image_path, input_category)
            
            # Get input item ID to exclude from recommendations
            input_item_id = Path(image_path).stem
            
            logger.info(f"Input analysis - Color: {input_color}, Category: {input_category}, "
                       f"Gender: {user_gender}, ID: {input_item_id}")
            
            # Get compatible categories based on input category
            compatible_categories = self._get_compatible_categories(input_category)
            
            # Filter items by gender compatibility
            compatible_genders = self.gender_classifier.get_compatible_genders(user_gender)
            
            # Collect and score recommendations per category
            recommendations_by_category = {}
            
            for category in compatible_categories:
                if category not in self.items_by_category:
                    continue
                
                category_items = self.items_by_category[category]
                
                # Filter by gender and exclude input item
                filtered_items = []
                for item in category_items:
                    # Exclude the input item itself
                    if item['id'] == input_item_id:
                        continue
                    
                    # Apply gender filtering
                    if item['gender'] not in compatible_genders:
                        continue
                    
                    filtered_items.append(item)
                
                if not filtered_items:
                    continue
                
                # Score and rank items in this category
                scored_items = []
                for item in filtered_items:
                    score_components = self._calculate_item_score(
                        input_color, input_category, user_gender, item
                    )
                    
                    scored_items.append((score_components['total'], item, score_components))
                
                # Sort by score and take top items
                scored_items.sort(key=lambda x: x[0], reverse=True)
                top_items = scored_items[:items_per_category]
                
                if top_items:
                    recommendations_by_category[category] = top_items
            
            # Format recommendations
            recommendations = []
            for category, scored_items in recommendations_by_category.items():
                for score, item, score_components in scored_items:
                    recommendation = {
                        'id': item['id'],
                        'path': item['path'],
                        'category': item['category'],
                        'gender': item['gender'],
                        'color': item['color'],
                        'score': round(score, 4),
                        'score_components': {
                            'color_harmony': round(score_components['color'], 4),
                            'category_compatibility': round(score_components['category'], 4),
                            'gender_compatibility': round(score_components['gender'], 4)
                        },
                        'match_reason': self._generate_match_reason(
                            score_components, input_color, item['color']
                        )
                    }
                    recommendations.append(recommendation)
            
            # Sort final recommendations by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Prepare response
            return {
                'query_item': {
                    'category': input_category,
                    'gender': user_gender,
                    'dominant_color': input_color,
                    'id': input_item_id
                },
                'recommendations': recommendations,
                'processing_info': {
                    'total_recommendations': len(recommendations),
                    'categories_found': len(recommendations_by_category),
                    'items_per_category': items_per_category,
                    'compatible_categories': compatible_categories,
                    'input_excluded': True  # Confirms input item was excluded
                }
            }
        
        except Exception as e:
            logger.error(f"Recommendation failed for {image_path}: {e}")
            return self._error_response(f"Recommendation failed: {str(e)}")
    
    def _infer_category_from_path(self, image_path: str) -> Optional[str]:
        """Infer category from image path."""
        path_str = str(image_path).lower()
        
        # Check if path contains any known category names
        for category in self.items_by_category.keys():
            if category.lower() in path_str:
                return category
        
        return None
    
    def _get_compatible_categories(self, input_category: Optional[str]) -> List[str]:
        """Get list of categories compatible with the input category."""
        if input_category and input_category in CATEGORY_COMPATIBILITY:
            return CATEGORY_COMPATIBILITY[input_category]['compatible']
        
        # If no specific input category, return all categories except same-type
        all_categories = list(self.items_by_category.keys())
        
        if input_category:
            # Remove categories that should be avoided
            compatibility = CATEGORY_COMPATIBILITY.get(input_category, {})
            avoid = compatibility.get('avoid', [])
            return [cat for cat in all_categories if cat not in avoid]
        
        return all_categories
    
    def _calculate_item_score(self, input_color: str, input_category: Optional[str], 
                             user_gender: str, item: Dict) -> Dict[str, float]:
        """Calculate comprehensive score for an item."""
        # Color harmony score
        color_score = self.color_processor.calculate_color_harmony(
            input_color, item['color']
        )
        
        # Category compatibility score
        category_score = 1.0  # All items in compatible categories get full score
        if input_category and input_category in CATEGORY_COMPATIBILITY:
            avoid_categories = CATEGORY_COMPATIBILITY[input_category]['avoid']
            if item['category'] in avoid_categories:
                category_score = 0.0  # Should not happen due to filtering
        
        # Gender compatibility score
        gender_score = self.gender_classifier.calculate_gender_score(
            user_gender, item['gender']
        )
        
        # Calculate weighted total score
        total_score = (
            self.weights['color'] * color_score +
            self.weights['category'] * category_score +
            self.weights['gender'] * gender_score
        )
        
        return {
            'color': color_score,
            'category': category_score,
            'gender': gender_score,
            'total': total_score
        }
    
    def _generate_match_reason(self, score_components: Dict[str, float], 
                              input_color: str, item_color: str) -> str:
        """Generate a human-readable explanation for the match."""
        reasons = []
        
        if score_components['color'] >= 0.85:
            if input_color == item_color:
                reasons.append(f"matching {input_color}")
            else:
                reasons.append(f"harmonious colors ({input_color} + {item_color})")
        elif score_components['color'] >= 0.70:
            reasons.append("good color harmony")
        
        if score_components['gender'] >= 0.85:
            reasons.append("gender appropriate")
        
        if score_components['category'] >= 0.90:
            reasons.append("perfect style match")
        
        if not reasons:
            reasons.append("style compatible")
        
        return ", ".join(reasons)
    
    def _error_response(self, error_message: str) -> Dict:
        """Generate error response."""
        return {
            'error': error_message,
            'query_item': None,
            'recommendations': [],
            'processing_info': None
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded dataset."""
        if not self.index_built:
            return {'error': 'Index not built'}
        
        gender_dist = {'male': 0, 'female': 0, 'unisex': 0}
        color_dist = {}
        
        for item in self.all_items:
            # Gender distribution
            gender = item.get('gender', 'unisex')
            if gender in gender_dist:
                gender_dist[gender] += 1
            
            # Color distribution
            color = item.get('color', 'unknown')
            color_dist[color] = color_dist.get(color, 0) + 1
        
        return {
            'total_items': len(self.all_items),
            'categories': {cat: len(items) for cat, items in self.items_by_category.items()},
            'gender_distribution': gender_dist,
            'color_distribution': dict(sorted(color_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            'items_per_category_setting': self.items_per_category,
            'scoring_weights': self.weights
        }


# Backward compatibility aliases
RecommendationEngine = FashionEngine
ProfessionalFashionRecommendationEngine = FashionEngine
