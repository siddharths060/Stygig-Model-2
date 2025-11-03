"""
Enterprise Recommendation Configuration for StyGig

This module provides configurable parameters for the recommendation system,
allowing flexible control over recommendation strategies, category limits,
and fallback mechanisms for enterprise deployment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class RecommendationConfig:
    """Enterprise-grade recommendation configuration."""
    
    # Core recommendation parameters
    items_per_category: int = 2
    max_total_recommendations: int = 26  # 13 categories * 2 items
    min_similarity_threshold: float = 0.5
    
    # Diversity and quality controls
    enforce_category_diversity: bool = True
    allow_same_item_as_query: bool = False
    quality_tiers: Dict[str, float] = None
    
    # Fallback strategies
    fallback_to_top_similar: bool = True
    fallback_min_items: int = 10
    cross_category_boost: float = 1.0
    
    # Performance settings
    faiss_search_multiplier: int = 5  # Search 5x more to ensure diversity
    batch_processing: bool = True
    
    # Enterprise features
    enable_logging: bool = True
    validate_recommendations: bool = True
    track_performance_metrics: bool = True
    
    def __post_init__(self):
        """Initialize default quality tiers if not provided."""
        if self.quality_tiers is None:
            self.quality_tiers = {
                'excellent': 0.95,
                'good': 0.85,
                'fair': 0.75,
                'minimum': self.min_similarity_threshold
            }
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.items_per_category < 1:
            raise ValueError("items_per_category must be at least 1")
        
        if self.max_total_recommendations < self.items_per_category:
            raise ValueError("max_total_recommendations must be >= items_per_category")
        
        if not 0.0 <= self.min_similarity_threshold <= 1.0:
            raise ValueError("min_similarity_threshold must be between 0 and 1")
        
        if self.faiss_search_multiplier < 1:
            raise ValueError("faiss_search_multiplier must be at least 1")
    
    def get_search_k(self, num_categories: int) -> int:
        """Calculate optimal search K for FAISS based on configuration."""
        base_k = self.items_per_category * num_categories
        return base_k * self.faiss_search_multiplier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'items_per_category': self.items_per_category,
            'max_total_recommendations': self.max_total_recommendations,
            'min_similarity_threshold': self.min_similarity_threshold,
            'enforce_category_diversity': self.enforce_category_diversity,
            'allow_same_item_as_query': self.allow_same_item_as_query,
            'quality_tiers': self.quality_tiers,
            'fallback_to_top_similar': self.fallback_to_top_similar,
            'fallback_min_items': self.fallback_min_items,
            'cross_category_boost': self.cross_category_boost,
            'faiss_search_multiplier': self.faiss_search_multiplier,
            'batch_processing': self.batch_processing,
            'enable_logging': self.enable_logging,
            'validate_recommendations': self.validate_recommendations,
            'track_performance_metrics': self.track_performance_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RecommendationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'RecommendationConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {filepath}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

# Predefined configurations for different use cases
ENTERPRISE_CONFIGS = {
    'default': RecommendationConfig(),
    
    'high_diversity': RecommendationConfig(
        items_per_category=2,
        min_similarity_threshold=0.6,
        cross_category_boost=1.2,
        faiss_search_multiplier=8
    ),
    
    'quality_focused': RecommendationConfig(
        items_per_category=1,
        min_similarity_threshold=0.8,
        quality_tiers={
            'excellent': 0.95,
            'good': 0.90,
            'fair': 0.80,
            'minimum': 0.8
        }
    ),
    
    'high_volume': RecommendationConfig(
        items_per_category=3,
        max_total_recommendations=39,  # 13 * 3
        min_similarity_threshold=0.4,
        fallback_min_items=20
    ),
    
    'demo_showcase': RecommendationConfig(
        items_per_category=2,
        min_similarity_threshold=0.7,
        enforce_category_diversity=True,
        track_performance_metrics=True,
        faiss_search_multiplier=6
    )
}

def get_config(config_name: str = 'default') -> RecommendationConfig:
    """Get a predefined configuration by name."""
    if config_name in ENTERPRISE_CONFIGS:
        return ENTERPRISE_CONFIGS[config_name]
    else:
        logger.warning(f"Unknown config '{config_name}', using default")
        return ENTERPRISE_CONFIGS['default']

def create_custom_config(items_per_category: int, **kwargs) -> RecommendationConfig:
    """Create a custom configuration with specified items per category."""
    return RecommendationConfig(
        items_per_category=items_per_category,
        max_total_recommendations=items_per_category * 13,  # Assume 13 categories
        **kwargs
    )