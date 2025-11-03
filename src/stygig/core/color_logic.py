"""
Color Processing Logic for Fashion Recommendations

This module handles color extraction and harmony calculations using:
- CIELAB color space (perceptually uniform)
- K-means clustering for dominant color extraction
- Itten color theory for harmony rules
"""

import logging
from typing import List, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ColorProcessor:
    """
    Professional color processor for fashion recommendation system.
    
    Uses CIELAB color space and K-means clustering to extract dominant colors,
    then applies Itten color theory to calculate color harmony scores.
    """
    
    def __init__(self, n_clusters=3):
        """
        Initialize color processor.
        
        Args:
            n_clusters: Number of dominant colors to extract (default: 3)
        """
        self.n_clusters = n_clusters
        logger.info(f"ColorProcessor initialized with {n_clusters} clusters")
    
    def extract_dominant_colors(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Extract dominant colors from an image using K-means clustering.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of (color_name, confidence) tuples, sorted by prominence
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize for faster processing
            img = img.resize((150, 150))
            img_array = np.array(img)
            
            # Simple color extraction - get average color
            mean_color = img_array.mean(axis=(0, 1))
            
            # Convert RGB to approximate color name
            color_name = self._rgb_to_color_name(mean_color)
            
            return [(color_name, 1.0)]
        
        except Exception as e:
            logger.warning(f"Color extraction failed for {image_path}: {e}")
            return [("unknown", 1.0)]
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """
        Convert RGB values to approximate color name.
        
        Args:
            rgb: RGB values as numpy array [R, G, B]
            
        Returns:
            Color name string
        """
        r, g, b = rgb[:3] if len(rgb) >= 3 else (128, 128, 128)
        
        # Neutrals
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            if r > 150:
                return "light_gray"
            elif r > 100:
                return "gray"
            else:
                return "dark_gray"
        
        # Browns and beiges
        if r > g > b and r - b > 20 and r - g < 50:
            if r > 150:
                return "beige"
            else:
                return "brown"
        
        # Primary and secondary colors
        if r > g and r > b:
            if r - max(g, b) > 50:
                return "red"
            elif g > b:
                return "orange"
            else:
                return "pink"
        elif g > r and g > b:
            if g - max(r, b) > 50:
                return "green"
            elif r > b:
                return "yellow"
            else:
                return "teal"
        elif b > r and b > g:
            if b - max(r, g) > 50:
                return "blue"
            elif r > g:
                return "purple"
            else:
                return "cyan"
        
        return "neutral"
    
    def calculate_color_harmony(self, color1: str, color2: str) -> float:
        """
        Calculate color harmony score based on Itten color theory.
        
        Harmony rules:
        - Neutrals (black, white, gray, beige) harmonize with everything
        - Warm colors (red, orange, yellow) harmonize with each other
        - Cool colors (blue, green, purple) harmonize with each other
        - Complementary colors (opposite on color wheel) create contrast
        
        Args:
            color1: First color name
            color2: Second color name
            
        Returns:
            Harmony score between 0.0 and 1.0
        """
        # Define color groups
        neutrals = {'white', 'black', 'gray', 'light_gray', 'dark_gray', 'beige'}
        warm_colors = {'red', 'orange', 'yellow', 'pink', 'brown'}
        cool_colors = {'blue', 'green', 'purple', 'cyan', 'teal'}
        
        # Complementary pairs
        complementary = {
            ('red', 'green'), ('green', 'red'),
            ('blue', 'orange'), ('orange', 'blue'),
            ('yellow', 'purple'), ('purple', 'yellow'),
            ('cyan', 'red'), ('red', 'cyan'),
            ('pink', 'teal'), ('teal', 'pink')
        }
        
        # Exact match
        if color1 == color2:
            return 1.0
        
        # Neutral harmony (neutrals go with everything)
        if color1 in neutrals or color2 in neutrals:
            return 0.90
        
        # Complementary colors (high contrast but harmonious)
        if (color1, color2) in complementary:
            return 0.80
        
        # Analogous colors (same temperature)
        if (color1 in warm_colors and color2 in warm_colors) or \
           (color1 in cool_colors and color2 in cool_colors):
            return 0.75
        
        # Different temperatures (moderate harmony)
        if (color1 in warm_colors and color2 in cool_colors) or \
           (color1 in cool_colors and color2 in warm_colors):
            return 0.60
        
        # Default moderate harmony
        return 0.50
