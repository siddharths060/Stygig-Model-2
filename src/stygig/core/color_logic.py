"""
Color Processing Logic for Fashion Recommendations

This module handles color extraction and harmony calculations using:
- HSL (Hue, Saturation, Lightness) color space for accurate color theory
- Advanced color harmony rules (Analogous, Complementary, Triadic)
- Neutral color detection via HSL analysis (no hard-coded names)

V4 UPDATE: Refactored to use RGB tuples directly (eliminates hard-coded color map)
"""

import colorsys
import logging
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ColorProcessor:
    """
    Professional color processor for fashion recommendation system.
    
    V4 UPDATE: Uses RGB tuples directly (no hard-coded color name map)
    - Neutral detection via HSL saturation/lightness analysis
    - Scalable to unlimited colors (no dictionary maintenance)
    - More accurate (uses exact RGB values from images)
    """
    
    def __init__(self, n_clusters=3):
        """
        Initialize color processor.
        
        Args:
            n_clusters: Number of dominant colors to extract (default: 3)
        """
        self.n_clusters = n_clusters
        logger.info(f"ColorProcessor initialized with {n_clusters} clusters (V4: RGB-based harmony)")
    
    def extract_dominant_colors(self, image_path: str) -> List[Tuple[Tuple[int, int, int], float]]:
        """
        Extract dominant colors from an image.
        
        V4 UPDATE: Returns RGB tuples instead of color names
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of ((R, G, B), confidence) tuples, sorted by prominence
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            
            # Resize for faster processing
            img = img.resize((150, 150))
            img_array = np.array(img)
            
            # Simple color extraction - get average color
            mean_color = img_array.mean(axis=(0, 1))
            
            # Convert to integer RGB tuple
            rgb_tuple = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
            
            return [(rgb_tuple, 1.0)]
        
        except Exception as e:
            logger.warning(f"Color extraction failed for {image_path}: {e}")
            return [((128, 128, 128), 1.0)]  # Return gray as fallback
    
    def _rgb_to_hsl(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert RGB tuple to HSL (Hue, Saturation, Lightness).
        
        V4 NEW METHOD: Core helper for RGB-based color theory
        
        Args:
            rgb: RGB tuple (0-255 range) e.g., (128, 128, 128)
            
        Returns:
            (hue, saturation, lightness) where all values are 0.0-1.0
        """
        # Normalize RGB from 0-255 to 0-1
        r, g, b = [x / 255.0 for x in rgb]
        
        # colorsys.rgb_to_hls returns (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Return in H, S, L order for clarity
        return h, s, l
    
    def _is_neutral(self, rgb: Tuple[int, int, int], 
                    sat_threshold: float = 0.2, 
                    light_threshold_low: float = 0.1,
                    light_threshold_high: float = 0.9) -> bool:
        """
        Determine if a color is neutral (black, white, or gray) via HSL analysis.
        
        V4 NEW METHOD: Replaces hard-coded NEUTRAL_COLORS set
        
        A color is neutral if:
        - Low saturation (grayscale colors)
        - Very dark (near black)
        - Very light (near white)
        
        Args:
            rgb: RGB tuple (0-255 range)
            sat_threshold: Saturation below this = neutral (default 0.2)
            light_threshold_low: Lightness below this = black (default 0.1)
            light_threshold_high: Lightness above this = white (default 0.9)
            
        Returns:
            True if color is neutral, False otherwise
        """
        h, s, l = self._rgb_to_hsl(rgb)
        
        # Low saturation = grayscale (gray, beige, taupe, etc.)
        if s < sat_threshold:
            return True
        
        # Very dark = black/near-black
        if l < light_threshold_low:
            return True
        
        # Very light = white/near-white
        if l > light_threshold_high:
            return True
        
        return False
    
    def calculate_color_harmony(self, color1_rgb: Tuple[int, int, int], 
                                  color2_rgb: Tuple[int, int, int]) -> float:
        """
        Calculate color harmony score based on HSL color theory.
        
        V4 REFACTORED: Now accepts RGB tuples directly (no color name strings)
        
        Scoring:
        - 1.0: Neutral Harmony (one color is neutral via HSL analysis)
        - 0.9: Analogous (colors are close on the wheel, within 30°)
        - 0.8: Complementary (colors are opposite, ~180° apart)
        - 0.7: Triadic (colors are evenly spaced, ~120° apart)
        - 0.2: No/Poor Harmony (no clear relationship)
        
        Args:
            color1_rgb: First color as RGB tuple (0-255), e.g., (128, 128, 128)
            color2_rgb: Second color as RGB tuple (0-255), e.g., (0, 0, 255)
            
        Returns:
            Harmony score between 0.0 and 1.0
        """
        # --- 1.0 (Best Match): Neutral Harmony ---
        # If either color is neutral (black/white/gray), it's a guaranteed match.
        # This is the foundation of fashion - neutrals go with everything.
        if self._is_neutral(color1_rgb) or self._is_neutral(color2_rgb):
            logger.debug(f"Neutral harmony detected: {color1_rgb} + {color2_rgb} = 1.0")
            return 1.0

        # Get HSL values for both colors
        h1, s1, l1 = self._rgb_to_hsl(color1_rgb)
        h2, s2, l2 = self._rgb_to_hsl(color2_rgb)

        # --- Calculate Hue Difference ---
        # Hue is a circle (0 to 1). The difference is the shortest path.
        hue_difference = abs(h1 - h2)
        if hue_difference > 0.5:
            hue_difference = 1.0 - hue_difference  # e.g., 0.9 and 0.1 are 0.2 apart, not 0.8

        # --- 0.9 (Excellent Match): Analogous Harmony ---
        # Colors are "next to" each other (e.g., within 30 degrees)
        # 30 degrees on a 360 wheel = 30/360 = ~0.083 on a 0-1 scale
        if hue_difference <= 0.084:
            logger.debug(f"Analogous harmony: {color1_rgb} + {color2_rgb} (Δ{hue_difference:.3f}) = 0.9")
            return 0.9

        # --- 0.8 (Bold Match): Complementary Harmony ---
        # Colors are "opposite" (e.g., ~180 degrees apart)
        # 180 degrees = 0.5. We give a window (e.g., 165-195 degrees or 0.45-0.55)
        if 0.45 <= hue_difference <= 0.55:
            logger.debug(f"Complementary harmony: {color1_rgb} + {color2_rgb} (Δ{hue_difference:.3f}) = 0.8")
            return 0.8

        # --- 0.7 (Good Match): Triadic Harmony ---
        # Colors are ~120 degrees apart (0.33)
        # Give a window (e.g., 105-135 degrees or 0.29-0.375)
        if 0.29 <= hue_difference <= 0.375:
            logger.debug(f"Triadic harmony: {color1_rgb} + {color2_rgb} (Δ{hue_difference:.3f}) = 0.7")
            return 0.7

        # --- 0.2 (Poor Match): No clear harmony ---
        # The colors are not neutral, analogous, complementary, or triadic.
        logger.debug(f"No harmony: {color1_rgb} + {color2_rgb} (Δ{hue_difference:.3f}) = 0.2")
        return 0.2
