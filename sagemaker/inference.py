#!/usr/bin/env python3
"""
SageMaker Inference Script for StyGig Fashion Recommendation System

This script implements the SageMaker inference interface with:
1. model_fn: Load FAISS index and pre-trained models from model artifacts
2. input_fn: Parse input data (image file or base64 encoded image)
3. predict_fn: Perform FAISS-based similarity search and rule-based scoring
4. output_fn: Format recommendations as JSON response

The inference pipeline:
1. Loads saved FAISS index, metadata, and model configuration
2. Accepts new fashion images via HTTP requests
3. Extracts image features using OpenCLIP
4. Performs fast FAISS approximate nearest neighbor search
5. Applies color harmony, gender, and category scoring rules
6. Returns ranked fashion recommendations

Usage:
    This script is automatically executed by SageMaker endpoints.
    The model artifacts are loaded from /opt/ml/model/
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional
import base64

import numpy as np
from PIL import Image
import faiss

# Add the stygig module to the path
sys.path.append('/opt/ml/model')
sys.path.append('/opt/ml/model/src')
sys.path.append('/opt/ml/model/src/stygig')

# Import enterprise configuration
try:
    from config.recommendation_config import RecommendationConfig, get_config
except ImportError:
    # Fallback for SageMaker environments where imports might be different
    RecommendationConfig = None
    get_config = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for loaded models and data
MODEL_DATA = {}

class FashionRecommendationInference:
    """Main inference class for fashion recommendations."""
    
    def __init__(self, model_dir: str, config=None):
        self.model_dir = Path(model_dir)
        self.enterprise_config = config  # Enterprise recommendation config
        self.device = None
        self.clip_model = None
        self.clip_preprocess = None
        self.color_processor = None
        self.gender_classifier = None
        self.faiss_index = None
        self.metadata = {}
        self.embeddings_dict = {}
        self.config = {}
        
        # Category compatibility and accessory mapping (from original code)
        self.ACCESSORY_MAP = {
            'footwear_shoes': ['bottomwear_pants', 'bottomwear_shorts', 'upperwear_shirt', 'upperwear_tshirt'],
            'footwear_sneakers': ['bottomwear_pants', 'bottomwear_shorts', 'upperwear_shirt', 'upperwear_tshirt'],
            'footwear_heels': ['bottomwear_skirt', 'one-piece_dress', 'upperwear_shirt'],
            'footwear_flats': ['bottomwear_pants', 'bottomwear_skirt', 'one-piece_dress'],
            'accessories_bag': ['upperwear_shirt', 'upperwear_tshirt', 'one-piece_dress'],
            'accessories_hat': ['upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket']
        }
        
        self.CATEGORY_COMPATIBILITY = {
            'upperwear_shirt': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt'],
            'upperwear_tshirt': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt'],
            'bottomwear_pants': ['upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket'],
            'bottomwear_shorts': ['upperwear_shirt', 'upperwear_tshirt'],
            'bottomwear_skirt': ['upperwear_shirt', 'upperwear_tshirt'],
            'one-piece_dress': ['accessories_bag', 'footwear_heels', 'footwear_flats'],
            'upperwear_jacket': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt']
        }
        
        # Scoring weights
        self.weights = {'color': 0.45, 'category': 0.25, 'gender': 0.3}
        
    def load_model_artifacts(self):
        """Load all model artifacts from the model directory."""
        try:
            logger.info(f"Loading model artifacts from: {self.model_dir}")
            
            # Load configuration
            config_path = self.model_dir / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Default configuration
                self.config = {
                    'clip_model': 'ViT-B-32',
                    'clip_pretrained': 'openai',
                    'embed_dim': 512,
                    'faiss_index_type': 'IndexFlatIP',
                    'n_clusters': 3
                }
            
            logger.info(f"Loaded configuration: {self.config}")
            
            # Initialize models
            self._initialize_models()
            
            # Load metadata
            metadata_path = self.model_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} categories")
            
            # Load embeddings
            embeddings_path = self.model_dir / 'embeddings.npz'
            if embeddings_path.exists():
                emb_data = np.load(embeddings_path, allow_pickle=True)
                self.embeddings_dict = {k: emb_data[k] for k in emb_data.files}
                logger.info(f"Loaded embeddings for {len(self.embeddings_dict)} categories")
            
            # Load FAISS index
            faiss_path = self.model_dir / 'faiss_index.index'
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                logger.warning("FAISS index not found, using flat index as fallback")
                embed_dim = self.config.get('embed_dim', 512)
                self.faiss_index = faiss.IndexFlatIP(embed_dim)
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize CLIP and other models."""
        try:
            import torch
            import open_clip
            from stygig.core.color_logic import ColorProcessor
            from stygig.core.gender_logic import GenderClassifier
            
            logger.info("Initializing models (this may take 30-60 seconds on first load)...")
            start_time = __import__('time').time()
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize CLIP model with caching and timeout handling
            try:
                logger.info(f"Loading CLIP model: {self.config['clip_model']}/{self.config['clip_pretrained']}")
                
                # Set cache directory to model dir to avoid re-downloads
                os.environ['TORCH_HOME'] = str(self.model_dir / '.cache')
                os.environ['HF_HOME'] = str(self.model_dir / '.cache')
                
                model_components = open_clip.create_model_and_transforms(
                    self.config['clip_model'],
                    pretrained=self.config['clip_pretrained'],
                    cache_dir=str(self.model_dir / '.cache')
                )
                
                if len(model_components) >= 2:
                    self.clip_model = model_components[0]
                    self.clip_preprocess = model_components[1]
                else:
                    raise ValueError("Unexpected return from create_model_and_transforms")
                
                self.clip_model = self.clip_model.to(self.device)
                self.clip_model.eval()
                
                logger.info(f"✓ CLIP model loaded in {__import__('time').time() - start_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                # Try to load from local cache if download fails
                logger.warning("Attempting to load from local cache...")
                raise
            
            # Initialize other processors (lightweight, fast)
            n_clusters = self.config.get('n_clusters', 3)
            self.color_processor = ColorProcessor(n_clusters=n_clusters)
            self.gender_classifier = GenderClassifier()
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def extract_image_features(self, image: Image.Image) -> Tuple[np.ndarray, str, str, float]:
        """
        Extract features from an input image with robust error handling.
        
        Returns:
            embedding: CLIP embedding vector
            dominant_color: Dominant color name
            predicted_gender: Predicted gender
            gender_confidence: Gender prediction confidence
        """
        import torch
        import tempfile
        
        # Validate input image
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError("Invalid image dimensions")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        dominant_color = 'unknown'
        gender = 'unknown'
        gender_conf = 0.0
        
        # Extract color features with error handling
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path, quality=95)
            
            try:
                colors = self.color_processor.extract_dominant_colors(temp_path)
                dominant_color = colors[0][0] if colors and len(colors) > 0 else 'unknown'
            except Exception as e:
                logger.warning(f"Color extraction failed: {e}")
                dominant_color = 'unknown'
            
            # Extract gender features with error handling
            try:
                gender, gender_conf = self.gender_classifier.predict_gender(temp_path)
            except Exception as e:
                logger.warning(f"Gender prediction failed: {e}")
                gender, gender_conf = 'unknown', 0.0
                
        except Exception as e:
            logger.error(f"Failed to save temporary image: {e}")
            # Continue with CLIP embedding extraction
        finally:
            # Clean up temp file
            try:
                if 'temp_path' in locals():
                    os.remove(temp_path)
            except:
                pass
        
        # Extract CLIP embedding
        try:
            img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Validate tensor
            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                raise ValueError("Invalid tensor values (NaN or Inf)")
            
            with torch.no_grad():
                embedding = self.clip_model.encode_image(img_tensor)
                
                # Validate embedding
                if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                    raise ValueError("Invalid embedding values (NaN or Inf)")
                
                embedding = embedding.cpu().numpy()[0]
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    logger.warning("Zero-norm embedding, using random normalized vector")
                    embedding = np.random.randn(len(embedding))
                    embedding = embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.error(f"CLIP embedding extraction failed: {e}")
            # Return a random normalized vector as fallback
            embed_dim = self.config.get('embed_dim', 512)
            embedding = np.random.randn(embed_dim)
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32), dominant_color, gender, float(gender_conf)
    
    def faiss_similarity_search(self, query_embedding: np.ndarray, k: int = 50) -> List[Tuple[int, float]]:
        """
        Perform FAISS-based similarity search.
        
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.faiss_index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search for top-k similar items
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] != -1:  # Valid index
                results.append((int(indices[0][i]), float(similarities[0][i])))
        
        return results
    
    def map_faiss_indices_to_items(self, faiss_results: List[Tuple[int, float]]) -> List[Tuple[Dict, float]]:
        """
        Map FAISS index results back to item metadata.
        """
        items_with_scores = []
        
        # Create a flat list of all items with their global index
        global_index = 0
        index_to_item = {}
        
        for category, items in self.metadata.items():
            for item in items:
                index_to_item[global_index] = item
                global_index += 1
        
        # Map FAISS results to items
        for faiss_idx, similarity in faiss_results:
            if faiss_idx in index_to_item:
                item = index_to_item[faiss_idx]
                items_with_scores.append((item, similarity))
        
        return items_with_scores
    
    def apply_enterprise_rule_based_scoring(self, query_color: str, query_gender: str, 
                                          candidates: List[Tuple[Dict, float]], 
                                          items_per_category: int = 2) -> List[Dict]:
        """
        Apply enterprise per-category rule-based scoring with configurable limits.
        
        Args:
            query_color: Dominant color of query image
            query_gender: Predicted gender of query image
            candidates: List of (item_dict, similarity_score) tuples
            items_per_category: Number of items to return per category
        
        Returns:
            List of recommendation dictionaries with per-category limits applied
        """
        compatible_genders = self._get_compatible_genders(query_gender)
        
        # Group candidates by category
        category_candidates = {}
        
        for item, similarity_score in candidates:
            # Skip items with incompatible genders
            if item['gender'] not in compatible_genders:
                continue
            
            category = item['category']
            if category not in category_candidates:
                category_candidates[category] = []
            
            # Calculate color harmony score
            color_score = self.color_processor.calculate_color_harmony(
                query_color, item.get('color', 'unknown')
            )
            
            # Calculate gender compatibility score
            gender_score = 1.0 if item['gender'] == query_gender else 0.75
            
            # Calculate final composite score
            final_score = (0.4 * similarity_score + 0.4 * color_score + 0.2 * gender_score)
            
            category_candidates[category].append({
                'id': item.get('id', 'unknown'),
                'category': category,
                'gender': item.get('gender', 'unisex'),
                'color': item.get('color', 'unknown'),
                'path': item.get('path', ''),
                'score': round(final_score, 4),
                'similarity_score': round(similarity_score, 4),
                'color_harmony_score': round(color_score, 4),
                'gender_compatibility_score': round(gender_score, 4),
                'match_reason': self._generate_match_reason(color_score, gender_score, query_color, item.get('color'))
            })
        
        # Select top items per category
        final_recommendations = []
        rank = 1
        
        # Sort categories by name for consistent ordering
        for category in sorted(category_candidates.keys()):
            items = category_candidates[category]
            
            # Sort by composite score and take top N
            items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
            top_items = items_sorted[:items_per_category]
            
            for item in top_items:
                item['rank'] = rank
                final_recommendations.append(item)
                rank += 1
        
        # Sort final results by score
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Enterprise scoring: {len(final_recommendations)} recommendations from {len(category_candidates)} categories")
        
        return final_recommendations
    
    def apply_rule_based_scoring(self, query_color: str, query_gender: str, 
                               candidates: List[Tuple[Dict, float]], 
                               n_recommendations: int = 5) -> List[Dict]:
        """
        Apply color harmony, gender compatibility, and category rules.
        """
        scored_items = []
        compatible_genders = self._get_compatible_genders(query_gender)
        
        for item, similarity_score in candidates:
            # Skip items with incompatible genders
            if item['gender'] not in compatible_genders:
                continue
            
            # Calculate color harmony score
            color_score = self.color_processor.calculate_color_harmony(
                query_color, item.get('color', 'unknown')
            )
            
            # Calculate gender compatibility score
            gender_score = 1.0 if item['gender'] == query_gender else 0.75
            
            # Category compatibility score (simplified to 1.0 for now)
            category_score = 1.0
            
            # Calculate final weighted score
            final_score = (
                self.weights['color'] * color_score +
                self.weights['category'] * category_score +
                self.weights['gender'] * gender_score
            )
            
            scored_items.append({
                'id': item['id'],
                'path': item['path'],
                'category': item['category'],
                'gender': item['gender'],
                'color': item.get('color', 'unknown'),
                'score': round(float(final_score), 4),
                'similarity_score': round(float(similarity_score), 4),
                'score_components': {
                    'color_harmony': round(float(color_score), 4),
                    'category_compatibility': round(float(category_score), 4),
                    'gender_compatibility': round(float(gender_score), 4),
                },
                'match_reason': self._generate_match_reason(color_score, category_score, gender_score, query_color, item.get('color'))
            })
        
        # Sort by final score and return top N
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_items = []
        for item in scored_items:
            if item['id'] not in seen_ids:
                unique_items.append(item)
                seen_ids.add(item['id'])
        
        return unique_items[:n_recommendations]
    
    def _get_compatible_genders(self, gender: str) -> List[str]:
        """Get list of compatible genders."""
        if gender == 'male':
            return ['male', 'unisex']
        elif gender == 'female':
            return ['female', 'unisex']
        else:
            return ['male', 'female', 'unisex']
    
    def _generate_match_reason(self, color_score: float, category_score: float, 
                             gender_score: float, color1: str, color2: str) -> str:
        """Generate human-readable match reason."""
        reasons = []
        
        if color_score >= 0.8:
            if color1 == color2:
                reasons.append(f"matching {color1}")
            else:
                reasons.append(f"harmonious colors ({color1}+{color2})")
        
        if gender_score >= 0.8:
            reasons.append('gender appropriate')
        
        if category_score >= 0.8:
            reasons.append('category compatible')
        
        if not reasons:
            reasons.append('similar style')
        
        return ', '.join(reasons)
    
    def predict(self, image: Image.Image, n_recommendations: int = 5) -> Dict[str, Any]:
        """
        Main enterprise prediction function with per-category configuration.
        
        Args:
            image: PIL Image object
            n_recommendations: Number of recommendations to return (overridden by enterprise config)
        
        Returns:
            Dictionary with query item info and recommendations
        """
        try:
            # Use enterprise config if available
            if self.enterprise_config:
                n_recommendations = self.enterprise_config.max_total_recommendations
                items_per_category = self.enterprise_config.items_per_category
                search_k = self.enterprise_config.get_search_k(len(self.metadata))
                logger.info(f"🏢 Enterprise inference: {items_per_category} items per category, {n_recommendations} max total")
            else:
                items_per_category = 2  # Default fallback
                search_k = min(200, self.faiss_index.ntotal)
                logger.info("📊 Standard inference: 2 items per category")
            
            # Extract features from query image
            embedding, color, gender, gender_conf = self.extract_image_features(image)
            
            # Perform FAISS similarity search with enterprise search K
            faiss_results = self.faiss_similarity_search(embedding, k=search_k)
            
            # Map FAISS results to item metadata
            candidates = self.map_faiss_indices_to_items(faiss_results)
            
            # Apply enterprise per-category rule-based scoring
            if self.enterprise_config:
                recommendations = self.apply_enterprise_rule_based_scoring(
                    color, gender, candidates, items_per_category
                )
            else:
                # Fallback to original method
                recommendations = self.apply_rule_based_scoring(
                    color, gender, candidates, n_recommendations
                )
            
            return {
                'query_item': {
                    'dominant_color': color,
                    'predicted_gender': gender,
                    'gender_confidence': round(float(gender_conf), 4)
                },
                'recommendations': recommendations,
                'total_candidates': len(candidates),
                'faiss_search_results': len(faiss_results),
                'enterprise_config': {
                    'items_per_category': items_per_category,
                    'max_total_recommendations': n_recommendations
                } if self.enterprise_config else None
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'query_item': None,
                'recommendations': []
            }


# SageMaker inference functions

def model_fn(model_dir: str):
    """
    Load model artifacts and initialize inference engine with enterprise configuration.
    
    This function is called once when the model is loaded by SageMaker.
    """
    try:
        logger.info("Loading enterprise model for inference...")
        
        # Load enterprise configuration
        enterprise_config = None
        if RecommendationConfig and get_config:
            try:
                # Try to load config from model artifacts
                config_path = Path(model_dir) / 'enterprise_config.json'
                if config_path.exists():
                    enterprise_config = RecommendationConfig.from_json_file(str(config_path))
                    logger.info(f"✅ Loaded enterprise config: {enterprise_config.items_per_category} items per category")
                else:
                    enterprise_config = get_config('default')
                    logger.info("📊 Using default enterprise config")
            except Exception as e:
                logger.warning(f"Failed to load enterprise config: {e}, using default")
                enterprise_config = get_config('default') if get_config else None
        
        # Initialize inference engine
        inference_engine = FashionRecommendationInference(model_dir, config=enterprise_config)
        inference_engine.load_model_artifacts()
        
        # Store in global variable
        global MODEL_DATA
        MODEL_DATA['inference_engine'] = inference_engine
        MODEL_DATA['enterprise_config'] = enterprise_config
        
        logger.info("✅ Enterprise model loaded successfully for inference")
        return inference_engine
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def input_fn(request_body: str, request_content_type: str):
    """
    Parse input data from HTTP request.
    
    Supports:
    - JSON with base64 encoded image
    - Direct image file upload
    """
    try:
        if request_content_type == 'application/json':
            # Parse JSON request
            data = json.loads(request_body)
            
            if 'image' in data:
                # Base64 encoded image
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data)).convert('RGB')
            else:
                raise ValueError("No 'image' field found in JSON request")
            
            # Extract other parameters
            n_recommendations = data.get('n_recommendations', 5)
            
        elif request_content_type.startswith('image/'):
            # Direct image upload
            image = Image.open(BytesIO(request_body)).convert('RGB')
            n_recommendations = 5  # Default
            
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        return {
            'image': image,
            'n_recommendations': n_recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to parse input: {e}")
        raise ValueError(f"Invalid input format: {e}")

def predict_fn(input_data: Dict, model):
    """
    Perform prediction using the loaded model.
    """
    try:
        image = input_data['image']
        n_recommendations = input_data.get('n_recommendations', 5)
        
        # Get inference engine
        if isinstance(model, FashionRecommendationInference):
            inference_engine = model
        else:
            # Fallback to global variable
            global MODEL_DATA
            inference_engine = MODEL_DATA.get('inference_engine')
            if inference_engine is None:
                raise RuntimeError("Inference engine not loaded")
        
        # Perform prediction
        result = inference_engine.predict(image, n_recommendations)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            'error': str(e),
            'query_item': None,
            'recommendations': []
        }

def output_fn(prediction: Dict, accept: str):
    """
    Format prediction output for HTTP response.
    """
    try:
        if accept == 'application/json':
            return json.dumps(prediction, indent=2)
        else:
            # Default to JSON
            return json.dumps(prediction)
            
    except Exception as e:
        logger.error(f"Failed to format output: {e}")
        error_response = {
            'error': f"Output formatting failed: {e}",
            'query_item': None,
            'recommendations': []
        }
        return json.dumps(error_response)

# For local testing
if __name__ == '__main__':
    # Test the inference pipeline locally
    model_dir = '/opt/ml/model'
    
    if os.path.exists(model_dir):
        # Load model
        model = model_fn(model_dir)
        
        # Test with a sample image
        test_image_path = '/tmp/test_image.jpg'
        if os.path.exists(test_image_path):
            with open(test_image_path, 'rb') as f:
                image_data = f.read()
            
            # Test input parsing
            json_input = json.dumps({
                'image': base64.b64encode(image_data).decode('utf-8'),
                'n_recommendations': 3
            })
            
            input_data = input_fn(json_input, 'application/json')
            
            # Test prediction
            prediction = predict_fn(input_data, model)
            
            # Test output formatting
            output = output_fn(prediction, 'application/json')
            
            print("Test successful!")
            print(output)
        else:
            print("No test image found")
    else:
        print("Model directory not found")