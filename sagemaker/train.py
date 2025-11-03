#!/usr/bin/env python3
"""
SageMaker Training Script for StyGig Fashion Recommendation System

This script serves as the SageMaker training entry point that:
1. Downloads S3 fashion dataset to local storage
2. Extracts features using OpenCLIP and other models
3. Builds FAISS index for efficient similarity search
4. Includes placeholder structure for Supervised Contrastive Learning (SCL)
5. Outputs trained model artifacts to /opt/ml/model

Usage in SageMaker:
    This script is executed by SageMaker Estimator with:
    - Input data from S3 mounted at /opt/ml/input/data/training
    - Hyperparameters passed via /opt/ml/input/config/hyperparameters.json
    - Output artifacts saved to /opt/ml/model/
"""

import os
import sys
import json
import argparse
import logging
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import faiss

# Add the stygig module to the path
# SageMaker uploads source_dir to /opt/ml/code/, so src/stygig will be at /opt/ml/code/src/stygig
sys.path.insert(0, '/opt/ml/code')
sys.path.insert(0, '/opt/ml/code/src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_numpy_compatibility():
    """Check and fix NumPy compatibility issues."""
    import subprocess
    import sys
    
    try:
        import numpy as np
        numpy_version = np.__version__
        logger.info(f"Current NumPy version: {numpy_version}")
        
        # Check if NumPy version is 2.x
        if numpy_version.startswith('2.'):
            logger.warning("Detected NumPy 2.x - using compatibility mode for local testing")
            # For local testing, we'll work with NumPy 2.x
            # In production SageMaker, NumPy 1.x will be used
            
    except Exception as e:
        logger.error(f"Failed to check/fix NumPy compatibility: {e}")
        
def verify_dependencies():
    """Verify all required dependencies are available and compatible."""
    required_packages = {
        'numpy': '1.26.4',
        'scipy': None,  # Use existing version
        'sklearn': None,  # Use existing version
        'faiss': None,  # Will be installed from requirements
        'open_clip_torch': None,  # Will be installed from requirements
        'colormath': None,  # Will be installed from requirements
    }
    
    missing_packages = []
    
    for package, expected_version in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                logger.info(f"✓ scikit-learn {sklearn.__version__} available")
            elif package == 'faiss':
                import faiss
                logger.info(f"✓ faiss available")
            elif package == 'open_clip_torch':
                import open_clip
                logger.info(f"✓ open_clip_torch available")
            elif package == 'colormath':
                import colormath
                logger.info(f"✓ colormath available")
            elif package == 'numpy':
                import numpy as np
                version = np.__version__
                if version.startswith('2.'):
                    logger.warning(f"NumPy {version} detected - using compatibility mode")
                else:
                    logger.info(f"✓ NumPy {version} compatible")
            elif package == 'scipy':
                import scipy
                logger.info(f"✓ scipy {scipy.__version__} available")
                
        except ImportError as e:
            missing_packages.append(package)
            logger.warning(f"✗ {package} not available: {e}")
    
    if missing_packages:
        logger.warning(f"Missing packages will be installed by SageMaker: {missing_packages}")
    
    return True

def parse_args():
    """Parse command line arguments and hyperparameters."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # Model hyperparameters
    parser.add_argument('--clip-model', type=str, default='ViT-B-32')
    parser.add_argument('--clip-pretrained', type=str, default='openai')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-items-per-category', type=int, default=None)
    parser.add_argument('--faiss-index-type', type=str, default='IndexFlatIP', choices=['IndexFlatIP', 'IndexIVFFlat', 'IndexPQ'])
    parser.add_argument('--n-clusters', type=int, default=3, help='Number of color clusters for K-means')
    
    # Supervised Contrastive Learning (SCL) placeholders
    parser.add_argument('--enable-scl', type=bool, default=False, help='Enable SCL training (placeholder)')
    parser.add_argument('--scl-temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--scl-epochs', type=int, default=10, help='Number of SCL training epochs')
    parser.add_argument('--scl-lr', type=float, default=1e-4, help='Learning rate for SCL')
    
    args = parser.parse_args()
    
    # Load hyperparameters from SageMaker if available
    hyperparameters_file = '/opt/ml/input/config/hyperparameters.json'
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            hyperparams = json.load(f)
            for key, value in hyperparams.items():
                key = key.replace('-', '_')
                if hasattr(args, key):
                    # Convert string values to appropriate types
                    current_value = getattr(args, key)
                    if isinstance(current_value, bool):
                        setattr(args, key, value.lower() == 'true')
                    elif isinstance(current_value, int):
                        setattr(args, key, int(value))
                    elif isinstance(current_value, float):
                        setattr(args, key, float(value))
                    else:
                        # Strip quotes from string values if present
                        cleaned_value = value.strip('"\'') if isinstance(value, str) else value
                        setattr(args, key, cleaned_value)
    
    return args

class FashionDatasetProcessor:
    """Processes fashion dataset for feature extraction and indexing."""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.clip_model = None
        self.clip_preprocess = None
        self.color_processor = None
        self.gender_classifier = None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize CLIP model and other processors."""
        try:
            import torch
            import open_clip
            from stygig.core.color_logic import ColorProcessor
            from stygig.core.gender_logic import GenderClassifier
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize CLIP model
            model_components = open_clip.create_model_and_transforms(
                self.args.clip_model, 
                pretrained=self.args.clip_pretrained
            )
            
            if len(model_components) >= 2:
                self.clip_model = model_components[0]
                self.clip_preprocess = model_components[1]
            else:
                raise ValueError("Unexpected return from create_model_and_transforms")
            
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Get embedding dimension
            self.embed_dim = getattr(self.clip_model.visual, 'output_dim', 512)
            logger.info(f"CLIP embedding dimension: {self.embed_dim}")
            
            # Initialize other processors
            self.color_processor = ColorProcessor(n_clusters=self.args.n_clusters)
            self.gender_classifier = GenderClassifier()
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def process_dataset(self, dataset_path: str) -> Tuple[Dict, Dict, faiss.Index]:
        """
        Process the entire dataset and build FAISS index.
        
        Returns:
            metadata: Dictionary with category metadata
            embeddings_dict: Dictionary with category embeddings
            faiss_index: Built FAISS index
        """
        dataset_path = Path(dataset_path)
        
        # Debug: Log the directory structure to understand what's available
        logger.info(f"Dataset root path: {dataset_path}")
        if dataset_path.exists():
            logger.info(f"Contents of {dataset_path}: {list(dataset_path.iterdir())}")
        
        # Check if there's a train subdirectory, otherwise use the dataset_path directly
        train_path = dataset_path / 'train' if (dataset_path / 'train').exists() else dataset_path
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training path not found: {train_path}")
        
        logger.info(f"Processing dataset from: {train_path}")
        logger.info(f"Contents of {train_path}: {list(train_path.iterdir())[:10]}...")  # Show first 10 items
        
        # Collect all categories and their images (handle nested structure)
        category_data = {}
        total_found_images = 0
        
        # Validate dataset path exists and is accessible
        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset path not found: {train_path}")
        
        # Get all directories in training path
        category_directories = [d for d in train_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(category_directories)} category directories: {[d.name for d in category_directories]}")
        
        if not category_directories:
            raise ValueError(f"No category directories found in {train_path}")
        
        for cat_path in category_directories:
            cat_name = cat_path.name
            logger.info(f"Processing category directory: {cat_name}")
            
            # Look for images directly in category directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            direct_images = []
            for ext in image_extensions:
                direct_images.extend(list(cat_path.glob(ext)))
            
            if direct_images:
                # Images found directly in category directory
                logger.info(f"Found {len(direct_images)} images directly in {cat_name}")
                
                if self.args.max_items_per_category:
                    direct_images = direct_images[:self.args.max_items_per_category]
                
                category_data[cat_name] = direct_images
                total_found_images += len(direct_images)
                logger.info(f"Added {len(direct_images)} images from category: {cat_name}")
            else:
                # No direct images, check subdirectories (nested structure)
                logger.info(f"No direct images in {cat_name}, scanning subdirectories...")
                subdirs = [d for d in cat_path.iterdir() if d.is_dir()]
                logger.info(f"Found {len(subdirs)} subdirectories in {cat_name}: {[d.name for d in subdirs]}")
                
                for subcat_path in subdirs:
                    subcat_name = f"{cat_name}_{subcat_path.name}"
                    logger.info(f"Processing subdirectory: {subcat_name}")
                    
                    # Search for images in subdirectory
                    sub_images = []
                    for ext in image_extensions:
                        sub_images.extend(list(subcat_path.glob(ext)))
                    
                    # Also check one level deeper (e.g., accessories/bag/images/)
                    if not sub_images:
                        for sub_subdir in subcat_path.iterdir():
                            if sub_subdir.is_dir():
                                for ext in image_extensions:
                                    sub_images.extend(list(sub_subdir.glob(ext)))
                    
                    if sub_images:
                        if self.args.max_items_per_category:
                            sub_images = sub_images[:self.args.max_items_per_category]
                        
                        category_data[subcat_name] = sub_images
                        total_found_images += len(sub_images)
                        logger.info(f"Added {len(sub_images)} images from subcategory: {subcat_name}")
                    else:
                        logger.warning(f"No images found in subdirectory: {subcat_name}")
        
        # Validate we found images
        if not category_data:
            raise ValueError(f"No images found in any category directories. Check dataset structure at: {train_path}")
        
        if total_found_images == 0:
            raise ValueError(f"Found {len(category_data)} categories but 0 images total. Check image file formats and accessibility.")
        
        logger.info(f"Dataset validation complete: {len(category_data)} categories, {total_found_images} total images")
        
        # Process each category
        metadata = {}
        embeddings_dict = {}
        all_embeddings = []
        all_item_indices = []  # Track which item each embedding belongs to
        
        item_counter = 0
        
        for cat_name, image_files in category_data.items():
            logger.info(f"Processing category: {cat_name}")
            
            cat_metadata = []
            cat_embeddings = []
            
            # Process images in batches
            for i in range(0, len(image_files), self.args.batch_size):
                batch_files = image_files[i:i + self.args.batch_size]
                batch_metadata, batch_embeddings = self._process_batch(batch_files, cat_name)
                
                cat_metadata.extend(batch_metadata)
                if len(batch_embeddings) > 0:
                    cat_embeddings.extend(batch_embeddings)
                    
                    # Add to global embeddings for FAISS index
                    for emb in batch_embeddings:
                        all_embeddings.append(emb)
                        all_item_indices.append(item_counter)
                        item_counter += 1
            
            metadata[cat_name] = cat_metadata
            if cat_embeddings:
                embeddings_dict[cat_name] = np.stack(cat_embeddings).astype(np.float32)
            else:
                embeddings_dict[cat_name] = np.zeros((0, self.embed_dim), dtype=np.float32)
        
        # Build FAISS index
        logger.info(f"Building FAISS index with {len(all_embeddings)} embeddings")
        faiss_index = self._build_faiss_index(all_embeddings)
        
        return metadata, embeddings_dict, faiss_index
    
    def _validate_image_file(self, img_path: Path) -> bool:
        """Validate if image file is readable and has correct format."""
        try:
            # Check file size (avoid empty files)
            if img_path.stat().st_size < 1024:  # Less than 1KB
                logger.warning(f"Image too small (< 1KB): {img_path}")
                return False
            
            # Check file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            if img_path.suffix not in valid_extensions:
                logger.warning(f"Invalid extension {img_path.suffix}: {img_path}")
                return False
            
            # Try to open and validate image
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
                
            # Reopen to check dimensions (verify() closes the image)
            with Image.open(img_path) as img:
                width, height = img.size
                if width < 32 or height < 32:  # Too small
                    logger.warning(f"Image dimensions too small ({width}x{height}): {img_path}")
                    return False
                    
                if width > 4096 or height > 4096:  # Suspiciously large
                    logger.warning(f"Image dimensions very large ({width}x{height}): {img_path}")
                    # Don't return False, just warn - large images are okay
            
            return True
            
        except Exception as e:
            logger.warning(f"Image validation failed for {img_path}: {e}")
            return False
    
    def _process_batch(self, image_files: List[Path], category: str) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of images and extract features with robust error handling."""
        import torch
        
        batch_metadata = []
        batch_images = []
        valid_indices = []
        failed_count = 0
        
        # Pre-validate images
        valid_image_files = []
        for img_path in image_files:
            if self._validate_image_file(img_path):
                valid_image_files.append(img_path)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.info(f"Pre-validation: {failed_count} invalid images filtered out from batch")
        
        if not valid_image_files:
            logger.warning(f"No valid images in batch for category: {category}")
            return [], []
        
        # Process valid images
        for idx, img_path in enumerate(valid_image_files):
            try:
                # Extract color features with error handling
                try:
                    colors = self.color_processor.extract_dominant_colors(str(img_path))
                    dominant_color = colors[0][0] if colors and len(colors) > 0 else 'unknown'
                except Exception as color_error:
                    logger.debug(f"Color extraction failed for {img_path}: {color_error}")
                    dominant_color = 'unknown'
                
                # Extract gender features with error handling
                try:
                    gender, gender_conf = self.gender_classifier.predict_gender(str(img_path), category)
                except Exception as gender_error:
                    logger.debug(f"Gender prediction failed for {img_path}: {gender_error}")
                    gender, gender_conf = 'unknown', 0.0
                
                metadata = {
                    'id': img_path.stem,
                    'path': str(img_path),
                    'category': category,
                    'gender': gender,
                    'color': dominant_color,
                    'gender_confidence': float(gender_conf),
                    'file_size': img_path.stat().st_size
                }
                
                # Preprocess image for CLIP with robust error handling
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # Validate image after opening
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(f"Zero dimension image: {img.size}")
                    
                    # Apply CLIP preprocessing
                    img_tensor = self.clip_preprocess(img)
                    
                    # Validate tensor
                    if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                        raise ValueError("NaN or Inf values in preprocessed tensor")
                    
                    batch_metadata.append(metadata)
                    batch_images.append(img_tensor)
                    valid_indices.append(idx)
                    
                except Exception as preprocessing_error:
                    logger.warning(f"Image preprocessing failed for {img_path}: {preprocessing_error}")
                    failed_count += 1
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
                failed_count += 1
                continue
        
        # Extract CLIP embeddings with robust error handling
        batch_embeddings = []
        if batch_images:
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Validate batch tensor
                if torch.isnan(batch_tensor).any() or torch.isinf(batch_tensor).any():
                    raise ValueError("NaN or Inf values in batch tensor")
                
                with torch.no_grad():
                    embeddings = self.clip_model.encode_image(batch_tensor)
                    
                    # Validate embeddings
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        raise ValueError("NaN or Inf values in embeddings")
                    
                    embeddings = embeddings.cpu().numpy()
                    
                    # Normalize embeddings with safety checks
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # Handle zero norms
                    zero_norm_mask = (norms == 0).flatten()
                    if zero_norm_mask.any():
                        logger.warning(f"Found {zero_norm_mask.sum()} zero-norm embeddings")
                        norms[zero_norm_mask] = 1.0
                    
                    embeddings = embeddings / norms
                    
                    # Final validation
                    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                        raise ValueError("NaN or Inf values in normalized embeddings")
                    
                    batch_embeddings = [embeddings[i] for i in range(len(embeddings))]
                    
            except Exception as e:
                logger.error(f"CLIP embedding extraction failed for batch in {category}: {e}")
                # Return metadata without embeddings rather than failing completely
                logger.warning(f"Continuing without embeddings for {len(batch_images)} images")
                batch_embeddings = []
        
        # Log batch processing summary
        success_count = len(batch_embeddings)
        logger.info(f"Batch processed - Category: {category}, Success: {success_count}, "
                   f"Failed: {failed_count}, Total attempted: {len(image_files)}")
        
        return batch_metadata, batch_embeddings
    
    def _build_faiss_index(self, embeddings: List[np.ndarray]) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        if not embeddings:
            logger.warning("No embeddings to index")
            return faiss.IndexFlatIP(self.embed_dim)
        
        embeddings_matrix = np.stack(embeddings).astype(np.float32)
        n_embeddings, dim = embeddings_matrix.shape
        
        logger.info(f"Building FAISS index: {self.args.faiss_index_type}")
        
        # Choose index type based on configuration
        if self.args.faiss_index_type == 'IndexFlatIP':
            # Flat index with inner product (cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(dim)
        elif self.args.faiss_index_type == 'IndexIVFFlat':
            # Inverted file index for faster search on large datasets
            nlist = min(100, n_embeddings // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_matrix)
        elif self.args.faiss_index_type == 'IndexPQ':
            # Product Quantization for memory efficiency
            m = 64  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            index = faiss.IndexPQ(dim, m, nbits)
            index.train(embeddings_matrix)
        else:
            raise ValueError(f"Unknown FAISS index type: {self.args.faiss_index_type}")
        
        # Add embeddings to index
        index.add(embeddings_matrix)
        
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
        return index

class SupervisedContrastiveLearning:
    """
    Placeholder implementation for Supervised Contrastive Learning.
    
    This class provides the structure for implementing SCL training to improve
    the fashion recommendation system by learning better representations.
    In the full implementation, this would:
    1. Create positive/negative pairs based on fashion compatibility
    2. Implement contrastive loss function
    3. Fine-tune the CLIP model on fashion-specific data
    """
    
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.temperature = args.scl_temperature
        
        logger.info("SCL Placeholder initialized - Full SCL training not implemented in MVP")
    
    def create_contrastive_pairs(self, metadata: Dict) -> List[Tuple]:
        """
        Placeholder: Create positive and negative pairs for contrastive learning.
        
        In full implementation, this would:
        - Create positive pairs from compatible fashion items
        - Create negative pairs from incompatible items
        - Use fashion rules (color harmony, category compatibility, gender)
        """
        logger.info("SCL: Creating contrastive pairs (placeholder)")
        
        # Placeholder implementation
        pairs = []
        # TODO: Implement actual pair creation logic
        # for category, items in metadata.items():
        #     # Create positive pairs within compatible categories
        #     # Create negative pairs across incompatible categories
        #     pass
        
        return pairs
    
    def contrastive_loss(self, embeddings, labels):
        """
        Placeholder: Implement supervised contrastive loss.
        
        Loss = -log(sum(exp(sim(i,p)/τ)) / sum(exp(sim(i,k)/τ)))
        where p are positive samples, k are all samples except i
        """
        import torch
        import torch.nn.functional as F
        
        # Placeholder implementation
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # TODO: Implement full contrastive loss with positive/negative sampling
        # This is a simplified placeholder
        loss = torch.tensor(0.0, device=self.device)
        
        return loss
    
    def train_epoch(self, dataloader):
        """
        Placeholder: Training loop for one epoch of SCL.
        """
        logger.info("SCL: Training epoch (placeholder)")
        
        # TODO: Implement actual training loop
        # - Forward pass through model
        # - Compute contrastive loss
        # - Backward pass and optimization
        
        return 0.0  # Placeholder loss

def save_model_artifacts(args, metadata: Dict, embeddings_dict: Dict, 
                        faiss_index: faiss.Index, scl_trainer: Optional[SupervisedContrastiveLearning] = None):
    """Save all model artifacts to the SageMaker model directory."""
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model artifacts to: {model_dir}")
    
    # Save metadata
    with open(model_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save embeddings dictionary
    np.savez_compressed(model_dir / 'embeddings.npz', **embeddings_dict)
    
    # Save FAISS index
    faiss.write_index(faiss_index, str(model_dir / 'faiss_index.index'))
    
    # Save model configuration
    config = {
        'clip_model': args.clip_model,
        'clip_pretrained': args.clip_pretrained,
        'embed_dim': getattr(faiss_index, 'd', 512),
        'faiss_index_type': args.faiss_index_type,
        'n_clusters': args.n_clusters,
        'scl_enabled': args.enable_scl
    }
    
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy source code for inference
    source_dir = Path('/opt/ml/code/stygig')
    target_dir = model_dir / 'stygig'
    
    if source_dir.exists():
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    
    logger.info("Model artifacts saved successfully")

def validate_training_environment():
    """Validate the training environment and dependencies."""
    validation_errors = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        validation_errors.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 2.0:
            validation_errors.append(f"Insufficient memory: {available_gb:.1f}GB available, 2GB+ recommended")
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
    
    # Check disk space in output directory
    try:
        output_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        stat = os.statvfs(output_dir)
        free_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
        if free_gb < 1.0:
            validation_errors.append(f"Insufficient disk space in {output_dir}: {free_gb:.1f}GB available")
    except Exception:
        logger.warning("Cannot check disk space")
    
    return validation_errors

def main():
    """Main training function with comprehensive error handling."""
    start_time = time.time()
    
    try:
        # Environment validation
        logger.info("Starting StyGig fashion recommendation training pipeline...")
        logger.info("="*60)
        
        env_errors = validate_training_environment()
        if env_errors:
            logger.error("Environment validation failed:")
            for error in env_errors:
                logger.error(f"  - {error}")
            if any("insufficient" in error.lower() for error in env_errors):
                logger.error("Critical environment issues detected, aborting training")
                sys.exit(1)
            else:
                logger.warning("Environment issues detected, but continuing training")
        
        # Dependency checks
        logger.info("Checking dependencies...")
        check_numpy_compatibility()
        
        if not verify_dependencies():
            logger.error("Dependency verification failed")
            sys.exit(1)
        
        # Parse arguments
        args = parse_args()
        logger.info("Training configuration:")
        logger.info(f"  - Dataset path: {args.train}")
        logger.info(f"  - Output path: {args.model_dir}")
        logger.info(f"  - CLIP model: {args.clip_model}/{args.clip_pretrained}")
        logger.info(f"  - Batch size: {args.batch_size}")
        logger.info(f"  - Max items per category: {args.max_items_per_category}")
        logger.info(f"  - FAISS index type: {args.faiss_index_type}")
        logger.info(f"  - SCL enabled: {args.enable_scl}")
        
        # Initialize dataset processor
        logger.info("Initializing models and processors...")
        try:
            processor = FashionDatasetProcessor(args)
            logger.info("✓ Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            logger.error("This could be due to:")
            logger.error("  - Missing dependencies (check requirements.txt)")
            logger.error("  - Insufficient memory for model loading")
            logger.error("  - Network issues downloading pretrained models")
            raise
        
        # Process dataset and build index
        logger.info("Processing dataset and building search index...")
        try:
            metadata, embeddings_dict, faiss_index = processor.process_dataset(args.train)
            
            # Validate processing results
            total_categories = len(metadata)
            total_items = sum(len(items) for items in metadata.values())
            total_embeddings = faiss_index.ntotal
            
            logger.info(f"✓ Dataset processing completed:")
            logger.info(f"  - Categories processed: {total_categories}")
            logger.info(f"  - Fashion items processed: {total_items}")
            logger.info(f"  - Embeddings generated: {total_embeddings}")
            
            # Validation checks
            if total_categories == 0:
                raise ValueError("No categories were processed. Check dataset structure and image formats.")
            
            if total_items == 0:
                raise ValueError("No fashion items were processed. Check image files and accessibility.")
            
            if total_embeddings == 0:
                raise ValueError("No embeddings were generated. Check CLIP model initialization and image preprocessing.")
            
            if total_embeddings != total_items:
                logger.warning(f"Embedding count ({total_embeddings}) != item count ({total_items})")
                logger.warning("Some items may have failed feature extraction")
            
            # Check category distribution
            items_per_category = [len(items) for items in metadata.values()]
            min_items = min(items_per_category) if items_per_category else 0
            max_items = max(items_per_category) if items_per_category else 0
            avg_items = sum(items_per_category) / len(items_per_category) if items_per_category else 0
            
            logger.info(f"  - Category distribution: min={min_items}, max={max_items}, avg={avg_items:.1f}")
            
            if min_items == 0:
                logger.warning("Some categories have 0 items - check dataset completeness")
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            logger.error("Common causes:")
            logger.error("  - Incorrect dataset path or structure")
            logger.error("  - Missing or corrupted image files")
            logger.error("  - Insufficient permissions to read files")
            logger.error("  - Memory issues during processing")
            raise
        
        # Supervised Contrastive Learning (placeholder)
        scl_trainer = None
        if args.enable_scl:
            logger.info("Initializing Supervised Contrastive Learning...")
            try:
                scl_trainer = SupervisedContrastiveLearning(args, processor.clip_model, processor.device)
                
                logger.info("Starting SCL training (placeholder implementation)")
                for epoch in range(args.scl_epochs):
                    loss = scl_trainer.train_epoch(None)
                    logger.info(f"SCL Epoch {epoch + 1}/{args.scl_epochs}, Loss: {loss:.4f}")
                
                logger.info("✓ SCL training completed")
            except Exception as e:
                logger.error(f"SCL training failed: {e}")
                logger.warning("Continuing without SCL training")
                scl_trainer = None
        
        # Save model artifacts
        logger.info("Saving model artifacts...")
        try:
            save_model_artifacts(args, metadata, embeddings_dict, faiss_index, scl_trainer)
            logger.info("✓ Model artifacts saved successfully")
            
            # Validate saved artifacts
            model_dir = Path(args.model_dir)
            required_files = ['metadata.pkl', 'embeddings.npz', 'faiss_index.index', 'config.json']
            
            for file_name in required_files:
                file_path = model_dir / file_name
                if not file_path.exists():
                    logger.error(f"Missing required artifact: {file_name}")
                    raise FileNotFoundError(f"Required model artifact not saved: {file_name}")
                
                file_size = file_path.stat().st_size
                logger.info(f"  - {file_name}: {file_size / 1024:.1f} KB")
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")
            logger.error("Possible causes:")
            logger.error("  - Insufficient disk space")
            logger.error("  - Permission issues in output directory")
            logger.error("  - Memory issues during serialization")
            raise
        
        # Final success summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("="*60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   Categories processed: {total_categories}")
        logger.info(f"   Fashion items indexed: {total_items}")
        logger.info(f"   Feature vectors generated: {total_embeddings}")
        logger.info(f"   Processing success rate: {(total_embeddings/total_items)*100:.1f}%")
        logger.info(f"")
        logger.info(f"⚡ Performance Metrics:")
        logger.info(f"   Total training time: {duration:.1f} seconds")
        logger.info(f"   Items per second: {total_items/duration:.1f}")
        logger.info(f"   Average batch processing: {duration/(total_items/args.batch_size):.2f}s")
        logger.info(f"")
        logger.info(f"💾 Model Artifacts:")
        logger.info(f"   Saved to: {args.model_dir}")
        logger.info(f"   FAISS index type: {args.faiss_index_type}")
        logger.info(f"   Embedding dimension: {processor.embed_dim}")
        logger.info("="*60)
        
        # Success indicator for SageMaker
        return 0
        
    except KeyboardInterrupt:
        logger.error("Training interrupted by user (Ctrl+C)")
        return 130  # Standard exit code for SIGINT
        
    except MemoryError:
        logger.error("Out of memory during training")
        logger.error("Try reducing batch size or max items per category")
        return 137  # Standard exit code for out of memory
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error("="*60)
        logger.error("❌ TRAINING FAILED!")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error(f"Duration before failure: {duration:.1f} seconds")
        
        # Include stack trace in debug mode
        if os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes'):
            import traceback
            logger.error("Stack trace:")
            logger.error(traceback.format_exc())
        
        logger.error("="*60)
        return 1

if __name__ == '__main__':
    main()