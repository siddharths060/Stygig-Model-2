#!/usr/bin/env python3
"""
Local Training Test Script for StyGig Fashion Recommendation System

This script allows local testing of the training pipeline without SageMaker dependencies.
It simulates the SageMaker environment locally for development and testing.

Usage:
    python local_train_test.py --dataset-path ./test_dataset --output-dir ./local_output
"""

import os
import sys
import json
import argparse
import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_local_args():
    """Parse command line arguments for local testing."""
    parser = argparse.ArgumentParser(description='Local StyGig Training Test')
    
    # Local directories (instead of SageMaker paths)
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to local dataset directory')
    parser.add_argument('--output-dir', type=str, default='../outputs',
                       help='Local output directory for model artifacts')
    
    # Model hyperparameters (same as SageMaker version)
    parser.add_argument('--clip-model', type=str, default='ViT-B-32')
    parser.add_argument('--clip-pretrained', type=str, default='openai')
    parser.add_argument('--batch-size', type=int, default=8)  # Smaller for local testing
    parser.add_argument('--max-items-per-category', type=int, default=10)  # Limit for testing
    parser.add_argument('--faiss-index-type', type=str, default='IndexFlatIP')
    parser.add_argument('--n-clusters', type=int, default=3)
    
    # SCL placeholders (same as SageMaker)
    parser.add_argument('--enable-scl', action='store_true', help='Enable SCL training')
    parser.add_argument('--scl-temperature', type=float, default=0.07)
    parser.add_argument('--scl-epochs', type=int, default=2)  # Fewer for testing
    parser.add_argument('--scl-lr', type=float, default=1e-4)
    
    # Local testing options
    parser.add_argument('--create-sample-dataset', action='store_true',
                       help='Create a sample dataset for testing')
    parser.add_argument('--skip-dependencies-check', action='store_true',
                       help='Skip dependency installation checks')
    
    return parser.parse_args()

def create_sample_dataset(dataset_path: Path):
    """Create a small sample dataset that mirrors the S3 structure."""
    logger.info(f"Creating sample dataset at: {dataset_path}")
    
    # Define the structure that matches your S3
    categories = {
        'accessories': ['bag', 'hat'],
        'bottomwear': ['pants', 'shorts', 'skirt'],
        'footwear': ['flats', 'heels', 'shoes', 'sneakers'],
        'one-piece': ['dress'],
        'upperwear': ['jacket', 'shirt', 'tshirt']
    }
    
    # Create directories
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample images using PIL
    try:
        from PIL import Image
        import random
        
        total_images = 0
        for main_cat, sub_cats in categories.items():
            main_dir = dataset_path / main_cat
            main_dir.mkdir(exist_ok=True)
            
            for sub_cat in sub_cats:
                sub_dir = main_dir / sub_cat
                sub_dir.mkdir(exist_ok=True)
                
                # Create 3-5 sample images per subcategory
                num_images = random.randint(3, 5)
                for i in range(num_images):
                    # Create a simple colored image
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    img = Image.new('RGB', (224, 224), color)
                    
                    # Add some simple pattern
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([50, 50, 174, 174], outline='white', width=3)
                    draw.text((80, 100), f"{main_cat[:3]}\n{sub_cat[:3]}\n{i+1}", fill='white')
                    
                    img_path = sub_dir / f"{main_cat}_{sub_cat}{i+1:03d}.png"
                    img.save(img_path)
                    total_images += 1
        
        logger.info(f"✅ Created {total_images} sample images across {sum(len(sc) for sc in categories.values())} subcategories")
        
    except ImportError:
        logger.error("❌ PIL not available. Please install: pip install Pillow")
        return False
    
    return True

def check_local_dependencies():
    """Check if required dependencies are available locally."""
    logger.info("🔍 Checking local dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('open_clip', 'OpenCLIP'), 
        ('faiss', 'FAISS'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing = []
    available = []
    
    for package, name in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'open_clip':
                import open_clip
            else:
                __import__(package)
            available.append(name)
            logger.info(f"✅ {name} available")
        except ImportError:
            missing.append(name)
            logger.warning(f"❌ {name} not available")
    
    if missing:
        logger.warning("Missing dependencies. Install them with:")
        logger.warning("pip install torch torchvision open_clip_torch faiss-cpu scikit-learn pillow numpy")
        return False
    
    logger.info(f"✅ All {len(available)} required dependencies available")
    return True

def setup_local_stygig_modules():
    """Set up local stygig modules for testing."""
    logger.info("🔧 Setting up local stygig modules...")
    
    # Add current directory and stygig package to path
    current_dir = Path.cwd()
    stygig_dir = current_dir / 'stygig'
    
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(stygig_dir))
    
    # Try to import stygig modules
    try:
        # Test basic imports
        logger.info("Testing stygig module imports...")
        
        # Create minimal versions of stygig modules if they don't exist
        if not stygig_dir.exists():
            logger.info("Creating minimal stygig package structure...")
            create_minimal_stygig_package(stygig_dir)
        
        # Test imports
        from stygig.core.color_enhanced import ColorProcessor
        from stygig.core.gender_enhanced import GenderClassifier
        logger.info("✅ StyGig modules imported successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  StyGig module import failed: {e}")
        logger.info("Creating minimal stygig modules for testing...")
        return create_minimal_stygig_package(stygig_dir)

def create_minimal_stygig_package(stygig_dir: Path):
    """Create minimal stygig package for local testing."""
    logger.info(f"Creating minimal stygig package at: {stygig_dir}")
    
    # Create package structure
    stygig_dir.mkdir(exist_ok=True)
    (stygig_dir / 'core').mkdir(exist_ok=True)
    
    # Create __init__.py files
    (stygig_dir / '__init__.py').write_text('')
    (stygig_dir / 'core' / '__init__.py').write_text('')
    
    # Create minimal color_enhanced.py
    color_enhanced_content = '''
import logging
from typing import List, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class ColorProcessor:
    """Minimal color processor for local testing."""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        logger.info(f"ColorProcessor initialized with {n_clusters} clusters")
    
    def extract_dominant_colors(self, image_path: str) -> List[Tuple[str, float]]:
        """Extract dominant colors from image (simplified for testing)."""
        try:
            img = Image.open(image_path)
            # Simple color extraction - get average color
            img_array = np.array(img)
            mean_color = img_array.mean(axis=(0, 1))
            
            # Convert to rough color name
            r, g, b = mean_color[:3] if len(mean_color) >= 3 else (128, 128, 128)
            
            if r > g and r > b:
                color_name = "red"
            elif g > r and g > b:
                color_name = "green"
            elif b > r and b > g:
                color_name = "blue"
            elif r > 150 and g > 150 and b > 150:
                color_name = "white"
            elif r < 100 and g < 100 and b < 100:
                color_name = "black"
            else:
                color_name = "neutral"
            
            return [(color_name, 1.0)]
        except Exception as e:
            logger.warning(f"Color extraction failed for {image_path}: {e}")
            return [("unknown", 1.0)]
'''
    
    # Create minimal gender_enhanced.py
    gender_enhanced_content = '''
import logging
from typing import Tuple
import random

logger = logging.getLogger(__name__)

class GenderClassifier:
    """Minimal gender classifier for local testing."""
    
    def __init__(self):
        logger.info("GenderClassifier initialized (minimal version)")
    
    def predict_gender(self, image_path: str, category: str) -> Tuple[str, float]:
        """Predict gender from image and category (simplified for testing)."""
        # Simple rule-based prediction for testing
        if 'dress' in category.lower() or 'skirt' in category.lower():
            return ("female", 0.8)
        elif 'jacket' in category.lower() or 'shirt' in category.lower():
            return ("unisex", 0.6)
        else:
            # Random for testing
            genders = ["male", "female", "unisex"]
            gender = random.choice(genders)
            confidence = random.uniform(0.5, 0.9)
            return (gender, confidence)
'''
    
    # Write the files
    (stygig_dir / 'core' / 'color_enhanced.py').write_text(color_enhanced_content)
    (stygig_dir / 'core' / 'gender_enhanced.py').write_text(gender_enhanced_content)
    
    logger.info("✅ Minimal stygig package created")
    return True

class LocalTrainingRunner:
    """Local version of the training runner."""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.clip_model = None
        self.clip_preprocess = None
        
    def run_local_training(self):
        """Run the complete local training pipeline."""
        logger.info("🚀 Starting Local StyGig Training Pipeline")
        logger.info("=" * 60)
        
        # Setup
        if not self.args.skip_dependencies_check:
            if not check_local_dependencies():
                logger.error("❌ Dependencies check failed")
                return False
        
        if not setup_local_stygig_modules():
            logger.error("❌ StyGig modules setup failed") 
            return False
        
        # Create sample dataset if requested
        if self.args.create_sample_dataset:
            dataset_path = Path(self.args.dataset_path)
            if not create_sample_dataset(dataset_path):
                return False
        
        # Initialize models (same as original train.py)
        if not self._initialize_models():
            return False
        
        # Process dataset
        try:
            logger.info("📊 Processing dataset...")
            metadata, embeddings_dict, faiss_index = self._process_local_dataset()
            
            # Save results
            logger.info("💾 Saving model artifacts...")
            self._save_local_artifacts(metadata, embeddings_dict, faiss_index)
            
            # Summary
            total_items = sum(len(items) for items in metadata.values())
            logger.info("\n" + "=" * 60)
            logger.info("✅ LOCAL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📊 Processed {len(metadata)} categories")
            logger.info(f"🖼️  Indexed {total_items} fashion items")
            logger.info(f"🔍 Built FAISS index with {faiss_index.ntotal} vectors")
            logger.info(f"💾 Artifacts saved to: {self.args.output_dir}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Local training failed: {e}")
            return False
    
    def _initialize_models(self):
        """Initialize CLIP and other models."""
        try:
            import torch
            import open_clip
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize CLIP model
            logger.info(f"Loading CLIP model: {self.args.clip_model}")
            model_components = open_clip.create_model_and_transforms(
                self.args.clip_model, 
                pretrained=self.args.clip_pretrained
            )
            
            self.clip_model = model_components[0]
            self.clip_preprocess = model_components[1]
            
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Get embedding dimension
            self.embed_dim = getattr(self.clip_model.visual, 'output_dim', 512)
            logger.info(f"CLIP embedding dimension: {self.embed_dim}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def _process_local_dataset(self):
        """Process local dataset (adapted from train.py)."""
        # Import the updated dataset processing logic from train.py
        # We'll reuse the same logic but with local paths
        
        from train import FashionDatasetProcessor
        
        # Create a mock args object that matches what train.py expects
        class MockArgs:
            def __init__(self, original_args):
                self.clip_model = original_args.clip_model
                self.clip_pretrained = original_args.clip_pretrained
                self.batch_size = original_args.batch_size
                self.max_items_per_category = original_args.max_items_per_category
                self.faiss_index_type = original_args.faiss_index_type
                self.n_clusters = original_args.n_clusters
                # Map local paths to SageMaker-style paths for compatibility
                self.model_dir = original_args.output_dir
                self.train = original_args.dataset_path
        
        mock_args = MockArgs(self.args)
        
        # Use the existing processor but with our initialized models
        processor = FashionDatasetProcessor(mock_args)
        processor.device = self.device
        processor.clip_model = self.clip_model
        processor.clip_preprocess = self.clip_preprocess
        processor.embed_dim = self.embed_dim
        
        return processor.process_dataset(self.args.dataset_path)
    
    def _save_local_artifacts(self, metadata, embeddings_dict, faiss_index):
        """Save artifacts locally."""
        import faiss
        import numpy as np
        
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save embeddings
        np.savez_compressed(output_dir / 'embeddings.npz', **embeddings_dict)
        
        # Save FAISS index
        faiss.write_index(faiss_index, str(output_dir / 'faiss_index.index'))
        
        # Save config
        config = {
            'clip_model': self.args.clip_model,
            'clip_pretrained': self.args.clip_pretrained,
            'embed_dim': self.embed_dim,
            'faiss_index_type': self.args.faiss_index_type,
            'n_clusters': self.args.n_clusters,
            'scl_enabled': self.args.enable_scl
        }
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Artifacts saved to {output_dir}")

def main():
    """Main function for local testing."""
    args = parse_local_args()
    
    logger.info("🧪 StyGig Local Training Test")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max items per category: {args.max_items_per_category}")
    
    runner = LocalTrainingRunner(args)
    success = runner.run_local_training()
    
    if success:
        logger.info("\n🎉 Local testing completed successfully!")
        logger.info("You can now run the AWS pipeline with confidence.")
    else:
        logger.error("\n❌ Local testing failed. Please fix issues before running on AWS.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)