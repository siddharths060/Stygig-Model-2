#!/usr/bin/env python3
"""
Local Training Test Script for StyGig Fashion Recommendation System

This script simulates SageMaker training locally for development and testing.
It's useful for rapid iteration without deploying to AWS.

Usage:
    python scripts/testing/local_train_test.py --dataset-path ./outfits_dataset --output-dir ./local_output

Examples:
    # Basic local training
    python scripts/testing/local_train_test.py --dataset-path outfits_dataset

    # With custom parameters
    python scripts/testing/local_train_test.py --dataset-path outfits_dataset --batch-size 16 --max-items 50
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'sagemaker'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for local testing."""
    parser = argparse.ArgumentParser(
        description='Local StyGig Training Test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Local directories
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to local dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Local output directory for model artifacts (default: outputs)'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B-32',
        help='CLIP model variant (default: ViT-B-32)'
    )
    parser.add_argument(
        '--clip-pretrained',
        type=str,
        default='openai',
        help='CLIP pretrained weights (default: openai)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=50,
        help='Max items per category for testing (default: 50, None for all)'
    )
    parser.add_argument(
        '--faiss-index-type',
        type=str,
        default='IndexFlatIP',
        choices=['IndexFlatIP', 'IndexIVFFlat', 'IndexPQ'],
        help='FAISS index type (default: IndexFlatIP)'
    )
    
    # Testing options
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with minimal data (max 10 items)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_environment():
    """Validate that required dependencies are available."""
    missing_deps = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import clip
        logger.info(f"✓ CLIP available")
    except ImportError:
        missing_deps.append('clip')
    
    try:
        import faiss
        logger.info(f"✓ FAISS available")
    except ImportError:
        missing_deps.append('faiss')
    
    try:
        from PIL import Image
        logger.info(f"✓ Pillow available")
    except ImportError:
        missing_deps.append('Pillow')
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install torch torchvision clip faiss-cpu Pillow")
        return False
    
    return True


def simulate_local_training(args):
    """Simulate SageMaker training locally."""
    logger.info("="*80)
    logger.info("StyGig Local Training Simulation")
    logger.info("="*80)
    logger.info("")
    
    # Validate dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 1
    
    logger.info(f"Dataset path: {dataset_path}")
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(f'*{ext}')))
    
    logger.info(f"Found {len(image_files)} images in dataset")
    
    if len(image_files) == 0:
        logger.error("No images found in dataset!")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Simulate training with the actual train.py logic
    logger.info("")
    logger.info("Starting training simulation...")
    logger.info(f"  CLIP Model: {args.clip_model}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Max Items: {args.max_items or 'All'}")
    logger.info(f"  FAISS Index: {args.faiss_index_type}")
    logger.info("")
    
    try:
        # Import training components
        from stygig.core.recommendation_engine import FashionProcessor
        
        # Initialize processor
        logger.info("Initializing fashion processor...")
        processor = FashionProcessor(
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained
        )
        
        # Build index
        logger.info("Building fashion index...")
        processor.build_index(
            dataset_path=str(dataset_path),
            batch_size=args.batch_size,
            max_items_per_category=args.max_items
        )
        
        # Get statistics
        stats = processor.get_statistics()
        logger.info("")
        logger.info("Training Statistics:")
        logger.info(f"  Total items: {stats['total_items']}")
        logger.info(f"  Categories: {len(stats['categories'])}")
        logger.info(f"  Embedding dim: {stats.get('embedding_dim', 'N/A')}")
        
        # Save model
        logger.info("")
        logger.info("Saving model artifacts...")
        
        model_path = output_dir / 'model.pkl'
        processor.save(str(model_path))
        logger.info(f"✓ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'clip_model': args.clip_model,
            'clip_pretrained': args.clip_pretrained,
            'total_items': stats['total_items'],
            'categories': list(stats['categories']),
            'faiss_index_type': args.faiss_index_type,
            'training_params': {
                'batch_size': args.batch_size,
                'max_items_per_category': args.max_items
            }
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved to: {metadata_path}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ Training simulation completed successfully!")
        logger.info("="*80)
        logger.info("")
        logger.info("Test your model:")
        logger.info(f"  python scripts/testing/integration_test.py <image_path> --dataset {dataset_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Quick test mode
    if args.quick_test:
        logger.info("Quick test mode: limiting to 10 items")
        args.max_items = 10
    
    # Validate environment
    logger.info("Validating environment...")
    if not validate_environment():
        return 1
    
    logger.info("✓ Environment validated")
    logger.info("")
    
    # Run local training
    return simulate_local_training(args)


if __name__ == '__main__':
    sys.exit(main())
