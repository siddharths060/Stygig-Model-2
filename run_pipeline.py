#!/usr/bin/env python3
"""
StyGig V4 - Complete MLOps Pipeline Runner
===========================================

This is the main "one-click" script for training and deploying a fresh V4 model.
It orchestrates the entire pipeline from data preparation to endpoint testing.

Usage:
    python run_pipeline.py

Features:
    - Trains a fresh V4 model from scratch using SageMaker Training Jobs
    - Deploys to a REAL-TIME (synchronous) endpoint
    - Tests the deployed endpoint with sample inference
    - Cleans up all resources to avoid unnecessary costs

Prerequisites:
    - AWS credentials configured
    - SageMaker execution role
    - Training data uploaded to S3
    - Test images in test_images/ directory

Author: StyGig MLOps Team
Date: November 15, 2025
Version: 4.0
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V4PipelineRunner:
    """
    Orchestrates the complete V4 training and deployment pipeline.
    """
    
    def __init__(self):
        """Initialize pipeline with AWS configurations."""
        logger.info("Initializing V4 Pipeline Runner...")
        
        # AWS Setup
        self.boto_session = boto3.Session()
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.region = self.boto_session.region_name
        self.role = sagemaker.get_execution_role()
        
        # SageMaker clients
        self.sm_client = boto3.client('sagemaker', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # Pipeline configuration
        self.s3_bucket = self.sagemaker_session.default_bucket()
        self.prefix = "stygig-v4-pipeline"
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.job_name = f"stygig-v4-training-{self.timestamp}"
        
        # Resource tracking for cleanup
        self.resources = {
            'endpoint_name': None,
            'endpoint_config_name': None,
            'model_name': None,
            'training_job_name': None
        }
        
        logger.info(f"✓ Region: {self.region}")
        logger.info(f"✓ S3 Bucket: {self.s3_bucket}")
        logger.info(f"✓ Job Name: {self.job_name}")
    
    def train_model(self, training_data_s3_uri):
        """
        Train a fresh V4 model using SageMaker Training Job.
        
        Args:
            training_data_s3_uri: S3 URI to training data (e.g., s3://bucket/data/)
        
        Returns:
            Trained PyTorch estimator with model artifacts
        """
        logger.info("="*70)
        logger.info("STEP 1: TRAINING V4 MODEL")
        logger.info("="*70)
        
        try:
            # Define hyperparameters for V4 training
            hyperparameters = {
                'batch-size': 32,
                'clip-model': 'ViT-B-32',
                'clip-pretrained': 'openai',
                'n-clusters': 3,
                'faiss-index-type': 'IndexFlatIP',
                'max-items-per-category': 1000  # Limit for faster CPU training
            }
            
            logger.info(f"Training data: {training_data_s3_uri}")
            logger.info(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
            logger.info("Note: Training limited to 1000 items per category for CPU optimization")
            
            # Create PyTorch Estimator for V4
            estimator = PyTorch(
                entry_point='train.py',
                source_dir='sagemaker',
                dependencies=['src', 'config'],
                role=self.role,
                instance_type='ml.m5.xlarge',  # CPU training (GPU not available in ap-south-1)
                instance_count=1,
                framework_version='2.0.0',
                py_version='py310',
                hyperparameters=hyperparameters,
                output_path=f"s3://{self.s3_bucket}/{self.prefix}/models",
                base_job_name='stygig-v4-training',
                sagemaker_session=self.sagemaker_session,
                keep_alive_period_in_seconds=0,  # No warm pools (cost optimization)
                metric_definitions=[
                    {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                    {'Name': 'validation:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'}
                ]
            )
            
            # Track training job name
            self.resources['training_job_name'] = estimator._current_job_name
            
            logger.info(f"Starting training job: {estimator._current_job_name}")
            logger.info("Training on CPU instance (ml.m5.xlarge) - expect 30-45 minutes...")
            
            # Start training (synchronous - waits for completion)
            estimator.fit(
                inputs={'training': training_data_s3_uri},
                wait=True,
                logs='All'
            )
            
            # Get model artifact URI
            model_data_uri = estimator.model_data
            
            logger.info("="*70)
            logger.info("✅ TRAINING COMPLETE")
            logger.info("="*70)
            logger.info(f"Model artifacts: {model_data_uri}")
            logger.info(f"Training job: {estimator._current_job_name}")
            
            return estimator
        
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def deploy_realtime_endpoint(self, estimator, endpoint_name=None):
        """
        Deploy trained model to a REAL-TIME (synchronous) SageMaker endpoint.
        
        Args:
            estimator: Trained PyTorch estimator
            endpoint_name: Optional custom endpoint name
        
        Returns:
            SageMaker Predictor object for the deployed endpoint
        """
        logger.info("="*70)
        logger.info("STEP 2: DEPLOYING REAL-TIME ENDPOINT")
        logger.info("="*70)
        
        try:
            # Generate endpoint name if not provided
            if endpoint_name is None:
                endpoint_name = f"stygig-v4-realtime-{self.timestamp}"
            
            self.resources['endpoint_name'] = endpoint_name
            
            logger.info(f"Endpoint name: {endpoint_name}")
            logger.info(f"Instance type: ml.g4dn.xlarge")
            # V4 SYNC REQUIRES GPU
            logger.info("Mode: REAL-TIME (synchronous)")
            logger.info("Deploying... (this takes 5-10 minutes)")
            
            # Deploy to real-time endpoint
            # CRITICAL: No AsyncInferenceConfig - this is synchronous
            # WARNING: ml.c5.large (CPU) will have 450-650ms latency
            predictor = estimator.deploy(
                initial_instance_count=1,
                instance_type='ml.g4dn.xlarge',  # V4 SYNC REQUIRES GPU
                endpoint_name=endpoint_name,
                serializer=IdentitySerializer(content_type='application/x-image'),
                deserializer=JSONDeserializer(),
                wait=True  # Wait for endpoint to be InService
            )
            
            # Track resource names for cleanup
            endpoint_context = predictor.endpoint_context()
            self.resources['endpoint_config_name'] = getattr(endpoint_context, 'endpoint_config_name', endpoint_name + '-config')
            self.resources['model_name'] = getattr(endpoint_context, 'model_name', endpoint_name + '-model')
            
            logger.info("="*70)
            logger.info("✅ DEPLOYMENT COMPLETE")
            logger.info("="*70)
            logger.info(f"Endpoint: {endpoint_name}")
            logger.info(f"Status: InService")
            logger.info(f"Endpoint ARN: {getattr(predictor.endpoint_context(), 'endpoint_arn', 'N/A')}")
            
            return predictor
        
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def test_realtime_endpoint(self, predictor):
        """
        Test the deployed real-time endpoint with sample inference.
        
        Args:
            predictor: SageMaker Predictor for the deployed endpoint
        """
        logger.info("="*70)
        logger.info("STEP 3: TESTING ENDPOINT")
        logger.info("="*70)
        
        try:
            # Find test image
            test_image_path = Path('test_images/gray_shirt.jpg')
            
            if not test_image_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path('outfits_dataset/test/gray_shirt.jpg'),
                    Path('outputs/test_image.jpg')
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        test_image_path = alt_path
                        break
            
            if not test_image_path.exists():
                logger.warning(f"⚠️  Test image not found: {test_image_path}")
                logger.warning("Skipping endpoint test (endpoint is still deployed)")
                return
            
            logger.info(f"Test image: {test_image_path}")
            
            # Load and serialize image
            with open(test_image_path, 'rb') as f:
                image_bytes = f.read()
            
            logger.info(f"Image size: {len(image_bytes)} bytes")
            logger.info("Invoking endpoint...")
            
            # Invoke real-time endpoint
            start_time = time.time()
            response = predictor.predict(image_bytes)
            latency = (time.time() - start_time) * 1000
            
            logger.info("="*70)
            logger.info("✅ INFERENCE SUCCESSFUL")
            logger.info("="*70)
            logger.info(f"Latency: {latency:.2f}ms")
            
            # Pretty-print response
            logger.info("\nQuery Item:")
            if 'query_item' in response:
                for key, value in response['query_item'].items():
                    logger.info(f"  {key}: {value}")
            
            logger.info(f"\nRecommendations: {len(response.get('recommendations', []))}")
            if response.get('recommendations'):
                for i, rec in enumerate(response['recommendations'][:3], 1):
                    logger.info(f"  {i}. {rec.get('category')} - Score: {rec.get('score'):.4f}")
            
            # Verify V4 features
            logger.info("\n✓ V4 Feature Verification:")
            
            if 'dominant_color_rgb' in response.get('query_item', {}):
                rgb = response['query_item']['dominant_color_rgb']
                logger.info(f"  ✓ RGB color tuples working: {rgb}")
            else:
                logger.warning("  ⚠️  RGB tuples not found (check metadata)")
            
            logger.info("  ✓ Real-time inference working")
            logger.info(f"  ✓ Latency < 200ms: {latency < 200}")
            
        except Exception as e:
            logger.error(f"❌ Endpoint test failed: {e}")
            logger.error("The endpoint is deployed but inference failed")
            raise
    
    def cleanup_resources(self):
        """
        Clean up all created AWS resources to avoid ongoing costs.
        """
        logger.info("="*70)
        logger.info("CLEANUP: DELETING AWS RESOURCES")
        logger.info("="*70)
        
        cleanup_summary = []
        
        # Delete endpoint
        if self.resources['endpoint_name']:
            try:
                logger.info(f"Deleting endpoint: {self.resources['endpoint_name']}")
                self.sm_client.delete_endpoint(
                    EndpointName=self.resources['endpoint_name']
                )
                cleanup_summary.append(f"✓ Deleted endpoint: {self.resources['endpoint_name']}")
            except Exception as e:
                cleanup_summary.append(f"✗ Failed to delete endpoint: {e}")
        
        # Delete endpoint config
        if self.resources['endpoint_config_name']:
            try:
                logger.info(f"Deleting endpoint config: {self.resources['endpoint_config_name']}")
                self.sm_client.delete_endpoint_config(
                    EndpointConfigName=self.resources['endpoint_config_name']
                )
                cleanup_summary.append(f"✓ Deleted endpoint config: {self.resources['endpoint_config_name']}")
            except Exception as e:
                cleanup_summary.append(f"✗ Failed to delete endpoint config: {e}")
        
        # Delete model
        if self.resources['model_name']:
            try:
                logger.info(f"Deleting model: {self.resources['model_name']}")
                self.sm_client.delete_model(
                    ModelName=self.resources['model_name']
                )
                cleanup_summary.append(f"✓ Deleted model: {self.resources['model_name']}")
            except Exception as e:
                cleanup_summary.append(f"✗ Failed to delete model: {e}")
        
        logger.info("="*70)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*70)
        for item in cleanup_summary:
            logger.info(item)
        
        logger.info("\nNote: Training job artifacts remain in S3:")
        logger.info(f"  s3://{self.s3_bucket}/{self.prefix}/models/")
        logger.info("  (Delete manually if no longer needed)")


def main():
    """
    Main pipeline execution with error handling and resource cleanup.
    """
    print("="*70)
    print("  StyGig V4 - Complete MLOps Pipeline")
    print("="*70)
    print("  This script will:")
    print("    1. Train a fresh V4 model (30-45 min on CPU)")
    print("    2. Deploy to real-time endpoint (5-10 min)")
    print("    3. Test with sample inference")
    print("    4. Clean up all resources")
    print("="*70)
    print()
    
    # Configuration
    TRAINING_DATA_S3 = os.getenv(
        'STYGIG_TRAINING_DATA',
        's3://your-bucket/stygig-data/train/'  # UPDATE THIS
    )
    
    if 'your-bucket' in TRAINING_DATA_S3:
        logger.error("❌ Please set the STYGIG_TRAINING_DATA environment variable")
        logger.error("   export STYGIG_TRAINING_DATA=s3://your-bucket/stygig-data/train/")
        sys.exit(1)
    
    pipeline = V4PipelineRunner()
    predictor = None
    
    try:
        # Step 1: Train model
        estimator = pipeline.train_model(TRAINING_DATA_S3)
        
        # Step 2: Deploy endpoint
        predictor = pipeline.deploy_realtime_endpoint(estimator)
        
        # Step 3: Test endpoint
        pipeline.test_realtime_endpoint(predictor)
        
        logger.info("="*70)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Endpoint: {pipeline.resources['endpoint_name']}")
        logger.info(f"Model: {estimator.model_data}")
        logger.info("\nThe endpoint is now running and incurring costs.")
        logger.info("Cleanup will begin in 5 seconds...")
        time.sleep(5)
    
    except Exception as e:
        logger.error("="*70)
        logger.error(f"❌ PIPELINE FAILED: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Always cleanup resources
        pipeline.cleanup_resources()
        
        logger.info("="*70)
        logger.info("Pipeline execution complete. All resources cleaned up.")
        logger.info("="*70)


if __name__ == '__main__':
    main()
