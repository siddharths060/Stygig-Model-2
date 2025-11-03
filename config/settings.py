# StyGig SageMaker Configuration
# Production-ready configuration with proper defaults for stygig-ml-s3 bucket

import os
import logging
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyGigConfig:
    """Centralized configuration management for StyGig SageMaker pipeline."""
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        
        # AWS Configuration
        self.AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
        self.SAGEMAKER_ROLE = os.environ.get('SAGEMAKER_ROLE', None)
        self.S3_BUCKET = os.environ.get('S3_BUCKET', 'stygig-ml-s3')
        
        # Dataset Configuration - Using train/ folder in stygig-ml-s3
        self.DATASET_S3_URI = os.environ.get('DATASET_S3_URI', 's3://stygig-ml-s3/train/')
        self.LOCAL_DATASET_PATH = os.environ.get('LOCAL_DATASET_PATH', None)
        
        # Training Configuration
        self.TRAINING_INSTANCE_TYPE = os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.large')
        self.TRAINING_VOLUME_SIZE = int(os.environ.get('TRAINING_VOLUME_SIZE', '30'))
        self.TRAINING_MAX_RUNTIME = int(os.environ.get('TRAINING_MAX_RUNTIME', '3600'))
        
        # Inference Configuration
        self.INFERENCE_INSTANCE_TYPE = os.environ.get('INFERENCE_INSTANCE_TYPE', 'ml.m5.large')
        self.ENDPOINT_NAME_PREFIX = os.environ.get('ENDPOINT_NAME_PREFIX', 'stygig-fashion')
        
        # Model Hyperparameters
        self.HYPERPARAMETERS = {
            'clip-model': os.environ.get('CLIP_MODEL', 'ViT-B-32'),
            'clip-pretrained': os.environ.get('CLIP_PRETRAINED', 'openai'),
            'batch-size': int(os.environ.get('BATCH_SIZE', '32')),
            'faiss-index-type': os.environ.get('FAISS_INDEX_TYPE', 'IndexFlatIP'),
            'n-clusters': int(os.environ.get('N_CLUSTERS', '3')),
            'enable-scl': os.environ.get('ENABLE_SCL', 'false').lower() == 'true',
            'scl-temperature': float(os.environ.get('SCL_TEMPERATURE', '0.07')),
            'scl-epochs': int(os.environ.get('SCL_EPOCHS', '5')),
            'max-items-per-category': int(os.environ.get('MAX_ITEMS_PER_CATEGORY', '500'))
        }
        
        # Cost Optimization
        self.USE_SPOT_INSTANCES = os.environ.get('USE_SPOT_INSTANCES', 'true').lower() == 'true'
        self.MAX_WAIT_TIME = int(os.environ.get('MAX_WAIT_TIME', '7200'))
        
        # Pipeline Behavior
        self.CLEANUP_AFTER_PIPELINE = os.environ.get('CLEANUP_AFTER_PIPELINE', 'false').lower() == 'true'
        self.TEST_ENDPOINT = os.environ.get('TEST_ENDPOINT', 'true').lower() == 'true'
        self.ENABLE_SAGEMAKER_METRICS = os.environ.get('ENABLE_SAGEMAKER_METRICS', 'true').lower() == 'true'
        
        # Development Settings
        self.DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        self.ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production').lower()
        
        # Apply environment-specific optimizations
        self._apply_environment_settings()
        
        # S3 Paths
        self.S3_PATHS = {
            'dataset': self.DATASET_S3_URI,
            'models': f's3://{self.S3_BUCKET}/models/',
            'outputs': f's3://{self.S3_BUCKET}/training-outputs/',
            'artifacts': f's3://{self.S3_BUCKET}/model-artifacts/',
            'logs': f's3://{self.S3_BUCKET}/logs/',
            'temp': f's3://{self.S3_BUCKET}/temp/',
            'checkpoints': f's3://{self.S3_BUCKET}/checkpoints/',
        }
        
        # Metric Definitions for SageMaker
        self.METRIC_DEFINITIONS = [
            {'Name': 'train:total_categories', 'Regex': 'Processed ([0-9]+) categories'},
            {'Name': 'train:total_items', 'Regex': 'Indexed ([0-9]+) fashion items'},
            {'Name': 'train:faiss_vectors', 'Regex': 'Built FAISS index with ([0-9]+) vectors'},
            {'Name': 'train:embedding_dim', 'Regex': 'CLIP embedding dimension: ([0-9]+)'},
            {'Name': 'train:processing_time', 'Regex': 'Processing completed in ([0-9.]+) seconds'},
        ]
        
        if self.HYPERPARAMETERS['enable-scl']:
            self.METRIC_DEFINITIONS.extend([
                {'Name': 'train:scl_loss', 'Regex': 'SCL Epoch [0-9]+/[0-9]+, Loss: ([0-9.]+)'},
                {'Name': 'train:scl_epoch', 'Regex': 'SCL Epoch ([0-9]+)/[0-9]+'},
            ])
    
    def _apply_environment_settings(self):
        """Apply environment-specific optimizations."""
        if self.DEBUG_MODE or self.ENVIRONMENT == 'development':
            # Development optimizations - use smaller batch sizes but same reliable instances
            self.HYPERPARAMETERS['batch-size'] = 8
            self.HYPERPARAMETERS['max-items-per-category'] = 50
            # Use ml.m5.large for development too - more reliable across regions
            if not os.environ.get('TRAINING_INSTANCE_TYPE'):
                self.TRAINING_INSTANCE_TYPE = 'ml.m5.large'
            if not os.environ.get('INFERENCE_INSTANCE_TYPE'):
                self.INFERENCE_INSTANCE_TYPE = 'ml.m5.large'
            
        elif self.ENVIRONMENT == 'production':
            # Production optimizations
            self.HYPERPARAMETERS['batch-size'] = int(os.environ.get('BATCH_SIZE', '64'))
            self.HYPERPARAMETERS['max-items-per-category'] = int(os.environ.get('MAX_ITEMS_PER_CATEGORY', '2000'))
            self.TRAINING_INSTANCE_TYPE = os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.c5.2xlarge')
            self.INFERENCE_INSTANCE_TYPE = os.environ.get('INFERENCE_INSTANCE_TYPE', 'ml.c5.xlarge')
            self.USE_SPOT_INSTANCES = os.environ.get('USE_SPOT_INSTANCES', 'false').lower() == 'true'
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate AWS region
        valid_regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-south-1', 'ap-southeast-1']
        if self.AWS_REGION not in valid_regions:
            errors.append(f"Invalid AWS region: {self.AWS_REGION}")
        
        # Validate instance types
        if not self.TRAINING_INSTANCE_TYPE.startswith('ml.'):
            errors.append(f"Invalid training instance type: {self.TRAINING_INSTANCE_TYPE}")
        
        if not self.INFERENCE_INSTANCE_TYPE.startswith('ml.'):
            errors.append(f"Invalid inference instance type: {self.INFERENCE_INSTANCE_TYPE}")
        
        # Validate S3 bucket
        if not self.S3_BUCKET or len(self.S3_BUCKET) < 3:
            errors.append("Invalid S3 bucket name")
        
        # Validate dataset URI
        if not self.DATASET_S3_URI.startswith('s3://'):
            errors.append("Dataset URI must start with s3://")
        
        # Validate hyperparameters
        if self.HYPERPARAMETERS['batch-size'] <= 0:
            errors.append("Batch size must be positive")
        
        return errors
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("StyGig SageMaker Configuration Summary")
        print("=" * 80)
        print(f"Environment: {self.ENVIRONMENT.upper()}")
        print(f"AWS Region: {self.AWS_REGION}")
        print(f"S3 Bucket: {self.S3_BUCKET}")
        print(f"Dataset URI: {self.DATASET_S3_URI}")
        print()
        print("Training Configuration:")
        print(f"  Instance Type: {self.TRAINING_INSTANCE_TYPE}")
        print(f"  Volume Size: {self.TRAINING_VOLUME_SIZE} GB")
        print(f"  Max Runtime: {self.TRAINING_MAX_RUNTIME} seconds")
        print(f"  Use Spot Instances: {self.USE_SPOT_INSTANCES}")
        print()
        print("Model Configuration:")
        print(f"  CLIP Model: {self.HYPERPARAMETERS['clip-model']}")
        print(f"  Batch Size: {self.HYPERPARAMETERS['batch-size']}")
        print(f"  Max Items/Category: {self.HYPERPARAMETERS['max-items-per-category']}")
        print(f"  Enable SCL: {self.HYPERPARAMETERS['enable-scl']}")
        print()
        print("Debug & Development:")
        print(f"  Debug Mode: {self.DEBUG_MODE}")
        print(f"  Log Level: {self.LOG_LEVEL}")
        print(f"  Test Endpoint: {self.TEST_ENDPOINT}")
        print("=" * 80)


# Create global config instance
config = StyGigConfig()

# Export commonly used values for backward compatibility
AWS_REGION = config.AWS_REGION
SAGEMAKER_ROLE = config.SAGEMAKER_ROLE  
S3_BUCKET = config.S3_BUCKET
DATASET_S3_URI = config.DATASET_S3_URI
LOCAL_DATASET_PATH = config.LOCAL_DATASET_PATH

TRAINING_INSTANCE_TYPE = config.TRAINING_INSTANCE_TYPE
TRAINING_VOLUME_SIZE = config.TRAINING_VOLUME_SIZE
TRAINING_MAX_RUNTIME = config.TRAINING_MAX_RUNTIME

INFERENCE_INSTANCE_TYPE = config.INFERENCE_INSTANCE_TYPE
ENDPOINT_NAME_PREFIX = config.ENDPOINT_NAME_PREFIX

HYPERPARAMETERS = config.HYPERPARAMETERS
USE_SPOT_INSTANCES = config.USE_SPOT_INSTANCES
MAX_WAIT_TIME = config.MAX_WAIT_TIME

CLEANUP_AFTER_PIPELINE = config.CLEANUP_AFTER_PIPELINE
TEST_ENDPOINT = config.TEST_ENDPOINT
ENABLE_SAGEMAKER_METRICS = config.ENABLE_SAGEMAKER_METRICS

DEBUG_MODE = config.DEBUG_MODE
LOG_LEVEL = config.LOG_LEVEL

METRIC_DEFINITIONS = config.METRIC_DEFINITIONS
S3_PATHS = config.S3_PATHS

if __name__ == "__main__":
    # Validate and display configuration
    errors = config.validate()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")
        config.print_summary()