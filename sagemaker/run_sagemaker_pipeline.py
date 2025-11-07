#!/usr/bin/env python3
"""
SageMaker Pipeline Orchestration Script for StyGig Fashion Recommendation System

This script orchestrates the complete SageMaker pipeline from the Code Editor:
1. Sets up IAM roles and S3 bucket configuration  
2. Creates SageMaker Estimator for training
3. Launches training job with fashion dataset
4. Deploys trained model to real-time endpoint
5. Tests the deployed endpoint with sample predictions

Usage:
    Run this script in the SageMaker Code Editor environment:
    
    # Basic usage with default config
    python run_sagemaker_pipeline.py
    
    # With environment variables
    export S3_BUCKET=stygig-ml-s3
    export DATASET_S3_URI=s3://stygig-ml-s3/train/
    python run_sagemaker_pipeline.py

Prerequisites:
    - SageMaker execution role with appropriate permissions
    - S3 bucket with fashion dataset uploaded 
    - AWS credentials configured in the environment
"""

import os
import sys
import json
import time
import boto3
import logging
import tempfile
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from io import BytesIO
from PIL import Image

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
from settings import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StyGigSageMakerPipeline:
    """Main orchestration class for the StyGig SageMaker pipeline."""
    
    def __init__(self,
                 region: str = None,
                 role: str = None,
                 bucket: str = None,
                 dataset_s3_uri: str = None):
        """
        Initialize the SageMaker pipeline with configuration.
        
        Args:
            region: AWS region (uses config default if None)
            role: SageMaker execution role ARN (auto-detected if None)
            bucket: S3 bucket name (uses config default if None)  
            dataset_s3_uri: S3 URI to fashion dataset (uses config default if None)
        """
        
        # Use config values with override capability
        self.region = region or config.AWS_REGION
        self.bucket = bucket or config.S3_BUCKET
        self.dataset_s3_uri = dataset_s3_uri or config.DATASET_S3_URI
        
        # Initialize AWS session and SageMaker
        self.session = boto3.Session(region_name=self.region)
        
        # Initialize SageMaker session (compatible with different SDK versions)
        try:
            # Try newer SDK version first
            self.sagemaker_session = sagemaker.Session(
                boto_session=self.session,
                sagemaker_client=self.session.client('sagemaker'),
                sagemaker_runtime_client=self.session.client('sagemaker-runtime')
            )
        except TypeError:
            try:
                # Fallback for older SDK versions
                self.sagemaker_session = sagemaker.Session(boto3_session=self.session)
            except TypeError:
                # Last resort - use default session with region override
                os.environ['AWS_DEFAULT_REGION'] = self.region
                self.sagemaker_session = sagemaker.Session()
        
        # Auto-detect role if not provided
        self.role = role or config.SAGEMAKER_ROLE or self._get_sagemaker_role()
        
        # S3 configuration using config
        self.output_s3_uri = config.S3_PATHS['artifacts']
        
        # Model and endpoint configuration
        timestamp = int(time.time())
        self.model_name = f"{config.ENDPOINT_NAME_PREFIX}-model-{timestamp}"
        self.endpoint_name = f"{config.ENDPOINT_NAME_PREFIX}-endpoint-{timestamp}"
        
        # Training configuration from config
        self.training_instance_type = config.TRAINING_INSTANCE_TYPE
        self.inference_instance_type = config.INFERENCE_INSTANCE_TYPE
        self.hyperparameters = config.HYPERPARAMETERS.copy()
        
        logger.info(f"Initialized StyGig pipeline:")
        logger.info(f"  Region: {self.region}")
        logger.info(f"  Role: {self.role}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Dataset S3 URI: {self.dataset_s3_uri}")
        
    def _get_sagemaker_role(self) -> str:
        """Auto-detect SageMaker execution role."""
        try:
            # Try to get role from SageMaker session
            return sagemaker.get_execution_role()
        except Exception:
            # Fallback: try to get from environment or instance metadata
            try:
                # Check environment variable
                role = os.environ.get('SAGEMAKER_ROLE')
                if role:
                    return role
                    
                # Try instance metadata service (if running on SageMaker instance)
                import urllib.request
                metadata_url = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
                response = urllib.request.urlopen(metadata_url, timeout=2)
                role_name = response.read().decode('utf-8')
                
                # Get the full role ARN
                sts_client = self.session.client('sts')
                account_id = sts_client.get_caller_identity()['Account']
                return f"arn:aws:iam::{account_id}:role/{role_name}"
                
            except Exception as e:
                logger.error(f"Failed to auto-detect SageMaker role: {e}")
                raise ValueError(
                    "Could not auto-detect SageMaker execution role. "
                    "Please provide role parameter or set SAGEMAKER_ROLE environment variable."
                )
    
    def prepare_dataset(self, local_dataset_path: str = None) -> str:
        """
        Upload dataset to S3 if needed.
        
        Args:
            local_dataset_path: Local path to dataset directory
            
        Returns:
            S3 URI of the dataset
        """
        if local_dataset_path and os.path.exists(local_dataset_path):
            logger.info(f"Uploading dataset from {local_dataset_path} to S3...")
            
            # Upload dataset to S3
            s3_client = self.session.client('s3')
            dataset_path = Path(local_dataset_path)
            
            # Upload all files in the dataset
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(dataset_path)
                    s3_key = f"train/{relative_path}"  # Use train/ prefix
                    
                    try:
                        s3_client.upload_file(
                            str(file_path),
                            self.bucket,
                            s3_key
                        )
                        logger.debug(f"Uploaded {file_path} to s3://{self.bucket}/{s3_key}")
                    except Exception as e:
                        logger.error(f"Failed to upload {file_path}: {e}")
            
            logger.info(f"Dataset uploaded to {self.dataset_s3_uri}")
        else:
            logger.info(f"Using existing dataset at {self.dataset_s3_uri}")
        
        return self.dataset_s3_uri
    
    def create_estimator(self) -> PyTorch:
        """Create SageMaker PyTorch Estimator for training."""
        
        logger.info("Creating SageMaker Estimator...")
        
        # Get the parent directory (project root) to include src/ in upload
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from sagemaker/
        
        # Define the estimator with config values
        estimator = PyTorch(
            entry_point='sagemaker/train.py',  # Relative to project root
            source_dir=project_root,  # Project root containing src/stygig
            role=self.role,
            instance_type=self.training_instance_type,
            instance_count=1,
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters=self.hyperparameters,
            output_path=self.output_s3_uri,
            sagemaker_session=self.sagemaker_session,
            enable_sagemaker_metrics=config.ENABLE_SAGEMAKER_METRICS,
            metric_definitions=config.METRIC_DEFINITIONS,
            volume_size=config.TRAINING_VOLUME_SIZE,
            max_run=config.TRAINING_MAX_RUNTIME,
            use_spot_instances=config.USE_SPOT_INSTANCES,
            max_wait=config.MAX_WAIT_TIME if config.USE_SPOT_INSTANCES else None
        )
        
        logger.info(f"Created estimator with training instance: {self.training_instance_type}")
        return estimator
    
    def launch_training_job(self, estimator: PyTorch) -> tuple:
        """Launch SageMaker training job and return both estimator and model_data."""
        
        logger.info("Launching SageMaker training job...")
        
        # Prepare input channels
        training_input = sagemaker.inputs.TrainingInput(
            s3_data=self.dataset_s3_uri,
            distribution='FullyReplicated',
            content_type='application/x-image',  # Indicate image data
            s3_data_type='S3Prefix'
        )
        
        # Start training
        job_name = f"stygig-training-{int(time.time())}"
        
        try:
            estimator.fit(
                inputs={'training': training_input},
                job_name=job_name,
                wait=True  # Wait for training to complete
            )
            
            logger.info(f"Training job {job_name} completed successfully!")
            
            # Validate training job completed successfully
            job_desc = estimator.sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            
            training_status = job_desc.get('TrainingJobStatus')
            if training_status != 'Completed':
                raise RuntimeError(f"Training job failed with status: {training_status}")
            
            # Get model data URI with multiple fallback strategies
            model_data = getattr(estimator, 'model_data', None)
            
            if model_data is None:
                # Strategy 1: Use job description
                model_artifacts = job_desc.get('ModelArtifacts', {})
                model_data = model_artifacts.get('S3ModelArtifacts')
                
            if model_data is None:
                # Strategy 2: Construct from output path and job name  
                model_data = f"{self.output_s3_uri.rstrip('/')}/{job_name}/output/model.tar.gz"
                logger.warning(f"Constructed model_data path: {model_data}")
            
            # Validate model artifacts exist in S3
            s3_client = self.session.client('s3')
            try:
                # Parse S3 URI
                s3_parts = model_data.replace('s3://', '').split('/', 1)
                bucket_name = s3_parts[0]
                object_key = s3_parts[1]
                
                # Check if object exists
                s3_client.head_object(Bucket=bucket_name, Key=object_key)
                logger.info(f"Model artifacts validated at: {model_data}")
                
            except Exception as e:
                logger.error(f"Model artifacts not found at {model_data}: {e}")
                # Try alternative paths
                alternative_path = f"{self.output_s3_uri.rstrip('/')}/{estimator.latest_training_job.name}/output/model.tar.gz"
                try:
                    s3_parts = alternative_path.replace('s3://', '').split('/', 1)
                    bucket_name = s3_parts[0]  
                    object_key = s3_parts[1]
                    s3_client.head_object(Bucket=bucket_name, Key=object_key)
                    model_data = alternative_path
                    logger.info(f"Found model artifacts at alternative path: {model_data}")
                except:
                    logger.warning(f"Could not validate model artifacts, proceeding with: {model_data}")
            
            # Note: estimator.model_data is read-only and already set after training
            # Just return the estimator with its model_data property and the validated URI
            
            return estimator, model_data
            
        except Exception as e:
            logger.error(f"Training job failed: {e}")
            raise
    
    def deploy_model(self, estimator: PyTorch, model_data_uri: str) -> Predictor:
        """Deploy trained model to SageMaker endpoint using the trained estimator."""
        
        logger.info("Deploying model to SageMaker endpoint...")
        logger.info(f"Using model data from: {model_data_uri}")
        
        # Validate model data URI
        if not model_data_uri or not model_data_uri.startswith('s3://'):
            raise ValueError(f"Invalid model data URI: {model_data_uri}")
        
        # Validate estimator has completed training
        if not hasattr(estimator, 'latest_training_job') or estimator.latest_training_job is None:
            raise ValueError("Estimator is not associated with a training job")
        
        try:
            # CRITICAL: Create a new PyTorchModel with inference.py entry point
            # Cannot use estimator.deploy() directly as it inherits train.py entry point
            logger.info("Deploying with optimized timeout settings for CLIP model loading...")
            logger.info("  - Model server timeout: 300s (5 minutes)")
            logger.info("  - Container startup: 600s (10 minutes)")
            logger.info("  - Model download: 600s (10 minutes)")
            
            # Get the project root for source_dir
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            
            # Create PyTorchModel with correct inference entry point
            # NOTE: inference.py must be in code/ directory inside model.tar.gz
            # The training script now copies it there automatically
            from sagemaker.pytorch import PyTorchModel
            model = PyTorchModel(
                model_data=model_data_uri,
                role=self.role,
                entry_point='inference.py',  # CRITICAL FIX: Path relative to code/ in model.tar.gz
                framework_version='2.0.0',
                py_version='py310',
                env={
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',  # CRITICAL: SageMaker-specific timeout
                    'MODEL_SERVER_TIMEOUT': '300',  # 5 minutes per request
                    'MODEL_SERVER_WORKERS': '1',  # Single worker to reduce memory
                    'TS_MAX_REQUEST_SIZE': '100000000',  # 100MB max request
                    'TS_MAX_RESPONSE_SIZE': '100000000',  # 100MB max response
                    'TS_DEFAULT_RESPONSE_TIMEOUT': '300',  # 5 minutes timeout
                    'TS_DEFAULT_WORKERS_PER_MODEL': '1',
                    # Python/PyTorch optimizations
                    'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2',
                    'TOKENIZERS_PARALLELISM': 'false',
                }
            )
            
            # Deploy the model
            predictor = model.deploy(
                initial_instance_count=1,
                instance_type=self.inference_instance_type,
                endpoint_name=self.endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
                # Extended timeouts for CLIP model loading (cold start)
                container_startup_health_check_timeout=600,  # 10 minutes for first startup
                model_data_download_timeout=600,  # 10 minutes to download model
            )
            
            logger.info(f"Model deployed to endpoint: {self.endpoint_name}")
            logger.info("⚠️  NOTE: First inference request will take 2-3 minutes (cold start)")
            logger.info("    Subsequent requests will be fast (~1-2 seconds)")
            return predictor
            
        except Exception as e:
            logger.error(f"Direct deployment failed, trying model creation approach: {e}")
            
            # Fallback: Create PyTorch model separately then deploy
            try:
                # Get the parent directory (project root) for fallback deployment
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                
                # NOTE: inference.py should be in code/ directory inside model.tar.gz
                model = PyTorch(
                    entry_point='inference.py',  # CRITICAL FIX: Path relative to code/ in model.tar.gz
                    role=self.role,
                    model_data=model_data_uri,
                    framework_version='2.0.0',
                    py_version='py310',
                    sagemaker_session=self.sagemaker_session,
                    env={
                        'MODEL_SERVER_TIMEOUT': '300',
                        'MODEL_SERVER_WORKERS': '1',
                        'TS_MAX_REQUEST_SIZE': '100000000',
                        'TS_MAX_RESPONSE_SIZE': '100000000',
                        'TS_DEFAULT_RESPONSE_TIMEOUT': '300',
                        'TS_DEFAULT_WORKERS_PER_MODEL': '1',
                        'OMP_NUM_THREADS': '2',
                        'MKL_NUM_THREADS': '2',
                        'TOKENIZERS_PARALLELISM': 'false',
                    }
                )
                
                predictor = model.deploy(
                    initial_instance_count=1,
                    instance_type=self.inference_instance_type,
                    endpoint_name=self.endpoint_name,
                    serializer=JSONSerializer(),
                    deserializer=JSONDeserializer(),
                    container_startup_health_check_timeout=600,
                    model_data_download_timeout=600
                )
                
                logger.info(f"Model deployed via fallback method to endpoint: {self.endpoint_name}")
                return predictor
                
            except Exception as fallback_error:
                logger.error(f"Fallback deployment also failed: {fallback_error}")
                raise RuntimeError(f"All deployment methods failed. Original error: {e}, Fallback error: {fallback_error}")
    
    def test_endpoint(self, predictor: Predictor, test_image_path: str = None) -> Dict:
        """Test the deployed endpoint with a sample image."""
        
        logger.info("Testing deployed endpoint...")
        
        # Create test image if none provided
        if test_image_path is None or not os.path.exists(test_image_path):
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='red')
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                test_image.save(tmp.name)
                test_image_path = tmp.name
        
        try:
            # Load and encode image
            with open(test_image_path, 'rb') as f:
                image_data = f.read()
            
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request
            request_data = {
                'image': image_b64,
                'n_recommendations': 3
            }
            
            # Make prediction
            response = predictor.predict(request_data)
            
            logger.info("Endpoint test successful!")
            logger.info(f"Response: {json.dumps(response, indent=2)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            raise
        finally:
            # Clean up test image if created
            if test_image_path and test_image_path.endswith('.jpg'):
                try:
                    os.unlink(test_image_path)
                except:
                    pass
    
    def cleanup_resources(self, predictor: Predictor = None):
        """Clean up AWS resources to avoid ongoing charges."""
        
        logger.info("Cleaning up AWS resources...")
        
        try:
            if predictor:
                # Delete endpoint
                predictor.delete_endpoint()
                logger.info(f"Deleted endpoint: {self.endpoint_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup resources: {e}")
    
    def run_complete_pipeline(self, 
                             local_dataset_path: str = None,
                             cleanup_after: bool = None,
                             test_endpoint_flag: bool = None) -> Dict:
        """
        Run the complete SageMaker pipeline end-to-end.
        
        Args:
            local_dataset_path: Path to local dataset (will upload to S3)
            cleanup_after: Whether to cleanup resources after completion (uses config default if None)
            test_endpoint_flag: Whether to test the endpoint (uses config default if None)
            
        Returns:
            Dictionary with pipeline results and endpoint information
        """
        
        logger.info("Starting complete StyGig SageMaker pipeline...")
        
        # Use config defaults if not specified
        cleanup_after = cleanup_after if cleanup_after is not None else config.CLEANUP_AFTER_PIPELINE
        test_endpoint_flag = test_endpoint_flag if test_endpoint_flag is not None else config.TEST_ENDPOINT
        
        results = {
            'start_time': datetime.now().isoformat(),
            'dataset_s3_uri': None,
            'model_data_uri': None,
            'endpoint_name': None,
            'test_results': None,
            'status': 'started'
        }
        
        try:
            # Step 1: Prepare dataset
            results['dataset_s3_uri'] = self.prepare_dataset(local_dataset_path)
            
            # Step 2: Create and run training job
            estimator = self.create_estimator()
            trained_estimator, model_data_uri = self.launch_training_job(estimator)
            results['model_data_uri'] = model_data_uri
            
            # Step 3: Deploy model
            predictor = self.deploy_model(trained_estimator, model_data_uri)
            results['endpoint_name'] = self.endpoint_name
            
            # Step 4: Test endpoint
            if test_endpoint_flag:
                test_results = self.test_endpoint(predictor)
                results['test_results'] = test_results
            
            results['status'] = 'completed'
            results['end_time'] = datetime.now().isoformat()
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Endpoint name: {self.endpoint_name}")
            logger.info(f"Model artifacts: {model_data_uri}")
            
            # Cleanup if requested
            if cleanup_after:
                self.cleanup_resources(predictor)
            else:
                logger.info(f"Endpoint {self.endpoint_name} is still running. "
                           f"Remember to delete it to avoid charges.")
            
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            
            logger.error(f"Pipeline failed: {e}")
            
            # Try to cleanup on failure
            try:
                if 'predictor' in locals():
                    self.cleanup_resources(predictor)
            except:
                pass
            
            raise


def main():
    """Main function to run the pipeline with configuration."""
    
    try:
        # Log SDK versions for debugging
        logger.info(f"SageMaker SDK version: {sagemaker.__version__}")
        logger.info(f"Boto3 version: {boto3.__version__}")
        logger.info(f"Python version: {sys.version}")
        
        # Validate configuration first
        errors = config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
        
        # Print configuration summary in debug mode
        if config.DEBUG_MODE:
            config.print_summary()
        
        # Initialize pipeline
        pipeline = StyGigSageMakerPipeline(
            region=config.AWS_REGION,
            role=config.SAGEMAKER_ROLE,
            bucket=config.S3_BUCKET,
            dataset_s3_uri=config.DATASET_S3_URI
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            local_dataset_path=config.LOCAL_DATASET_PATH,
            cleanup_after=config.CLEANUP_AFTER_PIPELINE,
            test_endpoint_flag=config.TEST_ENDPOINT
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Dataset S3 URI: {results['dataset_s3_uri']}")
        print(f"Model Artifacts: {results['model_data_uri']}")
        print(f"Endpoint Name: {results['endpoint_name']}")
        
        if results.get('test_results'):
            print(f"Test Status: Success")
            print(f"Recommendations: {len(results['test_results'].get('recommendations', []))}")
        
        print("="*60)
        
        # Save results to file
        with open('pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Pipeline results saved to pipeline_results.json")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())