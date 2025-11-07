#!/usr/bin/env python3
"""
Deploy Existing StyGig Model to SageMaker Endpoint

This script deploys an already-trained model without retraining.
Use this when you have a successful training job and just want to deploy.

Usage:
    python deploy_existing_model.py --model-uri s3://bucket/path/model.tar.gz
    python deploy_existing_model.py --training-job-name stygig-training-1762498189
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))

import boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker import Session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model_uri_from_training_job(job_name, region='ap-south-1', training_region=None):
    """Get model artifact S3 URI from training job name.
    
    Args:
        job_name: Training job name
        region: Region to search for training job (defaults to ap-south-1)
        training_region: Alternative region if training was in different region
    """
    # Try the specified region first
    regions_to_try = [region]
    if training_region and training_region != region:
        regions_to_try.insert(0, training_region)
    
    for try_region in regions_to_try:
        try:
            sagemaker_client = boto3.client('sagemaker', region_name=try_region)
            
            # Get training job details
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            model_uri = response['ModelArtifacts']['S3ModelArtifacts']
            logger.info(f"Found model artifacts in {try_region}: {model_uri}")
            
            return model_uri
        except Exception as e:
            logger.debug(f"Training job not found in {try_region}: {e}")
            continue
    
    # If not found in any region, raise error
    raise RuntimeError(f"Training job {job_name} not found in regions: {regions_to_try}")

def deploy_model(model_uri, endpoint_name=None, instance_type='ml.m5.large', region='ap-south-1'):
    """Deploy model from S3 URI to SageMaker endpoint."""
    try:
        # Get execution role
        session = Session(boto_session=boto3.Session(region_name=region))
        role = os.environ.get('SAGEMAKER_ROLE')
        
        if not role:
            # Try to get from SageMaker execution role
            sts_client = boto3.client('sts', region_name=region)
            account_id = sts_client.get_caller_identity()['Account']
            role = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-*"
            
            # Find the actual role
            iam_client = boto3.client('iam')
            paginator = iam_client.get_paginator('list_roles')
            for page in paginator.paginate():
                for iam_role in page['Roles']:
                    if 'AmazonSageMaker-ExecutionRole' in iam_role['RoleName']:
                        role = iam_role['Arn']
                        break
                if role and role.startswith('arn:'):
                    break
        
        logger.info(f"Using IAM role: {role}")
        
        # Generate endpoint name if not provided
        if not endpoint_name:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f'stygig-endpoint-{timestamp}'
        
        logger.info(f"Creating endpoint: {endpoint_name}")
        
        # Get the parent directory (project root) for source code
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Create PyTorch model with extended timeouts for CLIP model loading
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='sagemaker/inference.py',
            source_dir=project_root,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=session,
            model_server_workers=1,  # Single worker to reduce memory usage
            env={
                # Extended timeout for CLIP model loading
                'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',  # CRITICAL: SageMaker-specific timeout
                'MODEL_SERVER_TIMEOUT': '300',  # 5 minutes per request
                'MODEL_SERVER_WORKERS': '1',
                'TS_MAX_REQUEST_SIZE': '100000000',  # 100MB
                'TS_MAX_RESPONSE_SIZE': '100000000',
                'TS_DEFAULT_RESPONSE_TIMEOUT': '300',
                'TS_DEFAULT_WORKERS_PER_MODEL': '1',
                # Optimization flags
                'OMP_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2',
                'TOKENIZERS_PARALLELISM': 'false',
            }
        )
        
        logger.info("Deploying model with EXTENDED TIMEOUTS (this may take 5-10 minutes)...")
        logger.info("  Container startup: 600s (10 minutes)")
        logger.info("  Model download: 600s (10 minutes)")
        logger.info("  Model server: 300s (5 minutes per request)")
        
        # Deploy to endpoint with extended timeouts
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            # CRITICAL: Extended timeouts for CLIP model loading (cold start)
            container_startup_health_check_timeout=600,  # 10 minutes for first startup
            model_data_download_timeout=600  # 10 minutes to download model
        )
        
        logger.info(f"✅ Model deployed successfully!")
        logger.info(f"   Endpoint name: {endpoint_name}")
        logger.info(f"   Instance type: {instance_type}")
        logger.info(f"   Model URI: {model_uri}")
        
        return predictor, endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Deploy existing StyGig model to SageMaker')
    parser.add_argument('--model-uri', type=str, help='S3 URI to model.tar.gz')
    parser.add_argument('--training-job-name', type=str, 
                       default='stygig-training-1762498189',
                       help='Training job name to get model URI from')
    parser.add_argument('--endpoint-name', type=str, help='Custom endpoint name')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                       help='Instance type for endpoint')
    parser.add_argument('--region', type=str, default='ap-south-1',
                       help='AWS region for endpoint deployment')
    parser.add_argument('--training-region', type=str, default='ap-south-1',
                       help='AWS region where training job was created')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("StyGig Model Deployment - Existing Model")
    logger.info("=" * 80)
    
    # Get model URI
    if args.model_uri:
        model_uri = args.model_uri
        logger.info(f"Using provided model URI: {model_uri}")
    elif args.training_job_name:
        logger.info(f"Getting model URI from training job: {args.training_job_name}")
        logger.info(f"Searching in regions: {args.training_region}, {args.region}")
        model_uri = get_model_uri_from_training_job(
            args.training_job_name, 
            args.region,
            args.training_region
        )
    else:
        logger.error("Must provide either --model-uri or --training-job-name")
        return 1
    
    # Deploy model
    predictor, endpoint_name = deploy_model(
        model_uri=model_uri,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=args.region
    )
    
    # Save endpoint info
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'model_uri': model_uri,
        'instance_type': args.instance_type,
        'region': args.region
    }
    
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    logger.info(f"\n📄 Endpoint info saved to: endpoint_info.json")
    logger.info(f"\n🎉 Deployment complete! Your endpoint is ready for inference.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
