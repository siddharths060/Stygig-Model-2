#!/usr/bin/env python3
"""
Deploy Existing StyGig Model to SageMaker Endpoint

This script deploys an already-trained model without retraining.
Use this when you have a successful training job and just want to deploy.

Usage:
    python deploy_existing_model.py --model-uri s3://bucket/path/model.tar.gz
    python deploy_existing_model.py --training-job-name stygig-training-1762145223
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

def get_model_uri_from_training_job(job_name, region='us-east-1'):
    """Get model artifact S3 URI from training job name."""
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Get training job details
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        
        model_uri = response['ModelArtifacts']['S3ModelArtifacts']
        logger.info(f"Found model artifacts: {model_uri}")
        
        return model_uri
    except Exception as e:
        logger.error(f"Failed to get model URI from training job: {e}")
        raise

def deploy_model(model_uri, endpoint_name=None, instance_type='ml.m5.large', region='us-east-1'):
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
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='sagemaker/inference.py',
            source_dir=project_root,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=session
        )
        
        logger.info("Deploying model to endpoint (this may take 5-10 minutes)...")
        
        # Deploy to endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
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
                       default='stygig-training-1762145223',
                       help='Training job name to get model URI from')
    parser.add_argument('--endpoint-name', type=str, help='Custom endpoint name')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                       help='Instance type for endpoint')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    
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
        model_uri = get_model_uri_from_training_job(args.training_job_name, args.region)
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
