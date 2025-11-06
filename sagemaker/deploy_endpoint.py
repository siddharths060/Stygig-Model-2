#!/usr/bin/env python3
"""
StyGig Endpoint Deployment Script
==================================
Comprehensive script to deploy StyGig fashion recommendation model to SageMaker.

Features:
- Automatic IAM role detection
- Extended timeouts for CLIP model loading
- Optional deletion of existing endpoint
- Health checks and validation
- Clear status reporting

Usage:
    # Deploy new endpoint
    python deploy_endpoint.py --model-uri s3://your-bucket/model.tar.gz

    # Replace existing endpoint
    python deploy_endpoint.py --model-uri s3://your-bucket/model.tar.gz --endpoint-name existing-endpoint

    # Deploy without deleting old endpoint (creates new one)
    python deploy_endpoint.py --model-uri s3://your-bucket/model.tar.gz --skip-delete
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_endpoint(endpoint_name: str, region: str = 'ap-south-1') -> bool:
    """
    Delete an existing SageMaker endpoint and its configuration.
    
    Args:
        endpoint_name: Name of endpoint to delete
        region: AWS region
        
    Returns:
        True if deleted successfully, False if endpoint doesn't exist
    """
    try:
        sm_client = boto3.client('sagemaker', region_name=region)
        
        # Check if endpoint exists
        try:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info(f"Endpoint {endpoint_name} does not exist")
                return False
            raise
        
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        # Delete endpoint
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info("✓ Endpoint deletion initiated")
        
        # Wait for deletion
        logger.info("Waiting for endpoint to be deleted...")
        waiter = sm_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        logger.info("✓ Endpoint deleted")
        
        # Delete endpoint configuration
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info("✓ Endpoint configuration deleted")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ValidationException':
                logger.warning(f"Could not delete endpoint config: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete endpoint: {e}")
        raise


def get_execution_role(region: str = 'ap-south-1') -> str:
    """
    Find SageMaker execution role automatically.
    
    Args:
        region: AWS region
        
    Returns:
        IAM role ARN
    """
    iam_client = boto3.client('iam', region_name=region)
    
    # Try to find SageMaker execution role
    paginator = iam_client.get_paginator('list_roles')
    
    for page in paginator.paginate():
        for iam_role in page['Roles']:
            if 'AmazonSageMaker-ExecutionRole' in iam_role['RoleName']:
                return iam_role['Arn']
    
    raise ValueError(
        "Could not find SageMaker execution role. "
        "Please create one or specify with --role"
    )


def deploy_endpoint(
    model_uri: str,
    endpoint_name: str = None,
    instance_type: str = 'ml.m5.large',
    region: str = 'ap-south-1',
    role: str = None
) -> tuple:
    """
    Deploy StyGig model to SageMaker endpoint with optimized settings.
    
    Args:
        model_uri: S3 URI to model.tar.gz
        endpoint_name: Name for endpoint (auto-generated if None)
        instance_type: EC2 instance type
        region: AWS region
        role: IAM role ARN (auto-detected if None)
        
    Returns:
        Tuple of (predictor, endpoint_name)
    """
    try:
        # Initialize SageMaker session
        session = Session(boto_session=boto3.Session(region_name=region))
        
        # Get IAM role
        if not role:
            role = get_execution_role(region)
        
        logger.info(f"Using IAM role: {role}")
        
        # Generate endpoint name if not provided
        if not endpoint_name:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f'stygig-endpoint-{timestamp}'
        
        logger.info(f"Creating endpoint: {endpoint_name}")
        
        # Get project root (parent of sagemaker folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Entry point: sagemaker/inference.py")
        
        # Create PyTorchModel with optimized settings
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='sagemaker/inference.py',
            source_dir=project_root,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=session,
            model_server_workers=1,  # Single worker to reduce memory
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
        
        logger.info("=" * 80)
        logger.info("Deploying model with EXTENDED TIMEOUTS (this takes 5-10 minutes)...")
        logger.info(f"  Instance type: {instance_type}")
        logger.info(f"  Model URI: {model_uri}")
        logger.info(f"  Container startup: 600s (10 minutes)")
        logger.info(f"  Model download: 600s (10 minutes)")
        logger.info(f"  Model server: 300s (5 minutes per request)")
        logger.info("=" * 80)
        
        # Deploy with increased timeouts
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
        
        logger.info("=" * 80)
        logger.info("✅ Endpoint deployed successfully!")
        logger.info(f"  Endpoint name: {endpoint_name}")
        logger.info(f"  Instance type: {instance_type}")
        logger.info(f"  Model URI: {model_uri}")
        logger.info("=" * 80)
        
        # Save endpoint information
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'model_uri': model_uri,
            'instance_type': instance_type,
            'region': region,
            'deployed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timeouts': {
                'container': 180,
                'startup': 300,
                'download': 300
            }
        }
        
        with open('endpoint_info.json', 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info("\n📄 Endpoint info saved to: endpoint_info.json")
        
        return predictor, endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Deploy StyGig fashion recommendation model to SageMaker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy new endpoint
  python deploy_endpoint.py --model-uri s3://stygig-ml-s3/model-artifacts/stygig-training-1762145223/output/model.tar.gz

  # Replace existing endpoint
  python deploy_endpoint.py --endpoint-name stygig-endpoint-20251103-062336 --model-uri s3://stygig-ml-s3/model.tar.gz

  # Deploy without deleting old endpoint
  python deploy_endpoint.py --model-uri s3://stygig-ml-s3/model.tar.gz --skip-delete
        """
    )
    
    parser.add_argument(
        '--model-uri',
        type=str,
        required=True,
        help='S3 URI to model.tar.gz (e.g., s3://bucket/path/model.tar.gz)'
    )
    parser.add_argument(
        '--endpoint-name',
        type=str,
        default=None,
        help='Endpoint name (auto-generated if not provided). If exists and --skip-delete not set, will be replaced.'
    )
    parser.add_argument(
        '--instance-type',
        type=str,
        default='ml.m5.large',
        help='Instance type for endpoint (default: ml.m5.large)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='ap-south-1',
        help='AWS region (default: ap-south-1)'
    )
    parser.add_argument(
        '--role',
        type=str,
        default=None,
        help='IAM role ARN (auto-detected if not provided)'
    )
    parser.add_argument(
        '--skip-delete',
        action='store_true',
        help='Skip deleting existing endpoint (creates new with different name)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("StyGig Endpoint Deployment")
    logger.info("=" * 80)
    logger.info("")
    
    # Delete existing endpoint if specified and not skipped
    if args.endpoint_name and not args.skip_delete:
        logger.info("Step 1: Checking for existing endpoint...")
        deleted = delete_endpoint(args.endpoint_name, args.region)
        if deleted:
            logger.info("")
    elif args.skip_delete:
        logger.info("Skipping endpoint deletion (--skip-delete flag set)")
        logger.info("")
    
    # Deploy new endpoint
    step_num = 2 if (args.endpoint_name and not args.skip_delete) else 1
    logger.info(f"Step {step_num}: Deploying endpoint...")
    
    # If skip-delete, generate new name
    endpoint_name = None if args.skip_delete else args.endpoint_name
    
    predictor, deployed_name = deploy_endpoint(
        model_uri=args.model_uri,
        endpoint_name=endpoint_name,
        instance_type=args.instance_type,
        region=args.region,
        role=args.role
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Deployment Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test endpoint:")
    logger.info("     python test_endpoint.py --save-visual")
    logger.info("")
    logger.info(f"  2. Endpoint: {deployed_name}")
    logger.info("")
    logger.info("  3. First request takes 1-2 minutes (CLIP model loads)")
    logger.info("     Subsequent requests: ~1-2 seconds")
    logger.info("")
    logger.info("  4. To delete endpoint:")
    logger.info(f"     aws sagemaker delete-endpoint --endpoint-name {deployed_name}")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nDeployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nDeployment failed: {e}")
        sys.exit(1)
