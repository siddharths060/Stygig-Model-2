#!/usr/bin/env python3
"""
Redeploy StyGig Endpoint with Fixed Timeouts

This script deletes the old endpoint and creates a new one with:
- Increased container startup timeout (5 minutes)
- Increased model download timeout (5 minutes)  
- Same trained model artifacts
- Optimized inference code

Usage:
    python redeploy_endpoint.py
    python redeploy_endpoint.py --endpoint-name stygig-endpoint-20251103-062336
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker import Session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def delete_endpoint(endpoint_name, region='us-east-1'):
    """Delete existing endpoint and its configuration."""
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            # Delete endpoint
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info("✓ Endpoint deletion initiated")
            
            # Wait for deletion
            waiter = sagemaker_client.get_waiter('endpoint_deleted')
            logger.info("Waiting for endpoint to be deleted...")
            waiter.wait(EndpointName=endpoint_name)
            logger.info("✓ Endpoint deleted")
            
        except sagemaker_client.exceptions.ClientError as e:
            if 'Could not find endpoint' in str(e):
                logger.info("Endpoint already deleted or doesn't exist")
            else:
                raise
        
        # Delete endpoint configuration
        try:
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info("✓ Endpoint configuration deleted")
        except sagemaker_client.exceptions.ClientError as e:
            if 'Could not find endpoint configuration' in str(e):
                logger.info("Endpoint config already deleted or doesn't exist")
            else:
                logger.warning(f"Failed to delete endpoint config: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete endpoint: {e}")
        return False

def redeploy_with_timeouts(model_uri, endpoint_name=None, instance_type='ml.m5.large', region='us-east-1'):
    """Redeploy endpoint with increased timeouts."""
    try:
        session = Session(boto_session=boto3.Session(region_name=region))
        
        # Get execution role
        sts_client = boto3.client('sts', region_name=region)
        account_id = sts_client.get_caller_identity()['Account']
        
        # Find SageMaker execution role
        iam_client = boto3.client('iam')
        paginator = iam_client.get_paginator('list_roles')
        role = None
        for page in paginator.paginate():
            for iam_role in page['Roles']:
                if 'AmazonSageMaker-ExecutionRole' in iam_role['RoleName']:
                    role = iam_role['Arn']
                    break
            if role:
                break
        
        if not role:
            raise ValueError("Could not find SageMaker execution role")
        
        logger.info(f"Using IAM role: {role}")
        
        # Generate new endpoint name
        if not endpoint_name:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f'stygig-endpoint-{timestamp}'
        
        logger.info(f"Creating new endpoint: {endpoint_name}")
        
        # Get project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Create PyTorch model with extended timeouts
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='sagemaker/inference.py',
            source_dir=project_root,
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=session,
            # Extended timeouts for model loading
            model_server_workers=1,  # Single worker to reduce memory
            env={
                'MODEL_SERVER_TIMEOUT': '180',  # 3 minutes for model loading
                'MODEL_SERVER_WORKERS': '1',
                'TS_MAX_REQUEST_SIZE': '100000000',  # 100MB max request
                'TS_MAX_RESPONSE_SIZE': '100000000',
                'TS_DEFAULT_RESPONSE_TIMEOUT': '180'
            }
        )
        
        logger.info("Deploying model (this takes 5-10 minutes)...")
        logger.info(f"   Instance type: {instance_type}")
        logger.info(f"   Model URI: {model_uri}")
        logger.info("   Timeouts: 180s (container), 300s (client)")
        
        # Deploy with increased timeouts
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            # Health check and startup timeouts
            container_startup_health_check_timeout=300,  # 5 minutes
            model_data_download_timeout=300  # 5 minutes for S3 download
        )
        
        logger.info(f"✅ Endpoint deployed successfully!")
        logger.info(f"   Endpoint name: {endpoint_name}")
        logger.info(f"   Instance type: {instance_type}")
        logger.info(f"   Model URI: {model_uri}")
        
        # Save endpoint info
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'model_uri': model_uri,
            'instance_type': instance_type,
            'region': region,
            'timeouts': {
                'container': 180,
                'startup': 300,
                'download': 300
            }
        }
        
        with open('endpoint_info.json', 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info(f"\n📄 Endpoint info saved to: endpoint_info.json")
        
        return predictor, endpoint_name
        
    except Exception as e:
        logger.error(f"Redeployment failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Redeploy StyGig endpoint with fixed timeouts')
    parser.add_argument('--endpoint-name', type=str, default='stygig-endpoint-20251103-062336',
                       help='Existing endpoint name to replace')
    parser.add_argument('--model-uri', type=str,
                       default='s3://stygig-ml-s3/model-artifacts/stygig-training-1762145223/output/model.tar.gz',
                       help='S3 URI to model artifacts')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                       help='Instance type for endpoint')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    parser.add_argument('--skip-delete', action='store_true',
                       help='Skip deleting old endpoint (create new with different name)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("StyGig Endpoint Redeployment - With Fixed Timeouts")
    logger.info("=" * 80)
    logger.info("")
    
    # Delete old endpoint unless skipped
    if not args.skip_delete:
        logger.info(f"Step 1: Deleting old endpoint...")
        delete_endpoint(args.endpoint_name, args.region)
        logger.info("")
    
    # Redeploy with new configuration
    logger.info(f"Step 2: Deploying new endpoint with optimized settings...")
    new_endpoint_name = args.endpoint_name if not args.skip_delete else None
    
    predictor, endpoint_name = redeploy_with_timeouts(
        model_uri=args.model_uri,
        endpoint_name=new_endpoint_name,
        instance_type=args.instance_type,
        region=args.region
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Redeployment Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Test endpoint: python test_endpoint.py --save-visual")
    logger.info(f"  2. Endpoint name: {endpoint_name}")
    logger.info(f"  3. First request will take 1-2 minutes (CLIP model loading)")
    logger.info(f"  4. Subsequent requests: ~1-2 seconds")
    logger.info("")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
