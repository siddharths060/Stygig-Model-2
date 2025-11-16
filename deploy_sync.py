#!/usr/bin/env python3
"""
StyGig V4 - Real-Time (Synchronous) Endpoint Deployer
======================================================

Deploy any existing V4 model to a REAL-TIME SageMaker endpoint.
This creates a synchronous endpoint for immediate inference results.

Usage:
    python deploy_sync.py \\
        --model-uri s3://bucket/path/to/model.tar.gz \\
        --endpoint-name stygig-production-realtime \\
        --instance-type ml.m5.large

Features:
    - Deploys pre-trained V4 models
    - Creates real-time (synchronous) endpoints
    - Configurable instance types for cost/performance optimization
    - Automatic endpoint health validation

Use Cases:
    - Production deployments requiring low latency (<200ms)
    - Interactive user-facing applications
    - A/B testing new model versions
    - Replacing existing endpoints with updated models

Author: StyGig MLOps Team
Date: November 15, 2025
Version: 4.0
"""

import argparse
import logging
import sys
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_realtime_endpoint(
    model_uri,
    endpoint_name,
    instance_type='ml.g4dn.xlarge',  # GPU required for V4 CLIP inference
    instance_count=1
):
    """
    Deploy a V4 model to a real-time SageMaker endpoint.
    
    Args:
        model_uri: S3 URI to model.tar.gz (e.g., s3://bucket/model.tar.gz)
        endpoint_name: Name for the endpoint (must be unique in region)
        instance_type: EC2 instance type for hosting (default: ml.m5.large)
        instance_count: Number of instances (default: 1)
    
    Returns:
        Deployed predictor object
    """
    logger.info("="*70)
    logger.info("StyGig V4 - Real-Time Endpoint Deployment")
    logger.info("="*70)
    
    try:
        # Get AWS configurations
        role = sagemaker.get_execution_role()
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        
        logger.info(f"Region: {region}")
        logger.info(f"Role: {role}")
        logger.info(f"Model URI: {model_uri}")
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"Instance: {instance_type} (count: {instance_count})")
        logger.info("Mode: REAL-TIME (synchronous)")
        
        # Validate model URI
        if not model_uri.startswith('s3://'):
            raise ValueError(f"Invalid model URI: {model_uri} (must start with s3://)")
        
        if not model_uri.endswith('.tar.gz'):
            raise ValueError(f"Invalid model URI: {model_uri} (must end with .tar.gz)")
        
        # Create PyTorch Model pointing to V4 inference code
        logger.info("\nCreating SageMaker Model...")
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='inference.py',
            source_dir='sagemaker',
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=sagemaker_session,
            # V4 inference environment variables
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_uri,
                'MMS_DEFAULT_RESPONSE_TIMEOUT': '500'
            }
        )
        
        logger.info("✓ Model created successfully")
        
        # Deploy to real-time endpoint
        # CRITICAL: No AsyncInferenceConfig - this creates a synchronous endpoint
        logger.info("\nDeploying endpoint (this takes 5-10 minutes)...")
        logger.info("Status updates:")
        
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=IdentitySerializer(content_type='application/x-image'),
            deserializer=JSONDeserializer(),
            wait=True  # Block until endpoint is InService
        )
        
        logger.info("="*70)
        logger.info("✅ DEPLOYMENT SUCCESSFUL")
        logger.info("="*70)
        logger.info(f"Endpoint Name: {endpoint_name}")
        logger.info(f"Endpoint ARN: {predictor.endpoint_context().endpoint_arn}")
        logger.info(f"Instance Type: {instance_type}")
        logger.info(f"Instance Count: {instance_count}")
        logger.info("Status: InService")
        logger.info("Mode: Real-time (synchronous)")
        
        # Get endpoint details
        sm_client = boto3.client('sagemaker', region_name=region)
        endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        
        logger.info("\nEndpoint Details:")
        logger.info(f"  Creation Time: {endpoint_desc['CreationTime']}")
        logger.info(f"  Last Modified: {endpoint_desc['LastModifiedTime']}")
        logger.info(f"  Endpoint Config: {endpoint_desc['EndpointConfigName']}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Test the endpoint:")
        logger.info(f"     python sagemaker/test_endpoint.py --endpoint-name {endpoint_name}")
        logger.info("  2. Monitor CloudWatch metrics:")
        logger.info(f"     aws cloudwatch get-metric-statistics --namespace AWS/SageMaker --metric-name ModelLatency --dimensions Name=EndpointName,Value={endpoint_name}")
        logger.info("  3. Invoke from application:")
        logger.info(f"     boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName='{endpoint_name}', Body=image_bytes)")
        
        return predictor
    
    except Exception as e:
        logger.error("="*70)
        logger.error(f"❌ DEPLOYMENT FAILED: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        raise


def main():
    """Parse arguments and deploy endpoint."""
    parser = argparse.ArgumentParser(
        description='Deploy V4 model to a real-time SageMaker endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy to production endpoint
  python deploy_sync.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-production-realtime

  # Deploy with larger instance for higher throughput
  python deploy_sync.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-production-realtime \\
      --instance-type ml.m5.xlarge

  # Deploy with multiple instances for load balancing
  python deploy_sync.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-production-realtime \\
      --instance-type ml.m5.large \\
      --instance-count 2

Instance Type Recommendations:
  - ml.t2.medium:  Development/testing (lowest cost, ~$0.065/hr)
  - ml.m5.large:   Production (balanced, ~$0.115/hr)
  - ml.m5.xlarge:  High traffic (2x throughput, ~$0.23/hr)
  - ml.c5.2xlarge: Low latency (compute optimized, ~$0.408/hr)
        """
    )
    
    parser.add_argument(
        '--model-uri',
        type=str,
        required=True,
        help='S3 URI to model.tar.gz (e.g., s3://bucket/path/to/model.tar.gz)'
    )
    
    parser.add_argument(
        '--endpoint-name',
        type=str,
        required=True,
        help='Name for the SageMaker endpoint (e.g., stygig-production-realtime)'
    )
    
    parser.add_argument(
        '--instance-type',
        type=str,
        default='ml.g4dn.xlarge',
        help='EC2 instance type for hosting (default: ml.g4dn.xlarge, GPU required for V4)'
    )
    
    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Number of instances for load balancing (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate endpoint name
    if len(args.endpoint_name) > 63:
        logger.error("❌ Endpoint name must be ≤63 characters")
        sys.exit(1)
    
    if not args.endpoint_name.replace('-', '').replace('_', '').isalnum():
        logger.error("❌ Endpoint name can only contain alphanumeric characters, hyphens, and underscores")
        sys.exit(1)
    
    # Deploy endpoint
    try:
        predictor = deploy_realtime_endpoint(
            model_uri=args.model_uri,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count
        )
        
        logger.info("\n✅ Endpoint is ready for inference!")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
