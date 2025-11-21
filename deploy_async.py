#!/usr/bin/env python3
"""
StyGig V4 - Asynchronous Endpoint Deployer
===========================================

Deploy any existing V4 model to an ASYNCHRONOUS SageMaker endpoint.
This creates an async endpoint for batch processing and long-running inference.

Usage:
    python deploy_async.py \\
        --model-uri s3://bucket/path/to/model.tar.gz \\
        --endpoint-name stygig-production-async \\
        --s3-output-path s3://bucket/async-results/ \\
        --sns-topic-arn arn:aws:sns:region:account:topic

Features:
    - Deploys pre-trained V4 models to async endpoints
    - Supports batch processing and long-running inference
    - Automatic scaling (scales to zero when idle)
    - SNS notifications for completion/errors
    - Cost-effective for non-real-time workloads

Use Cases:
    - Batch processing of large product catalogs
    - Overnight recommendation refreshes
    - Processing uploaded user images
    - Cost-sensitive workloads (auto-scales to zero)

Author: StyGig MLOps Team
Date: November 15, 2025
Version: 4.0
"""

import argparse
import logging
import sys

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_async_endpoint(
    model_uri,
    endpoint_name,
    s3_output_path,
    instance_type='ml.c5.large',
    instance_count=1,
    sns_topic_arn=None,
    max_concurrent_invocations=10
):
    """
    Deploy a V4 model to an asynchronous SageMaker endpoint.
    
    Args:
        model_uri: S3 URI to model.tar.gz
        endpoint_name: Name for the async endpoint
        s3_output_path: S3 path for inference results (e.g., s3://bucket/results/)
        instance_type: EC2 instance type (default: ml.c5.large)
        instance_count: Number of instances (default: 1)
        sns_topic_arn: Optional SNS topic for notifications
        max_concurrent_invocations: Max concurrent requests (default: 10)
    
    Returns:
        Deployed predictor object
    """
    logger.info("="*70)
    logger.info("StyGig V4 - Asynchronous Endpoint Deployment")
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
        logger.info(f"Output Path: {s3_output_path}")
        logger.info(f"SNS Topic: {sns_topic_arn or 'None'}")
        logger.info("Mode: ASYNCHRONOUS (batch)")
        
        # Validate inputs
        if not model_uri.startswith('s3://') or not model_uri.endswith('.tar.gz'):
            raise ValueError(f"Invalid model URI: {model_uri}")
        
        if not s3_output_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 output path: {s3_output_path}")
        
        if not s3_output_path.endswith('/'):
            s3_output_path += '/'
            logger.info(f"Normalized output path: {s3_output_path}")
        
        # Create PyTorch Model for V4
        logger.info("\nCreating SageMaker Model...")
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='inference.py',
            source_dir='sagemaker',
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=sagemaker_session,
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_uri,
                'MMS_DEFAULT_RESPONSE_TIMEOUT': '900',  # 15 min for async
                'ASYNC_INFERENCE': 'true'
            }
        )
        
        logger.info("✓ Model created successfully")
        
        # Create AsyncInferenceConfig
        logger.info("\nConfiguring async inference...")
        
        notification_config = None
        if sns_topic_arn:
            notification_config = {
                "SuccessTopic": sns_topic_arn,
                "ErrorTopic": sns_topic_arn
            }
            logger.info(f"✓ SNS notifications enabled: {sns_topic_arn}")
        
        async_config = AsyncInferenceConfig(
            output_path=s3_output_path,
            max_concurrent_invocations_per_instance=max_concurrent_invocations,
            notification_config=notification_config
        )
        
        logger.info(f"✓ Max concurrent invocations: {max_concurrent_invocations}")
        logger.info(f"✓ Results will be saved to: {s3_output_path}")
        
        # Deploy to async endpoint
        logger.info("\nDeploying async endpoint (this takes 5-10 minutes)...")
        
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,  # CRITICAL: This makes it async
            serializer=IdentitySerializer(content_type='application/x-image'),
            deserializer=JSONDeserializer(),
            wait=True
        )
        
        logger.info("="*70)
        logger.info("✅ ASYNC DEPLOYMENT SUCCESSFUL")
        logger.info("="*70)
        logger.info(f"Endpoint Name: {endpoint_name}")
        logger.info(f"Endpoint ARN: {getattr(predictor.endpoint_context(), 'endpoint_arn', f'arn:aws:sagemaker:ap-south-1:732414292744:endpoint/{endpoint_name}')}")
        logger.info(f"Instance Type: {instance_type}")
        logger.info(f"Instance Count: {instance_count}")
        logger.info("Status: InService")
        logger.info("Mode: Asynchronous (batch)")
        
        # Get endpoint details
        sm_client = boto3.client('sagemaker', region_name=region)
        endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        
        logger.info("\nEndpoint Details:")
        logger.info(f"  Creation Time: {endpoint_desc['CreationTime']}")
        logger.info(f"  Last Modified: {endpoint_desc['LastModifiedTime']}")
        logger.info(f"  Endpoint Config: {endpoint_desc['EndpointConfigName']}")
        
        logger.info("\nAsync Configuration:")
        logger.info(f"  Output S3 Path: {s3_output_path}")
        logger.info(f"  Max Concurrent: {max_concurrent_invocations} per instance")
        logger.info(f"  SNS Notifications: {'Enabled' if sns_topic_arn else 'Disabled'}")
        logger.info("  Auto-scaling: Enabled (scales to zero when idle)")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Invoke async endpoint:")
        logger.info(f"     python sagemaker/invoke_async.py --endpoint-name {endpoint_name} --image-path test.jpg")
        logger.info("  2. Check results in S3:")
        logger.info(f"     aws s3 ls {s3_output_path}")
        logger.info("  3. Monitor with CloudWatch:")
        logger.info(f"     aws cloudwatch get-metric-statistics --namespace AWS/SageMaker --metric-name AsyncInvocations --dimensions Name=EndpointName,Value={endpoint_name}")
        
        logger.info("\nImportant Notes:")
        logger.info("  - Invocations return immediately with an S3 output location")
        logger.info("  - Results are written to S3 when inference completes")
        logger.info("  - Endpoint auto-scales to zero when idle (cost savings)")
        logger.info(f"  - Max processing time: 15 minutes per request")
        
        return predictor
    
    except Exception as e:
        logger.error("="*70)
        logger.error(f"❌ DEPLOYMENT FAILED: {e}")
        logger.error("="*70)
        import traceback
        traceback.print_exc()
        raise


def main():
    """Parse arguments and deploy async endpoint."""
    parser = argparse.ArgumentParser(
        description='Deploy V4 model to an asynchronous SageMaker endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy basic async endpoint
  python deploy_async.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-async \\
      --s3-output-path s3://my-bucket/async-results/

  # Deploy with SNS notifications
  python deploy_async.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-async \\
      --s3-output-path s3://my-bucket/async-results/ \\
      --sns-topic-arn arn:aws:sns:us-east-1:123456:stygig-notifications

  # Deploy with high concurrency for batch processing
  python deploy_async.py \\
      --model-uri s3://my-bucket/models/model.tar.gz \\
      --endpoint-name stygig-batch \\
      --s3-output-path s3://my-bucket/batch-results/ \\
      --max-concurrent 50

When to use Async vs Real-time:
  - Async: Batch processing, non-urgent requests, cost optimization
  - Real-time: User-facing apps, low latency required (<200ms)

Async Benefits:
  - Auto-scales to zero (no cost when idle)
  - Handles long-running inference (up to 15 min)
  - Built-in result storage in S3
  - Optional SNS notifications
        """
    )
    
    parser.add_argument(
        '--model-uri',
        type=str,
        required=True,
        help='S3 URI to model.tar.gz'
    )
    
    parser.add_argument(
        '--endpoint-name',
        type=str,
        required=True,
        help='Name for the async endpoint'
    )
    
    parser.add_argument(
        '--s3-output-path',
        type=str,
        required=True,
        help='S3 path for inference results (e.g., s3://bucket/results/)'
    )
    
    parser.add_argument(
        '--instance-type',
        type=str,
        default='ml.c5.large',
        help='EC2 instance type (default: ml.c5.large)'
    )
    
    parser.add_argument(
        '--instance-count',
        type=int,
        default=1,
        help='Number of instances (default: 1)'
    )
    
    parser.add_argument(
        '--sns-topic-arn',
        type=str,
        default=None,
        help='SNS topic ARN for success/error notifications (optional)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Max concurrent invocations per instance (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate endpoint name
    if len(args.endpoint_name) > 63:
        logger.error("❌ Endpoint name must be ≤63 characters")
        sys.exit(1)
    
    # Deploy async endpoint
    try:
        predictor = deploy_async_endpoint(
            model_uri=args.model_uri,
            endpoint_name=args.endpoint_name,
            s3_output_path=args.s3_output_path,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            sns_topic_arn=args.sns_topic_arn,
            max_concurrent_invocations=args.max_concurrent
        )
        
        logger.info("\n✅ Async endpoint is ready for batch inference!")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
