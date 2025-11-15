#!/usr/bin/env python3
"""
StyGig Asynchronous Inference Endpoint Deployment
=================================================
Deploy StyGig fashion recommendation model using SageMaker Asynchronous Inference
to handle long-running predictions (CLIP model cold start > 60 seconds).

Asynchronous Inference Benefits:
- Handles predictions that exceed 60-second timeout limit
- Scales to zero when not in use (cost-effective)
- Queues requests during high traffic
- Returns results via S3 + SNS notifications

Usage:
    # Deploy new async endpoint
    python deploy_async_endpoint.py --model-uri s3://your-bucket/model.tar.gz

    # Deploy with custom SNS topic
    python deploy_async_endpoint.py --model-uri s3://bucket/model.tar.gz \
        --sns-topic-arn arn:aws:sns:region:account:topic-name

    # Deploy with custom S3 output path
    python deploy_async_endpoint.py --model-uri s3://bucket/model.tar.gz \
        --s3-output-path s3://your-bucket/async-inference-results/
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_REGION = 'ap-south-1'
DEFAULT_INSTANCE_TYPE = 'ml.m5.large'
DEFAULT_MAX_CONCURRENT_INVOCATIONS = 5


def get_execution_role(region: str = DEFAULT_REGION) -> str:
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


def create_sns_topic(topic_name: str, region: str = DEFAULT_REGION) -> str:
    """
    Create SNS topic for async inference notifications.
    
    Args:
        topic_name: Name for the SNS topic
        region: AWS region
        
    Returns:
        SNS topic ARN
    """
    sns_client = boto3.client('sns', region_name=region)
    
    try:
        response = sns_client.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        logger.info(f"✓ Created SNS topic: {topic_arn}")
        return topic_arn
    except ClientError as e:
        if e.response['Error']['Code'] == 'TopicAlreadyExists':
            # Get existing topic ARN
            response = sns_client.create_topic(Name=topic_name)
            logger.info(f"✓ Using existing SNS topic: {response['TopicArn']}")
            return response['TopicArn']
        raise


def ensure_s3_bucket_exists(bucket_name: str, region: str = DEFAULT_REGION) -> None:
    """
    Ensure S3 bucket exists, create if not.
    
    Args:
        bucket_name: Name of S3 bucket
        region: AWS region
    """
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"✓ S3 bucket exists: {bucket_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"Creating S3 bucket: {bucket_name}")
            if region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            logger.info(f"✓ Created S3 bucket: {bucket_name}")
        else:
            raise


def delete_endpoint(endpoint_name: str, region: str = DEFAULT_REGION) -> bool:
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


def deploy_async_endpoint(
    model_uri: str,
    endpoint_name: str = None,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    region: str = DEFAULT_REGION,
    role: str = None,
    s3_output_path: str = None,
    sns_topic_arn: str = None,
    max_concurrent_invocations: int = DEFAULT_MAX_CONCURRENT_INVOCATIONS
) -> str:
    """
    Deploy StyGig model to SageMaker Asynchronous Inference endpoint.
    
    Args:
        model_uri: S3 URI to model.tar.gz
        endpoint_name: Name for endpoint (auto-generated if None)
        instance_type: EC2 instance type
        region: AWS region
        role: IAM role ARN (auto-detected if None)
        s3_output_path: S3 path for async inference results
        sns_topic_arn: SNS topic ARN for notifications (auto-created if None)
        max_concurrent_invocations: Max concurrent async invocations
        
    Returns:
        Endpoint name
    """
    try:
        sm_client = boto3.client('sagemaker', region_name=region)
        
        # Get IAM role
        if not role:
            role = get_execution_role(region)
        
        logger.info(f"Using IAM role: {role}")
        
        # Generate endpoint name if not provided
        if not endpoint_name:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f'stygig-async-endpoint-{timestamp}'
        
        logger.info(f"Creating async endpoint: {endpoint_name}")
        
        # Setup S3 output path
        if not s3_output_path:
            s3_output_bucket = 'stygig-ml-s3'
            ensure_s3_bucket_exists(s3_output_bucket, region)
            s3_output_path = f's3://{s3_output_bucket}/async-inference-results/'
        
        logger.info(f"Async inference results will be saved to: {s3_output_path}")
        
        # Setup SNS topic for notifications
        if not sns_topic_arn:
            sns_topic_name = f'stygig-async-inference-notifications'
            sns_topic_arn = create_sns_topic(sns_topic_name, region)
        
        logger.info(f"Using SNS topic for notifications: {sns_topic_arn}")
        
        # Get project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        logger.info(f"Project root: {project_root}")
        
        # Parse model URI to get bucket and key
        model_uri_parts = model_uri.replace('s3://', '').split('/', 1)
        model_bucket = model_uri_parts[0]
        model_key = model_uri_parts[1]
        
        # Step 1: Create Model
        model_name = f'{endpoint_name}-model'
        logger.info(f"Creating SageMaker model: {model_name}")
        
        # Get PyTorch DLC image URI
        from sagemaker import image_uris
        image_uri = image_uris.retrieve(
            framework='pytorch',
            region=region,
            version='2.0.0',
            py_version='py310',
            instance_type=instance_type,
            image_scope='inference'
        )
        
        # Package source code
        import tarfile
        import tempfile
        
        source_tar_path = os.path.join(tempfile.gettempdir(), 'source.tar.gz')
        with tarfile.open(source_tar_path, 'w:gz') as tar:
            # Add inference.py from sagemaker directory
            inference_path = os.path.join(current_dir, 'inference.py')
            tar.add(inference_path, arcname='inference.py')
            
            # Add config and src directories
            config_dir = os.path.join(project_root, 'config')
            src_dir = os.path.join(project_root, 'src')
            
            if os.path.exists(config_dir):
                tar.add(config_dir, arcname='config')
            if os.path.exists(src_dir):
                tar.add(src_dir, arcname='src')
        
        # Upload source code to S3
        s3_client = boto3.client('s3', region_name=region)
        source_s3_key = f'async-inference-code/{endpoint_name}/source.tar.gz'
        s3_client.upload_file(source_tar_path, model_bucket, source_s3_key)
        source_s3_uri = f's3://{model_bucket}/{source_s3_key}'
        
        logger.info(f"Uploaded source code to: {source_s3_uri}")
        
        create_model_response = sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_uri,
                'Environment': {
                    # Extended timeout for CLIP model loading
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',
                    'MODEL_SERVER_TIMEOUT': '300',
                    'MODEL_SERVER_WORKERS': '1',
                    'TS_MAX_REQUEST_SIZE': '100000000',
                    'TS_MAX_RESPONSE_SIZE': '100000000',
                    'TS_DEFAULT_RESPONSE_TIMEOUT': '300',
                    'TS_DEFAULT_WORKERS_PER_MODEL': '1',
                    'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '2',
                    'TOKENIZERS_PARALLELISM': 'false',
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': source_s3_uri
                }
            },
            ExecutionRoleArn=role
        )
        
        logger.info(f"✓ Model created: {model_name}")
        
        # Step 2: Create Endpoint Configuration with AsyncInferenceConfig
        endpoint_config_name = f'{endpoint_name}-config'
        logger.info(f"Creating endpoint configuration: {endpoint_config_name}")
        
        async_config = {
            'OutputConfig': {
                'S3OutputPath': s3_output_path,
                'NotificationConfig': {
                    'SuccessTopic': sns_topic_arn,
                    'ErrorTopic': sns_topic_arn
                }
            },
            'ClientConfig': {
                'MaxConcurrentInvocationsPerInstance': max_concurrent_invocations
            }
        }
        
        logger.info("=" * 80)
        logger.info("ASYNCHRONOUS INFERENCE CONFIGURATION:")
        logger.info("=" * 80)
        logger.info(f"  S3 Output Path: {s3_output_path}")
        logger.info(f"  Success Notifications: {sns_topic_arn}")
        logger.info(f"  Error Notifications: {sns_topic_arn}")
        logger.info(f"  Max Concurrent Invocations: {max_concurrent_invocations}")
        logger.info("=" * 80)
        
        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ],
            AsyncInferenceConfig=async_config
        )
        
        logger.info(f"✓ Endpoint configuration created: {endpoint_config_name}")
        
        # Step 3: Create Endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        logger.info("This will take 5-10 minutes...")
        
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        logger.info("Waiting for endpoint to be in service...")
        
        # Wait for endpoint to be in service
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={
                'Delay': 30,  # Check every 30 seconds
                'MaxAttempts': 60  # Wait up to 30 minutes
            }
        )
        
        logger.info("=" * 80)
        logger.info("✅ ASYNC ENDPOINT DEPLOYED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"  Endpoint name: {endpoint_name}")
        logger.info(f"  Instance type: {instance_type}")
        logger.info(f"  Model URI: {model_uri}")
        logger.info(f"  S3 Output: {s3_output_path}")
        logger.info(f"  SNS Topic: {sns_topic_arn}")
        logger.info("=" * 80)
        
        # Save endpoint information
        endpoint_info = {
            'endpoint_name': endpoint_name,
            'endpoint_type': 'async',
            'model_uri': model_uri,
            'instance_type': instance_type,
            'region': region,
            'deployed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'async_config': {
                's3_output_path': s3_output_path,
                'sns_topic_arn': sns_topic_arn,
                'max_concurrent_invocations': max_concurrent_invocations
            }
        }
        
        with open('async_endpoint_info.json', 'w') as f:
            json.dump(endpoint_info, f, indent=2)
        
        logger.info("\n📄 Endpoint info saved to: async_endpoint_info.json")
        
        return endpoint_name
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Deploy StyGig model using SageMaker Asynchronous Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy new async endpoint
  python deploy_async_endpoint.py --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz

  # Replace existing async endpoint
  python deploy_async_endpoint.py --endpoint-name stygig-async-endpoint-20251115-120000 \
      --model-uri s3://stygig-ml-s3/model.tar.gz

  # Deploy with custom SNS topic
  python deploy_async_endpoint.py --model-uri s3://bucket/model.tar.gz \
      --sns-topic-arn arn:aws:sns:ap-south-1:123456789:my-topic
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
        help='Endpoint name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--instance-type',
        type=str,
        default=DEFAULT_INSTANCE_TYPE,
        help=f'Instance type for endpoint (default: {DEFAULT_INSTANCE_TYPE})'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=DEFAULT_REGION,
        help=f'AWS region (default: {DEFAULT_REGION})'
    )
    parser.add_argument(
        '--role',
        type=str,
        default=None,
        help='IAM role ARN (auto-detected if not provided)'
    )
    parser.add_argument(
        '--s3-output-path',
        type=str,
        default=None,
        help='S3 path for async inference results (default: s3://stygig-ml-s3/async-inference-results/)'
    )
    parser.add_argument(
        '--sns-topic-arn',
        type=str,
        default=None,
        help='SNS topic ARN for notifications (auto-created if not provided)'
    )
    parser.add_argument(
        '--max-concurrent-invocations',
        type=int,
        default=DEFAULT_MAX_CONCURRENT_INVOCATIONS,
        help=f'Max concurrent invocations per instance (default: {DEFAULT_MAX_CONCURRENT_INVOCATIONS})'
    )
    parser.add_argument(
        '--skip-delete',
        action='store_true',
        help='Skip deleting existing endpoint (creates new with different name)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("StyGig Asynchronous Inference Endpoint Deployment")
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
    logger.info(f"Step {step_num}: Deploying async endpoint...")
    
    # If skip-delete, generate new name
    endpoint_name = None if args.skip_delete else args.endpoint_name
    
    deployed_name = deploy_async_endpoint(
        model_uri=args.model_uri,
        endpoint_name=endpoint_name,
        instance_type=args.instance_type,
        region=args.region,
        role=args.role,
        s3_output_path=args.s3_output_path,
        sns_topic_arn=args.sns_topic_arn,
        max_concurrent_invocations=args.max_concurrent_invocations
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 Deployment Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Test async endpoint:")
    logger.info("     python sagemaker/invoke_async.py")
    logger.info("")
    logger.info(f"  2. Endpoint: {deployed_name}")
    logger.info("")
    logger.info("  3. Monitor SNS notifications for completion")
    logger.info("")
    logger.info("  4. Results will be saved to:")
    logger.info(f"     {args.s3_output_path or 's3://stygig-ml-s3/async-inference-results/'}")
    logger.info("")
    logger.info("  5. To delete endpoint:")
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
