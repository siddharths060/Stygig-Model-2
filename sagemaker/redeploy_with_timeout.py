#!/usr/bin/env python3
"""
Redeploy StyGig Endpoint with Extended Timeouts for CLIP Model Loading

The current endpoint times out during cold start because:
- Loading CLIP model (ViT-B-32) takes ~60-90 seconds on first request
- Loading FAISS index takes ~10-15 seconds
- Loading metadata and embeddings takes ~5-10 seconds
- Initializing color/gender processors takes ~5 seconds
Total: ~80-120 seconds, exceeding default 60s timeout

This script redeploys with proper timeout configuration:
- Container startup: 600s (10 minutes)
- Model server timeout: 300s (5 minutes per request)
- Model download: 600s (10 minutes)
"""

import boto3
import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Configuration
DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_REGION = "ap-south-1"

# Extended timeouts for CLIP model loading
CONTAINER_STARTUP_TIMEOUT = 600  # 10 minutes for first startup
MODEL_DATA_DOWNLOAD_TIMEOUT = 600  # 10 minutes to download model
MODEL_SERVER_TIMEOUT = 300  # 5 minutes per request


def get_latest_model_uri(bucket='stygig-ml-s3', prefix='model-artifacts/'):
    """Find the most recent model artifact in S3."""
    s3_client = boto3.client('s3', region_name=DEFAULT_REGION)
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return None
        
        # Find all model.tar.gz files
        model_files = [obj for obj in response['Contents'] if obj['Key'].endswith('model.tar.gz')]
        
        if not model_files:
            return None
        
        # Sort by last modified date
        latest_model = sorted(model_files, key=lambda x: x['LastModified'], reverse=True)[0]
        
        return f"s3://{bucket}/{latest_model['Key']}"
        
    except Exception as e:
        print(f"Error finding model: {e}")
        return None


def delete_endpoint(endpoint_name, region=DEFAULT_REGION):
    """Delete an existing endpoint and its configuration."""
    sm_client = boto3.client('sagemaker', region_name=region)
    
    try:
        print(f"Deleting endpoint: {endpoint_name}")
        
        # Delete endpoint
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"  ✓ Endpoint deletion initiated")
        
        # Wait for deletion
        waiter = sm_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        print(f"  ✓ Endpoint deleted")
        
        # Delete endpoint configuration
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            print(f"  ✓ Endpoint configuration deleted")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"Note: {e}")
        return False


def get_execution_role(region=DEFAULT_REGION):
    """Find SageMaker execution role automatically."""
    iam_client = boto3.client('iam', region_name=region)
    
    # Try to find SageMaker execution role
    paginator = iam_client.get_paginator('list_roles')
    
    for page in paginator.paginate():
        for iam_role in page['Roles']:
            if 'AmazonSageMaker-ExecutionRole' in iam_role['RoleName']:
                return iam_role['Arn']
    
    raise ValueError("Could not find SageMaker execution role")


def deploy_with_timeout(model_uri, endpoint_name=None, instance_type=DEFAULT_INSTANCE_TYPE, 
                       region=DEFAULT_REGION, role=None):
    """Deploy endpoint with proper timeout configuration for CLIP model loading."""
    
    from sagemaker.pytorch import PyTorchModel
    from sagemaker import Session
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer
    
    # Get execution role
    if not role:
        role = get_execution_role(region)
    
    print(f"Using IAM role: {role}")
    
    # Initialize session
    session = Session(boto_session=boto3.Session(region_name=region))
    
    # Generate endpoint name if not provided
    if not endpoint_name:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f'stygig-endpoint-{timestamp}'
    
    print(f"\nCreating endpoint: {endpoint_name}")
    print(f"Model URI: {model_uri}")
    print(f"Instance type: {instance_type}")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Create model with extended timeouts and optimizations
    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        entry_point='sagemaker/inference.py',
        source_dir=str(project_root),
        framework_version='2.0.0',
        py_version='py310',
        sagemaker_session=session,
        model_server_workers=1,  # Single worker to reduce memory usage
        env={
            # Extended timeout for CLIP model loading
            'MODEL_SERVER_TIMEOUT': str(MODEL_SERVER_TIMEOUT),
            'MODEL_SERVER_WORKERS': '1',
            'TS_MAX_REQUEST_SIZE': '100000000',  # 100MB
            'TS_MAX_RESPONSE_SIZE': '100000000',
            'TS_DEFAULT_RESPONSE_TIMEOUT': str(MODEL_SERVER_TIMEOUT),
            'TS_DEFAULT_WORKERS_PER_MODEL': '1',
            # Optimization flags
            'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    
    print("\n" + "=" * 80)
    print("Deploying model with EXTENDED TIMEOUTS (this takes 5-10 minutes)...")
    print("=" * 80)
    print(f"  Container startup timeout: {CONTAINER_STARTUP_TIMEOUT}s (10 minutes)")
    print(f"  Model download timeout: {MODEL_DATA_DOWNLOAD_TIMEOUT}s (10 minutes)")
    print(f"  Model server timeout: {MODEL_SERVER_TIMEOUT}s (5 minutes per request)")
    print("=" * 80)
    
    # Deploy with extended timeouts
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        # CRITICAL: Extended timeouts for CLIP model loading
        container_startup_health_check_timeout=CONTAINER_STARTUP_TIMEOUT,
        model_data_download_timeout=MODEL_DATA_DOWNLOAD_TIMEOUT
    )
    
    print("\n" + "=" * 80)
    print("✅ Endpoint deployed successfully!")
    print("=" * 80)
    print(f"  Endpoint name: {endpoint_name}")
    print(f"  Container timeout: {CONTAINER_STARTUP_TIMEOUT}s")
    print(f"  Model server timeout: {MODEL_SERVER_TIMEOUT}s")
    print("=" * 80)
    
    # Save endpoint info
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'model_uri': model_uri,
        'instance_type': instance_type,
        'region': region,
        'deployed_at': datetime.now().isoformat(),
        'timeouts': {
            'container_startup': CONTAINER_STARTUP_TIMEOUT,
            'model_download': MODEL_DATA_DOWNLOAD_TIMEOUT,
            'model_server': MODEL_SERVER_TIMEOUT
        }
    }
    
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    print(f"\n📄 Endpoint info saved to: endpoint_info.json")
    
    return endpoint_name


def main():
    parser = argparse.ArgumentParser(
        description='Redeploy StyGig endpoint with extended timeouts for CLIP model loading'
    )
    parser.add_argument('--model-uri', type=str, help='S3 URI to model.tar.gz')
    parser.add_argument('--old-endpoint', type=str, help='Name of old endpoint to delete')
    parser.add_argument('--new-endpoint', type=str, help='Name for new endpoint')
    parser.add_argument('--instance-type', type=str, default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument('--region', type=str, default=DEFAULT_REGION)
    parser.add_argument('--role', type=str, help='IAM role ARN')
    parser.add_argument('--skip-delete', action='store_true', help='Skip deleting old endpoint')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("StyGig Endpoint Redeployment with Extended Timeouts")
    print("=" * 80)
    print("\nThis script fixes the timeout issue by deploying with:")
    print(f"  • Container startup timeout: {CONTAINER_STARTUP_TIMEOUT}s (10 minutes)")
    print(f"  • Model server timeout: {MODEL_SERVER_TIMEOUT}s (5 minutes per request)")
    print(f"  • Model download timeout: {MODEL_DATA_DOWNLOAD_TIMEOUT}s (10 minutes)")
    print()
    
    # Find model URI if not provided
    model_uri = args.model_uri
    if not model_uri:
        print("No model URI provided, searching for latest model in S3...")
        model_uri = get_latest_model_uri()
        if model_uri:
            print(f"Found latest model: {model_uri}")
        else:
            print("ERROR: No model found. Please provide --model-uri")
            return 1
    
    # Delete old endpoint if specified
    if args.old_endpoint and not args.skip_delete:
        print(f"\nStep 1: Deleting old endpoint...")
        delete_endpoint(args.old_endpoint, args.region)
        print("\nWaiting 30 seconds for cleanup...")
        time.sleep(30)
    elif args.skip_delete:
        print("\nSkipping endpoint deletion (--skip-delete flag set)")
    
    # Deploy new endpoint
    step_num = 2 if (args.old_endpoint and not args.skip_delete) else 1
    print(f"\nStep {step_num}: Deploying new endpoint with extended timeouts...")
    
    endpoint_name = deploy_with_timeout(
        model_uri=model_uri,
        endpoint_name=args.new_endpoint,
        instance_type=args.instance_type,
        region=args.region,
        role=args.role
    )
    
    print()
    print("=" * 80)
    print("🎉 Redeployment Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Test endpoint (first request takes 2-3 minutes):")
    print(f"     python sagemaker/test_endpoint.py --endpoint-name {endpoint_name}")
    print()
    print(f"  2. Check CloudWatch logs if issues persist:")
    print(f"     https://console.aws.amazon.com/cloudwatch/home?region={args.region}#logStream:group=/aws/sagemaker/Endpoints/{endpoint_name}")
    print()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nRedeployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nRedeployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
