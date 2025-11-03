#!/usr/bin/env python3
"""
Redeploy StyGig Endpoint with Extended Timeout

The current endpoint times out during cold start because:
- Loading CLIP model (ViT-B-32) takes ~30-40 seconds
- Loading FAISS index (25,169 items) takes ~10-15 seconds
- Loading metadata and embeddings takes ~5-10 seconds
Total: ~60+ seconds, exceeding default 60s timeout

This script redeploys with proper timeout configuration.
"""

import boto3
import json
from datetime import datetime

# Configuration
ENDPOINT_NAME_OLD = "stygig-endpoint-20251103-062336"
MODEL_URI = "s3://stygig-ml-s3/model-artifacts/stygig-training-1762145223/output/model.tar.gz"
INSTANCE_TYPE = "ml.m5.large"
REGION = "us-east-1"

# Extended timeouts for model loading
CONTAINER_STARTUP_TIMEOUT = 600  # 10 minutes for first request
MODEL_DATA_DOWNLOAD_TIMEOUT = 600  # 10 minutes to download model

def delete_old_endpoint():
    """Delete the old endpoint that's timing out."""
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    try:
        print(f"Deleting old endpoint: {ENDPOINT_NAME_OLD}")
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME_OLD)
        print(f"✓ Endpoint {ENDPOINT_NAME_OLD} deleted")
    except Exception as e:
        print(f"Note: {e}")

def deploy_with_timeout():
    """Deploy endpoint with proper timeout configuration."""
    import sys
    sys.path.insert(0, '/home/sagemaker-user/Stygig-Model-2')
    
    from sagemaker.pytorch import PyTorchModel
    from sagemaker import Session
    
    # Get execution role
    session = Session(boto_session=boto3.Session(region_name=REGION))
    role = "arn:aws:iam::732414292744:role/service-role/AmazonSageMaker-ExecutionRole-20251025T234436"
    
    # New endpoint name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    endpoint_name = f'stygig-endpoint-{timestamp}'
    
    print(f"Creating new endpoint: {endpoint_name}")
    print(f"Model URI: {MODEL_URI}")
    
    # Create model with extended timeouts
    model = PyTorchModel(
        model_data=MODEL_URI,
        role=role,
        entry_point='sagemaker/inference.py',
        source_dir='/home/sagemaker-user/Stygig-Model-2',
        framework_version='2.0.0',
        py_version='py310',
        sagemaker_session=session,
        # Extended timeouts
        env={
            'MODEL_SERVER_TIMEOUT': '600',  # 10 minutes
            'MODEL_SERVER_WORKERS': '1'     # Single worker to avoid memory issues
        }
    )
    
    print("Deploying model (this takes 5-10 minutes)...")
    
    # Deploy with extended timeout
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        model_data_download_timeout=MODEL_DATA_DOWNLOAD_TIMEOUT,
        container_startup_health_check_timeout=CONTAINER_STARTUP_TIMEOUT
    )
    
    print(f"\n✅ Endpoint deployed successfully!")
    print(f"   Endpoint name: {endpoint_name}")
    print(f"   Container timeout: {CONTAINER_STARTUP_TIMEOUT}s")
    
    # Save endpoint info
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'model_uri': MODEL_URI,
        'instance_type': INSTANCE_TYPE,
        'region': REGION,
        'container_timeout': CONTAINER_STARTUP_TIMEOUT
    }
    
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    print(f"\n📄 Endpoint info saved to: endpoint_info.json")
    return endpoint_name

if __name__ == '__main__':
    print("=" * 80)
    print("StyGig Endpoint Redeployment with Extended Timeout")
    print("=" * 80)
    print()
    
    # Delete old endpoint
    delete_old_endpoint()
    
    print()
    print("Waiting 30 seconds for cleanup...")
    import time
    time.sleep(30)
    
    # Deploy new endpoint with timeout
    endpoint_name = deploy_with_timeout()
    
    print()
    print("=" * 80)
    print("🎉 Redeployment Complete!")
    print("=" * 80)
    print()
    print(f"Test your new endpoint with:")
    print(f"  python test_endpoint.py --endpoint-name {endpoint_name} --save-visual")
