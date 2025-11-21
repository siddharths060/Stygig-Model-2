#!/usr/bin/env python3
"""
Redeploy the async endpoint with updated inference code
"""

import boto3
import time
from datetime import datetime

def redeploy_async_endpoint():
    print("=" * 60)
    print("StyGig V4 Async Endpoint Redeployment")
    print("=" * 60)
    print()
    
    sagemaker = boto3.client('sagemaker', region_name='ap-south-1')
    
    endpoint_name = 'stygig-async-production'
    model_name = f'stygig-v4-async-model-{int(time.time())}'
    config_name = f'stygig-async-config-{int(time.time())}'
    
    # Model S3 location from successful training
    model_data = 's3://sagemaker-ap-south-1-732414292744/stygig-v4-pipeline/models/stygig-v4-training-2025-11-21-13-25-44-446/output/model.tar.gz'
    
    print(f"Creating new model: {model_name}")
    print(f"Model data: {model_data}")
    
    # 1. Create new model
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.0-cpu-py310',
                'ModelDataUrl': model_data,
                'Environment': {
                    'PYTHONPATH': '/opt/ml/model:/opt/ml/model/src',
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code'
                }
            },
            ExecutionRoleArn='arn:aws:iam::732414292744:role/sagemaker-execution-role'
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # 2. Create new endpoint configuration
    print(f"\nCreating endpoint configuration: {config_name}")
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': 'ml.c5.large',
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0
                }
            ],
            AsyncInferenceConfig={
                'OutputConfig': {
                    'S3OutputPath': 's3://sagemaker-ap-south-1-732414292744/async-results/',
                    'S3FailurePath': 's3://sagemaker-ap-south-1-732414292744/async-endpoint-failures/'
                },
                'ClientConfig': {
                    'MaxConcurrentInvocationsPerInstance': 10
                }
            }
        )
        print("✅ Endpoint configuration created successfully")
    except Exception as e:
        print(f"❌ Endpoint config creation failed: {e}")
        return False
    
    # 3. Update existing endpoint
    print(f"\nUpdating endpoint: {endpoint_name}")
    try:
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print("✅ Endpoint update initiated")
    except Exception as e:
        print(f"❌ Endpoint update failed: {e}")
        return False
    
    # 4. Wait for endpoint to be InService
    print("\nWaiting for endpoint to be InService...")
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            elapsed = int(time.time() - start_time)
            print(f"Status: {status} (elapsed: {elapsed}s)")
            
            if status == 'InService':
                print("✅ Endpoint is now InService!")
                break
            elif status in ['Failed', 'OutOfService']:
                print(f"❌ Endpoint update failed with status: {status}")
                return False
            
            time.sleep(30)
            
            # Timeout after 20 minutes
            if elapsed > 1200:
                print("⏱️ Timeout waiting for endpoint")
                return False
                
        except Exception as e:
            print(f"Error checking endpoint status: {e}")
            time.sleep(10)
    
    print()
    print("🎉 REDEPLOYMENT COMPLETE!")
    print(f"Endpoint: {endpoint_name}")
    print(f"Model: {model_name}")
    print(f"Config: {config_name}")
    print()
    print("The endpoint now supports both:")
    print("- Base64 image data: {'image': 'base64_data', 'top_k': 5}")
    print("- S3 URIs: {'image_s3_uri': 's3://bucket/key', 'top_k': 5}")
    
    return True

if __name__ == '__main__':
    success = redeploy_async_endpoint()
    if success:
        print("\n✅ Ready to test with S3 URIs!")
    else:
        print("\n❌ Redeployment failed - check logs")