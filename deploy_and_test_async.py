#!/usr/bin/env python3
"""
Deploy Async Endpoint and Test with Sample Image
===============================================

This script combines async deployment and testing in one workflow.
"""

import os
import sys
import json
import time
import boto3
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig

def get_latest_model_uri():
    """Find the latest trained model from the training job."""
    try:
        # Read from the pipeline output or search S3
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        
        # Look for the latest model in the pipeline output
        s3_client = boto3.client('s3')
        prefix = 'stygig-v4-pipeline/models/'
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter='/'
        )
        
        # Find the most recent training job folder
        folders = []
        for prefix_info in response.get('CommonPrefixes', []):
            folders.append(prefix_info['Prefix'])
        
        if not folders:
            print("❌ No trained models found. Please run training first.")
            return None
            
        # Get the latest folder (they're timestamped)
        latest_folder = sorted(folders)[-1]
        model_uri = f"s3://{bucket}/{latest_folder}output/model.tar.gz"
        
        print(f"✓ Found latest model: {model_uri}")
        return model_uri
        
    except Exception as e:
        print(f"❌ Error finding model: {e}")
        return None

def deploy_async_endpoint(model_uri, endpoint_name="stygig-async-test"):
    """Deploy the model to an async endpoint."""
    try:
        print(f"\n🚀 Deploying async endpoint: {endpoint_name}")
        
        # Get SageMaker role
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()
        bucket = sagemaker_session.default_bucket()
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_uri,
            role=role,
            entry_point='inference.py',
            source_dir='sagemaker',
            framework_version='2.0.0',
            py_version='py310',
            sagemaker_session=sagemaker_session
        )
        
        # Configure async inference
        async_config = AsyncInferenceConfig(
            output_path=f"s3://{bucket}/async-inference-output/",
            max_concurrent_invocations_per_instance=4,
            # Optional: Add SNS topic for notifications
            # success_topic="arn:aws:sns:region:account:topic-name",
            # error_topic="arn:aws:sns:region:account:topic-name"
        )
        
        # Deploy to async endpoint
        predictor = model.deploy(
            instance_type='ml.c5.large',
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True  # Wait for deployment to complete
        )
        
        print(f"✅ Async endpoint deployed: {endpoint_name}")
        return predictor
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return None

def download_test_image(s3_uri):
    """Download test image from S3."""
    try:
        print(f"\n📥 Downloading test image: {s3_uri}")
        
        # Parse S3 URI
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Download image
        s3_client = boto3.client('s3')
        local_path = 'test_image.jpg'
        s3_client.download_file(bucket, key, local_path)
        
        # Load and display info
        image = Image.open(local_path)
        print(f"✓ Downloaded: {local_path}")
        print(f"  Image size: {image.size}")
        print(f"  Format: {image.format}")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Failed to download image: {e}")
        return None

def test_async_endpoint(endpoint_name, image_path):
    """Test the async endpoint with the image."""
    try:
        print(f"\n🧪 Testing async endpoint: {endpoint_name}")
        
        # Load and encode image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare payload
        payload = {
            "image": image_b64,
            "top_k": 5
        }
        
        # Create SageMaker runtime client
        runtime_client = boto3.client('sagemaker-runtime')
        
        # Start async inference
        print("🔄 Starting async inference...")
        response = runtime_client.invoke_endpoint_async(
            EndpointName=endpoint_name,
            ContentType='application/json',
            InputLocation=None,  # We're sending data directly
            Body=json.dumps(payload)
        )
        
        output_location = response['OutputLocation']
        print(f"✓ Inference started!")
        print(f"  Output will be saved to: {output_location}")
        
        # Wait for completion and fetch result
        print("⏳ Waiting for inference to complete...")
        return wait_for_async_result(output_location)
        
    except Exception as e:
        print(f"❌ Async inference failed: {e}")
        return None

def wait_for_async_result(output_location, timeout=300):
    """Wait for async inference result and fetch it."""
    try:
        # Parse output location
        parts = output_location.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        s3_client = boto3.client('s3')
        
        # Poll for result
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if result exists
                response = s3_client.head_object(Bucket=bucket, Key=key)
                print("✅ Inference completed!")
                
                # Download result
                result_obj = s3_client.get_object(Bucket=bucket, Key=key)
                result_json = result_obj['Body'].read().decode('utf-8')
                result = json.loads(result_json)
                
                return result
                
            except s3_client.exceptions.NoSuchKey:
                # Result not ready yet
                print("⏳ Still processing...")
                time.sleep(10)
                continue
        
        print("⚠️  Timeout waiting for result")
        return None
        
    except Exception as e:
        print(f"❌ Error waiting for result: {e}")
        return None

def display_recommendations(result):
    """Display the recommendation results."""
    if not result:
        print("❌ No results to display")
        return
    
    print("\n🎯 RECOMMENDATION RESULTS")
    print("=" * 50)
    
    try:
        recommendations = result.get('recommendations', [])
        query_info = result.get('query_image_info', {})
        
        print(f"Query Image Category: {query_info.get('predicted_category', 'N/A')}")
        print(f"Query Image Gender: {query_info.get('predicted_gender', 'N/A')}")
        print(f"Processing Time: {result.get('processing_time_ms', 'N/A')} ms")
        print(f"Total Recommendations: {len(recommendations)}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. Score: {rec.get('similarity_score', 0):.4f}")
            print(f"   Category: {rec.get('category', 'N/A')}")
            print(f"   Gender: {rec.get('gender', 'N/A')}")
            print(f"   Image: {rec.get('image_path', 'N/A')}")
            
            # Show color harmony info if available
            colors = rec.get('color_harmony', {})
            if colors:
                print(f"   Colors: {colors.get('primary', 'N/A')} + {colors.get('secondary', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error displaying results: {e}")
        print(f"Raw result: {json.dumps(result, indent=2)}")

def cleanup_endpoint(endpoint_name):
    """Clean up the test endpoint."""
    try:
        print(f"\n🧹 Cleaning up endpoint: {endpoint_name}")
        
        sagemaker_client = boto3.client('sagemaker')
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        print("✅ Endpoint cleanup initiated")
        print("   (Endpoint will be deleted in a few minutes)")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main workflow."""
    print("🎯 StyGig V4 - Async Endpoint Test")
    print("=" * 50)
    
    # Configuration
    test_image_s3 = "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png"
    endpoint_name = "stygig-async-test"
    
    try:
        # Step 1: Find latest model
        model_uri = get_latest_model_uri()
        if not model_uri:
            return
        
        # Step 2: Deploy async endpoint
        predictor = deploy_async_endpoint(model_uri, endpoint_name)
        if not predictor:
            return
        
        # Step 3: Download test image
        image_path = download_test_image(test_image_s3)
        if not image_path:
            cleanup_endpoint(endpoint_name)
            return
        
        # Step 4: Test endpoint
        result = test_async_endpoint(endpoint_name, image_path)
        
        # Step 5: Display results
        display_recommendations(result)
        
        # Step 6: Ask about cleanup
        response = input("\n🤔 Delete the test endpoint? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_endpoint(endpoint_name)
        else:
            print(f"✓ Keeping endpoint: {endpoint_name}")
            print("  Remember to delete it later to avoid charges!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        cleanup_endpoint(endpoint_name)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_endpoint(endpoint_name)

if __name__ == '__main__':
    main()