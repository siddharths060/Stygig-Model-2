#!/usr/bin/env python3
"""
StyGig Asynchronous Inference Invocation Script
===============================================
Test script to invoke SageMaker Asynchronous Inference endpoint.

This script demonstrates:
1. Uploading input payload to S3 (InputLocation)
2. Invoking the async endpoint with InvokeEndpointAsync
3. Retrieving the OutputLocation from the response
4. Polling S3 for the result
5. Downloading and displaying the recommendation results

Usage:
    # Invoke with auto-detected endpoint
    python invoke_async.py

    # Invoke specific endpoint
    python invoke_async.py --endpoint-name stygig-async-endpoint-20251115-120000

    # Use specific image
    python invoke_async.py --image test_image.jpg

    # Use S3 image
    python invoke_async.py --s3-image s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg

    # Wait for result and display
    python invoke_async.py --wait --display-results
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_REGION = 'ap-south-1'
DEFAULT_S3_BUCKET = 'stygig-ml-s3'
DEFAULT_INPUT_PREFIX = 'async-inference-input/'


def get_endpoint_name() -> str:
    """Get endpoint name from saved async endpoint info."""
    info_file = Path('async_endpoint_info.json')
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            return info.get('endpoint_name')
    
    # Fallback to regular endpoint info
    info_file = Path('endpoint_info.json')
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            return info.get('endpoint_name')
    
    return None


def download_s3_image(s3_uri: str, local_path: str = 'test_image.jpg') -> str:
    """
    Download image from S3.
    
    Args:
        s3_uri: S3 URI (s3://bucket/key)
        local_path: Local path to save image
        
    Returns:
        Local file path
    """
    try:
        # Parse S3 URI
        s3_parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        logger.info(f"Downloading from S3: {s3_uri}")
        
        s3_client = boto3.client('s3', region_name=DEFAULT_REGION)
        s3_client.download_file(bucket, key, local_path)
        
        logger.info(f"✓ Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download S3 image: {e}")
        raise


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')


def upload_payload_to_s3(
    payload: dict,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = DEFAULT_INPUT_PREFIX,
    region: str = DEFAULT_REGION
) -> str:
    """
    Upload inference payload to S3.
    
    Args:
        payload: Inference request payload
        bucket: S3 bucket name
        prefix: S3 key prefix
        region: AWS region
        
    Returns:
        S3 URI of uploaded payload
    """
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Generate unique key
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        s3_key = f'{prefix}request-{timestamp}.json'
        
        # Upload payload
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(payload),
            ContentType='application/json'
        )
        
        s3_uri = f's3://{bucket}/{s3_key}'
        logger.info(f"✓ Uploaded payload to: {s3_uri}")
        
        return s3_uri
        
    except Exception as e:
        logger.error(f"Failed to upload payload to S3: {e}")
        raise


def invoke_async_endpoint(
    endpoint_name: str,
    input_location: str,
    region: str = DEFAULT_REGION
) -> str:
    """
    Invoke asynchronous inference endpoint.
    
    Args:
        endpoint_name: Name of the async endpoint
        input_location: S3 URI of input payload
        region: AWS region
        
    Returns:
        S3 URI where output will be saved (OutputLocation)
    """
    try:
        runtime_client = boto3.client('sagemaker-runtime', region_name=region)
        
        logger.info(f"Invoking async endpoint: {endpoint_name}")
        logger.info(f"Input location: {input_location}")
        
        # Invoke async endpoint
        response = runtime_client.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=input_location,
            ContentType='application/json'
        )
        
        output_location = response['OutputLocation']
        
        logger.info("=" * 80)
        logger.info("✅ ASYNC INVOCATION SUBMITTED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"  Output Location: {output_location}")
        
        if 'FailureLocation' in response:
            logger.info(f"  Failure Location: {response['FailureLocation']}")
        
        logger.info("=" * 80)
        
        return output_location
        
    except Exception as e:
        logger.error(f"Failed to invoke async endpoint: {e}")
        raise


def wait_for_result(
    output_location: str,
    max_wait_seconds: int = 600,
    poll_interval: int = 10,
    region: str = DEFAULT_REGION
) -> dict:
    """
    Wait for async inference result to be available in S3.
    
    Args:
        output_location: S3 URI where output will be saved
        max_wait_seconds: Maximum time to wait (seconds)
        poll_interval: Time between polls (seconds)
        region: AWS region
        
    Returns:
        Inference result as dict
    """
    try:
        # Parse S3 URI
        s3_parts = output_location.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        s3_client = boto3.client('s3', region_name=region)
        
        logger.info("")
        logger.info("⏳ Waiting for inference result...")
        logger.info(f"   Checking: {output_location}")
        logger.info(f"   Max wait time: {max_wait_seconds}s")
        logger.info(f"   Poll interval: {poll_interval}s")
        logger.info("")
        
        start_time = time.time()
        attempt = 0
        
        while (time.time() - start_time) < max_wait_seconds:
            attempt += 1
            elapsed = int(time.time() - start_time)
            
            try:
                # Check if object exists
                response = s3_client.head_object(Bucket=bucket, Key=key)
                
                # Object exists, download result
                logger.info(f"✓ Result available after {elapsed}s (attempt {attempt})")
                
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                result = json.loads(obj['Body'].read().decode('utf-8'))
                
                return result
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # Object doesn't exist yet
                    logger.info(f"   Attempt {attempt}: Not ready yet ({elapsed}s elapsed)")
                    time.sleep(poll_interval)
                else:
                    raise
        
        # Timeout
        logger.error(f"⚠️  Timeout after {max_wait_seconds}s")
        logger.error(f"   Result not available at: {output_location}")
        logger.error("")
        logger.error("   The inference may still be processing. You can:")
        logger.error("   1. Wait longer and check the S3 location manually")
        logger.error("   2. Subscribe to the SNS topic for notifications")
        logger.error("   3. Check CloudWatch logs for errors")
        
        return None
        
    except Exception as e:
        logger.error(f"Error while waiting for result: {e}")
        raise


def display_results(result: dict, input_image_path: str = None) -> None:
    """
    Display inference results in a readable format.
    
    Args:
        result: Inference result dictionary
        input_image_path: Path to input image (optional)
    """
    try:
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 INFERENCE RESULTS")
        logger.info("=" * 80)
        
        if 'error' in result:
            logger.error(f"❌ Error: {result['error']}")
            return
        
        # Display input item info
        if 'input_item' in result:
            input_info = result['input_item']
            logger.info("")
            logger.info("📸 INPUT ITEM:")
            logger.info(f"   Category: {input_info.get('category', 'Unknown')}")
            logger.info(f"   Gender: {input_info.get('gender', 'Unknown')}")
            logger.info(f"   Dominant Colors: {', '.join(input_info.get('colors', []))}")
        
        # Display recommendations
        recommendations = result.get('recommendations', [])
        logger.info("")
        logger.info(f"🎯 TOP {len(recommendations)} RECOMMENDATIONS:")
        logger.info("")
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec['item_id']}")
            logger.info(f"   Category: {rec['category']}")
            logger.info(f"   Gender: {rec['gender']}")
            logger.info(f"   Overall Score: {rec['score']:.4f}")
            logger.info(f"   Color Match: {rec['color_score']:.4f}")
            logger.info(f"   Category Match: {rec['category_score']:.4f}")
            logger.info(f"   Gender Match: {rec['gender_score']:.4f}")
            logger.info(f"   Colors: {', '.join(rec.get('colors', []))}")
            logger.info("")
        
        # Display metadata
        metadata = result.get('metadata', {})
        logger.info("📈 METADATA:")
        logger.info(f"   Processing time: {metadata.get('processing_time_ms', 0):.2f}ms")
        logger.info(f"   Total items searched: {metadata.get('total_items', 0)}")
        logger.info(f"   Model version: {metadata.get('model_version', 'unknown')}")
        logger.info("")
        logger.info("=" * 80)
        
        # Save detailed results
        output_file = f'async_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"\n💾 Full results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error displaying results: {e}")


def invoke_async_inference(
    endpoint_name: str,
    image_path: str = None,
    s3_image: str = None,
    top_k: int = 5,
    wait: bool = False,
    display: bool = False,
    region: str = DEFAULT_REGION
) -> dict:
    """
    Complete async inference workflow.
    
    Args:
        endpoint_name: Name of async endpoint
        image_path: Local image path
        s3_image: S3 URI of image
        top_k: Number of recommendations
        wait: Whether to wait for result
        display: Whether to display results
        region: AWS region
        
    Returns:
        Response with OutputLocation
    """
    try:
        # Handle S3 image if provided
        if s3_image:
            image_path = download_s3_image(s3_image)
        
        # If no image provided, find a sample from S3
        if not image_path:
            logger.info("No image provided, selecting random sample from S3...")
            s3_client = boto3.client('s3', region_name=region)
            
            # List some objects from train folder
            response = s3_client.list_objects_v2(
                Bucket=DEFAULT_S3_BUCKET,
                Prefix='train/upperwear/shirt/',
                MaxKeys=10
            )
            
            if response.get('Contents'):
                sample_key = response['Contents'][0]['Key']
                s3_uri = f"s3://{DEFAULT_S3_BUCKET}/{sample_key}"
                logger.info(f"Selected sample: {s3_uri}")
                image_path = download_s3_image(s3_uri)
            else:
                raise ValueError("No sample images found in S3")
        
        # Get image info
        img = Image.open(image_path)
        logger.info(f"Input image: {image_path} ({img.size[0]}x{img.size[1]})")
        
        # Convert to base64
        image_b64 = image_to_base64(image_path)
        
        # Prepare request payload
        payload = {
            'image': image_b64,
            'top_k': top_k,
            'min_score': 0.5
        }
        
        logger.info(f"Preparing async inference request (top_k={top_k})...")
        
        # Upload payload to S3
        input_location = upload_payload_to_s3(payload, region=region)
        
        # Invoke async endpoint
        output_location = invoke_async_endpoint(endpoint_name, input_location, region)
        
        # Optionally wait for result
        if wait:
            result = wait_for_result(output_location, region=region)
            
            if result and display:
                display_results(result, image_path)
            
            return {
                'output_location': output_location,
                'result': result
            }
        
        return {
            'output_location': output_location,
            'input_location': input_location
        }
        
    except Exception as e:
        logger.error(f"Async inference failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Invoke StyGig asynchronous inference endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--endpoint-name',
        type=str,
        help='Async endpoint name (auto-detected if not provided)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Local image path'
    )
    parser.add_argument(
        '--s3-image',
        type=str,
        help='S3 URI of image (e.g., s3://bucket/key)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of recommendations (default: 5)'
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for inference result to be available'
    )
    parser.add_argument(
        '--display-results',
        action='store_true',
        help='Display results (requires --wait)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=DEFAULT_REGION,
        help=f'AWS region (default: {DEFAULT_REGION})'
    )
    
    args = parser.parse_args()
    
    # Get endpoint name
    endpoint_name = args.endpoint_name or get_endpoint_name()
    
    if not endpoint_name:
        logger.error("No endpoint name provided. Use --endpoint-name or ensure async_endpoint_info.json exists")
        return 1
    
    logger.info("=" * 80)
    logger.info("StyGig Asynchronous Inference - Test Invocation")
    logger.info("=" * 80)
    logger.info("")
    
    # Invoke async endpoint
    response = invoke_async_inference(
        endpoint_name=endpoint_name,
        image_path=args.image,
        s3_image=args.s3_image,
        top_k=args.top_k,
        wait=args.wait,
        display=args.display_results,
        region=args.region
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Async Invocation Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Check output at: {response['output_location']}")
    logger.info("")
    logger.info("  2. Or run again with --wait --display-results:")
    logger.info(f"     python invoke_async.py --wait --display-results")
    logger.info("")
    logger.info("  3. Subscribe to SNS topic for notifications")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nInvocation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nInvocation failed: {e}")
        sys.exit(1)
