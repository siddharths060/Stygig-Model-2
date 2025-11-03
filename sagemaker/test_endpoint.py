#!/usr/bin/env python3
"""
Test StyGig Fashion Recommendation Endpoint

This script tests your deployed endpoint with sample fashion items.
It downloads an image from S3, sends it to the endpoint, and displays recommendations.

Usage:
    python test_endpoint.py
    python test_endpoint.py --endpoint-name stygig-endpoint-20251103-062336
    python test_endpoint.py --s3-image s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import base64
from io import BytesIO

import boto3
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_endpoint_name():
    """Get endpoint name from saved info or environment."""
    # Try to read from endpoint_info.json
    info_file = Path('endpoint_info.json')
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            return info.get('endpoint_name')
    
    return None

def download_s3_image(s3_uri, local_path='test_image.jpg'):
    """Download image from S3."""
    try:
        # Parse S3 URI
        s3_parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        
        logger.info(f"Downloading from S3: {s3_uri}")
        
        s3_client = boto3.client('s3', region_name='ap-south-1')
        s3_client.download_file(bucket, key, local_path)
        
        logger.info(f"✓ Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download S3 image: {e}")
        raise

def image_to_base64(image_path):
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_endpoint(endpoint_name, image_path=None, s3_uri=None, top_k=5):
    """Send test request to endpoint and display results."""
    try:
        # Create SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        # Handle S3 image if provided
        if s3_uri:
            image_path = download_s3_image(s3_uri)
        
        # If no image provided, find a sample from S3
        if not image_path:
            logger.info("No image provided, selecting random sample from S3...")
            s3_client = boto3.client('s3', region_name='ap-south-1')
            
            # List some objects from train folder
            response = s3_client.list_objects_v2(
                Bucket='stygig-ml-s3',
                Prefix='train/upperwear/shirt/',
                MaxKeys=10
            )
            
            if response.get('Contents'):
                sample_key = response['Contents'][0]['Key']
                s3_uri = f"s3://stygig-ml-s3/{sample_key}"
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
        
        logger.info(f"Sending inference request (top_k={top_k})...")
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        logger.info("✅ Inference successful!")
        logger.info("=" * 80)
        logger.info("RECOMMENDATION RESULTS")
        logger.info("=" * 80)
        
        if 'error' in result:
            logger.error(f"Error from endpoint: {result['error']}")
            return
        
        # Display input item info
        if 'input_item' in result:
            input_info = result['input_item']
            logger.info(f"\n📸 INPUT ITEM:")
            logger.info(f"   Category: {input_info.get('category', 'Unknown')}")
            logger.info(f"   Gender: {input_info.get('gender', 'Unknown')}")
            logger.info(f"   Dominant Colors: {', '.join(input_info.get('colors', []))}")
        
        # Display recommendations
        recommendations = result.get('recommendations', [])
        logger.info(f"\n🎯 TOP {len(recommendations)} RECOMMENDATIONS:\n")
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec['item_id']}")
            logger.info(f"   Category: {rec['category']}")
            logger.info(f"   Gender: {rec['gender']}")
            logger.info(f"   Score: {rec['score']:.4f}")
            logger.info(f"   Color Match: {rec['color_score']:.4f}")
            logger.info(f"   Category Match: {rec['category_score']:.4f}")
            logger.info(f"   Gender Match: {rec['gender_score']:.4f}")
            logger.info(f"   Colors: {', '.join(rec.get('colors', []))}")
            logger.info("")
        
        # Display metadata
        metadata = result.get('metadata', {})
        logger.info(f"📊 METADATA:")
        logger.info(f"   Processing time: {metadata.get('processing_time_ms', 0):.2f}ms")
        logger.info(f"   Total items searched: {metadata.get('total_items', 0)}")
        logger.info(f"   Model version: {metadata.get('model_version', 'unknown')}")
        logger.info("")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Endpoint test failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Test StyGig fashion recommendation endpoint')
    parser.add_argument('--endpoint-name', type=str, help='SageMaker endpoint name')
    parser.add_argument('--image', type=str, help='Local image path')
    parser.add_argument('--s3-image', type=str, help='S3 URI of image (e.g., s3://bucket/key)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Get endpoint name
    endpoint_name = args.endpoint_name or get_endpoint_name()
    
    if not endpoint_name:
        logger.error("No endpoint name provided. Use --endpoint-name or ensure endpoint_info.json exists")
        return 1
    
    logger.info("=" * 80)
    logger.info("StyGig Fashion Recommendation - Endpoint Test")
    logger.info("=" * 80)
    logger.info("")
    
    # Test endpoint
    result = test_endpoint(
        endpoint_name=endpoint_name,
        image_path=args.image,
        s3_uri=args.s3_image,
        top_k=args.top_k
    )
    
    logger.info("✅ Test completed successfully!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
