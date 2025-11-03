#!/usr/bin/env python3
"""
Test StyGig Fashion Recommendation Endpoint

This script tests your deployed endpoint with sample fashion items.
It downloads an image from S3, sends it to the endpoint, and displays recommendations.

Usage:
    python test_endpoint.py
    python test_endpoint.py --endpoint-name stygig-endpoint-20251103-062336
    python test_endpoint.py --s3-image s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg
    python test_endpoint.py --save-visual  # Creates visual output with images + JSON
    python test_endpoint.py --top-k 10 --save-visual
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
from PIL import Image, ImageDraw, ImageFont

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

def create_visual_output(input_image_path, recommendations, result, output_dir='test_results'):
    """Create visual output with input image and recommended items side by side."""
    try:
        import os
        from datetime import datetime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info("Creating visual output...")
        
        # Download recommended images from S3
        s3_client = boto3.client('s3', region_name='ap-south-1')
        rec_images = []
        
        for i, rec in enumerate(recommendations[:5]):  # Max 5 recommendations
            try:
                # Get image path from recommendation
                image_key = rec.get('image_path', '')
                if not image_key:
                    continue
                
                # Download from S3
                local_path = os.path.join(output_dir, f'rec_{i}_{timestamp}.jpg')
                s3_client.download_file('stygig-ml-s3', image_key, local_path)
                rec_images.append((local_path, rec))
                
            except Exception as e:
                logger.warning(f"Failed to download recommendation {i}: {e}")
                continue
        
        # Create combined image
        # Load input image
        input_img = Image.open(input_image_path)
        
        # Resize to standard size
        img_size = (300, 400)
        input_img = input_img.resize(img_size, Image.Resampling.LANCZOS)
        
        # Calculate output image dimensions
        num_images = len(rec_images) + 1  # input + recommendations
        cols = min(3, num_images)  # Max 3 columns
        rows = (num_images + cols - 1) // cols
        
        # Image dimensions with labels
        label_height = 120
        combined_width = cols * img_size[0]
        combined_height = rows * (img_size[1] + label_height)
        
        # Create blank canvas
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        draw = ImageDraw.Draw(combined_img)
        
        # Try to use a font, fallback to default
        try:
            font_large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
            font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Place input image
        combined_img.paste(input_img, (0, 0))
        
        # Add input label
        input_info = result.get('input_item', {})
        label_y = img_size[1] + 5
        draw.text((5, label_y), "INPUT IMAGE", fill='black', font=font_large)
        draw.text((5, label_y + 25), f"Category: {input_info.get('category', 'Unknown')}", fill='black', font=font_small)
        draw.text((5, label_y + 45), f"Gender: {input_info.get('gender', 'Unknown')}", fill='black', font=font_small)
        draw.text((5, label_y + 65), f"Colors: {', '.join(input_info.get('colors', [])[:3])}", fill='black', font=font_small)
        
        # Place recommendation images
        for idx, (rec_path, rec) in enumerate(rec_images):
            # Calculate position
            pos_idx = idx + 1
            col = pos_idx % cols
            row = pos_idx // cols
            x = col * img_size[0]
            y = row * (img_size[1] + label_height)
            
            # Load and resize recommendation image
            rec_img = Image.open(rec_path)
            rec_img = rec_img.resize(img_size, Image.Resampling.LANCZOS)
            
            # Paste image
            combined_img.paste(rec_img, (x, y))
            
            # Add label with recommendation info
            label_y = y + img_size[1] + 5
            draw.text((x + 5, label_y), f"RANK #{idx + 1} - Score: {rec['score']:.3f}", fill='green', font=font_large)
            draw.text((x + 5, label_y + 25), f"Category: {rec['category']}", fill='black', font=font_small)
            draw.text((x + 5, label_y + 45), f"Gender: {rec['gender']}", fill='black', font=font_small)
            draw.text((x + 5, label_y + 65), f"Colors: {', '.join(rec.get('colors', [])[:3])}", fill='black', font=font_small)
            draw.text((x + 5, label_y + 85), f"Color: {rec['color_score']:.2f} | Cat: {rec['category_score']:.2f} | Gen: {rec['gender_score']:.2f}", 
                     fill='blue', font=font_small)
        
        # Save combined image
        output_image_path = os.path.join(output_dir, f'recommendations_{timestamp}.jpg')
        combined_img.save(output_image_path, 'JPEG', quality=95)
        logger.info(f"✓ Visual output saved: {output_image_path}")
        
        # Save JSON result
        output_json_path = os.path.join(output_dir, f'recommendations_{timestamp}.json')
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"✓ JSON output saved: {output_json_path}")
        
        return output_image_path, output_json_path
        
    except Exception as e:
        logger.error(f"Failed to create visual output: {e}")
        return None, None

def test_endpoint(endpoint_name, image_path=None, s3_uri=None, top_k=5, save_visual=False):
    """Send test request to endpoint and display results."""
    try:
        # Create SageMaker runtime client with extended timeout
        from botocore.config import Config
        config = Config(read_timeout=300)  # 5 minutes timeout for first request (cold start)
        runtime = boto3.client('sagemaker-runtime', region_name='us-east-1', config=config)
        
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
        logger.info("⏳ Note: First request may take 1-2 minutes (cold start - loading model)")
        
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
        
        # Create visual output if requested
        if save_visual and recommendations:
            logger.info("")
            visual_path, json_path = create_visual_output(image_path, recommendations, result)
            if visual_path:
                logger.info(f"\n📸 VISUAL OUTPUT:")
                logger.info(f"   Image: {visual_path}")
                logger.info(f"   JSON: {json_path}")
        
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
    parser.add_argument('--save-visual', action='store_true', help='Save visual output with images and JSON')
    
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
        top_k=args.top_k,
        save_visual=args.save_visual
    )
    
    logger.info("✅ Test completed successfully!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
