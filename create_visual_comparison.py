#!/usr/bin/env python3
"""
create_visual_comparison.py

Create a visual comparison of your StyGig V4 async endpoint results.
Downloads query image and recommendations, creates a side-by-side comparison.

Usage:
    python create_visual_comparison.py --result-location s3://bucket/path/to/result.json
    python create_visual_comparison.py --result-location s3://bucket/path/to/result.json --output comparison.png
"""

import argparse
import json
import os
import sys
from io import BytesIO
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import boto3
from PIL import Image, ImageDraw, ImageFont


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse S3 URI into bucket and key."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip('/')


def download_image_from_s3(s3_client, s3_uri: str) -> Image.Image:
    """Download image from S3 and return as PIL Image."""
    bucket, key = parse_s3_uri(s3_uri)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"❌ Error downloading {s3_uri}: {e}")
        raise


def download_json_from_s3(s3_client, s3_uri: str) -> dict:
    """Download JSON results from S3."""
    bucket, key = parse_s3_uri(s3_uri)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_data = response['Body'].read().decode('utf-8')
        return json.loads(json_data)
    except Exception as e:
        print(f"❌ Error downloading JSON {s3_uri}: {e}")
        raise


def resize_image_proportional(image: Image.Image, target_size: int) -> Image.Image:
    """Resize image maintaining aspect ratio."""
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = target_size
        new_height = int(target_size * original_height / original_width)
    else:
        new_height = target_size
        new_width = int(target_size * original_width / original_height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_stygig_comparison(
    query_image: Image.Image,
    recommendation_images: List[Image.Image],
    recommendation_data: List[Dict],
    query_info: Dict,
    processing_time_ms: float
) -> Image.Image:
    """Create StyGig-specific comparison sheet."""
    
    # Resize images
    img_size = 200
    query_resized = resize_image_proportional(query_image, img_size)
    rec_resized = [resize_image_proportional(img, img_size) for img in recommendation_images[:5]]
    
    # Canvas dimensions
    padding = 20
    title_height = 60
    label_height = 80
    
    canvas_width = (img_size + padding) * 6 + padding  # Query + 5 recommendations
    canvas_height = title_height + img_size + label_height + padding * 3
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = label_font = small_font = ImageFont.load_default()
    
    # Title
    title = f"StyGig V4 Fashion Recommendations (Processing: {processing_time_ms:.0f}ms)"
    draw.text((padding, padding), title, fill='darkblue', font=title_font)
    
    # Query image
    query_x = padding
    query_y = title_height + padding
    canvas.paste(query_resized, (query_x, query_y))
    
    # Query labels
    label_y = query_y + img_size + 5
    category = query_info.get('predicted_category', 'Unknown')
    gender = query_info.get('predicted_gender', 'Unknown')
    
    draw.text((query_x, label_y), "QUERY IMAGE", fill='black', font=label_font)
    draw.text((query_x, label_y + 15), f"Category: {category}", fill='darkgreen', font=small_font)
    draw.text((query_x, label_y + 28), f"Gender: {gender}", fill='darkgreen', font=small_font)
    
    # Recommendation images
    for idx, (img, data) in enumerate(zip(rec_resized, recommendation_data[:5])):
        x = padding + (idx + 1) * (img_size + padding)
        y = query_y
        
        canvas.paste(img, (x, y))
        
        # Labels
        label_y = y + img_size + 5
        score = data.get('similarity_score', 0.0)
        rec_category = data.get('category', 'Unknown')
        rec_gender = data.get('gender', 'Unknown')
        
        # Color harmony
        colors = data.get('color_harmony', {})
        primary = colors.get('primary', 'N/A')
        secondary = colors.get('secondary', 'N/A')
        
        draw.text((x, label_y), f"#{idx + 1} ({score:.3f})", fill='black', font=label_font)
        draw.text((x, label_y + 15), f"{rec_category}", fill='darkgreen', font=small_font)
        draw.text((x, label_y + 28), f"{rec_gender}", fill='darkgreen', font=small_font)
        draw.text((x, label_y + 41), f"{primary}+{secondary}", fill='purple', font=small_font)
    
    return canvas


def convert_training_path_to_s3(training_path: str, base_s3_uri: str = "s3://stygig-ml-s3/train/") -> str:
    """Convert training data path to S3 URI."""
    if training_path.startswith('s3://'):
        return training_path
    
    # Extract relative path from training data path
    if '/opt/ml/input/data/training/' in training_path:
        relative_path = training_path.split('/opt/ml/input/data/training/')[-1]
        return f"{base_s3_uri}{relative_path}"
    
    # Handle other path formats
    if training_path.startswith('/'):
        return f"{base_s3_uri}{training_path.lstrip('/')}"
    
    return f"{base_s3_uri}{training_path}"


def main():
    parser = argparse.ArgumentParser(description='Create StyGig V4 visual comparison')
    parser.add_argument('--result-location', required=True, help='S3 URI to async inference result JSON')
    parser.add_argument('--output', default='stygig_comparison.png', help='Output image file')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    
    args = parser.parse_args()
    
    print("🎯 StyGig V4 Visual Comparison Creator")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=args.region)
    
    # Download results JSON
    print(f"📥 Downloading results: {args.result_location}")
    results_data = download_json_from_s3(s3_client, args.result_location)
    
    # Extract data
    recommendations = results_data.get('recommendations', [])
    query_info = results_data.get('query_image_info', {})
    processing_time = results_data.get('processing_time_ms', 0)
    
    if not recommendations:
        print("❌ No recommendations found in results")
        return 1
    
    print(f"✅ Found {len(recommendations)} recommendations")
    print(f"📊 Processing time: {processing_time}ms")
    print(f"🔍 Query: {query_info.get('predicted_category')} ({query_info.get('predicted_gender')})")
    
    # Get query image path from first recommendation's context or use default
    query_image_s3 = "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png"
    
    # Download query image
    print(f"📥 Downloading query image: {query_image_s3}")
    query_image = download_image_from_s3(s3_client, query_image_s3)
    
    # Download recommendation images
    print(f"📥 Downloading {min(5, len(recommendations))} recommendation images...")
    recommendation_images = []
    
    for idx, rec in enumerate(recommendations[:5], 1):
        image_path = rec.get('image_path', '')
        
        # Convert training path to S3 URI
        s3_uri = convert_training_path_to_s3(image_path)
        
        try:
            print(f"  [{idx}/5] {s3_uri.split('/')[-1]}", end='... ')
            img = download_image_from_s3(s3_client, s3_uri)
            recommendation_images.append(img)
            print("✅")
        except Exception as e:
            print(f"❌ ({e})")
            continue
    
    if not recommendation_images:
        print("❌ No recommendation images could be downloaded")
        return 1
    
    # Create comparison
    print(f"🎨 Creating visual comparison...")
    comparison = create_stygig_comparison(
        query_image,
        recommendation_images,
        recommendations[:len(recommendation_images)],
        query_info,
        processing_time
    )
    
    # Save
    print(f"💾 Saving to: {args.output}")
    comparison.save(args.output, quality=95)
    
    file_size = os.path.getsize(args.output) / 1024
    print(f"✅ Saved! ({file_size:.1f} KB)")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Query: {query_info.get('predicted_category')} t-shirt")
    print(f"   Recommendations: {len(recommendation_images)} similar items")
    print(f"   Best match: {recommendations[0].get('similarity_score', 0):.4f} similarity")
    print(f"   Visual comparison: {args.output}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)