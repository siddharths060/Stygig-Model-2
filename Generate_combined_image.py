#!/usr/bin/env python3
"""
visualize_s3_results.py

Standalone utility script to visualize fashion recommendation results.
Downloads query image and recommended images from S3, creates a comparison sheet.

Usage:
    python visualize_s3_results.py \
        --input_image_s3 s3://bucket/path/to/query.jpg \
        --json_s3 s3://bucket/path/to/results.json \
        --output_file comparison_sheet.png
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
    """
    Parse S3 URI into bucket and key.
    
    Args:
        s3_uri: Full S3 URI (e.g., s3://bucket/path/to/file.jpg)
    
    Returns:
        Tuple of (bucket_name, object_key)
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with s3://")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    return bucket, key


def download_image_from_s3(s3_client, s3_uri: str) -> Image.Image:
    """
    Download image from S3 and return as PIL Image.
    
    Args:
        s3_client: Boto3 S3 client
        s3_uri: Full S3 URI to image
    
    Returns:
        PIL Image object
    """
    bucket, key = parse_s3_uri(s3_uri)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        print(f"Error downloading {s3_uri}: {e}", file=sys.stderr)
        raise


def download_json_from_s3(s3_client, s3_uri: str) -> dict:
    """
    Download JSON file from S3 and parse it.
    
    Args:
        s3_client: Boto3 S3 client
        s3_uri: Full S3 URI to JSON file
    
    Returns:
        Parsed JSON as dictionary
    """
    bucket, key = parse_s3_uri(s3_uri)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        json_data = response['Body'].read().decode('utf-8')
        return json.loads(json_data)
    except Exception as e:
        print(f"Error downloading JSON {s3_uri}: {e}", file=sys.stderr)
        raise


def resize_image_proportional(image: Image.Image, target_height: int) -> Image.Image:
    """
    Resize image to target height while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_height: Desired height in pixels
    
    Returns:
        Resized PIL Image
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_width = int(target_height * aspect_ratio)
    
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def create_comparison_sheet(
    query_image: Image.Image,
    recommendation_images: List[Image.Image],
    recommendation_data: List[Dict],
    target_height: int = 300,
    padding: int = 20,
    font_size: int = 14
) -> Image.Image:
    """
    Create a single image with query image on left and recommendations grid on right.
    
    Args:
        query_image: The input/query image
        recommendation_images: List of recommended images
        recommendation_data: List of recommendation metadata (id, score, etc.)
        target_height: Height for each image thumbnail
        padding: Padding between images in pixels
        font_size: Font size for labels
    
    Returns:
        Combined PIL Image
    """
    # Resize query image
    query_resized = resize_image_proportional(query_image, target_height)
    
    # Resize all recommendation images
    rec_resized = [resize_image_proportional(img, target_height) for img in recommendation_images]
    
    # Calculate dimensions
    query_width = query_resized.width
    
    # Grid layout for recommendations (2 columns)
    num_cols = 2
    num_rows = (len(rec_resized) + num_cols - 1) // num_cols
    
    # Find max width in each column for alignment
    max_rec_width = max([img.width for img in rec_resized]) if rec_resized else 0
    
    # Calculate total dimensions
    label_height = 60  # Space for text labels
    
    grid_width = (max_rec_width * num_cols) + (padding * (num_cols + 1))
    grid_height = (target_height + label_height + padding) * num_rows + padding
    
    total_width = query_width + padding * 3 + grid_width
    total_height = max(query_resized.height + padding * 2, grid_height)
    
    # Create canvas
    canvas = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size + 2)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Paste query image on left with title
    query_x = padding
    query_y = padding + 30
    canvas.paste(query_resized, (query_x, query_y))
    
    # Draw "Query Image" title
    draw.text((query_x, padding), "QUERY IMAGE", fill='black', font=title_font)
    
    # Paste recommendations in grid on right
    rec_start_x = query_width + padding * 3
    
    draw.text((rec_start_x, padding), "RECOMMENDATIONS", fill='black', font=title_font)
    
    for idx, (img, data) in enumerate(zip(rec_resized, recommendation_data)):
        row = idx // num_cols
        col = idx % num_cols
        
        x = rec_start_x + col * (max_rec_width + padding)
        y = padding + 30 + row * (target_height + label_height + padding)
        
        # Paste image
        canvas.paste(img, (x, y))
        
        # Draw labels below image
        label_y = y + target_height + 5
        
        item_id = data.get('id', 'Unknown')
        score = data.get('score', 0.0)
        match_reason = data.get('match_reason', 'N/A')
        
        draw.text((x, label_y), f"#{idx + 1}: {item_id}", fill='black', font=font)
        draw.text((x, label_y + 18), f"Score: {score:.4f}", fill='darkgreen', font=font)
        draw.text((x, label_y + 36), f"{match_reason[:30]}...", fill='darkblue', font=font)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description='Visualize fashion recommendation results from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_s3_results.py \\
        --input_image_s3 s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \\
        --json_s3 s3://stygig-ml-s3/async-inference-results/results.json \\
        --output_file comparison_sheet.png
        """
    )
    
    parser.add_argument(
        '--input_image_s3',
        type=str,
        required=True,
        help='Full S3 URI to the query/input image (e.g., s3://bucket/path/image.jpg)'
    )
    
    parser.add_argument(
        '--json_s3',
        type=str,
        required=True,
        help='Full S3 URI to the results JSON file (e.g., s3://bucket/path/results.json)'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Local file path to save the comparison sheet (e.g., comparison_sheet.png)'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='ap-south-1',
        help='AWS region (default: ap-south-1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Fashion Recommendation Visualization Tool")
    print("=" * 80)
    print()
    
    # Initialize S3 client
    print(f"Initializing AWS S3 client (region: {args.region})...")
    s3_client = boto3.client('s3', region_name=args.region)
    
    # Download JSON results
    print(f"Downloading results JSON from: {args.json_s3}")
    results_data = download_json_from_s3(s3_client, args.json_s3)
    
    # Extract recommendations
    recommendations = results_data.get('recommendations', [])
    if not recommendations:
        print("ERROR: No recommendations found in JSON file", file=sys.stderr)
        return 1
    
    print(f"Found {len(recommendations)} recommendations")
    
    # Download query image
    print(f"Downloading query image from: {args.input_image_s3}")
    query_image = download_image_from_s3(s3_client, args.input_image_s3)
    print(f"✓ Query image loaded: {query_image.size}")
    
    # Download recommendation images
    print(f"Downloading {len(recommendations)} recommendation images...")
    recommendation_images = []
    
    for idx, rec in enumerate(recommendations, 1):
        rec_path = rec.get('path', '')
        
        # Handle both full S3 URIs and local paths
        if not rec_path.startswith('s3://'):
            # Path is a local filesystem path from training, need to construct S3 URI
            # Extract the relative path and construct S3 URI
            # Assuming training data is in s3://stygig-ml-s3/train/
            if '/opt/ml/input/data/training/' in rec_path:
                relative_path = rec_path.split('/opt/ml/input/data/training/')[-1]
                rec_path = f"s3://stygig-ml-s3/train/{relative_path}"
            else:
                print(f"WARNING: Cannot convert path to S3 URI: {rec_path}", file=sys.stderr)
                print(f"Skipping recommendation #{idx}", file=sys.stderr)
                continue
        
        try:
            print(f"  [{idx}/{len(recommendations)}] Downloading: {rec.get('id', 'Unknown')}...", end=' ')
            img = download_image_from_s3(s3_client, rec_path)
            recommendation_images.append(img)
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})", file=sys.stderr)
    
    if not recommendation_images:
        print("ERROR: No recommendation images could be downloaded", file=sys.stderr)
        return 1
    
    print(f"✓ Downloaded {len(recommendation_images)} images successfully")
    
    # Create comparison sheet
    print("Creating comparison sheet...")
    comparison_sheet = create_comparison_sheet(
        query_image,
        recommendation_images,
        recommendations[:len(recommendation_images)]  # Match downloaded images
    )
    
    # Save output
    print(f"Saving comparison sheet to: {args.output_file}")
    comparison_sheet.save(args.output_file, quality=95)
    
    file_size_kb = os.path.getsize(args.output_file) / 1024
    print(f"✓ Saved successfully ({file_size_kb:.1f} KB)")
    
    print()
    print("=" * 80)
    print("✅ Visualization Complete!")
    print("=" * 80)
    print(f"Output file: {args.output_file}")
    print(f"Query image: {query_image.size}")
    print(f"Recommendations visualized: {len(recommendation_images)}")
    print()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)