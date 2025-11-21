#!/usr/bin/env python3
"""
Check async inference results and list available result files.
"""

import boto3
import json
from datetime import datetime

def main():
    print("=" * 80)
    print("SageMaker Async Inference Results Checker")
    print("=" * 80)
    print()
    
    # Initialize clients
    sagemaker = boto3.client('sagemaker', region_name='ap-south-1')
    s3 = boto3.client('s3', region_name='ap-south-1')
    
    # Check endpoint status
    endpoint_name = 'stygig-async-production'
    print(f"Checking endpoint status: {endpoint_name}")
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"✓ Endpoint Status: {status}")
        print(f"  Creation Time: {response['CreationTime']}")
        print(f"  Last Modified: {response['LastModifiedTime']}")
        print()
    except Exception as e:
        print(f"✗ Error checking endpoint: {e}")
        return
    
    # List async results in S3
    bucket_name = 'sagemaker-ap-south-1-732414292744'
    async_prefix = 'async-results/'
    
    print(f"Checking S3 bucket: s3://{bucket_name}/{async_prefix}")
    print("Recent async inference results:")
    print("-" * 60)
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=async_prefix,
            StartAfter=async_prefix
        )
        
        results = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    results.append({
                        'Key': obj['Key'],
                        'LastModified': obj['LastModified'],
                        'Size': obj['Size']
                    })
        
        # Sort by last modified (newest first)
        results.sort(key=lambda x: x['LastModified'], reverse=True)
        
        if not results:
            print("No async results found yet.")
            print("\nThis could mean:")
            print("1. The async inference is still processing")
            print("2. The job failed")
            print("3. Results are stored in a different location")
            return
        
        print(f"Found {len(results)} result files:")
        print()
        
        for idx, result in enumerate(results[:10], 1):  # Show top 10
            key = result['Key']
            modified = result['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
            size_kb = result['Size'] / 1024
            
            print(f"{idx:2d}. {key}")
            print(f"    Modified: {modified}")
            print(f"    Size: {size_kb:.1f} KB")
            
            # Try to read the content if it's small enough
            if result['Size'] < 10000 and key.endswith('.out'):
                try:
                    obj_response = s3.get_object(Bucket=bucket_name, Key=key)
                    content = obj_response['Body'].read().decode('utf-8')
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(content)
                        if 'recommendations' in data:
                            num_recs = len(data['recommendations'])
                            print(f"    Content: ✓ Valid JSON with {num_recs} recommendations")
                        else:
                            print(f"    Content: JSON (no recommendations key)")
                    except json.JSONDecodeError:
                        print(f"    Content: Text ({len(content)} chars)")
                        if len(content) < 200:
                            print(f"    Preview: {content[:100]}...")
                
                except Exception as e:
                    print(f"    Content: Error reading - {e}")
            
            print()
        
        # Show the most recent valid result
        if results:
            latest_result = results[0]
            print("=" * 60)
            print("LATEST RESULT:")
            print(f"s3://{bucket_name}/{latest_result['Key']}")
            print("=" * 60)
            
            # Provide the corrected command
            print()
            print("Use this command to create the visualization:")
            print()
            print(f"python Generate_combined_image.py \\")
            print(f"    --input_image_s3 s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \\")
            print(f"    --json_s3 s3://{bucket_name}/{latest_result['Key']} \\")
            print(f"    --output_file stygig_v4_comparison.png")
            print()
    
    except Exception as e:
        print(f"✗ Error listing S3 objects: {e}")
        return

if __name__ == '__main__':
    main()