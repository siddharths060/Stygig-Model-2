#!/usr/bin/env python3
"""
Test async endpoint with S3 URI
"""

import boto3
import json
import time

def test_async_with_s3():
    print("=" * 60)
    print("Testing Async Endpoint with S3 URI")
    print("=" * 60)
    print()
    
    runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    s3 = boto3.client('s3', region_name='ap-south-1')
    
    bucket = 'sagemaker-ap-south-1-732414292744'
    endpoint_name = 'stygig-async-v4'
    
    # Create test payload with S3 URI
    payload = {
        "image_s3_uri": "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png",
        "top_k": 5
    }
    
    timestamp = int(time.time())
    input_key = f'async-inputs/test-{timestamp}.json'
    
    print(f"Test payload: {payload}")
    print(f"Uploading to: s3://{bucket}/{input_key}")
    
    # Upload input
    s3.put_object(
        Bucket=bucket,
        Key=input_key,
        Body=json.dumps(payload),
        ContentType='application/json'
    )
    
    print("✓ Input uploaded")
    print()
    
    # Submit async inference
    print(f"Invoking endpoint: {endpoint_name}")
    try:
        response = runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            ContentType='application/json',
            InputLocation=f's3://{bucket}/{input_key}'
        )
        
        output_location = response.get('OutputLocation', '')
        failure_location = response.get('FailureLocation', '')
        
        print("✅ ASYNC INFERENCE SUBMITTED!")
        print()
        print("Response:")
        print(f"  Output Location: {output_location}")
        if failure_location:
            print(f"  Failure Location: {failure_location}")
        print()
        
        # Parse output location
        if output_location:
            parts = output_location.replace('s3://', '').split('/', 1)
            result_bucket = parts[0]
            result_key = parts[1] if len(parts) > 1 else ''
            
            print("Monitoring for results...")
            print(f"Will check: s3://{result_bucket}/{result_key}")
            print()
            
            # Wait and check for results
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    # Check if result exists
                    s3.head_object(Bucket=result_bucket, Key=result_key)
                    
                    # Result exists! Download it
                    elapsed = int(time.time() - start_time)
                    print(f"✅ Result ready after {elapsed} seconds!")
                    print()
                    
                    obj = s3.get_object(Bucket=result_bucket, Key=result_key)
                    result_data = json.loads(obj['Body'].read().decode('utf-8'))
                    
                    print("Results:")
                    print(json.dumps(result_data, indent=2))
                    print()
                    
                    if 'recommendations' in result_data:
                        recs = result_data['recommendations']
                        print(f"Found {len(recs)} recommendations:")
                        for i, rec in enumerate(recs, 1):
                            print(f"  {i}. {rec.get('id', 'Unknown')} - Score: {rec.get('score', 0):.4f}")
                        
                        print()
                        print("=" * 60)
                        print("✅ SUCCESS! Now create visualization:")
                        print("=" * 60)
                        print()
                        print("python Generate_combined_image.py \\")
                        print(f"    --input_image_s3 {payload['image_s3_uri']} \\")
                        print(f"    --json_s3 {output_location} \\")
                        print("    --output_file stygig_v4_comparison.png")
                        print()
                    
                    return True
                    
                except s3.exceptions.ClientError:
                    # Not ready yet
                    elapsed = int(time.time() - start_time)
                    print(f"Waiting... ({elapsed}s)", end='\r')
                    time.sleep(5)
            
            print()
            print("⏱️ Timeout waiting for results")
            print(f"Check manually: aws s3 ls s3://{result_bucket}/{result_key.rsplit('/', 1)[0]}/")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    test_async_with_s3()
