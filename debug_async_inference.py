#!/usr/bin/env python3
"""
Check SageMaker async inference job status and troubleshoot issues.
"""

import boto3
import json
from datetime import datetime, timedelta

def main():
    print("=" * 80)
    print("SageMaker Async Inference Job Status Checker")
    print("=" * 80)
    print()
    
    # Initialize clients
    sagemaker = boto3.client('sagemaker', region_name='ap-south-1')
    s3 = boto3.client('s3', region_name='ap-south-1') 
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    
    endpoint_name = 'stygig-async-production'
    
    # 1. Check endpoint status
    print("1. ENDPOINT STATUS")
    print("-" * 40)
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"✓ Endpoint: {endpoint_name}")
        print(f"  Status: {response['EndpointStatus']}")
        print(f"  Created: {response['CreationTime']}")
        print(f"  Instance Type: {response['ProductionVariants'][0]['InstanceType']}")
        print(f"  Current Weight: {response['ProductionVariants'][0]['CurrentWeight']}")
        print()
    except Exception as e:
        print(f"✗ Error checking endpoint: {e}")
        return
    
    # 2. Check recent async invocations (CloudWatch logs would be better, but check S3 patterns)
    print("2. S3 BUCKET INVESTIGATION")
    print("-" * 40)
    
    bucket_name = 'sagemaker-ap-south-1-732414292744'
    
    # Check all prefixes that might contain async results
    prefixes_to_check = [
        'async-results/',
        'async-inference-output/',
        'sagemaker-async-inference/',
        'stygig-async-output/',
        'inference-results/',
        ''  # Check root level too
    ]
    
    found_any = False
    
    for prefix in prefixes_to_check:
        try:
            print(f"Checking s3://{bucket_name}/{prefix}")
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=50
            )
            
            if 'Contents' in response:
                objects = response['Contents']
                recent_objects = [obj for obj in objects 
                                if obj['LastModified'] > datetime.now(obj['LastModified'].tzinfo) - timedelta(hours=2)]
                
                if recent_objects:
                    found_any = True
                    print(f"  ✓ Found {len(recent_objects)} recent objects:")
                    for obj in sorted(recent_objects, key=lambda x: x['LastModified'], reverse=True)[:5]:
                        print(f"    - {obj['Key']} ({obj['Size']} bytes, {obj['LastModified']})")
                else:
                    print(f"  - No recent objects (found {len(objects)} total)")
            else:
                print(f"  - Empty")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()
    
    if not found_any:
        print("⚠️  No recent async results found in any common location")
        print()
    
    # 3. Check if we can manually invoke the endpoint to test it
    print("3. MANUAL ENDPOINT TEST")
    print("-" * 40)
    
    # Create a simple test payload
    test_image_s3 = "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png"
    test_payload = {
        "image_s3_uri": test_image_s3,
        "top_k": 5
    }
    
    print(f"Testing with payload: {test_payload}")
    
    try:
        response = sagemaker_runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            ContentType='application/json',
            InputLocation=f's3://{bucket_name}/async-test-input.json'
        )
        
        # First, upload the test payload
        s3.put_object(
            Bucket=bucket_name,
            Key='async-test-input.json',
            Body=json.dumps(test_payload),
            ContentType='application/json'
        )
        
        print(f"✓ Async invocation submitted")
        print(f"  Output Location: {response.get('OutputLocation', 'Not specified')}")
        print(f"  Invocation Status: Started")
        print()
        
        # Extract the result location
        if 'OutputLocation' in response:
            output_location = response['OutputLocation']
            print(f"Monitor this location for results: {output_location}")
            
            # Parse S3 location
            if output_location.startswith('s3://'):
                parts = output_location[5:].split('/', 1)
                result_bucket = parts[0]
                result_key = parts[1] if len(parts) > 1 else ''
                
                print(f"Result will be at: s3://{result_bucket}/{result_key}")
                print()
                print("Wait 2-5 minutes, then check:")
                print(f"aws s3 ls s3://{result_bucket}/{result_key.rsplit('/', 1)[0]}/ --human-readable")
        
    except Exception as e:
        print(f"✗ Error invoking endpoint: {e}")
        print()
        print("This could indicate:")
        print("- Endpoint configuration issues")
        print("- IAM permission problems")
        print("- Model loading failures")
    
    # 4. Provide next steps
    print("4. NEXT STEPS")
    print("-" * 40)
    print("If no results appear after 5-10 minutes:")
    print("1. Check CloudWatch logs for the endpoint")
    print("2. Verify the model artifacts are correct")
    print("3. Check SageMaker console for any error messages")
    print("4. Consider trying synchronous inference first for debugging")
    print()
    print("Commands to run:")
    print(f"aws logs describe-log-groups --log-group-name-prefix '/aws/sagemaker/Endpoints/{endpoint_name}'")
    print(f"aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
    print()

if __name__ == '__main__':
    main()