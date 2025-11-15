#!/usr/bin/env python3
"""
Quick script to verify AWS setup and find model artifacts
"""
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import sys

def check_credentials():
    """Verify AWS credentials are configured"""
    try:
        sts_client = boto3.client('sts', region_name='ap-south-1')
        identity = sts_client.get_caller_identity()
        print("✅ AWS Credentials Configured")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        return True
    except NoCredentialsError:
        print("❌ No AWS credentials found")
        print("   Configure credentials with: aws configure")
        return False
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def find_latest_model():
    """Find the latest model artifact in S3"""
    try:
        s3_client = boto3.client('s3', region_name='ap-south-1')
        bucket = 'stygig-ml-s3'
        prefix = 'model-artifacts/'
        
        print(f"\n🔍 Searching for models in s3://{bucket}/{prefix}")
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            print(f"❌ No objects found in s3://{bucket}/{prefix}")
            return None
        
        # Find all model.tar.gz files
        model_files = [
            obj for obj in response['Contents'] 
            if obj['Key'].endswith('model.tar.gz')
        ]
        
        if not model_files:
            print(f"❌ No model.tar.gz files found")
            return None
        
        # Sort by last modified date
        latest_model = sorted(
            model_files, 
            key=lambda x: x['LastModified'], 
            reverse=True
        )[0]
        
        model_uri = f"s3://{bucket}/{latest_model['Key']}"
        size_mb = latest_model['Size'] / (1024 * 1024)
        
        print("✅ Latest Model Found:")
        print(f"   URI: {model_uri}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Last Modified: {latest_model['LastModified']}")
        
        return model_uri
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"❌ Bucket not found: {bucket}")
        else:
            print(f"❌ Error accessing S3: {e}")
        return None
    except Exception as e:
        print(f"❌ Error finding model: {e}")
        return None

def check_sagemaker_role():
    """Check if SageMaker execution role exists"""
    try:
        iam_client = boto3.client('iam', region_name='ap-south-1')
        paginator = iam_client.get_paginator('list_roles')
        
        print("\n🔍 Searching for SageMaker execution role...")
        
        for page in paginator.paginate():
            for role in page['Roles']:
                if 'AmazonSageMaker-ExecutionRole' in role['RoleName']:
                    print("✅ SageMaker Execution Role Found:")
                    print(f"   Role Name: {role['RoleName']}")
                    print(f"   Role ARN: {role['Arn']}")
                    return role['Arn']
        
        print("❌ No SageMaker execution role found")
        print("   Create one in the AWS Console or with aws iam create-role")
        return None
        
    except Exception as e:
        print(f"❌ Error checking IAM roles: {e}")
        return None

def main():
    print("=" * 80)
    print("AWS Setup Verification for SageMaker Async Inference")
    print("=" * 80)
    
    # Check credentials
    if not check_credentials():
        print("\n⚠️  Please configure AWS credentials before proceeding")
        sys.exit(1)
    
    # Find model
    model_uri = find_latest_model()
    
    # Check role
    role_arn = check_sagemaker_role()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if model_uri and role_arn:
        print("✅ All prerequisites met!")
        print("\nNext step - Deploy async endpoint:")
        print(f"\npython sagemaker/deploy_async_endpoint.py \\")
        print(f"    --model-uri {model_uri}")
        
        # Save to file for easy access
        with open('model_uri.txt', 'w') as f:
            f.write(model_uri)
        print(f"\n📄 Model URI saved to: model_uri.txt")
        
    else:
        print("⚠️  Missing prerequisites:")
        if not model_uri:
            print("   - Model artifact not found in S3")
        if not role_arn:
            print("   - SageMaker execution role not found")

if __name__ == '__main__':
    main()
