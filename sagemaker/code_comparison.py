"""
SageMaker Asynchronous Inference - Code Comparison
==================================================

This file shows the key code differences between real-time and async inference.
"""

# =============================================================================
# REAL-TIME INFERENCE (OLD - Times out)
# =============================================================================

# Deployment Code (deploy_endpoint.py)
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data=model_uri,
    role=role,
    entry_point='sagemaker/inference.py',
    framework_version='2.0.0',
    py_version='py310',
    env={
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',
        # ... other env vars ...
    }
)

# Deploy with timeout settings
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name,
    container_startup_health_check_timeout=600,
    model_data_download_timeout=600
)
# ❌ PROBLEM: Still times out after 60s during cold start

# Invocation Code (test_endpoint.py)
import boto3

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)
# ❌ PROBLEM: Times out waiting for response

result = json.loads(response['Body'].read())


# =============================================================================
# ASYNCHRONOUS INFERENCE (NEW - Solves timeout)
# =============================================================================

# Deployment Code (deploy_async_endpoint.py)
import boto3

sm_client = boto3.client('sagemaker')

# Step 1: Create Model (same as before)
model_name = f'{endpoint_name}-model'
sm_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_uri,
        'Environment': {
            'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',
            # ... other env vars ...
        }
    },
    ExecutionRoleArn=role
)

# Step 2: Create Endpoint Config with AsyncInferenceConfig
# ⭐ KEY CHANGE: Add AsyncInferenceConfig
async_config = {
    'OutputConfig': {
        'S3OutputPath': s3_output_path,  # Where results are saved
        'NotificationConfig': {
            'SuccessTopic': sns_topic_arn,  # SNS for success
            'ErrorTopic': sns_topic_arn     # SNS for errors
        }
    },
    'ClientConfig': {
        'MaxConcurrentInvocationsPerInstance': 5
    }
}

endpoint_config_name = f'{endpoint_name}-config'
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large',
            'InitialVariantWeight': 1.0
        }
    ],
    AsyncInferenceConfig=async_config  # ⭐ CRITICAL: Enables async inference
)

# Step 3: Create Endpoint (same as before)
sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Invocation Code (invoke_async.py)
import boto3
import json

s3_client = boto3.client('s3')
runtime_client = boto3.client('sagemaker-runtime')

# ⭐ KEY CHANGE 1: Upload payload to S3 first
payload = {
    'image': image_base64,
    'top_k': 5,
    'min_score': 0.5
}

bucket = 'stygig-ml-s3'
key = 'async-inference-input/request-12345.json'

s3_client.put_object(
    Bucket=bucket,
    Key=key,
    Body=json.dumps(payload),
    ContentType='application/json'
)

input_location = f's3://{bucket}/{key}'

# ⭐ KEY CHANGE 2: Use invoke_endpoint_async instead of invoke_endpoint
response = runtime_client.invoke_endpoint_async(
    EndpointName=endpoint_name,
    InputLocation=input_location,  # ⭐ S3 URI instead of Body
    ContentType='application/json'
)

# ⭐ KEY CHANGE 3: Get OutputLocation instead of immediate result
output_location = response['OutputLocation']
print(f"Result will be saved to: {output_location}")

# ⭐ KEY CHANGE 4: Poll S3 for result (or wait for SNS notification)
import time

while True:
    try:
        # Parse output location
        output_bucket, output_key = parse_s3_uri(output_location)
        
        # Try to get result
        obj = s3_client.get_object(Bucket=output_bucket, Key=output_key)
        result = json.loads(obj['Body'].read().decode('utf-8'))
        
        print("✅ Result ready!")
        print(result)
        break
        
    except s3_client.exceptions.NoSuchKey:
        # Result not ready yet
        print("⏳ Waiting for result...")
        time.sleep(10)  # Wait 10 seconds before checking again


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

"""
DEPLOYMENT CHANGES:
-------------------
1. Use boto3 SageMaker client instead of SageMaker SDK
2. Add AsyncInferenceConfig to create_endpoint_config()
3. Specify S3OutputPath for results
4. Specify NotificationConfig with SNS topics

INVOCATION CHANGES:
-------------------
1. Upload payload to S3 instead of sending in request body
2. Use invoke_endpoint_async() instead of invoke_endpoint()
3. Pass InputLocation (S3 URI) instead of Body
4. Get OutputLocation from response instead of immediate result
5. Poll S3 or subscribe to SNS for result notification

BENEFITS:
---------
✅ No 60-second timeout limit (supports up to 15 minutes)
✅ Solves CLIP model cold start timeout issue
✅ Scales to zero when idle (cost-effective)
✅ Queues requests during high traffic
✅ SNS notifications for async workflow integration

TRADE-OFFS:
-----------
❌ Latency: Results not immediate (need to poll or subscribe)
❌ Complexity: More moving parts (S3, SNS, polling)
❌ State management: Need to track request/response mapping
"""
