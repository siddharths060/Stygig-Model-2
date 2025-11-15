# SageMaker Asynchronous Inference - Implementation Guide

**Date:** November 15, 2025  
**Status:** ✅ **MVP COMPLETE**

---

## Overview

This implementation migrates the StyGig Fashion Recommendation System from **real-time inference** to **asynchronous inference** to solve the `ModelError: invocation timed out` issue caused by CLIP model cold start exceeding 60 seconds.

### Problem Statement

```
ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation: 
Received server error (0) from primary with message "Your invocation timed out while 
waiting for a response from container primary."
```

**Root Cause:** The CLIP model (ViT-B-32) takes 2-3 minutes to load during cold start, exceeding the 60-second real-time inference timeout limit.

### Solution

**SageMaker Asynchronous Inference** removes the timeout constraint by:
- Queueing requests and processing them asynchronously
- Returning results via S3 + SNS notifications
- Supporting processing times > 60 seconds (up to 15 minutes)
- Scaling to zero when idle (cost-effective)

---

## Architecture

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
│   Client    │──1──▶ │   Upload Input   │       │   S3 Bucket │
│             │       │   Payload to S3  │──────▶│   (Input)   │
└─────────────┘       └──────────────────┘       └─────────────┘
                               │
                               │ 2. InvokeEndpointAsync
                               ▼
                      ┌──────────────────┐
                      │  SageMaker Async │
                      │    Endpoint      │
                      └──────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
           ┌─────────────┐       ┌─────────────┐
           │ S3 Bucket   │       │  SNS Topic  │
           │  (Output)   │       │(Notification)│
           └─────────────┘       └─────────────┘
                    │                     │
                    └──────────┬──────────┘
                               │ 3. Retrieve Result
                               ▼
                      ┌──────────────────┐
                      │     Client       │
                      └──────────────────┘
```

### Workflow

1. **Upload Input**: Client uploads payload (JSON with base64 image) to S3
2. **Invoke Async**: Client calls `invoke_endpoint_async()` with S3 InputLocation
3. **Get OutputLocation**: SageMaker returns S3 OutputLocation immediately
4. **Process**: SageMaker processes request asynchronously (can take minutes)
5. **Notify**: SNS sends notification when processing completes
6. **Retrieve**: Client downloads result from S3 OutputLocation

---

## Files Created

### 1. `sagemaker/deploy_async_endpoint.py`

**Purpose:** Deploy SageMaker endpoint with Asynchronous Inference configuration

**Key Features:**
- ✅ Creates `AsyncInferenceConfig` with S3OutputPath and NotificationConfig
- ✅ Configures SNS topics for success/error notifications
- ✅ Uses `boto3` SageMaker client with proper error handling
- ✅ Maintains all timeout settings for CLIP model loading
- ✅ Auto-creates SNS topics and S3 buckets if needed

**Critical Code Section:**

```python
# Create Endpoint Configuration with AsyncInferenceConfig
async_config = {
    'OutputConfig': {
        'S3OutputPath': s3_output_path,  # Where results are saved
        'NotificationConfig': {
            'SuccessTopic': sns_topic_arn,  # SNS for successful predictions
            'ErrorTopic': sns_topic_arn     # SNS for failed predictions
        }
    },
    'ClientConfig': {
        'MaxConcurrentInvocationsPerInstance': max_concurrent_invocations
    }
}

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[...],
    AsyncInferenceConfig=async_config  # ⭐ CRITICAL: Enables async inference
)
```

**Usage:**

```bash
# Deploy new async endpoint
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz

# Deploy with custom settings
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://bucket/model.tar.gz \
    --s3-output-path s3://my-bucket/results/ \
    --sns-topic-arn arn:aws:sns:ap-south-1:123456789:my-topic \
    --max-concurrent-invocations 10
```

---

### 2. `sagemaker/invoke_async.py`

**Purpose:** Invoke asynchronous endpoint and retrieve results

**Key Features:**
- ✅ Uses `invoke_endpoint_async()` method from `sagemaker-runtime`
- ✅ Uploads payload to S3 to get InputLocation
- ✅ Prints OutputLocation from response
- ✅ Optional: Wait for result and display recommendations
- ✅ Handles timeout gracefully (polls S3 for result)

**Critical Code Section:**

```python
# 1. Upload payload to S3
s3_client.put_object(
    Bucket=bucket,
    Key=s3_key,
    Body=json.dumps(payload),
    ContentType='application/json'
)
input_location = f's3://{bucket}/{s3_key}'

# 2. Invoke async endpoint
runtime_client = boto3.client('sagemaker-runtime', region_name=region)
response = runtime_client.invoke_endpoint_async(
    EndpointName=endpoint_name,
    InputLocation=input_location,  # ⭐ S3 URI of input payload
    ContentType='application/json'
)

# 3. Get output location
output_location = response['OutputLocation']  # ⭐ Where result will be saved
print(f"Output Location: {output_location}")
```

**Usage:**

```bash
# Basic invocation (returns OutputLocation immediately)
python sagemaker/invoke_async.py

# Wait for result and display
python sagemaker/invoke_async.py --wait --display-results

# Use specific image
python sagemaker/invoke_async.py --image my_image.jpg --wait --display-results

# Use S3 image
python sagemaker/invoke_async.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg \
    --wait --display-results
```

---

## Configuration Details

### AsyncInferenceConfig Structure

```python
{
    'OutputConfig': {
        'S3OutputPath': 's3://bucket/async-inference-results/',
        'NotificationConfig': {
            'SuccessTopic': 'arn:aws:sns:region:account:success-topic',
            'ErrorTopic': 'arn:aws:sns:region:account:error-topic'
        },
        # Optional: KMS key for encryption
        'KmsKeyId': 'arn:aws:kms:region:account:key/...'
    },
    'ClientConfig': {
        'MaxConcurrentInvocationsPerInstance': 5  # Concurrent requests per instance
    }
}
```

### Environment Variables (Maintained for CLIP Model)

```python
env = {
    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',      # 5 min for model_fn()
    'MODEL_SERVER_TIMEOUT': '300',                # 5 min per request
    'TS_DEFAULT_RESPONSE_TIMEOUT': '300',         # TorchServe timeout
    'MODEL_SERVER_WORKERS': '1',                  # Single worker
    'TS_DEFAULT_WORKERS_PER_MODEL': '1',
    'OMP_NUM_THREADS': '2',                       # CPU optimization
    'MKL_NUM_THREADS': '2',
    'TOKENIZERS_PARALLELISM': 'false',
    'SAGEMAKER_PROGRAM': 'inference.py',
    'SAGEMAKER_SUBMIT_DIRECTORY': source_s3_uri
}
```

---

## SNS Notification Format

### Success Notification

```json
{
  "invocationStatus": "Completed",
  "requestId": "abc123...",
  "outputLocation": "s3://bucket/async-inference-results/abc123.out",
  "invocationTime": "2025-11-15T12:00:00Z",
  "completionTime": "2025-11-15T12:03:45Z"
}
```

### Error Notification

```json
{
  "invocationStatus": "Failed",
  "requestId": "abc123...",
  "failureLocation": "s3://bucket/async-inference-results/abc123.err",
  "invocationTime": "2025-11-15T12:00:00Z",
  "completionTime": "2025-11-15T12:02:30Z"
}
```

---

## Testing & Validation

### Step 1: Deploy Async Endpoint

```bash
# Use the latest model artifact
MODEL_URI="s3://stygig-ml-s3/model-artifacts/stygig-training-1762145223/output/model.tar.gz"

python sagemaker/deploy_async_endpoint.py --model-uri $MODEL_URI
```

**Expected Output:**

```
✅ ASYNC ENDPOINT DEPLOYED SUCCESSFULLY!
  Endpoint name: stygig-async-endpoint-20251115-120000
  S3 Output: s3://stygig-ml-s3/async-inference-results/
  SNS Topic: arn:aws:sns:ap-south-1:123456789:stygig-async-inference-notifications
```

### Step 2: Test Async Invocation

```bash
# Quick test (returns OutputLocation immediately)
python sagemaker/invoke_async.py

# Wait for result
python sagemaker/invoke_async.py --wait --display-results
```

**Expected Output:**

```
✅ ASYNC INVOCATION SUBMITTED SUCCESSFULLY
  Output Location: s3://stygig-ml-s3/async-inference-results/abc123.out

⏳ Waiting for inference result...
✓ Result available after 180s (attempt 18)

📊 INFERENCE RESULTS
🎯 TOP 5 RECOMMENDATIONS:
1. train/upperwear/shirt/0042.jpg
   Category: upperwear
   Gender: Men
   Overall Score: 0.9234
   ...
```

### Step 3: Verify S3 Results

```bash
# List output files
aws s3 ls s3://stygig-ml-s3/async-inference-results/

# Download result manually
aws s3 cp s3://stygig-ml-s3/async-inference-results/abc123.out result.json
cat result.json | jq .
```

### Step 4: Subscribe to SNS Notifications

```bash
# Subscribe email to SNS topic
aws sns subscribe \
    --topic-arn arn:aws:sns:ap-south-1:123456789:stygig-async-inference-notifications \
    --protocol email \
    --notification-endpoint your-email@example.com

# Confirm subscription via email
```

---

## Comparison: Real-Time vs Asynchronous

| Feature | Real-Time Inference | Asynchronous Inference |
|---------|---------------------|------------------------|
| **Timeout Limit** | 60 seconds | 15 minutes |
| **Response** | Immediate (synchronous) | Via S3 + SNS (async) |
| **Cold Start** | ❌ Times out (2-3 min) | ✅ Works (2-3 min OK) |
| **Cost** | Always running | Scales to zero |
| **Use Case** | Low-latency (<60s) | Long-running (>60s) |
| **Queueing** | No | Yes |
| **Scaling** | Manual | Automatic |
| **Notification** | N/A | SNS topic |

---

## Migration Checklist

### Before Migration

- [x] Identify timeout errors in real-time endpoint
- [x] Confirm CLIP model cold start time (2-3 minutes)
- [x] Review current endpoint configuration

### During Migration

- [x] Create feature branch: `git checkout -b feature/sagemaker-async-inference`
- [x] Deploy async endpoint with `deploy_async_endpoint.py`
- [x] Create SNS topics for notifications
- [x] Configure S3 bucket for input/output
- [x] Test async invocation with `invoke_async.py`

### After Migration

- [ ] Verify async endpoint handles cold start without timeout
- [ ] Subscribe to SNS notifications
- [ ] Update application code to use async API
- [ ] Monitor CloudWatch logs for errors
- [ ] Delete old real-time endpoint (optional)

---

## Troubleshooting

### Issue 1: "Endpoint not found"

**Solution:** Verify endpoint name in `async_endpoint_info.json`

```bash
cat async_endpoint_info.json | grep endpoint_name
```

### Issue 2: "Access denied to S3 bucket"

**Solution:** Update IAM role with S3 permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject"
  ],
  "Resource": [
    "arn:aws:s3:::stygig-ml-s3/*"
  ]
}
```

### Issue 3: "SNS topic not found"

**Solution:** Auto-created by deployment script, or specify with `--sns-topic-arn`

### Issue 4: Result not available after 10 minutes

**Possible Causes:**
1. Inference still processing (wait longer)
2. Error during inference (check FailureLocation in S3)
3. CloudWatch logs show error

**Check Logs:**

```bash
# Get log stream name
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/Endpoints/stygig-async-endpoint-20251115-120000 \
    --order-by LastEventTime \
    --descending \
    --max-items 1

# Tail logs
aws logs tail /aws/sagemaker/Endpoints/stygig-async-endpoint-20251115-120000 --follow
```

---

## Cost Optimization

### Asynchronous Inference Pricing

**Components:**
1. **Instance Hours:** Only when processing requests (scales to zero)
2. **S3 Storage:** Input/output payloads (~$0.023/GB/month)
3. **SNS Notifications:** ~$0.50 per 1M notifications
4. **Data Transfer:** Standard S3/SNS pricing

**Example Monthly Cost (100 requests/day):**

```
Instance time: 100 requests × 3 min × 30 days = 150 hours
Instance cost: 150 hours × $0.096/hour (ml.m5.large) = $14.40

S3 storage: 100 × 30 × 1MB × 2 (input+output) = 6GB × $0.023 = $0.14
SNS: 100 × 30 × 2 (success+error) = 6,000 × $0.0000005 = $0.003

Total: ~$14.54/month
```

**vs Real-Time Endpoint (24/7):**

```
Instance cost: 720 hours × $0.096/hour = $69.12/month
```

**Savings: ~76% cost reduction** for low-traffic workloads

---

## Next Steps

### Phase 1: MVP Testing (Current)

- [x] Deploy async endpoint
- [x] Test basic invocation
- [ ] Validate results match real-time endpoint
- [ ] Measure end-to-end latency

### Phase 2: Integration

- [ ] Update API Gateway to use async endpoint
- [ ] Implement webhook for SNS notifications
- [ ] Add retry logic for failed invocations
- [ ] Create monitoring dashboard

### Phase 3: Production

- [ ] Setup CloudWatch alarms
- [ ] Configure auto-scaling policies
- [ ] Implement request throttling
- [ ] Add comprehensive logging

---

## Reference Commands

### Deploy Async Endpoint

```bash
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz \
    --instance-type ml.m5.large \
    --max-concurrent-invocations 5
```

### Invoke Async Endpoint

```bash
# Quick invocation
python sagemaker/invoke_async.py

# With waiting
python sagemaker/invoke_async.py --wait --display-results

# Custom image
python sagemaker/invoke_async.py \
    --image my_fashion_item.jpg \
    --top-k 10 \
    --wait --display-results
```

### Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
    --endpoint-name stygig-async-endpoint-20251115-120000
```

### Delete Endpoint

```bash
aws sagemaker delete-endpoint \
    --endpoint-name stygig-async-endpoint-20251115-120000

aws sagemaker delete-endpoint-config \
    --endpoint-config-name stygig-async-endpoint-20251115-120000-config

aws sagemaker delete-model \
    --model-name stygig-async-endpoint-20251115-120000-model
```

---

## Additional Resources

### AWS Documentation

- [SageMaker Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
- [InvokeEndpointAsync API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointAsync.html)
- [AsyncInferenceConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AsyncInferenceConfig.html)

### Project Documentation

- `Docs/TIMEOUT_FIX_COMPLETE.md` - Real-time endpoint timeout fixes
- `Docs/deploy_sagemaker.md` - General deployment guide
- `Docs/QUICKSTART.md` - Quick start guide

---

**Implementation Completed:** November 15, 2025  
**Status:** ✅ MVP Ready for Testing  
**Branch:** `feature/sagemaker-async-inference`

---

## Summary

This implementation provides a complete solution to the timeout issue by:

1. ✅ **Creating feature branch** for async inference work
2. ✅ **Refactoring deployment script** with `AsyncInferenceConfig`
   - S3OutputPath for results
   - NotificationConfig for SNS topics
   - Professional error handling
3. ✅ **Creating invocation script** (`invoke_async.py`)
   - Uploads payload to S3 (InputLocation)
   - Calls `invoke_endpoint_async()`
   - Prints OutputLocation
   - Optional: Wait and display results

**All MVP requirements met!** 🎉
