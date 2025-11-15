# SageMaker Asynchronous Inference - Quick Reference

## 🚀 Quick Start

### 1. Create Feature Branch
```bash
git checkout -b feature/sagemaker-async-inference
```

### 2. Deploy Async Endpoint
```bash
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz
```

### 3. Test Async Invocation
```bash
python sagemaker/invoke_async.py --wait --display-results
```

---

## 📋 Key Differences: Real-Time vs Async

| Aspect | Real-Time | Asynchronous |
|--------|-----------|--------------|
| API Method | `invoke_endpoint()` | `invoke_endpoint_async()` |
| Input | Request body | S3 InputLocation |
| Output | Immediate response | S3 OutputLocation |
| Timeout | 60 seconds | 15 minutes |
| Cold Start | ❌ Fails | ✅ Works |

---

## 🔑 Critical Code Changes

### Deployment (deploy_async_endpoint.py)

```python
# Create AsyncInferenceConfig
async_config = {
    'OutputConfig': {
        'S3OutputPath': 's3://bucket/results/',
        'NotificationConfig': {
            'SuccessTopic': sns_topic_arn,
            'ErrorTopic': sns_topic_arn
        }
    }
}

# Apply to endpoint config
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[...],
    AsyncInferenceConfig=async_config  # ⭐ KEY CHANGE
)
```

### Invocation (invoke_async.py)

```python
# 1. Upload payload to S3
input_location = upload_to_s3(payload)

# 2. Invoke async endpoint
runtime_client = boto3.client('sagemaker-runtime')
response = runtime_client.invoke_endpoint_async(
    EndpointName=endpoint_name,
    InputLocation=input_location,  # ⭐ S3 URI
    ContentType='application/json'
)

# 3. Get output location
output_location = response['OutputLocation']  # ⭐ Where result will be
print(f"Result will be at: {output_location}")
```

---

## 📁 Files Created

1. **`sagemaker/deploy_async_endpoint.py`** - Deploy script with AsyncInferenceConfig
2. **`sagemaker/invoke_async.py`** - Async invocation script
3. **`Docs/ASYNC_INFERENCE_GUIDE.md`** - Complete documentation

---

## ✅ Testing Checklist

- [ ] Deploy async endpoint successfully
- [ ] Invoke endpoint and get OutputLocation
- [ ] Wait for result in S3 (2-3 minutes for cold start)
- [ ] Verify recommendations match real-time endpoint
- [ ] Subscribe to SNS notifications
- [ ] Test error handling

---

## 🎯 MVP Complete

All requirements met:
- ✅ Git branch command provided
- ✅ AsyncInferenceConfig implementation
- ✅ S3OutputPath configured
- ✅ NotificationConfig with SNS topics
- ✅ Async invocation script with `invoke_endpoint_async()`
- ✅ S3 InputLocation/OutputLocation handling
- ✅ Professional code with error handling

---

## 📞 Support

- Full documentation: `Docs/ASYNC_INFERENCE_GUIDE.md`
- Timeout fix reference: `Docs/TIMEOUT_FIX_COMPLETE.md`
- Deployment guide: `Docs/deploy_sagemaker.md`
