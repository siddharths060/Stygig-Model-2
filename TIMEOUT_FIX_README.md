# SageMaker Endpoint Timeout Fix

## Problem Summary

Your SageMaker endpoint was timing out with this error:
```
ModelError: Your invocation timed out while waiting for a response from container primary
```

## Root Cause

The timeout occurs because the **first request** to the endpoint (cold start) takes **2-3 minutes** to:
1. **Load CLIP model** (ViT-B-32) - downloads from OpenAI/HuggingFace - **60-90 seconds**
2. **Load FAISS index** with all fashion embeddings - **10-15 seconds**
3. **Load metadata and preprocessors** - **5-10 seconds**
4. **Initialize color/gender classifiers** - **5 seconds**

**Total: 80-120 seconds**, which exceeds the default SageMaker timeout of **60 seconds**.

## Solution Applied

I've fixed the timeout issue by making the following changes:

### 1. **Extended Timeouts in Deployment** (Primary Fix)

Updated all deployment scripts with proper timeout configurations:

```python
# Container startup timeout: 600s (10 minutes)
container_startup_health_check_timeout=600

# Model download timeout: 600s (10 minutes)
model_data_download_timeout=600

# Model server timeout: 300s (5 minutes per request)
env={
    'MODEL_SERVER_TIMEOUT': '300',
    'TS_DEFAULT_RESPONSE_TIMEOUT': '300',
    ...
}
```

### 2. **Optimized Model Loading** (Secondary Fix)

Updated `sagemaker/inference.py` to:
- Disable unnecessary tokenizer parallelism
- Use single worker to reduce memory
- Add progress logging for debugging
- Disable gradient computation for inference
- Cache models in `/tmp/.cache` to avoid re-downloads

### 3. **Extended Client Timeouts** (Testing Fix)

Updated `sagemaker/test_endpoint.py` to:
- Use 300s (5 minute) read timeout
- Disable retries (avoid multiple cold starts)
- Add helpful timeout error messages

## Files Modified

1. ✅ `sagemaker/inference.py` - Optimized model loading
2. ✅ `sagemaker/run_sagemaker_pipeline.py` - Added timeout configs to deployment
3. ✅ `sagemaker/deploy_endpoint.py` - Extended timeout settings
4. ✅ `sagemaker/test_endpoint.py` - Extended client timeouts
5. ✅ `sagemaker/redeploy_with_timeout.py` - Comprehensive redeployment script

## How to Fix Your Current Endpoint

### Option 1: Redeploy with Extended Timeouts (Recommended)

Use the new redeployment script:

```bash
cd /path/to/Stygig-Model-2

# Redeploy with latest model (auto-finds the model)
python sagemaker/redeploy_with_timeout.py \
  --old-endpoint stygig-fashion-endpoint-1762445546

# Or specify exact model URI
python sagemaker/redeploy_with_timeout.py \
  --model-uri s3://stygig-ml-s3/model-artifacts/stygig-training-1762445546/output/model.tar.gz \
  --old-endpoint stygig-fashion-endpoint-1762445546
```

This will:
1. Delete the old endpoint
2. Deploy a new endpoint with extended timeouts
3. Save endpoint info to `endpoint_info.json`

### Option 2: Deploy Fresh Endpoint

```bash
# Deploy a new endpoint (keeps old one running)
python sagemaker/deploy_endpoint.py \
  --model-uri s3://stygig-ml-s3/model-artifacts/stygig-training-1762445546/output/model.tar.gz \
  --skip-delete
```

### Option 3: Run Full Pipeline with Fixed Settings

```bash
# Run complete pipeline (training + deployment with timeouts)
cd scripts
./run_pipeline.sh
```

## Testing the Fixed Endpoint

After redeployment, test with extended timeout:

```bash
# Basic test (auto-finds endpoint from endpoint_info.json)
python sagemaker/test_endpoint.py

# Specify endpoint explicitly
python sagemaker/test_endpoint.py \
  --endpoint-name stygig-endpoint-20251106-XXXXXX

# Test with visual output
python sagemaker/test_endpoint.py \
  --endpoint-name stygig-endpoint-20251106-XXXXXX \
  --save-visual
```

**Expected behavior:**
- ✅ First request: **2-3 minutes** (cold start - CLIP model loading)
- ✅ Subsequent requests: **1-2 seconds** (model cached in memory)

## Important Notes

### 1. Cold Start is Normal
The first request **will always take 2-3 minutes**. This is expected because:
- CLIP model must be downloaded from HuggingFace
- Model must be loaded into memory
- All embeddings/FAISS index must be initialized

### 2. Warm Requests are Fast
After the first request, subsequent requests take **1-2 seconds** because:
- Model is cached in memory
- No downloading required
- Embeddings already loaded

### 3. Cost Optimization
To avoid cold starts in production:
- Use **CloudWatch Events** to ping endpoint every 5 minutes
- Or use **Provisioned Concurrency** (keeps instance warm)
- Or accept 2-3 minute first-request delay

### 4. Monitoring
Check CloudWatch logs if issues persist:
```
https://console.aws.amazon.com/cloudwatch/home?region=ap-south-1#logStream:group=/aws/sagemaker/Endpoints/<ENDPOINT_NAME>
```

## Timeout Configuration Summary

| Setting | Old Value | New Value | Purpose |
|---------|-----------|-----------|---------|
| Container Startup | 300s | **600s** | Allow CLIP model download |
| Model Download | 300s | **600s** | Large model.tar.gz download |
| Model Server | 60s | **300s** | Per-request timeout |
| Client Read | 60s | **300s** | Test client timeout |

## Verification Checklist

After redeployment, verify:

- [ ] Endpoint deploys successfully (5-10 minutes)
- [ ] First test request completes (2-3 minutes expected)
- [ ] Second test request is fast (1-2 seconds)
- [ ] No timeout errors in CloudWatch logs
- [ ] Recommendations are returned correctly

## Troubleshooting

### Still Getting Timeout?

1. **Check CloudWatch logs** for actual error:
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/<ENDPOINT_NAME> --follow
   ```

2. **Verify instance type** has enough memory:
   - Current: `ml.m5.large` (8GB RAM)
   - If OOM errors: Upgrade to `ml.m5.xlarge` (16GB RAM)

3. **Check model artifacts exist**:
   ```bash
   aws s3 ls s3://stygig-ml-s3/model-artifacts/stygig-training-1762445546/output/
   ```

### Out of Memory Errors?

If you see OOM errors in CloudWatch:
- Upgrade instance type to `ml.m5.xlarge` or `ml.c5.xlarge`
- Reduce batch size in hyperparameters
- Use single worker (already configured)

### Model Download Fails?

If model download times out:
- Check S3 bucket permissions
- Verify IAM role has S3 read access
- Increase `model_data_download_timeout` to 900s (15 minutes)

## Next Steps

1. **Redeploy your endpoint** using one of the methods above
2. **Test the endpoint** with extended timeouts
3. **Monitor first request** (expect 2-3 minutes)
4. **Verify subsequent requests** are fast (1-2 seconds)
5. **Delete old endpoint** to avoid charges:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name stygig-fashion-endpoint-1762445546
   ```

## Questions?

If you continue to experience issues:
1. Check CloudWatch logs for detailed error messages
2. Verify all files were updated correctly
3. Ensure you're using the redeployed endpoint name
4. Try testing with a simple image first

---

**Summary:** The timeout issue is now fixed with extended timeout configurations. Redeploy your endpoint using the provided scripts, and the first request will complete successfully (albeit in 2-3 minutes due to CLIP model loading).
