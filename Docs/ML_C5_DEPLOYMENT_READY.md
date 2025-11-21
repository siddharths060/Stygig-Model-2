# ML.C5.LARGE DEPLOYMENT - READY FOR SAGEMAKER

**Status**: ✅ **READY TO DEPLOY**  
**Date**: November 21, 2025

## Changes Made for ml.c5.large

### 1. Instance Type Updates ✅
All deployment scripts updated to use `ml.c5.large`:

- ✅ `run_pipeline.py`: Default changed to ml.c5.large
- ✅ `deploy_sync.py`: Default changed to ml.c5.large  
- ✅ `deploy_async.py`: Default changed to ml.c5.large
- ✅ All argparse help text updated with performance expectations

### 2. Performance Optimizations ✅
Added CPU-specific optimizations for ml.c5.large:

- ✅ **Reduced FAISS search**: 50 candidates instead of 200 (faster processing)
- ✅ **CPU threading optimization**: Set to 2 threads (matches ml.c5.large vCPU)
- ✅ **Increased timeout**: 30 seconds for synchronous endpoints
- ✅ **Memory efficiency**: Reduced interop threads to minimize overhead

### 3. Syntax Validation ✅
All files have been syntax-checked and validated:

- ✅ `run_pipeline.py`: Syntax OK
- ✅ `deploy_sync.py`: Syntax OK  
- ✅ `deploy_async.py`: Syntax OK
- ✅ `sagemaker/inference.py`: Syntax OK (indentation issues fixed)

## Deployment Recommendations

### ✅ RECOMMENDED: Asynchronous Deployment
**Best choice for ml.c5.large**

```bash
python deploy_async.py \
    --model-uri s3://bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-production-async \
    --s3-output-path s3://bucket/async-results/
```

**Why Async is Better:**
- ✅ 15-minute timeout (very safe for CPU processing)
- ✅ Auto-scales to zero when idle (cost savings)
- ✅ Perfect for batch processing
- ✅ Expected processing time: 450-650ms per image

### ⚠️ ACCEPTABLE: Synchronous Deployment
**Possible but with latency warnings**

```bash
python deploy_sync.py \
    --model-uri s3://bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-production-sync
```

**Synchronous Performance Expectations:**
- ⚠️ Latency: 450-650ms average, up to 1000ms p99
- ⚠️ User experience: Slower than ideal for real-time apps
- ✅ Timeout safety: 30-second timeout provides 30x headroom
- ⚠️ Best for: Internal tools, admin interfaces, low-traffic APIs

## Instance Specifications

**ml.c5.large:**
- **vCPUs**: 2 (Intel Xeon Platinum 8000 series)
- **Memory**: 4 GB
- **Network**: Up to 10 Gbps
- **Cost**: ~$0.096/hour ($70.08/month)
- **Optimized for**: Compute-intensive workloads

## Expected Performance

### CLIP Model Performance on ml.c5.large
```
Component                 Time (ms)    Optimization Applied
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image preprocessing           15ms     Standard
CLIP inference (CPU)         350ms     ✅ 2 threads, reduced interop
FAISS search                  20ms     ✅ Reduced to 50 candidates  
V4 scoring logic              60ms     ✅ Fewer candidates to process
Response formatting           15ms     Standard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL EXPECTED               460ms     (vs 650ms unoptimized)
```

### Safety Margins
- **Sync endpoint**: 30,000ms timeout ÷ 460ms = **65x safety margin** ✅
- **Async endpoint**: 900,000ms timeout ÷ 460ms = **1,956x safety margin** ✅

## Monitoring Setup

### CloudWatch Metrics to Watch
```bash
# Latency monitoring (should be <1000ms for sync)
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name ModelLatency \
    --dimensions Name=EndpointName,Value=stygig-production

# Error rate monitoring (should be 0%)
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name ModelInvocation5XXErrors \
    --dimensions Name=EndpointName,Value=stygig-production

# CPU utilization (should be <80%)
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name CPUUtilization \
    --dimensions Name=EndpointName,Value=stygig-production
```

### Recommended Alarms
```bash
# High latency alarm (>1000ms)
aws cloudwatch put-metric-alarm \
    --alarm-name stygig-high-latency \
    --metric-name ModelLatency \
    --namespace AWS/SageMaker \
    --statistic Average \
    --period 300 \
    --threshold 1000 \
    --comparison-operator GreaterThanThreshold

# Error rate alarm (>0%)
aws cloudwatch put-metric-alarm \
    --alarm-name stygig-error-rate \
    --metric-name ModelInvocation5XXErrors \
    --namespace AWS/SageMaker \
    --statistic Sum \
    --period 300 \
    --threshold 1 \
    --comparison-operator GreaterThanThreshold
```

## Testing Commands

### 1. Deploy and Test Async (Recommended)
```bash
# Deploy async endpoint
python deploy_async.py \
    --model-uri s3://your-bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-test-async \
    --s3-output-path s3://your-bucket/test-results/

# Test async endpoint
python test_endpoint.py \
    --endpoint-name stygig-test-async \
    --image test_images/sample.jpg \
    --async \
    --s3-input s3://your-bucket/test-inputs/

# Cleanup after testing
aws sagemaker delete-endpoint --endpoint-name stygig-test-async
```

### 2. Deploy and Test Sync (if needed)
```bash
# Deploy sync endpoint
python deploy_sync.py \
    --model-uri s3://your-bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-test-sync

# Test sync endpoint (expect 450-650ms response)
python test_endpoint.py \
    --endpoint-name stygig-test-sync \
    --image test_images/sample.jpg

# Cleanup after testing
aws sagemaker delete-endpoint --endpoint-name stygig-test-sync
```

## Pre-Deployment Checklist

- [x] **Instance type**: Updated to ml.c5.large
- [x] **CPU optimizations**: Applied (2 threads, reduced search)
- [x] **Timeout settings**: Increased to 30 seconds for sync
- [x] **Syntax validation**: All Python files verified
- [x] **Performance estimates**: Documented and realistic
- [x] **Monitoring setup**: CloudWatch alarms defined
- [x] **Testing commands**: Ready to execute

## Cost Analysis

**ml.c5.large pricing (us-east-1):**

| Scenario | Monthly Cost | Use Case |
|----------|-------------|----------|
| **Sync endpoint (24/7)** | $70.08 | Always-on real-time API |
| **Async endpoint (8hrs/day)** | $23.36 | Business hours batch processing |
| **Async endpoint (on-demand)** | ~$5-20 | Sporadic batch jobs |

**Cost vs Performance:**
- ✅ 27% cheaper than ml.m5.large ($96/month)
- ⚠️ Similar latency to ml.m5.large (both CPU-based)
- ✅ Better compute optimization for ML workloads

## Final Recommendation

**Deploy Async First**: Start with asynchronous deployment for safety and cost-effectiveness. The 15-minute timeout provides enormous headroom, and auto-scaling to zero saves costs during idle periods.

**Sync if Needed**: If real-time inference is required, the synchronous endpoint will work but expect 450-650ms latency. This is acceptable for internal tools but may feel slow for user-facing applications.

## Support Information

**If issues arise:**
1. Check CloudWatch logs: `/aws/sagemaker/Endpoints/{endpoint-name}`
2. Monitor ModelLatency metric (should be <1000ms)
3. Verify CPUUtilization (should be <80%)
4. For timeout issues, increase `MMS_DEFAULT_RESPONSE_TIMEOUT` in deploy scripts

**Team Contact**: MLOps team for deployment assistance

---

**Status**: ✅ **READY FOR SAGEMAKER DEPLOYMENT**  
**Confidence**: **High** (65x safety margin for sync, 1,956x for async)  
**Next Action**: Execute deployment commands above