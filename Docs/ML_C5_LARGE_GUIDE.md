# ML.C5.LARGE DEPLOYMENT GUIDE

## Instance Specifications
- **Instance Type**: ml.c5.large
- **vCPUs**: 2
- **Memory**: 4 GB
- **Network**: Up to 10 Gbps
- **Cost**: ~$0.096/hour ($70/month)

## Performance Expectations

### Synchronous Endpoint (ml.c5.large)
```
Expected Latency:
- Average: 450-650ms
- 99th percentile: 800ms
- Cold start: 1-2 seconds
```

**User Experience**: ⚠️ **BORDERLINE** - Acceptable for internal tools, may feel slow for user-facing apps

### Asynchronous Endpoint (ml.c5.large)
```
Processing Time: 450-650ms per image
Timeout: 15 minutes (SAFE)
Cost: Very economical for batch processing
```

**Recommendation**: ✅ **EXCELLENT** for batch processing

## Deployment Commands

### For Real-Time (Synchronous)
```bash
python deploy_sync.py \
    --model-uri s3://bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-c5-realtime
```

### For Batch Processing (Asynchronous) - RECOMMENDED
```bash
python deploy_async.py \
    --model-uri s3://bucket/models/v4-model.tar.gz \
    --endpoint-name stygig-c5-batch \
    --s3-output-path s3://bucket/results/
```

## Optimizations Applied

1. **Reduced FAISS search**: 50 candidates instead of 200
2. **CPU threading**: Optimized for 2 vCPU
3. **Increased timeout**: 30 seconds for sync endpoints
4. **Memory efficiency**: Reduced batch processing overhead

## Monitoring

Watch these CloudWatch metrics:
- ModelLatency (should be <1000ms for sync)
- ModelInvocation5XXErrors (should be 0%)
- CPUUtilization (should be <80%)

## Recommendation Summary

✅ **ASYNC DEPLOYMENT**: Highly recommended - safe, cost-effective
⚠️ **SYNC DEPLOYMENT**: Possible but expect 450-650ms latency