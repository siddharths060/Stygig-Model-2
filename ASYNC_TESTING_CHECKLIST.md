# SageMaker Asynchronous Inference - Testing Checklist

## Pre-Deployment Checklist

- [ ] **Git Branch Created**
  ```bash
  git checkout -b feature/sagemaker-async-inference
  git status  # Verify on correct branch
  ```

- [ ] **Model Artifact Available**
  ```bash
  # Verify model exists in S3
  aws s3 ls s3://stygig-ml-s3/model-artifacts/ --recursive | grep model.tar.gz
  ```

- [ ] **IAM Role Permissions**
  - [ ] S3 read/write access
  - [ ] SageMaker full access
  - [ ] SNS publish permissions

- [ ] **AWS Credentials Configured**
  ```bash
  aws sts get-caller-identity  # Verify AWS credentials
  ```

---

## Deployment Testing

### Test 1: Deploy Async Endpoint

**Command:**
```bash
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz
```

**Expected Output:**
```
✅ ASYNC ENDPOINT DEPLOYED SUCCESSFULLY!
  Endpoint name: stygig-async-endpoint-YYYYMMDD-HHMMSS
  S3 Output: s3://stygig-ml-s3/async-inference-results/
  SNS Topic: arn:aws:sns:ap-south-1:...:stygig-async-inference-notifications
```

**Verification:**
```bash
# Check endpoint status
aws sagemaker describe-endpoint \
    --endpoint-name stygig-async-endpoint-YYYYMMDD-HHMMSS

# Verify endpoint is InService
# Expected: EndpointStatus: "InService"
```

**Pass Criteria:**
- [ ] Deployment completes without errors
- [ ] Endpoint status is "InService"
- [ ] `async_endpoint_info.json` created
- [ ] SNS topic created
- [ ] S3 output path accessible

---

### Test 2: Basic Async Invocation

**Command:**
```bash
python sagemaker/invoke_async.py
```

**Expected Output:**
```
✅ ASYNC INVOCATION SUBMITTED SUCCESSFULLY
  Output Location: s3://stygig-ml-s3/async-inference-results/abc123.out
```

**Verification:**
```bash
# Check if input uploaded to S3
aws s3 ls s3://stygig-ml-s3/async-inference-input/

# Verify output location format
# Expected: S3 URI with .out extension
```

**Pass Criteria:**
- [ ] Invocation returns OutputLocation
- [ ] InputLocation uploaded to S3
- [ ] No errors during invocation
- [ ] OutputLocation is valid S3 URI

---

### Test 3: Wait for Result

**Command:**
```bash
python sagemaker/invoke_async.py --wait --display-results
```

**Expected Output:**
```
⏳ Waiting for inference result...
✓ Result available after 180s (attempt 18)

📊 INFERENCE RESULTS
🎯 TOP 5 RECOMMENDATIONS:
1. train/upperwear/shirt/0042.jpg
   Category: upperwear
   Score: 0.9234
   ...
```

**Verification:**
```bash
# Check S3 for output file
aws s3 cp s3://stygig-ml-s3/async-inference-results/abc123.out result.json
cat result.json | jq .

# Verify result structure
# Expected: JSON with recommendations array
```

**Pass Criteria:**
- [ ] Result appears in S3 within 5 minutes
- [ ] Result contains valid recommendations
- [ ] Scores are between 0 and 1
- [ ] Input item metadata present

---

### Test 4: Custom Image Invocation

**Command:**
```bash
# Download a test image first
aws s3 cp s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg test_image.jpg

python sagemaker/invoke_async.py \
    --image test_image.jpg \
    --top-k 10 \
    --wait --display-results
```

**Pass Criteria:**
- [ ] Custom image processed successfully
- [ ] Returns 10 recommendations (top-k)
- [ ] Recommendations relevant to input category

---

### Test 5: Error Handling

**Test 5a: Invalid Image**
```bash
# Create invalid image file
echo "not an image" > invalid.jpg

python sagemaker/invoke_async.py --image invalid.jpg
```

**Expected:** Error message or error notification in SNS

**Test 5b: Missing Endpoint**
```bash
python sagemaker/invoke_async.py --endpoint-name nonexistent-endpoint
```

**Expected:** "Endpoint not found" error

**Pass Criteria:**
- [ ] Errors are caught and logged properly
- [ ] No uncaught exceptions
- [ ] Clear error messages displayed

---

### Test 6: SNS Notifications

**Setup:**
```bash
# Subscribe to SNS topic
SNS_TOPIC=$(cat async_endpoint_info.json | jq -r '.async_config.sns_topic_arn')

aws sns subscribe \
    --topic-arn $SNS_TOPIC \
    --protocol email \
    --notification-endpoint your-email@example.com
```

**Test:**
```bash
# Invoke endpoint
python sagemaker/invoke_async.py

# Wait for email notification
```

**Pass Criteria:**
- [ ] Confirmation email received
- [ ] Success notification received after inference
- [ ] Notification contains OutputLocation
- [ ] Notification arrives within 1 minute of completion

---

## Performance Testing

### Test 7: Cold Start Timeout Resolution

**Objective:** Verify async endpoint handles cold start without timeout

**Steps:**
1. Delete endpoint to force cold start:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
   ```

2. Redeploy endpoint:
   ```bash
   python sagemaker/deploy_async_endpoint.py --model-uri <model-uri>
   ```

3. Immediately invoke (triggers cold start):
   ```bash
   python sagemaker/invoke_async.py --wait --display-results
   ```

**Expected:**
- Cold start takes 2-3 minutes
- No timeout errors
- Result successfully returned

**Pass Criteria:**
- [ ] No "ModelError: invocation timed out" error
- [ ] Result available after 2-5 minutes
- [ ] Subsequent requests faster (<1 minute)

---

### Test 8: Concurrent Requests

**Command:**
```bash
# Run 5 invocations in parallel
for i in {1..5}; do
    python sagemaker/invoke_async.py &
done
wait
```

**Pass Criteria:**
- [ ] All 5 requests accepted
- [ ] All 5 results available in S3
- [ ] No throttling errors
- [ ] Max concurrent invocations respected

---

### Test 9: Large Payload

**Test:**
```bash
# Use high-resolution image
python sagemaker/invoke_async.py \
    --s3-image s3://stygig-ml-s3/large-image.jpg \
    --wait --display-results
```

**Pass Criteria:**
- [ ] Large images processed successfully
- [ ] No payload size errors
- [ ] Results consistent with smaller images

---

## Integration Testing

### Test 10: End-to-End Workflow

**Scenario:** Simulate production workflow

**Steps:**
1. Upload new image to S3
2. Invoke async endpoint
3. Subscribe to SNS notifications
4. Retrieve result from S3
5. Validate recommendations

**Pass Criteria:**
- [ ] Complete workflow executes without manual intervention
- [ ] Results available within SLA (5 minutes)
- [ ] Recommendations are accurate and relevant

---

## Comparison Testing

### Test 11: Results Match Real-Time Endpoint

**Objective:** Verify async endpoint returns identical results to real-time

**Steps:**
1. Deploy both real-time and async endpoints with same model
2. Send identical request to both
3. Compare results

**Command:**
```bash
# Real-time endpoint
python sagemaker/test_endpoint.py --endpoint-name <real-time-endpoint>

# Async endpoint
python sagemaker/invoke_async.py --wait --display-results

# Compare outputs
diff realtime_output.json async_output.json
```

**Pass Criteria:**
- [ ] Recommendations order matches
- [ ] Scores match (within 0.001 tolerance)
- [ ] Metadata consistent

---

## Monitoring & Cleanup

### Test 12: CloudWatch Logs

**Command:**
```bash
# View endpoint logs
aws logs tail /aws/sagemaker/Endpoints/stygig-async-endpoint-YYYYMMDD-HHMMSS --follow
```

**Pass Criteria:**
- [ ] Logs show model loading successfully
- [ ] Inference requests logged
- [ ] No error messages in logs

---

### Test 13: Cost Verification

**Check:**
```bash
# Check endpoint configuration
aws sagemaker describe-endpoint-config \
    --endpoint-config-name stygig-async-endpoint-YYYYMMDD-HHMMSS-config
```

**Pass Criteria:**
- [ ] Instance type correct (ml.m5.large)
- [ ] Instance count = 1
- [ ] AsyncInferenceConfig present

---

### Test 14: Cleanup

**Commands:**
```bash
# Delete endpoint
aws sagemaker delete-endpoint \
    --endpoint-name stygig-async-endpoint-YYYYMMDD-HHMMSS

# Delete endpoint config
aws sagemaker delete-endpoint-config \
    --endpoint-config-name stygig-async-endpoint-YYYYMMDD-HHMMSS-config

# Delete model
aws sagemaker delete-model \
    --model-name stygig-async-endpoint-YYYYMMDD-HHMMSS-model

# Optional: Clean S3
aws s3 rm s3://stygig-ml-s3/async-inference-input/ --recursive
aws s3 rm s3://stygig-ml-s3/async-inference-results/ --recursive
```

**Pass Criteria:**
- [ ] All resources deleted successfully
- [ ] No orphaned resources
- [ ] S3 cleaned up (optional)

---

## Sign-Off

### Testing Summary

- **Total Tests:** 14
- **Tests Passed:** _____ / 14
- **Tests Failed:** _____
- **Blockers:** _____

### Deployment Decision

- [ ] **APPROVED** - Ready for production
- [ ] **NEEDS WORK** - Issues to resolve
- [ ] **BLOCKED** - Critical issues

### Notes

```
Date: _______________
Tested By: _______________
Environment: Development / Staging / Production
Model Version: _______________

Issues Found:
-
-
-

Additional Comments:
-
-
```

---

## Quick Commands Reference

```bash
# Deploy
python sagemaker/deploy_async_endpoint.py --model-uri s3://bucket/model.tar.gz

# Invoke (quick)
python sagemaker/invoke_async.py

# Invoke (wait for result)
python sagemaker/invoke_async.py --wait --display-results

# Check status
aws sagemaker describe-endpoint --endpoint-name <endpoint-name>

# View logs
aws logs tail /aws/sagemaker/Endpoints/<endpoint-name> --follow

# Delete
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```
