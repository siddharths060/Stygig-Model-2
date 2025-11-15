# SageMaker Async Inference - Troubleshooting Guide

## Problem 1: Worker Process Crashes (CRITICAL)

### Symptoms
- Endpoint deploys successfully but fails health checks
- CloudWatch logs show:
  ```
  [WARN] Auto recovery failed again
  [ERROR] Worker disconnected
  io.netty.channel.unix.Errors$NativeIoException: Connection reset by peer
  ```
- `/ping` health checks return HTTP 500
- Async invocations never complete (timeout after 10+ minutes)

### Root Cause
The `model.tar.gz` archive was **missing the source code dependencies** (`src/stygig/` directory).

The `inference.py` script imports:
```python
from stygig.core.color_logic import ColorProcessor
from stygig.core.gender_logic import GenderClassifier
```

These modules don't exist in the model archive, causing the TorchServe worker to crash during initialization.

### Why It Happened
Bug in `sagemaker/train.py` at line 669:
```python
# ❌ INCORRECT - Wrong path
source_dir = Path('/opt/ml/code/stygig')

# ✅ CORRECT - Proper path
source_dir = Path('/opt/ml/code/src/stygig')
```

The training script tried to copy from `/opt/ml/code/stygig` but the actual location is `/opt/ml/code/src/stygig`, so nothing was copied.

### Solution

#### Quick Fix (Immediate - 10 minutes)
Manually rebuild the model archive with missing dependencies:

```bash
# On SageMaker Studio
cd ~/Stygig-Model-2
python sagemaker/rebuild_model_archive.py
```

This script:
1. Downloads existing `model.tar.gz` from S3
2. Extracts to temporary directory
3. Adds missing `src/stygig/` directory
4. Adds missing `config/` directory  
5. Rebuilds and uploads to: `s3://stygig-ml-s3/model-artifacts/stygig-async-fixed/model.tar.gz`

#### Permanent Fix (Production - 30 minutes)
Fix the training script and retrain:

**Edit `sagemaker/train.py`:**
```python
# Line 669 - Fix source directory path
source_dir = Path('/opt/ml/code/src/stygig')  # Changed from '/opt/ml/code/stygig'
target_dir = model_dir / 'stygig'

if source_dir.exists():
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    logger.info(f"✅ Copied stygig source to {target_dir}")

# Add config directory copy
config_source = Path('/opt/ml/code/config')
config_target = model_dir / 'config'
if config_source.exists():
    shutil.copytree(config_source, config_target, dirs_exist_ok=True)
    logger.info(f"✅ Copied config to {config_target}")
```

Then retrain:
```bash
python sagemaker/run_sagemaker_pipeline.py \
    --instance-type ml.m5.xlarge \
    --instance-count 1
```

### Verification
After deploying the fixed model, check CloudWatch logs should show:
```
✓ CLIP model loaded in 2.3s
✓ All models initialized in 2.5s
✓ Loaded metadata for 7 categories
```

---

## Problem 2: Function Parameter Mismatch

### Symptoms
- Endpoint deploys successfully
- Health checks pass
- Async invocation completes in ~10 seconds
- Result contains error:
  ```
  ❌ Error: FashionRecommendationInference._generate_match_reason() 
  missing 1 required positional argument: 'color2'
  ```

### Root Cause
Function signature mismatch in `sagemaker/inference.py`:

**Function definition (line 517):**
```python
def _generate_match_reason(self, color_score: float, category_score: float, 
                         gender_score: float, color1: str, color2: str) -> str:
```

**Function call in `apply_enterprise_rule_based_scoring` (line 420):**
```python
# ❌ INCORRECT - Only 4 args, missing category_score
'match_reason': self._generate_match_reason(color_score, gender_score, query_color, item.get('color'))
```

### Solution

**Edit `sagemaker/inference.py` line 420:**
```python
# Change from:
'match_reason': self._generate_match_reason(color_score, gender_score, query_color, item.get('color'))

# To:
'match_reason': self._generate_match_reason(color_score, 1.0, gender_score, query_color, item.get('color'))
```

The second parameter is `category_score` - we use `1.0` as default since enterprise scoring doesn't calculate category compatibility separately.

### Deployment Steps

1. **Commit the fix to Git:**
   ```bash
   # On local machine
   cd G:/Stygig/stygig_project
   git add sagemaker/inference.py
   git commit -m "fix: Add missing category_score parameter to _generate_match_reason"
   git push origin main
   ```

2. **Pull on SageMaker and rebuild:**
   ```bash
   # On SageMaker Studio
   cd ~/Stygig-Model-2
   git pull origin main
   python sagemaker/rebuild_model_archive.py
   ```

3. **Deploy fixed endpoint:**
   ```bash
   python sagemaker/deploy_async_endpoint.py \
       --model-uri s3://stygig-ml-s3/model-artifacts/stygig-async-fixed/model.tar.gz \
       --sns-topic-arn arn:aws:sns:ap-south-1:732414292744:SNSMLTopic \
       --endpoint-name stygig-async-final
   ```

4. **Test:**
   ```bash
   python sagemaker/invoke_async.py \
       --endpoint-name stygig-async-final \
       --s3-image s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \
       --wait \
       --display-results
   ```

### Expected Output
```json
{
  "query_item": {
    "dominant_color": "blue",
    "predicted_gender": "male",
    "gender_confidence": 0.92
  },
  "recommendations": [
    {
      "id": "bottomwear_pants_123",
      "category": "bottomwear_pants",
      "gender": "male",
      "color": "navy",
      "score": 0.8745,
      "similarity_score": 0.9234,
      "color_harmony_score": 0.8890,
      "gender_compatibility_score": 1.0,
      "match_reason": "harmonious colors (blue+navy), gender appropriate"
    }
  ],
  "total_candidates": 150,
  "faiss_search_results": 200
}
```

---

## Testing Checklist

After deploying the fully fixed endpoint:

- [ ] **Health Checks Pass** - CloudWatch shows successful model loading
- [ ] **Fast Inference** - Cold start: 2-3 minutes, Warm: 5-10 seconds
- [ ] **Valid Results** - JSON response with recommendations (no errors)
- [ ] **Per-Category Limits** - 2 items per category as configured
- [ ] **Color Harmony** - Recommendations show complementary colors
- [ ] **Gender Filtering** - Male/female items filtered correctly
- [ ] **SNS Notifications** - Success messages arrive at SNS topic

---

## Cleanup Commands

Delete old failing endpoints:
```bash
# List all endpoints
aws sagemaker list-endpoints --region ap-south-1

# Delete specific endpoints
aws sagemaker delete-endpoint --endpoint-name stygig-async-endpoint-20251115-101546
aws sagemaker delete-endpoint --endpoint-name stygig-async-fixed-test
aws sagemaker delete-endpoint --endpoint-name stygig-async-endpoint-20251115-105400

# List and delete endpoint configs
aws sagemaker list-endpoint-configs --region ap-south-1
aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>

# List and delete models
aws sagemaker list-models --region ap-south-1
aws sagemaker delete-model --model-name <model-name>
```

---

## Summary

| Issue | Impact | Fix Time | Status |
|-------|--------|----------|--------|
| Missing source code in model.tar.gz | Worker crashes, endpoint unusable | 10 min | ✅ Fixed |
| Function parameter mismatch | Inference fails with error | 5 min | ✅ Fixed |

**Total Resolution Time:** ~15 minutes (quick fix) or ~30 minutes (permanent fix with retraining)

**Key Learnings:**
1. Always verify model.tar.gz contents before deployment
2. Test inference locally before deploying to SageMaker
3. Monitor CloudWatch logs during first invocation
4. Use `rebuild_model_archive.py` for quick fixes during development
5. Fix training script for production deployments
