# Async Endpoint Worker Crash - Root Cause & Fix

## Problem Identified

Your async endpoint is **crashing during model initialization** because the `model.tar.gz` is missing the source code dependencies.

### CloudWatch Logs Evidence
```
2025-11-15T10:25:31,929 [WARN] Auto recovery failed again
2025-11-15T10:25:35,397 io.netty.channel.unix.Errors$NativeIoException: readAddress(..) failed: Connection reset by peer
```

The TorchServe worker starts, tries to load the model, crashes immediately, and enters an infinite retry loop.

### Root Cause

The `model.tar.gz` contains:
- ✅ `embeddings.npz` - Image embeddings
- ✅ `faiss_index.index` - FAISS similarity index
- ✅ `code/inference.py` - Inference script
- ✅ `code/requirements.txt` - Python dependencies
- ✅ `config.json` - Model configuration
- ✅ `metadata.pkl` - Item metadata
- ❌ **MISSING**: `src/stygig/` - Source code directory

The `inference.py` imports:
```python
from stygig.core.color_logic import ColorProcessor
from stygig.core.gender_logic import GenderClassifier
```

These imports **fail** because the `stygig` package is not in the model archive, causing the worker to crash.

### Why This Happened

Bug in `sagemaker/train.py` line 669:
```python
source_dir = Path('/opt/ml/code/stygig')  # ❌ WRONG PATH
```

Should be:
```python
source_dir = Path('/opt/ml/code/src/stygig')  # ✅ CORRECT PATH
```

## Solution Options

### Option 1: Quick Fix (Recommended for Testing) - 10 minutes

Rebuild the model archive with the missing source code:

```bash
# On SageMaker Studio terminal
cd ~/Stygig-Model-2
python sagemaker/rebuild_model_archive.py
```

This will:
1. Download existing `model.tar.gz` from S3
2. Extract it to a temp directory
3. Add the missing `src/stygig/` directory
4. Add the missing `config/` directory
5. Rebuild and upload to `s3://stygig-ml-s3/model-artifacts/stygig-async-fixed/model.tar.gz`

Then deploy a new endpoint with the fixed model:
```bash
python sagemaker/deploy_async_endpoint.py \
    --model-data s3://stygig-ml-s3/model-artifacts/stygig-async-fixed/model.tar.gz \
    --sns-topic-arn arn:aws:sns:ap-south-1:732414292744:SNSMLTopic \
    --endpoint-name stygig-async-fixed-test
```

### Option 2: Fix Training Script (Recommended for Production) - 30+ minutes

Fix the bug in `train.py` and retrain:

1. Fix the path in `sagemaker/train.py`:
```python
# Line 669 - Change from:
source_dir = Path('/opt/ml/code/stygig')

# To:
source_dir = Path('/opt/ml/code/src/stygig')
```

2. Also add explicit config copy:
```python
# After line 673, add:
config_source = Path('/opt/ml/code/config')
config_target = model_dir / 'config'
if config_source.exists():
    shutil.copytree(config_source, config_target, dirs_exist_ok=True)
    logger.info(f"✅ Copied config to {config_target}")
```

3. Retrain the model:
```bash
python sagemaker/run_sagemaker_pipeline.py \
    --instance-type ml.m5.xlarge \
    --instance-count 1
```

4. Deploy with the new model artifacts

## Verification Steps

After deploying the fixed endpoint:

1. **Check CloudWatch logs** - should see:
```
✓ CLIP model loaded in X.Xs
✓ All models initialized
✓ Loaded metadata for N categories
```

2. **Test async invocation**:
```bash
python sagemaker/invoke_async.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \
    --wait \
    --display-results
```

3. **Expected timeline**:
   - Cold start: 2-3 minutes (model loading)
   - Warm inference: 5-10 seconds
   - Result file appears in S3 within timeout window

## Current Status

- ❌ Endpoint: `stygig-async-endpoint-20251115-101546` - **FAILING** (worker crash loop)
- 🔄 Next action: Run `rebuild_model_archive.py` to create fixed model
- 📋 After fix: Deploy new endpoint with complete model artifacts

## Commands Reference

### Check if endpoint is still failing:
```bash
aws logs tail /aws/sagemaker/Endpoints/stygig-async-endpoint-20251115-101546 --follow --since 5m
```

### Delete failing endpoint (after deploying fixed one):
```bash
aws sagemaker delete-endpoint --endpoint-name stygig-async-endpoint-20251115-101546
aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>
```

### List S3 async results:
```bash
aws s3 ls s3://stygig-ml-s3/async-inference-results/ --recursive
```
