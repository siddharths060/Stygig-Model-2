# Quick Fix Guide: Deploy Working Endpoint

**Problem:** Workers crashing with `ValueError: Exactly one .pth or .pt file is required`  
**Solution:** Retrain with fixed code, then redeploy

---

## ⚡ Quick Fix (5 Steps)

### Step 1: Pull Latest Code
```bash
cd ~/Stygig-Model-2
git pull origin main
```
**Verify:** You should have commit `b78b171` or later

### Step 2: Train New Model
```bash
python sagemaker/run_sagemaker_pipeline.py
```
**Wait:** ~20 minutes for training  
**Look for:** `✅ Copied inference.py to /opt/ml/model/code` in logs  
**Note:** Save the training job name (e.g., `stygig-training-1762501234`)

### Step 3: Delete Old Broken Endpoint
```bash
# Get endpoint details
aws sagemaker describe-endpoint --endpoint-name stygig-endpoint-20251107-074457

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name stygig-endpoint-20251107-074457

# Wait for deletion
aws sagemaker wait endpoint-deleted --endpoint-name stygig-endpoint-20251107-074457

# Optional: Clean up endpoint config and model
aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>
aws sagemaker delete-model --model-name <model-name>
```

### Step 4: Deploy New Endpoint
```bash
python sagemaker/deploy_existing_model.py \
  --training-job-name stygig-training-XXXXX  # Use training job from Step 2
```
**Wait:** ~5 minutes for deployment  
**Note:** New endpoint name will be shown in output

### Step 5: Test Endpoint
```bash
python sagemaker/test_endpoint.py \
  --endpoint-name stygig-endpoint-YYYYMMDD-HHMMSS \  # Use new endpoint name
  --s3-image s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \
  --top-k 5 \
  --save-visual
```
**Expected:**
- First request: Takes 2-3 minutes (CLIP model loading)
- Returns 5 recommendations
- No errors about `.pth` files
- Workers stay running (check CloudWatch)

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] **Training logs** show: `✅ Copied inference.py to /opt/ml/model/code`
- [ ] **CloudWatch** has NO errors about `.pth` files
- [ ] **Workers** start and STAY running (no crash loop)
- [ ] **First inference** completes successfully (2-3 min)
- [ ] **Second inference** is fast (~1-2 sec)
- [ ] **Test output** has recommendations with scores

---

## 🔍 Troubleshooting

### Issue: Training fails
**Check:**
- Dataset at `s3://stygig-ml-s3/train/` exists
- Training instance has enough memory (ml.m5.large recommended)
- Requirements.txt dependencies installed

### Issue: Deployment times out
**Check:**
- Training completed successfully
- Model artifacts saved to S3
- SageMaker execution role has S3 access
- Increase `container_startup_health_check_timeout` to 600s

### Issue: Workers still crashing
**Check CloudWatch logs:**
```bash
aws logs tail /aws/sagemaker/Endpoints/<endpoint-name> --follow
```

**Look for:**
- `ValueError: Exactly one .pth` → Old model artifacts (retrain needed)
- `ModuleNotFoundError` → Dependencies missing (check requirements.txt)
- `MemoryError` → Increase instance size
- `CLIP model loading` → Normal, wait 2-3 minutes

### Issue: Inference returns errors
**Common causes:**
1. **Image format issues:** Convert to RGB, check size
2. **S3 access:** Verify endpoint IAM role has S3 read permissions
3. **Timeout:** First request takes 2-3 min (CLIP cold start)

---

## 📞 Quick Commands

### Check endpoint status:
```bash
aws sagemaker describe-endpoint --endpoint-name <name> --query 'EndpointStatus'
```

### Watch CloudWatch logs:
```bash
aws logs tail /aws/sagemaker/Endpoints/<name> --follow --format short
```

### Test endpoint quickly:
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name <name> \
  --body '{"image":"<base64-encoded-image>"}' \
  --content-type application/json \
  output.json
```

### List all endpoints:
```bash
aws sagemaker list-endpoints --query 'Endpoints[*].[EndpointName,EndpointStatus]' --output table
```

---

## 🚀 Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Pull code | 10 sec | Git pull |
| Training | 18-22 min | Processing 25K items |
| Delete old endpoint | 2-3 min | Cleanup |
| Deploy new endpoint | 4-6 min | Container startup |
| First inference test | 2-3 min | CLIP cold start |
| **Total** | **~30 min** | End-to-end fix |

---

## 💡 Key Changes (What Was Fixed)

1. **Training now copies `inference.py`** into `model.tar.gz`
2. **Deployment uses correct entry_point** (`inference.py` not `sagemaker/inference.py`)
3. **Model artifacts have proper structure** with `code/` directory

**Result:** SageMaker uses our custom handler instead of default PyTorch handler

---

## 🔗 More Information

- Full analysis: `Docs/CRITICAL_BUG_FIX.md`
- Training docs: `Docs/QUICKSTART.md`
- Deployment docs: `Docs/deploy_sagemaker.md`
- Testing guide: `scripts/README.md`

---

**Last Updated:** November 7, 2025  
**Fix Version:** b78b171 and later
