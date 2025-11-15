# 🚀 V4 Deployment Quick Start Guide

## ✅ You Are Here

You've completed all V4 code changes. Now it's time to deploy to AWS SageMaker.

---

## 🎯 Deployment Strategy

### **Recommended: Update Inference Code Only** (No Retraining)

Since V4 changes are only in the **inference logic** (not the model/embeddings), you can deploy without retraining:

✅ **Pros:** Fast (minutes), cheap, low risk  
❌ **Cons:** Metadata needs migration (gray fallback if images unavailable)

### **Alternative: Full Retrain** (Complete Rebuild)

Only do this if you have updated images/dataset:

✅ **Pros:** Fresh embeddings, perfect V4 metadata  
❌ **Cons:** Slow (hours), expensive, requires GPU

---

## 📋 Step-by-Step: Deploy V4 (Recommended Path)

### **Prerequisites**

```bash
# 1. Install AWS CLI + boto3
pip install boto3 awscli

# 2. Configure AWS credentials
aws configure

# 3. Verify access to your SageMaker endpoint
aws sagemaker describe-endpoint --endpoint-name YOUR_ENDPOINT_NAME --region YOUR_REGION
```

---

### **Option 1: Automated Python Script** ⚡ (Easiest)

```bash
cd /g/Stygig/stygig_project

# Deploy V4 to SageMaker
python deploy_v4_sagemaker.py \
    --endpoint stygig-production \
    --region us-east-1 \
    --s3-bucket your-sagemaker-bucket

# This will:
# ✓ Download your current V3 model from S3
# ✓ Migrate metadata to V4 format
# ✓ Package V4 inference code
# ✓ Upload to S3
# ✓ Print deployment commands
```

**Then create staging endpoint:**

```bash
# Use the S3 path printed by the script
python sagemaker/deploy_existing_model.py \
    --model-data s3://your-bucket/stygig-models/v4/model-v4-TIMESTAMP.tar.gz \
    --endpoint-name stygig-v4-staging \
    --instance-type ml.m5.xlarge
```

---

### **Option 2: Manual Step-by-Step** 🛠️ (Full Control)

#### Step 1: Download Current Model from S3

```bash
# Get your current model S3 path
aws sagemaker describe-endpoint --endpoint-name YOUR_ENDPOINT_NAME --region YOUR_REGION

# Look for "ModelDataUrl" in the output, something like:
# s3://sagemaker-us-east-1-123456/stygig/model.tar.gz

# Download it
aws s3 cp s3://YOUR_BUCKET/YOUR_MODEL_PATH/model.tar.gz model_v3.tar.gz
```

#### Step 2: Extract and Check Metadata

```bash
# Extract
mkdir model_v3_extracted
tar -xzf model_v3.tar.gz -C model_v3_extracted

# Check metadata format
python -c "
import pickle
with open('model_v3_extracted/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
first = metadata[list(metadata.keys())[0]]
print('Fields:', list(first.keys()))
print('Has color_rgb?', 'color_rgb' in first)
"
```

#### Step 3: Migrate Metadata (if needed)

```bash
# If metadata doesn't have 'color_rgb', run migration
python migrate_metadata_v3_to_v4.py \
    --input model_v3_extracted/metadata.pkl \
    --output model_v3_extracted/metadata.pkl
```

#### Step 4: Package V4 Model

```bash
# Create package directory
mkdir model_v4_package

# Copy V4 code
cp sagemaker/inference.py model_v4_package/
cp -r src model_v4_package/
cp -r config model_v4_package/

# Copy existing artifacts
cp model_v3_extracted/metadata.pkl model_v4_package/
cp model_v3_extracted/*.index model_v4_package/ 2>/dev/null || true
cp model_v3_extracted/*.json model_v4_package/ 2>/dev/null || true

# Create requirements.txt
cat > model_v4_package/requirements.txt << EOF
torch==2.0.0
open-clip-torch==2.20.0
faiss-cpu==1.7.4
Pillow==10.0.0
numpy==1.24.0
EOF

# Package it
tar -czf model_v4.tar.gz -C model_v4_package .
```

#### Step 5: Upload to S3

```bash
# Upload
aws s3 cp model_v4.tar.gz s3://YOUR_BUCKET/stygig-models/v4/model_v4.tar.gz

# Note the S3 URI for next steps
echo "s3://YOUR_BUCKET/stygig-models/v4/model_v4.tar.gz"
```

#### Step 6: Deploy to Staging

```bash
# Create staging endpoint
python sagemaker/deploy_existing_model.py \
    --model-data s3://YOUR_BUCKET/stygig-models/v4/model_v4.tar.gz \
    --endpoint-name stygig-v4-staging \
    --instance-type ml.m5.xlarge
```

#### Step 7: Test Staging Endpoint

```bash
# Run tests
python sagemaker/test_endpoint.py --endpoint-name stygig-v4-staging

# Check CloudWatch logs for V4 features:
# - "🔍 Inferred query category by vote" (top-5 voting)
# - "Applied 1.15x boost" (category boost)
# - RGB tuples in response
```

#### Step 8: Update Production (if tests pass)

```bash
# Gradual rollout: Update production endpoint
python sagemaker/redeploy_endpoint.py \
    --endpoint-name stygig-production \
    --model-data s3://YOUR_BUCKET/stygig-models/v4/model_v4.tar.gz

# Monitor for 24 hours
# If issues: roll back to V3 using old model S3 path
```

---

## 🧪 Testing V4 Endpoint

### Basic Test

```bash
python sagemaker/test_endpoint.py --endpoint-name stygig-v4-staging
```

### Check Logs for V4 Features

```bash
# View CloudWatch logs
aws logs tail /aws/sagemaker/Endpoints/stygig-v4-staging --follow

# Look for these V4 indicators:
# ✓ "Inferred query category by vote: shirt (from Counter({'shirt': 4, 't-shirt': 1}))"
# ✓ "Applied 1.15x boost to bottomwear_pants (compatible with upperwear_shirt)"
# ✓ "dominant_color_rgb": [128, 128, 128]
```

### Manual Inference Test

```python
import boto3
import json
from PIL import Image
import io
import base64

# Load test image
image = Image.open('test_image.jpg')
buffer = io.BytesIO()
image.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

# Invoke endpoint
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
response = runtime.invoke_endpoint(
    EndpointName='stygig-v4-staging',
    ContentType='application/x-image',
    Body=image_bytes
)

result = json.loads(response['Body'].read())
print(json.dumps(result, indent=2))

# Check for V4 features:
# - result['query_item']['dominant_color_rgb'] should be [R, G, B]
# - Logs should show "vote" and "boost" messages
```

---

## ⚠️ Important Notes

### Metadata Migration Caveat

The migration script sets default gray color `(128, 128, 128)` for all items. For production use, you should:

**Option A:** Extract colors from actual images (best quality)
```bash
# Modify migrate script to access image files
# This requires downloading images from S3 or having them locally
```

**Option B:** Use V3 color names as approximation (quick fix)
```python
# In migrate script, map color names to RGB:
COLOR_MAP = {
    'gray': (128, 128, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    # ... etc
}
item['color_rgb'] = COLOR_MAP.get(item.get('color', 'gray'), (128, 128, 128))
```

**Option C:** Retrain with fresh metadata (most accurate)

---

## 🔄 If You Need to Retrain Instead

```bash
# 1. Update your data preprocessing to output color_rgb
# 2. Re-run training pipeline
python sagemaker/run_sagemaker_pipeline.py

# 3. This will generate fresh:
#    - FAISS index
#    - CLIP embeddings
#    - V4 metadata.pkl with actual RGB values
#    - config.json

# 4. Deploy the trained model
python sagemaker/deploy_endpoint.py --endpoint-name stygig-v4-production
```

---

## 📊 Success Metrics to Monitor

After deployment, track these metrics:

| Metric | Target |
|--------|--------|
| **Inference Latency** | <120ms (same as V3) |
| **Error Rate** | <1% |
| **Category Accuracy** | >90% (check logs) |
| **User CTR** | +10% vs V3 |

---

## 🆘 Troubleshooting

### Issue: "color_rgb not found in metadata"

**Solution:** Re-run metadata migration or use V3 color name mapping

### Issue: Import errors in inference.py

**Solution:** Verify `src/` and `config/` folders are in model package

### Issue: High latency (>200ms)

**Solution:** Check instance type (use ml.m5.xlarge or better)

### Issue: Category voting not showing in logs

**Solution:** Set logging level to INFO in CloudWatch

---

## ✅ Deployment Checklist

- [ ] AWS credentials configured
- [ ] Current V3 model downloaded from S3
- [ ] Metadata migrated to V4 format
- [ ] V4 code packaged with artifacts
- [ ] Uploaded to S3
- [ ] Staging endpoint created
- [ ] Tests pass on staging
- [ ] Logs show V4 features (voting, boost)
- [ ] Latency <120ms
- [ ] Error rate <1%
- [ ] Production updated
- [ ] Monitor for 24 hours

---

## 🎉 You're Ready!

**Run this command to start:**

```bash
python deploy_v4_sagemaker.py \
    --endpoint YOUR_ENDPOINT_NAME \
    --region YOUR_REGION \
    --s3-bucket YOUR_BUCKET
```

---

**Questions? Check:**
- `V4_IMPLEMENTATION_COMPLETE.md` - Technical details
- `V4_DEPLOYMENT_CHECKLIST.md` - Full checklist
- `V4_RELEASE_NOTES.md` - What's new in V4
