# 🚀 Quick Start Guide - StyGig Enterprise MVP

## Welcome!

You now have a **clean, enterprise-grade** fashion recommendation system. Here's how to get started in 5 minutes.

---

## ✅ Step 1: Verify Structure

```bash
cd stygig_project
./verify_structure.sh
```

**Expected output**: ✅ All 37 checks passed!

---

## ✅ Step 2: Choose Your Path

### Option A: AWS SageMaker Deployment (Production)

**Prerequisites:**
- AWS account with SageMaker access
- Dataset uploaded to `s3://stygig-ml-s3/train/`
- AWS CLI configured

**Run:**
```bash
./run_project.sh
```

This will:
1. ✅ Validate environment
2. ✅ Check AWS credentials
3. ✅ Verify S3 dataset
4. ✅ Install dependencies
5. ✅ Train model on SageMaker
6. ✅ Deploy to endpoint
7. ✅ Test the endpoint

**Time**: ~15-20 minutes (depending on dataset size)

---

### Option B: Local Testing (Development)

**Prerequisites:**
- Python 3.8+
- Local dataset in `outfits_dataset/train/`

**Run:**
```bash
# Install dependencies
pip install -r requirements_local.txt

# Test the engine
cd testing
python integration_test.py ../outfits_dataset/train/upperwear_shirt/sample.jpg \
    --gender male \
    --items-per-category 2
```

**Time**: ~2-3 minutes

---

## 📊 Example Output

```json
{
  "query_item": {
    "category": "upperwear_shirt",
    "gender": "male",
    "dominant_color": "blue"
  },
  "recommendations": [
    {
      "id": "pants_001",
      "category": "bottomwear_pants",
      "color": "black",
      "score": 0.8542,
      "match_reason": "harmonious colors (blue + black), gender appropriate"
    },
    {
      "id": "sneakers_042",
      "category": "footwear_sneakers",
      "color": "white",
      "score": 0.8321,
      "match_reason": "harmonious colors (blue + white), perfect style match"
    }
  ]
}
```

---

## 🔧 Configuration

### Quick Config Changes

Edit `config/settings.py` or set environment variables:

```bash
export TRAINING_INSTANCE_TYPE=ml.m5.large  # Or ml.c5.2xlarge for faster training
export BATCH_SIZE=32                        # Larger = faster but more memory
export DEBUG_MODE=true                      # Enable verbose logging
```

### Recommendation Tuning

Edit `config/recommendation_config.py`:

```python
RecommendationConfig(
    items_per_category=3,              # More items per category
    min_similarity_threshold=0.6,      # Higher = stricter matches
    enforce_category_diversity=True    # Keep diversity enabled
)
```

---

## 🧪 Testing Checklist

- [ ] Run `./verify_structure.sh` - All checks pass
- [ ] Test locally with sample image
- [ ] Verify dataset is in S3 (if using AWS)
- [ ] Run `./run_project.sh` for full pipeline
- [ ] Check outputs in `outputs/` directory

---

## 📚 Key Files to Know

| File | Purpose |
|------|---------|
| `run_project.sh` | Main execution script |
| `README.md` | Full documentation |
| `config/settings.py` | SageMaker configuration |
| `src/stygig/core/recommendation_engine.py` | Core logic |
| `src/stygig/core/rules/category_compatibility.py` | Category rules |

---

## 🆘 Troubleshooting

### "AWS credentials not configured"
```bash
aws configure
# Enter your Access Key ID, Secret Access Key, and Region
```

### "Dataset not found in S3"
```bash
# Upload your dataset
aws s3 sync outfits_dataset/train/ s3://stygig-ml-s3/train/ --region ap-south-1
```

### "Import errors"
```bash
# Install dependencies
pip install -r sagemaker/requirements.txt  # For AWS
pip install -r requirements_local.txt      # For local testing
```

### "Permission denied: run_project.sh"
```bash
chmod +x run_project.sh
```

---

## 🎯 What's Different from Old Code?

| Before | After |
|--------|-------|
| `ProfessionalFashionRecommendationEngine` | `FashionEngine` (clean name!) |
| Scattered files in root | Organized in `src/`, `sagemaker/`, `testing/` |
| Multiple setup scripts | Single `run_project.sh` |
| `local_output/` | `outputs/` (gitignored) |
| `mvp_integration.py` | `testing/integration_test.py` |

---

## 📖 Learn More

- **Full Documentation**: See `README.md`
- **Refactoring Details**: See `REFACTORING_SUMMARY.md`
- **API Reference**: See `README.md` → API Reference section
- **Architecture**: See `README.md` → How It Works section

---

## ✨ Next Steps

1. **Test locally** to verify everything works
2. **Upload dataset** to S3 if deploying to AWS
3. **Run pipeline** with `./run_project.sh`
4. **Customize** category rules in `src/stygig/core/rules/`
5. **Add features** to `src/stygig/api/` for REST endpoints

---

## 🤝 Need Help?

Check the troubleshooting section in `README.md` or review the comprehensive documentation.

---

**You're all set! 🎉**

Start with: `./verify_structure.sh` → `./run_project.sh`
