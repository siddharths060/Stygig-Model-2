# StyGig Scripts Quick Reference

## 🚀 Most Common Commands

```bash
# Full ML Pipeline
./scripts/run_pipeline.sh

# Deploy Model Only
./scripts/deploy_model.sh --training-job-name <job-name>

# Test Endpoint
python scripts/testing/test_endpoint.py --save-visual

# List Endpoints
python scripts/manage_endpoints.py list

# Delete All Endpoints (Cost Savings)
python scripts/manage_endpoints.py delete-all

# Verify Project Structure
./scripts/testing/verify_structure.sh

# Set Script Permissions
./scripts/set_permissions.sh
```

---

## 📁 Quick Navigation

| Task | Command |
|------|---------|
| **Train + Deploy** | `./scripts/run_pipeline.sh` |
| **Deploy Only** | `./scripts/deploy_model.sh --model-uri s3://...` |
| **Train Only** | `./scripts/run_pipeline.sh --skip-deployment` |
| **Test Endpoint** | `python scripts/testing/test_endpoint.py` |
| **Local Test** | `python scripts/testing/integration_test.py <image>` |
| **List Endpoints** | `python scripts/manage_endpoints.py list` |
| **Cleanup AWS** | `python scripts/manage_endpoints.py delete-all` |
| **Verify Setup** | `./scripts/testing/verify_structure.sh` |

---

## 🔧 Configuration

```bash
# Set these before running scripts
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1
```

---

## 📚 Documentation

- **Full Guide:** `scripts/README.md`
- **Migration:** `MIGRATION_GUIDE.md`
- **Summary:** `SCRIPT_REFACTORING_SUMMARY.md`

---

## ⚡ First Time Setup

```bash
chmod +x scripts/set_permissions.sh
./scripts/set_permissions.sh
./scripts/testing/verify_structure.sh
```
