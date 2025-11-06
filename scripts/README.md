# StyGig Scripts Directory

This directory contains all consolidated, production-ready scripts for the StyGig Fashion Recommendation System. These scripts replace the old, scattered scripts in the project root.

## 📁 Directory Structure

```
scripts/
├── run_pipeline.sh          # Main ML pipeline orchestrator
├── deploy_model.sh          # Model deployment script
├── manage_endpoints.py      # Endpoint management utility
├── set_permissions.sh       # Script permissions setter
└── testing/                 # Testing & validation scripts
    ├── test_endpoint.py         # Endpoint testing
    ├── integration_test.py      # Local engine testing
    ├── local_train_test.py      # Local training simulation
    └── verify_structure.sh      # Project structure validation
```

---

## 🚀 Main Scripts

### 1. `run_pipeline.sh` - Complete ML Pipeline

**Purpose:** Orchestrates the complete machine learning pipeline from environment validation to endpoint deployment.

**Usage:**
```bash
# Full pipeline (train + deploy + test)
./scripts/run_pipeline.sh

# Train only (skip deployment)
./scripts/run_pipeline.sh --skip-deployment

# Deploy only (skip training)
./scripts/run_pipeline.sh --skip-training --training-job-name stygig-training-xxxxx

# Deploy without testing
./scripts/run_pipeline.sh --skip-testing
```

**Features:**
- ✅ Environment validation (AWS CLI, Python, credentials)
- ✅ S3 dataset verification
- ✅ Dependency installation
- ✅ SageMaker training job execution
- ✅ Model deployment
- ✅ Endpoint testing
- ✅ Comprehensive error handling and logging

---

### 2. `deploy_model.sh` - Model Deployment

**Purpose:** Deploy a trained model to SageMaker endpoint. Consolidates all deployment functionality into one clean script.

**Usage:**
```bash
# Deploy from model URI
./scripts/deploy_model.sh --model-uri s3://stygig-ml-s3/model.tar.gz

# Deploy from training job name
./scripts/deploy_model.sh --training-job-name stygig-training-1762145223

# Replace existing endpoint
./scripts/deploy_model.sh --model-uri s3://bucket/model.tar.gz --endpoint-name existing-endpoint --delete-existing

# Custom instance type
./scripts/deploy_model.sh --model-uri s3://bucket/model.tar.gz --instance-type ml.m5.xlarge
```

**Features:**
- 🎯 Deploy from S3 URI or training job name
- 🔄 Optional endpoint replacement
- ⚙️ Auto-detect IAM role
- ⏱️ Extended timeout configuration for CLIP model
- 📄 Saves endpoint info to JSON

---

### 3. `manage_endpoints.py` - Endpoint Management

**Purpose:** Manage and clean up AWS SageMaker endpoints for cost savings and resource management.

**Usage:**
```bash
# List all StyGig endpoints
python scripts/manage_endpoints.py list

# List only active endpoints
python scripts/manage_endpoints.py list --status InService

# Get detailed endpoint info
python scripts/manage_endpoints.py info --endpoint-name stygig-endpoint-20251103-062336

# Delete specific endpoint
python scripts/manage_endpoints.py delete --endpoint-name stygig-endpoint-20251103-062336

# Delete ALL StyGig endpoints (with confirmation)
python scripts/manage_endpoints.py delete-all

# Delete all without confirmation (BE CAREFUL!)
python scripts/manage_endpoints.py delete-all --yes

# JSON output
python scripts/manage_endpoints.py list --json
```

**Features:**
- 📋 List all project endpoints
- 🔍 Detailed endpoint information
- 🗑️ Delete individual endpoints
- 💰 Bulk delete for cost savings
- 🎯 Filter by status and prefix
- 📊 JSON output support

---

### 4. `set_permissions.sh` - Set Script Permissions

**Purpose:** Make all shell scripts executable. Run this after cloning the repository.

**Usage:**
```bash
./scripts/set_permissions.sh
```

**What it does:**
- Makes all `.sh` files in `scripts/` executable
- Makes all `.sh` files in `scripts/testing/` executable
- Reports count of files updated

---

## 🧪 Testing Scripts (`scripts/testing/`)

### 1. `test_endpoint.py` - Endpoint Testing

**Purpose:** Test deployed SageMaker endpoints with sample images and generate visual outputs.

**Usage:**
```bash
# Auto-detect endpoint from endpoint_info.json
python scripts/testing/test_endpoint.py --save-visual

# Specify endpoint name
python scripts/testing/test_endpoint.py --endpoint-name stygig-endpoint-20251103-062336

# Test with custom image
python scripts/testing/test_endpoint.py --image path/to/image.jpg --save-visual

# Test with S3 image
python scripts/testing/test_endpoint.py --s3-image s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg

# Get more recommendations
python scripts/testing/test_endpoint.py --top-k 10 --save-visual
```

**Features:**
- 🖼️ Visual output generation (side-by-side comparisons)
- 📊 Detailed scoring breakdowns
- 📄 JSON result export
- 🎨 Automatic S3 sample selection
- ⏱️ Cold start timeout handling

---

### 2. `integration_test.py` - Local Engine Testing

**Purpose:** Test the recommendation engine locally without SageMaker dependencies.

**Usage:**
```bash
# Basic test
python scripts/testing/integration_test.py path/to/image.jpg

# With gender filtering
python scripts/testing/integration_test.py image.jpg --gender male

# Custom items per category
python scripts/testing/integration_test.py image.jpg --items-per-category 3

# Save results to file
python scripts/testing/integration_test.py image.jpg --output results.json
```

**Features:**
- ✅ Self-exclusion validation
- 🎨 Color harmony testing
- 👥 Gender filtering validation
- 📊 Category diversity checks
- 💾 JSON result export

---

### 3. `local_train_test.py` - Local Training Simulation

**Purpose:** Simulate SageMaker training locally for rapid development and testing.

**Usage:**
```bash
# Basic local training
python scripts/testing/local_train_test.py --dataset-path outfits_dataset

# With custom parameters
python scripts/testing/local_train_test.py --dataset-path outfits_dataset --batch-size 16 --max-items 50

# Quick test (only 10 items)
python scripts/testing/local_train_test.py --dataset-path outfits_dataset --quick-test

# Custom output directory
python scripts/testing/local_train_test.py --dataset-path outfits_dataset --output-dir ./my_output
```

**Features:**
- 🏠 Local training without AWS
- 🧪 Environment validation
- 📊 Training statistics
- 💾 Model artifact saving
- 🔧 Configurable parameters

---

### 4. `verify_structure.sh` - Project Structure Validation

**Purpose:** Verify that the project structure is correct and all required files exist.

**Usage:**
```bash
./scripts/testing/verify_structure.sh
```

**What it checks:**
- ✅ All core directories
- ✅ Python package files
- ✅ Configuration files
- ✅ SageMaker scripts
- ✅ New consolidated scripts
- ✅ Documentation files

---

## 🔄 Migration from Old Scripts

The new scripts replace these old scripts from the project root:

| Old Script (DEPRECATED) | New Script | Notes |
|------------------------|------------|-------|
| `run_project.sh` | `scripts/run_pipeline.sh` | Enhanced with more options |
| `deploy_only.sh` | `scripts/deploy_model.sh` | Unified deployment |
| `deploy_endpoint.py` | `scripts/deploy_model.sh` | Consolidated |
| `deploy_existing_model.py` | `scripts/deploy_model.sh` | Consolidated |
| `redeploy_endpoint.py` | `scripts/deploy_model.sh` | Consolidated |
| `redeploy_with_timeout.py` | `scripts/deploy_model.sh` | Built-in timeout handling |
| `make_executable.sh` | `scripts/set_permissions.sh` | Moved |
| `verify_structure.sh` | `scripts/testing/verify_structure.sh` | Moved |
| `test_endpoint.py` (in sagemaker/) | `scripts/testing/test_endpoint.py` | Moved |
| `integration_test.py` (in testing/) | `scripts/testing/integration_test.py` | Moved |
| `local_train_test.py` (in testing/) | `scripts/testing/local_train_test.py` | Moved |

**⚠️ IMPORTANT:** After verifying the new scripts work, delete the old scripts to avoid confusion.

---

## 📋 Quick Start Guide

### First Time Setup

1. **Set permissions:**
   ```bash
   chmod +x scripts/set_permissions.sh
   ./scripts/set_permissions.sh
   ```

2. **Verify structure:**
   ```bash
   ./scripts/testing/verify_structure.sh
   ```

3. **Run full pipeline:**
   ```bash
   ./scripts/run_pipeline.sh
   ```

### Daily Workflow

**Training + Deployment:**
```bash
./scripts/run_pipeline.sh
```

**Deploy Existing Model:**
```bash
./scripts/deploy_model.sh --training-job-name stygig-training-xxxxx
```

**Test Endpoint:**
```bash
python scripts/testing/test_endpoint.py --save-visual
```

**List Endpoints:**
```bash
python scripts/manage_endpoints.py list
```

**Clean Up (cost savings):**
```bash
python scripts/manage_endpoints.py delete-all
```

---

## 🛠️ Configuration

All scripts use environment variables for configuration:

### Common Variables

```bash
# AWS Configuration
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1

# SageMaker Configuration
export TRAINING_INSTANCE_TYPE=ml.m5.large
export INFERENCE_INSTANCE_TYPE=ml.m5.large
export USE_SPOT_INSTANCES=false

# Pipeline Behavior
export TEST_ENDPOINT=true
export CLEANUP_AFTER_PIPELINE=false
export DEBUG_MODE=false
```

You can set these in your shell profile or pass them when running scripts:
```bash
S3_BUCKET=my-bucket ./scripts/run_pipeline.sh
```

---

## 🐛 Troubleshooting

### Script Permission Errors
```bash
./scripts/set_permissions.sh
```

### AWS Credentials Issues
```bash
aws sts get-caller-identity  # Verify credentials
aws configure  # Reconfigure if needed
```

### Endpoint Timeout Issues
The new `deploy_model.sh` script includes extended timeouts (600s) for CLIP model loading.

### Missing Dependencies
```bash
cd sagemaker
pip install -r requirements.txt
```

---

## 📊 Script Ratings (from script.md)

| Script | Rating | Strengths |
|--------|--------|-----------|
| `run_pipeline.sh` | 9/10 | Comprehensive orchestration, excellent error handling |
| `deploy_model.sh` | 9/10 | Unified deployment, flexible configuration |
| `manage_endpoints.py` | 9/10 | Professional endpoint management, cost-saving features |
| `test_endpoint.py` | 9/10 | Visual outputs, comprehensive testing |
| `integration_test.py` | 8/10 | Good local testing, validation checks |
| `local_train_test.py` | 8/10 | Useful for development, good simulation |
| `verify_structure.sh` | 8/10 | Thorough validation, clear output |

---

## 📝 Notes

- All scripts are designed to be run from the project root directory
- Scripts automatically change to the correct directory
- Comprehensive logging and error messages included
- Both S3 and SageMaker are configured to run in ap-south-1 region
- All deployment scripts include extended timeouts for CLIP model loading

---

## 🤝 Contributing

When adding new scripts:
1. Place them in the appropriate directory (`scripts/` or `scripts/testing/`)
2. Add comprehensive help text and usage examples
3. Update this README
4. Run `./scripts/set_permissions.sh` to make scripts executable
5. Test thoroughly before committing

---

## 📚 Additional Resources

- Main README: `../README.md`
- Script Analysis: `../script.md`
- Refactoring Summary: `../REFACTORING_SUMMARY.md`
- SageMaker Documentation: `../Docs/SAGEMAKER_FIX_SUMMARY.md`
