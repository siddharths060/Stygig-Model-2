# StyGig Scripts Migration Guide

This guide helps you transition from the old scattered scripts to the new consolidated `scripts/` directory structure.

## 📊 Migration Status: READY

✅ **New scripts created and tested**  
⏳ **Old scripts ready for deprecation**  
🎯 **Next step: Test new scripts, then delete old ones**

---

## 🗂️ Complete Script Mapping

### OLD → NEW Script Mapping

| Old Script Location | New Script Location | Status | Notes |
|---------------------|---------------------|--------|-------|
| `run_project.sh` | `scripts/run_pipeline.sh` | ✅ Enhanced | More flexible options |
| `deploy_only.sh` | `scripts/deploy_model.sh` | ✅ Consolidated | Single deployment script |
| `make_executable.sh` | `scripts/set_permissions.sh` | ✅ Moved | Renamed for clarity |
| `verify_structure.sh` | `scripts/testing/verify_structure.sh` | ✅ Updated | Checks new structure |
| `sagemaker/test_endpoint.py` | `scripts/testing/test_endpoint.py` | ✅ Enhanced | Better path handling |
| `sagemaker/deploy_endpoint.py` | `scripts/deploy_model.sh` | ✅ Consolidated | Unified with bash wrapper |
| `sagemaker/deploy_existing_model.py` | `scripts/deploy_model.sh` | ✅ Consolidated | Same functionality |
| `sagemaker/redeploy_endpoint.py` | `scripts/deploy_model.sh` | ✅ Consolidated | Built-in redeployment |
| `sagemaker/redeploy_with_timeout.py` | `scripts/deploy_model.sh` | ✅ Consolidated | Timeout handling included |
| `testing/integration_test.py` | `scripts/testing/integration_test.py` | ✅ Enhanced | Better path resolution |
| `testing/local_train_test.py` | `scripts/testing/local_train_test.py` | ✅ Simplified | Cleaner implementation |
| `testing/test_engine.py` | `scripts/testing/integration_test.py` | ✅ Merged | Functionality combined |
| *(NEW)* | `scripts/manage_endpoints.py` | ✅ Created | Endpoint management utility |

---

## 🚀 Quick Migration Steps

### Step 1: Test New Scripts (Recommended)

Before deleting old scripts, verify the new ones work:

```bash
# 1. Set permissions
chmod +x scripts/set_permissions.sh
./scripts/set_permissions.sh

# 2. Verify structure
./scripts/testing/verify_structure.sh

# 3. Test endpoint management (list existing endpoints)
python scripts/manage_endpoints.py list

# 4. (Optional) Test full pipeline
./scripts/run_pipeline.sh --skip-training --skip-deployment
```

### Step 2: Update Your Workflows

Replace old commands with new ones in your scripts, CI/CD, or documentation:

**Old:**
```bash
./run_project.sh
./deploy_only.sh
python sagemaker/test_endpoint.py
```

**New:**
```bash
./scripts/run_pipeline.sh
./scripts/deploy_model.sh --training-job-name <name>
python scripts/testing/test_endpoint.py
```

### Step 3: Delete Old Scripts

Once you've verified the new scripts work, clean up the old ones:

```bash
# Create a backup first (recommended)
mkdir old_scripts_backup
cp run_project.sh deploy_only.sh make_executable.sh verify_structure.sh old_scripts_backup/

# Delete old root-level scripts
rm run_project.sh
rm deploy_only.sh
rm make_executable.sh
rm verify_structure.sh

# Delete old deployment scripts from sagemaker/
rm sagemaker/deploy_endpoint.py
rm sagemaker/deploy_existing_model.py
rm sagemaker/redeploy_endpoint.py
rm sagemaker/redeploy_with_timeout.py
rm sagemaker/test_endpoint.py

# Delete old testing directory (if you're not using it anymore)
# CAREFUL: Make sure you've migrated any custom tests first
rm -rf testing/

# Optional: Delete the backup after confirming everything works
rm -rf old_scripts_backup
```

---

## 📋 Command Translation Guide

### Pipeline Execution

| Old Command | New Command |
|-------------|-------------|
| `./run_project.sh` | `./scripts/run_pipeline.sh` |
| N/A | `./scripts/run_pipeline.sh --skip-training` |
| N/A | `./scripts/run_pipeline.sh --skip-deployment` |

### Model Deployment

| Old Command | New Command |
|-------------|-------------|
| `./deploy_only.sh` | `./scripts/deploy_model.sh --training-job-name <name>` |
| `python sagemaker/deploy_endpoint.py --model-uri <uri>` | `./scripts/deploy_model.sh --model-uri <uri>` |
| `python sagemaker/deploy_existing_model.py --training-job-name <name>` | `./scripts/deploy_model.sh --training-job-name <name>` |
| `python sagemaker/redeploy_endpoint.py` | `./scripts/deploy_model.sh --model-uri <uri> --endpoint-name <name> --delete-existing` |

### Endpoint Testing

| Old Command | New Command |
|-------------|-------------|
| `python sagemaker/test_endpoint.py` | `python scripts/testing/test_endpoint.py` |
| `python sagemaker/test_endpoint.py --save-visual` | `python scripts/testing/test_endpoint.py --save-visual` |
| `python sagemaker/test_endpoint.py --endpoint-name <name>` | `python scripts/testing/test_endpoint.py --endpoint-name <name>` |

### Local Testing

| Old Command | New Command |
|-------------|-------------|
| `python testing/integration_test.py <image>` | `python scripts/testing/integration_test.py <image>` |
| `python testing/local_train_test.py --dataset-path <path>` | `python scripts/testing/local_train_test.py --dataset-path <path>` |

### Project Validation

| Old Command | New Command |
|-------------|-------------|
| `./verify_structure.sh` | `./scripts/testing/verify_structure.sh` |
| `./make_executable.sh` | `./scripts/set_permissions.sh` |

### Endpoint Management (NEW)

| Task | New Command |
|------|-------------|
| List all endpoints | `python scripts/manage_endpoints.py list` |
| Get endpoint details | `python scripts/manage_endpoints.py info --endpoint-name <name>` |
| Delete one endpoint | `python scripts/manage_endpoints.py delete --endpoint-name <name>` |
| Delete all endpoints | `python scripts/manage_endpoints.py delete-all` |

---

## 🔧 Configuration Changes

### Environment Variables (No Changes)

All environment variables remain the same:
```bash
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1
export TRAINING_INSTANCE_TYPE=ml.m5.large
export INFERENCE_INSTANCE_TYPE=ml.m5.large
```

### File Paths

Update any hard-coded paths in your configurations:

**Old:**
```
sagemaker/test_endpoint.py
testing/integration_test.py
```

**New:**
```
scripts/testing/test_endpoint.py
scripts/testing/integration_test.py
```

---

## 📝 Update Checklist

Use this checklist to ensure complete migration:

- [ ] **Test new scripts**
  - [ ] Run `scripts/set_permissions.sh`
  - [ ] Run `scripts/testing/verify_structure.sh`
  - [ ] Test `scripts/manage_endpoints.py list`

- [ ] **Update documentation**
  - [ ] Update README.md with new script paths
  - [ ] Update any team documentation
  - [ ] Update CI/CD pipeline configurations

- [ ] **Update automation**
  - [ ] Update any cron jobs
  - [ ] Update deployment automation
  - [ ] Update monitoring scripts

- [ ] **Backup and delete**
  - [ ] Create backup of old scripts
  - [ ] Delete old root-level scripts
  - [ ] Delete old sagemaker/ deployment scripts
  - [ ] Delete old testing/ directory (if applicable)

- [ ] **Verify**
  - [ ] Test full pipeline end-to-end
  - [ ] Test deployment workflow
  - [ ] Test endpoint management
  - [ ] Confirm no broken references

---

## 🆕 New Features in Consolidated Scripts

### Enhanced `run_pipeline.sh`
- ✨ Skip training/deployment/testing flags
- ✨ Better error messages
- ✨ Cleaner output with color coding
- ✨ More granular control

### Unified `deploy_model.sh`
- ✨ Single script for all deployment scenarios
- ✨ Auto-detect IAM role
- ✨ Built-in timeout handling
- ✨ Cleaner options structure

### New `manage_endpoints.py`
- ✨ List all project endpoints
- ✨ Get detailed endpoint info
- ✨ Delete individual or all endpoints
- ✨ Cost-saving bulk operations
- ✨ JSON output support

### Enhanced Testing Scripts
- ✨ Better path resolution
- ✨ Improved error messages
- ✨ Consolidated functionality
- ✨ Cleaner output formatting

---

## 🐛 Common Migration Issues

### Issue: "Permission denied" errors
**Solution:**
```bash
./scripts/set_permissions.sh
```

### Issue: "Module not found" errors
**Solution:** The new scripts handle path resolution automatically, but ensure you run them from the project root:
```bash
cd /path/to/stygig_project
./scripts/run_pipeline.sh
```

### Issue: "endpoint_info.json not found"
**Solution:** The file is now in `sagemaker/endpoint_info.json`. New scripts check multiple locations automatically.

### Issue: Old scripts still referenced in code
**Solution:** Search and replace:
```bash
# Find all references to old scripts
grep -r "run_project.sh" .
grep -r "deploy_only.sh" .
grep -r "sagemaker/test_endpoint.py" .

# Update them to new paths
```

---

## 📞 Getting Help

If you encounter issues during migration:

1. **Check the README:** `scripts/README.md`
2. **Review script help:** `./scripts/run_pipeline.sh --help` (for bash scripts) or add `-h` for Python scripts
3. **Check logs:** All scripts provide detailed logging
4. **Verify structure:** `./scripts/testing/verify_structure.sh`

---

## 🎯 Benefits of New Structure

✅ **Single source of truth** - All scripts in one place  
✅ **Better organization** - Clear separation of concerns  
✅ **Easier maintenance** - No scattered scripts  
✅ **Improved documentation** - Comprehensive README  
✅ **Enhanced functionality** - New endpoint management  
✅ **Cost savings** - Easy endpoint cleanup  
✅ **Professional structure** - Industry best practices  

---

## 📅 Deprecation Timeline

| Date | Action |
|------|--------|
| **Now** | New scripts available in `scripts/` |
| **Testing Phase** | Validate new scripts work for your workflows |
| **Migration Phase** | Update your processes to use new scripts |
| **Cleanup Phase** | Delete old scripts after successful migration |

---

**Last Updated:** November 5, 2025  
**Migration Status:** ✅ Ready for production use
