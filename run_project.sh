#!/bin/bash

# StyGig Enterprise Project Runner
# This script sets up the environment and runs the complete SageMaker pipeline
# Combines functionality from setup.sh and run_pipeline.sh

set -e  # Exit immediately if a command exits with a non-zero status

echo "════════════════════════════════════════════════════════════════════════════════"
echo "   StyGig Fashion Recommendation System - Enterprise Pipeline Runner"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# =============================================================================
# STEP 1: ENVIRONMENT VALIDATION
# =============================================================================
echo "🔧 STEP 1: Validating Environment"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check if we're in the correct directory
if [ ! -f "run_project.sh" ]; then
    echo "❌ ERROR: Please run this script from the stygig_project root directory"
    exit 1
fi
echo "✅ Current directory verified"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ ERROR: AWS CLI is not installed or not in PATH"
    echo "   Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi
echo "✅ AWS CLI found"

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python is not installed or not in PATH"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION found"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ ERROR: AWS credentials not configured or invalid"
    echo "   Please configure AWS credentials: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CURRENT_REGION=$(aws configure get region || echo "us-east-1")
echo "✅ AWS credentials configured"
echo "   Account ID: $ACCOUNT_ID"
echo "   Region: $CURRENT_REGION"
echo ""

# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================
echo "🔧 STEP 2: Loading Configuration"
echo "─────────────────────────────────────────────────────────────────────────────"

# Set environment variables for stygig-ml-s3 bucket configuration
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1

# Training configuration
export TRAINING_INSTANCE_TYPE=ml.m5.large
export INFERENCE_INSTANCE_TYPE=ml.m5.large

# Pipeline behavior
export TEST_ENDPOINT=true
export CLEANUP_AFTER_PIPELINE=false
export DEBUG_MODE=false
export ENVIRONMENT=production

# Cross-region optimizations for ap-south-1
# Spot instances may have limited availability in ap-south-1
export USE_SPOT_INSTANCES=false

echo "📊 Configuration:"
echo "   S3_BUCKET: $S3_BUCKET"
echo "   DATASET_S3_URI: $DATASET_S3_URI"
echo "   AWS_REGION: $AWS_REGION (S3 bucket region)"
echo "   SAGEMAKER_REGION: $CURRENT_REGION"
echo "   TRAINING_INSTANCE: $TRAINING_INSTANCE_TYPE"
echo "   INFERENCE_INSTANCE: $INFERENCE_INSTANCE_TYPE"
echo "   USE_SPOT_INSTANCES: $USE_SPOT_INSTANCES"
echo ""

if [ "$AWS_REGION" != "$CURRENT_REGION" ]; then
    echo "⚠️  NOTICE: Cross-region setup detected"
    echo "   S3 bucket in $AWS_REGION, SageMaker in $CURRENT_REGION"
    echo "   Using on-demand instances for better availability"
    echo ""
fi

# =============================================================================
# STEP 3: DATASET VERIFICATION
# =============================================================================
echo "🔍 STEP 3: Verifying Dataset"
echo "─────────────────────────────────────────────────────────────────────────────"

# Ensure AWS CLI uses the correct region for S3 operations
export AWS_DEFAULT_REGION="$AWS_REGION"

# Check if bucket is accessible
if aws s3 ls s3://stygig-ml-s3/ &> /dev/null; then
    echo "✅ stygig-ml-s3 bucket is accessible"
    
    # Check if train/ folder exists and has content
    if aws s3 ls s3://stygig-ml-s3/train/ &> /dev/null; then
        echo "✅ train/ folder found in bucket"
        
        # Count objects to ensure dataset is available
        OBJECT_COUNT=$(aws s3 ls s3://stygig-ml-s3/train/ --recursive 2>/dev/null | wc -l)
        echo "   📊 Found $OBJECT_COUNT objects in train/ folder"
        
        if [ "$OBJECT_COUNT" -gt 0 ]; then
            echo "✅ Dataset appears to be available"
            
            # Show sample structure
            echo ""
            echo "📁 Dataset sample (first 5 items):"
            aws s3 ls s3://stygig-ml-s3/train/ --recursive 2>/dev/null | head -5 || true
            echo ""
        else
            echo "❌ ERROR: train/ folder is empty"
            echo "   Please upload your dataset to s3://stygig-ml-s3/train/"
            exit 1
        fi
    else
        echo "❌ ERROR: train/ folder not found in bucket"
        echo "   Please upload your dataset to s3://stygig-ml-s3/train/"
        exit 1
    fi
else
    echo "❌ ERROR: Cannot access stygig-ml-s3 bucket"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  • Check AWS credentials: aws sts get-caller-identity"
    echo "  • Verify bucket region: $AWS_REGION"
    echo "  • Cross-region access may require specific permissions"
    exit 1
fi

# =============================================================================
# STEP 4: DEPENDENCY INSTALLATION
# =============================================================================
echo "📦 STEP 4: Installing Dependencies"
echo "─────────────────────────────────────────────────────────────────────────────"

cd sagemaker

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found in sagemaker directory"
    exit 1
fi

echo "Installing Python packages..."
pip install -r requirements.txt --quiet || {
    echo "⚠️  WARNING: Some packages may have failed to install"
    echo "   Continuing with pipeline execution..."
}

echo "✅ Dependencies installed"
echo ""

# =============================================================================
# STEP 5: SAGEMAKER PERMISSIONS CHECK
# =============================================================================
echo "🔍 STEP 5: Checking SageMaker Permissions"
echo "─────────────────────────────────────────────────────────────────────────────"

if aws sagemaker list-training-jobs --max-items 1 &> /dev/null; then
    echo "✅ SageMaker permissions appear to be working"
else
    echo "⚠️  WARNING: SageMaker permissions may be limited"
    echo "   Ensure your role has SageMaker execution permissions"
fi
echo ""

# =============================================================================
# STEP 6: PIPELINE EXECUTION
# =============================================================================
echo "🚀 STEP 6: Running SageMaker Pipeline"
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

python run_sagemaker_pipeline.py

exit_code=$?

# =============================================================================
# FINAL STATUS
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

if [ $exit_code -eq 0 ]; then
    echo "🎉 SUCCESS: Pipeline completed successfully!"
    echo ""
    echo "📄 Results:"
    echo "   • Check pipeline_results.json for detailed results"
    echo "   • Model artifacts saved to S3"
    echo "   • Endpoint (if deployed) is ready for inference"
else
    echo "❌ FAILURE: Pipeline failed with exit code $exit_code"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • Check the logs above for error details"
    echo "   • Verify AWS permissions and credentials"
    echo "   • Ensure dataset is properly uploaded to S3"
fi

echo "════════════════════════════════════════════════════════════════════════════════"

# Return to root directory
cd ..

exit $exit_code
