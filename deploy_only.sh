#!/bin/bash

# Deploy Existing Trained Model (No Training Required)
# This script deploys your already-trained model to a SageMaker endpoint

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "   StyGig Model Deployment - Using Existing Trained Model"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
TRAINING_JOB_NAME="stygig-training-1762145223"  # Your successful training job
INSTANCE_TYPE="ml.m5.large"
REGION="us-east-1"

echo "📦 Deployment Configuration:"
echo "   Training Job: $TRAINING_JOB_NAME"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   Region: $REGION"
echo ""

# Change to sagemaker directory
cd sagemaker

echo "🚀 Starting deployment (this takes 5-10 minutes)..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Run deployment script
python deploy_existing_model.py \
    --training-job-name "$TRAINING_JOB_NAME" \
    --instance-type "$INSTANCE_TYPE" \
    --region "$REGION"

exit_code=$?

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

if [ $exit_code -eq 0 ]; then
    echo "🎉 SUCCESS: Model deployed to endpoint!"
    echo ""
    echo "📄 Endpoint details saved to: sagemaker/endpoint_info.json"
    echo ""
    echo "✅ Your fashion recommendation API is now live and ready for inference!"
else
    echo "❌ FAILURE: Deployment failed with exit code $exit_code"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   • Check AWS permissions for SageMaker endpoints"
    echo "   • Verify the training job name is correct"
    echo "   • Ensure model artifacts exist in S3"
fi

echo "════════════════════════════════════════════════════════════════════════════════"

# Return to root directory
cd ..

exit $exit_code
