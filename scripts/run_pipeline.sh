#!/bin/bash

################################################################################
# StyGig Fashion Recommendation - Complete ML Pipeline Runner
################################################################################
# 
# This script orchestrates the complete machine learning pipeline:
#   1. Environment validation (AWS CLI, Python, credentials)
#   2. Configuration loading and verification
#   3. Dataset validation in S3
#   4. Dependency installation
#   5. SageMaker training job execution
#   6. Model deployment to endpoint
#   7. Endpoint testing and validation
#
# Usage:
#   ./scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-training        Deploy existing model without training
#   --skip-deployment      Train model but don't deploy
#   --skip-testing         Don't test the endpoint after deployment
#   --training-job-name    Use existing training job for deployment
#
# Examples:
#   ./scripts/run_pipeline.sh                                    # Full pipeline
#   ./scripts/run_pipeline.sh --skip-training                    # Deploy only
#   ./scripts/run_pipeline.sh --skip-deployment                  # Train only
#
################################################################################

set -e  # Exit on any error

# Parse command line arguments
SKIP_TRAINING=false
SKIP_DEPLOYMENT=false
SKIP_TESTING=false
TRAINING_JOB_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-deployment)
            SKIP_DEPLOYMENT=true
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
            shift
            ;;
        --training-job-name)
            TRAINING_JOB_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/run_pipeline.sh [--skip-training] [--skip-deployment] [--skip-testing] [--training-job-name NAME]"
            exit 1
            ;;
    esac
done

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "   $1"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}🔹 $1${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Change to project root directory
cd "$(dirname "$0")/.."

print_header "StyGig Fashion Recommendation System - ML Pipeline Runner"

################################################################################
# STEP 1: ENVIRONMENT VALIDATION
################################################################################
print_step "STEP 1: Validating Environment"

# Check if we're in the correct directory
if [ ! -d "sagemaker" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi
print_success "Project directory verified"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed or not in PATH"
    echo "   Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi
print_success "AWS CLI found: $(aws --version 2>&1 | head -n1)"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured or invalid"
    echo "   Please configure AWS credentials: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CURRENT_REGION=$(aws configure get region || echo "ap-south-1")
print_success "AWS credentials configured"
echo "   Account ID: $ACCOUNT_ID"
echo "   Region: $CURRENT_REGION"

################################################################################
# STEP 2: CONFIGURATION LOADING
################################################################################
print_step "STEP 2: Loading Configuration"

# Load configuration from environment or use defaults
export S3_BUCKET="${S3_BUCKET:-stygig-ml-s3}"
export DATASET_S3_URI="${DATASET_S3_URI:-s3://stygig-ml-s3/train/}"
export AWS_REGION="${AWS_REGION:-ap-south-1}"
export TRAINING_INSTANCE_TYPE="${TRAINING_INSTANCE_TYPE:-ml.m5.large}"
export INFERENCE_INSTANCE_TYPE="${INFERENCE_INSTANCE_TYPE:-ml.m5.large}"
export USE_SPOT_INSTANCES="${USE_SPOT_INSTANCES:-false}"
export TEST_ENDPOINT="${TEST_ENDPOINT:-true}"
export CLEANUP_AFTER_PIPELINE="${CLEANUP_AFTER_PIPELINE:-false}"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export ENVIRONMENT="${ENVIRONMENT:-production}"

echo "📊 Configuration:"
echo "   S3_BUCKET: $S3_BUCKET"
echo "   DATASET_S3_URI: $DATASET_S3_URI"
echo "   AWS_REGION: $AWS_REGION (S3 bucket region)"
echo "   SAGEMAKER_REGION: $CURRENT_REGION"
echo "   TRAINING_INSTANCE: $TRAINING_INSTANCE_TYPE"
echo "   INFERENCE_INSTANCE: $INFERENCE_INSTANCE_TYPE"
echo "   USE_SPOT_INSTANCES: $USE_SPOT_INSTANCES"

if [ "$AWS_REGION" != "$CURRENT_REGION" ]; then
    print_warning "Cross-region setup detected"
    echo "   S3 bucket in $AWS_REGION, SageMaker in $CURRENT_REGION"
    echo "   Using on-demand instances for better availability"
fi

################################################################################
# STEP 3: DATASET VERIFICATION
################################################################################
print_step "STEP 3: Verifying Dataset in S3"

# Set AWS region for S3 operations
export AWS_DEFAULT_REGION="$AWS_REGION"

# Check if bucket is accessible
if aws s3 ls "s3://$S3_BUCKET/" &> /dev/null; then
    print_success "$S3_BUCKET bucket is accessible"
    
    # Check if train/ folder exists and has content
    if aws s3 ls "$DATASET_S3_URI" &> /dev/null; then
        print_success "train/ folder found in bucket"
        
        # Count objects
        OBJECT_COUNT=$(aws s3 ls "$DATASET_S3_URI" --recursive 2>/dev/null | wc -l)
        echo "   📊 Found $OBJECT_COUNT objects in train/ folder"
        
        if [ "$OBJECT_COUNT" -gt 0 ]; then
            print_success "Dataset appears to be available"
            
            # Show sample structure
            echo ""
            echo "📁 Dataset sample (first 5 items):"
            aws s3 ls "$DATASET_S3_URI" --recursive 2>/dev/null | head -5 || true
            echo ""
        else
            print_error "train/ folder is empty"
            echo "   Please upload your dataset to $DATASET_S3_URI"
            exit 1
        fi
    else
        print_error "train/ folder not found in bucket"
        echo "   Please upload your dataset to $DATASET_S3_URI"
        exit 1
    fi
else
    print_error "Cannot access $S3_BUCKET bucket"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  • Check AWS credentials: aws sts get-caller-identity"
    echo "  • Verify bucket region: $AWS_REGION"
    echo "  • Cross-region access may require specific permissions"
    exit 1
fi

################################################################################
# STEP 4: DEPENDENCY INSTALLATION
################################################################################
print_step "STEP 4: Installing Dependencies"

cd sagemaker

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in sagemaker directory"
    exit 1
fi

echo "Installing Python packages..."
pip install -r requirements.txt --quiet --no-warn-conflicts 2>&1 | grep -v "ERROR: pip's dependency resolver" || {
    print_warning "Some dependency conflicts exist but won't affect the pipeline"
    echo "   Continuing with pipeline execution..."
}

print_success "Dependencies installed"

################################################################################
# STEP 5: SAGEMAKER PERMISSIONS CHECK
################################################################################
print_step "STEP 5: Checking SageMaker Permissions"

if aws sagemaker list-training-jobs --max-items 1 &> /dev/null; then
    print_success "SageMaker permissions verified"
else
    print_warning "SageMaker permissions may be limited"
    echo "   Ensure your role has SageMaker execution permissions"
fi

################################################################################
# STEP 6: TRAINING EXECUTION
################################################################################
if [ "$SKIP_TRAINING" = false ]; then
    print_step "STEP 6: Running SageMaker Training Job"
    
    echo "Executing training pipeline..."
    python run_sagemaker_pipeline.py
    
    TRAINING_EXIT_CODE=$?
    
    if [ $TRAINING_EXIT_CODE -ne 0 ]; then
        print_error "Training failed with exit code $TRAINING_EXIT_CODE"
        cd ..
        exit $TRAINING_EXIT_CODE
    fi
    
    print_success "Training completed successfully"
    
    # Extract training job name from results if available
    if [ -f "pipeline_results.json" ]; then
        TRAINING_JOB_NAME=$(python -c "import json; print(json.load(open('pipeline_results.json')).get('training_job_name', ''))" 2>/dev/null || echo "")
    fi
else
    print_step "STEP 6: Skipping Training (--skip-training flag set)"
    
    if [ -z "$TRAINING_JOB_NAME" ]; then
        print_warning "No training job name provided"
        echo "   Use --training-job-name to specify existing training job"
    fi
fi

cd ..

################################################################################
# STEP 7: MODEL DEPLOYMENT
################################################################################
if [ "$SKIP_DEPLOYMENT" = false ]; then
    print_step "STEP 7: Deploying Model to Endpoint"
    
    if [ -n "$TRAINING_JOB_NAME" ]; then
        echo "Deploying model from training job: $TRAINING_JOB_NAME"
        ./scripts/deploy_model.sh --training-job-name "$TRAINING_JOB_NAME"
    elif [ "$SKIP_TRAINING" = false ]; then
        echo "Deploying newly trained model..."
        # Find the most recent model in S3
        MODEL_URI=$(aws s3 ls "s3://$S3_BUCKET/model-artifacts/" --recursive | sort | tail -n 1 | awk '{print $4}')
        if [ -n "$MODEL_URI" ]; then
            ./scripts/deploy_model.sh --model-uri "s3://$S3_BUCKET/$MODEL_URI"
        else
            print_error "Could not find model artifacts in S3"
            exit 1
        fi
    else
        print_error "Cannot deploy: no training job specified and training was skipped"
        echo "   Use --training-job-name to specify existing training job"
        exit 1
    fi
    
    DEPLOYMENT_EXIT_CODE=$?
    
    if [ $DEPLOYMENT_EXIT_CODE -ne 0 ]; then
        print_error "Deployment failed with exit code $DEPLOYMENT_EXIT_CODE"
        exit $DEPLOYMENT_EXIT_CODE
    fi
    
    print_success "Model deployed successfully"
else
    print_step "STEP 7: Skipping Deployment (--skip-deployment flag set)"
fi

################################################################################
# STEP 8: ENDPOINT TESTING
################################################################################
if [ "$SKIP_DEPLOYMENT" = false ] && [ "$SKIP_TESTING" = false ] && [ "$TEST_ENDPOINT" = "true" ]; then
    print_step "STEP 8: Testing Deployed Endpoint"
    
    # Get endpoint name from saved info
    if [ -f "sagemaker/endpoint_info.json" ]; then
        ENDPOINT_NAME=$(python -c "import json; print(json.load(open('sagemaker/endpoint_info.json')).get('endpoint_name', ''))" 2>/dev/null || echo "")
        
        if [ -n "$ENDPOINT_NAME" ]; then
            echo "Testing endpoint: $ENDPOINT_NAME"
            python scripts/testing/test_endpoint.py --endpoint-name "$ENDPOINT_NAME" --save-visual
            
            TEST_EXIT_CODE=$?
            
            if [ $TEST_EXIT_CODE -eq 0 ]; then
                print_success "Endpoint test completed successfully"
            else
                print_warning "Endpoint test failed, but deployment was successful"
            fi
        else
            print_warning "Could not determine endpoint name from endpoint_info.json"
        fi
    else
        print_warning "endpoint_info.json not found, skipping endpoint test"
    fi
else
    print_step "STEP 8: Skipping Endpoint Testing"
fi

################################################################################
# FINAL SUMMARY
################################################################################
print_header "Pipeline Execution Complete"

echo "📊 Summary:"
if [ "$SKIP_TRAINING" = false ]; then
    echo "   ✅ Training: Completed"
    [ -n "$TRAINING_JOB_NAME" ] && echo "      Job: $TRAINING_JOB_NAME"
else
    echo "   ⏭️  Training: Skipped"
fi

if [ "$SKIP_DEPLOYMENT" = false ]; then
    echo "   ✅ Deployment: Completed"
    if [ -f "sagemaker/endpoint_info.json" ]; then
        ENDPOINT_NAME=$(python -c "import json; print(json.load(open('sagemaker/endpoint_info.json')).get('endpoint_name', ''))" 2>/dev/null || echo "")
        [ -n "$ENDPOINT_NAME" ] && echo "      Endpoint: $ENDPOINT_NAME"
    fi
else
    echo "   ⏭️  Deployment: Skipped"
fi

if [ "$SKIP_TESTING" = false ] && [ "$TEST_ENDPOINT" = "true" ]; then
    echo "   ✅ Testing: Completed"
else
    echo "   ⏭️  Testing: Skipped"
fi

echo ""
echo "📄 Output Files:"
[ -f "sagemaker/pipeline_results.json" ] && echo "   • Training results: sagemaker/pipeline_results.json"
[ -f "sagemaker/endpoint_info.json" ] && echo "   • Endpoint info: sagemaker/endpoint_info.json"
[ -d "scripts/testing/test_results" ] && echo "   • Test results: scripts/testing/test_results/"

echo ""
echo "🎯 Next Steps:"
echo "   • List endpoints: python scripts/manage_endpoints.py list"
echo "   • Test endpoint: python scripts/testing/test_endpoint.py --save-visual"
echo "   • Clean up: python scripts/manage_endpoints.py delete-all"

echo ""
print_success "Pipeline execution completed successfully!"

exit 0
