#!/bin/bash

################################################################################
# StyGig Fashion Recommendation - Model Deployment Script
################################################################################
#
# This script deploys a trained StyGig model to a SageMaker endpoint.
# It consolidates functionality from multiple deployment scripts into one
# clean, configurable solution.
#
# Usage:
#   ./scripts/deploy_model.sh [OPTIONS]
#
# Options:
#   --model-uri URI              S3 URI to model.tar.gz artifact
#   --training-job-name NAME     Training job name (will fetch model URI)
#   --endpoint-name NAME         Custom endpoint name (auto-generated if not provided)
#   --instance-type TYPE         Instance type (default: ml.m5.large)
#   --region REGION              AWS region (default: ap-south-1)
#   --role ARN                   IAM role ARN (auto-detected if not provided)
#   --skip-delete                Don't delete existing endpoint
#   --delete-existing            Delete existing endpoint with same name
#
# Examples:
#   # Deploy from model URI
#   ./scripts/deploy_model.sh --model-uri s3://bucket/model.tar.gz
#
#   # Deploy from training job
#   ./scripts/deploy_model.sh --training-job-name stygig-training-1762145223
#
#   # Replace existing endpoint
#   ./scripts/deploy_model.sh --model-uri s3://bucket/model.tar.gz --delete-existing
#
################################################################################

set -e

# Parse command line arguments
MODEL_URI=""
TRAINING_JOB_NAME=""
ENDPOINT_NAME=""
INSTANCE_TYPE="ml.m5.large"
REGION="ap-south-1"
ROLE=""
SKIP_DELETE=false
DELETE_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-uri)
            MODEL_URI="$2"
            shift 2
            ;;
        --training-job-name)
            TRAINING_JOB_NAME="$2"
            shift 2
            ;;
        --endpoint-name)
            ENDPOINT_NAME="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --role)
            ROLE="$2"
            shift 2
            ;;
        --skip-delete)
            SKIP_DELETE=true
            shift
            ;;
        --delete-existing)
            DELETE_EXISTING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: ./scripts/deploy_model.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-uri URI              S3 URI to model.tar.gz"
            echo "  --training-job-name NAME     Training job name"
            echo "  --endpoint-name NAME         Custom endpoint name"
            echo "  --instance-type TYPE         Instance type (default: ml.m5.large)"
            echo "  --region REGION              AWS region (default: ap-south-1)"
            echo "  --role ARN                   IAM role ARN"
            echo "  --skip-delete                Don't delete existing endpoint"
            echo "  --delete-existing            Delete existing endpoint"
            exit 1
            ;;
    esac
done

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "   $1"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Change to project root
cd "$(dirname "$0")/.."

print_header "StyGig Model Deployment"

################################################################################
# STEP 1: VALIDATE INPUTS AND RESOLVE MODEL URI
################################################################################
echo "📋 Step 1: Validating inputs..."
echo ""

# Resolve model URI from training job if needed
if [ -z "$MODEL_URI" ] && [ -n "$TRAINING_JOB_NAME" ]; then
    print_info "Fetching model URI from training job: $TRAINING_JOB_NAME"
    
    # Try to get model URI from training job
    MODEL_URI=$(aws sagemaker describe-training-job \
        --training-job-name "$TRAINING_JOB_NAME" \
        --region ap-south-1 \
        --query 'ModelArtifacts.S3ModelArtifacts' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$MODEL_URI" ]; then
        print_error "Could not retrieve model URI from training job: $TRAINING_JOB_NAME"
        exit 1
    fi
    
    print_success "Model URI resolved: $MODEL_URI"
fi

# Validate model URI
if [ -z "$MODEL_URI" ]; then
    print_error "No model URI provided"
    echo ""
    echo "Please provide either:"
    echo "  --model-uri s3://bucket/path/model.tar.gz"
    echo "  --training-job-name stygig-training-xxxxx"
    exit 1
fi

echo "📦 Deployment Configuration:"
echo "   Model URI: $MODEL_URI"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   Region: $REGION"
[ -n "$ENDPOINT_NAME" ] && echo "   Endpoint Name: $ENDPOINT_NAME"
[ -n "$ROLE" ] && echo "   IAM Role: $ROLE"
echo ""

################################################################################
# STEP 2: DELETE EXISTING ENDPOINT (if requested)
################################################################################
if [ "$DELETE_EXISTING" = true ] && [ -n "$ENDPOINT_NAME" ]; then
    echo "🗑️  Step 2: Deleting existing endpoint..."
    echo ""
    
    if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" --region "$REGION" &>/dev/null; then
        print_info "Found existing endpoint: $ENDPOINT_NAME"
        echo "   Deleting..."
        
        aws sagemaker delete-endpoint \
            --endpoint-name "$ENDPOINT_NAME" \
            --region "$REGION"
        
        print_success "Endpoint deleted"
        echo "   Waiting 30 seconds for cleanup..."
        sleep 30
    else
        print_info "No existing endpoint found with name: $ENDPOINT_NAME"
    fi
    echo ""
fi

################################################################################
# STEP 3: DEPLOY MODEL USING PYTHON SCRIPT
################################################################################
echo "🚀 Step 3: Deploying model to SageMaker endpoint..."
echo ""

# Change to sagemaker directory
cd sagemaker

# Prepare Python deployment command
PYTHON_CMD="python -c \"
import sys
import json
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configuration
MODEL_URI = '$MODEL_URI'
INSTANCE_TYPE = '$INSTANCE_TYPE'
REGION = '$REGION'
ENDPOINT_NAME = '$ENDPOINT_NAME' or None
ROLE = '$ROLE' or None

print('Initializing SageMaker session...')
session = Session(boto_session=boto3.Session(region_name=REGION))

# Auto-detect IAM role if not provided
if not ROLE:
    try:
        # Try to get role from SageMaker execution role
        sts_client = boto3.client('sts', region_name=REGION)
        caller_id = sts_client.get_caller_identity()
        account_id = caller_id['Account']
        
        # Common SageMaker execution role pattern
        iam_client = boto3.client('iam', region_name=REGION)
        roles = iam_client.list_roles()
        
        for role_info in roles['Roles']:
            role_name = role_info['RoleName']
            if 'SageMaker' in role_name and 'ExecutionRole' in role_name:
                ROLE = role_info['Arn']
                print(f'Auto-detected IAM role: {ROLE}')
                break
        
        if not ROLE:
            print('ERROR: Could not auto-detect IAM role')
            print('Please provide --role ARN')
            sys.exit(1)
    except Exception as e:
        print(f'ERROR: Failed to auto-detect IAM role: {e}')
        sys.exit(1)

# Generate endpoint name if not provided
if not ENDPOINT_NAME:
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    ENDPOINT_NAME = f'stygig-endpoint-{timestamp}'
    print(f'Generated endpoint name: {ENDPOINT_NAME}')

print(f'Creating PyTorch model...')

# Get project root directory
project_root = Path(__file__).parent.parent

# Create PyTorch model
model = PyTorchModel(
    model_data=MODEL_URI,
    role=ROLE,
    entry_point='inference.py',
    source_dir=str(project_root / 'sagemaker'),
    framework_version='2.0.0',
    py_version='py310',
    sagemaker_session=session,
    env={
        'MODEL_SERVER_TIMEOUT': '600',  # 10 minutes for cold start
        'MODEL_SERVER_WORKERS': '1'
    }
)

print(f'Deploying to endpoint: {ENDPOINT_NAME}')
print('This will take 5-10 minutes...')
print('')

# Deploy model
try:
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        model_data_download_timeout=600,
        container_startup_health_check_timeout=600
    )
    
    print('')
    print('✅ Deployment successful!')
    
    # Save endpoint information
    endpoint_info = {
        'endpoint_name': ENDPOINT_NAME,
        'model_uri': MODEL_URI,
        'instance_type': INSTANCE_TYPE,
        'region': REGION,
        'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('endpoint_info.json', 'w') as f:
        json.dump(endpoint_info, f, indent=2)
    
    print(f'📄 Endpoint info saved to: endpoint_info.json')
    print('')
    
except Exception as e:
    print(f'ERROR: Deployment failed: {e}')
    sys.exit(1)
\""

# Execute Python deployment
eval "$PYTHON_CMD"

DEPLOY_EXIT_CODE=$?

cd ..

if [ $DEPLOY_EXIT_CODE -ne 0 ]; then
    print_error "Deployment failed"
    exit $DEPLOY_EXIT_CODE
fi

################################################################################
# FINAL SUMMARY
################################################################################
print_header "Deployment Complete"

if [ -f "sagemaker/endpoint_info.json" ]; then
    DEPLOYED_ENDPOINT=$(python -c "import json; print(json.load(open('sagemaker/endpoint_info.json')).get('endpoint_name', ''))" 2>/dev/null || echo "")
    
    echo "🎉 Your fashion recommendation API is now live!"
    echo ""
    echo "📊 Endpoint Details:"
    echo "   Name: $DEPLOYED_ENDPOINT"
    echo "   Region: $REGION"
    echo "   Instance: $INSTANCE_TYPE"
    echo ""
    echo "🧪 Next Steps:"
    echo ""
    echo "  1. Test endpoint:"
    echo "     python scripts/testing/test_endpoint.py --endpoint-name $DEPLOYED_ENDPOINT --save-visual"
    echo ""
    echo "  2. List all endpoints:"
    echo "     python scripts/manage_endpoints.py list"
    echo ""
    echo "  3. First request takes 1-2 minutes (CLIP model loads)"
    echo "     Subsequent requests: ~1-2 seconds"
    echo ""
    echo "  4. To delete endpoint later:"
    echo "     python scripts/manage_endpoints.py delete --endpoint-name $DEPLOYED_ENDPOINT"
    echo ""
    
    print_success "Deployment completed successfully!"
else
    print_warning "Deployment may have succeeded but endpoint_info.json not found"
fi

exit 0
