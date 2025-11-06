#!/bin/bash
# Quick fix script for SageMaker endpoint timeout issue
# This script redeploys your endpoint with extended timeouts

set -e

echo "=============================================================================="
echo "StyGig SageMaker Endpoint Timeout Fix"
echo "=============================================================================="
echo ""
echo "This script will:"
echo "  1. Delete the old endpoint with timeout issues"
echo "  2. Redeploy with extended timeouts (10 min startup, 5 min per request)"
echo "  3. Test the new endpoint"
echo ""
echo "Expected behavior:"
echo "  • First request: 2-3 minutes (CLIP model cold start)"
echo "  • Subsequent requests: 1-2 seconds (cached)"
echo ""
echo "=============================================================================="
echo ""

# Configuration
OLD_ENDPOINT="stygig-fashion-endpoint-1762445546"
MODEL_URI="s3://stygig-ml-s3/model-artifacts/stygig-training-1762445546/output/model.tar.gz"

echo "Step 1: Redeploying endpoint with extended timeouts..."
echo ""

# Change to sagemaker directory
cd "$(dirname "$0")/sagemaker"

# Run redeployment script
python redeploy_with_timeout.py \
  --model-uri "$MODEL_URI" \
  --old-endpoint "$OLD_ENDPOINT"

# Get new endpoint name from endpoint_info.json
NEW_ENDPOINT=$(python -c "import json; print(json.load(open('endpoint_info.json'))['endpoint_name'])")

echo ""
echo "=============================================================================="
echo "Step 2: Testing new endpoint (first request takes 2-3 minutes)..."
echo "=============================================================================="
echo ""

# Wait a bit for endpoint to stabilize
sleep 10

# Test the endpoint
python test_endpoint.py --endpoint-name "$NEW_ENDPOINT"

echo ""
echo "=============================================================================="
echo "✅ Fix Complete!"
echo "=============================================================================="
echo ""
echo "Your new endpoint: $NEW_ENDPOINT"
echo ""
echo "Expected performance:"
echo "  • First request: 2-3 minutes (one-time CLIP model loading)"
echo "  • All subsequent requests: 1-2 seconds"
echo ""
echo "To test again:"
echo "  python sagemaker/test_endpoint.py --endpoint-name $NEW_ENDPOINT"
echo ""
echo "To delete old endpoint and save costs:"
echo "  aws sagemaker delete-endpoint --endpoint-name $OLD_ENDPOINT"
echo ""
echo "=============================================================================="
