#!/bin/bash
# Quick diagnostic script to check async inference status

echo "============================================================"
echo "StyGig Async Inference Diagnostic"
echo "============================================================"
echo

# Check endpoint status
echo "1. Checking endpoint status..."
aws sagemaker describe-endpoint --endpoint-name stygig-async-production --query 'EndpointStatus' --output text

echo
echo "2. Checking all possible S3 locations for results..."

# Check multiple possible result locations
BUCKET="sagemaker-ap-south-1-732414292744"

echo "Checking main async-results folder..."
aws s3 ls s3://$BUCKET/async-results/ --recursive --human-readable

echo
echo "Checking root bucket for any recent files..."
aws s3 ls s3://$BUCKET/ --recursive --human-readable | grep "$(date '+%Y-%m-%d')"

echo
echo "3. Looking for any files modified in the last 2 hours..."
aws s3api list-objects-v2 \
    --bucket $BUCKET \
    --query "Contents[?LastModified >= '$(date -u -d '2 hours ago' '+%Y-%m-%dT%H:%M:%SZ')'].{Key: Key, LastModified: LastModified, Size: Size}" \
    --output table

echo
echo "4. Checking CloudWatch logs for endpoint errors..."

# Get log group name for the endpoint
LOG_GROUP="/aws/sagemaker/Endpoints/stygig-async-production"

echo "Recent log events (last 30 minutes):"
aws logs filter-log-events \
    --log-group-name $LOG_GROUP \
    --start-time $(date -d '30 minutes ago' +%s)000 \
    --query 'events[].message' \
    --output text 2>/dev/null || echo "No logs found or log group doesn't exist"

echo
echo "5. Manual test - submitting a new async inference..."

# Create a test input file
TEST_INPUT='{"image_s3_uri": "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png", "top_k": 5}'
echo $TEST_INPUT > /tmp/test_input.json

# Upload to S3
aws s3 cp /tmp/test_input.json s3://$BUCKET/test-input-$(date +%s).json

# Submit async inference
echo "Submitting new async inference request..."
aws sagemaker-runtime invoke-endpoint-async \
    --endpoint-name stygig-async-production \
    --content-type application/json \
    --input-location s3://$BUCKET/test-input-$(date +%s).json \
    --output-location s3://$BUCKET/async-results/test-output-$(date +%s).json

echo
echo "✅ New async request submitted. Wait 3-5 minutes then run:"
echo "aws s3 ls s3://$BUCKET/async-results/ --recursive --human-readable"

echo
echo "============================================================"