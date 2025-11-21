#!/bin/bash
# Check the error details and test fresh inference

echo "============================================================"
echo "Error Analysis & Fresh Inference Test"
echo "============================================================"
echo

# 1. Download and examine the error file
echo "1. EXAMINING THE ERROR FROM FAILED INFERENCE:"
echo "============================================================"
aws s3 cp s3://sagemaker-ap-south-1-732414292744/async-endpoint-failures/pytorch-inference-2025-11-21-13-53-28-530-1763733209-4e4c/41744303-4f13-45cb-8207-ca3b38dbd486-error.out error.txt

echo "Error details:"
cat error.txt
echo
echo "============================================================"

# 2. Test synchronous inference first
echo
echo "2. TESTING SYNCHRONOUS INFERENCE:"
echo "============================================================"

# Create test payload
TEST_PAYLOAD='{
    "image_s3_uri": "s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png",
    "top_k": 5
}'

echo "Payload: $TEST_PAYLOAD"
echo

echo "Invoking synchronous endpoint..."
echo "$TEST_PAYLOAD" | aws sagemaker-runtime invoke-endpoint \
    --endpoint-name stygig-async-production \
    --content-type application/json \
    --body file:///dev/stdin \
    sync_result.json

if [ $? -eq 0 ]; then
    echo "✅ SYNC INFERENCE SUCCESS!"
    echo "Response:"
    cat sync_result.json | jq . 2>/dev/null || cat sync_result.json
    echo
    
    # If sync works, try async
    echo
    echo "3. TESTING NEW ASYNC INFERENCE:"
    echo "============================================================"
    
    # Create unique input file
    TIMESTAMP=$(date +%s)
    INPUT_FILE="async-test-input-$TIMESTAMP.json"
    echo "$TEST_PAYLOAD" > $INPUT_FILE
    
    # Upload to S3
    aws s3 cp $INPUT_FILE s3://sagemaker-ap-south-1-732414292744/async-inputs/$INPUT_FILE
    
    # Submit async inference
    echo "Submitting new async inference..."
    aws sagemaker-runtime invoke-endpoint-async \
        --endpoint-name stygig-async-production \
        --content-type application/json \
        --input-location s3://sagemaker-ap-south-1-732414292744/async-inputs/$INPUT_FILE > async_response.json
    
    if [ $? -eq 0 ]; then
        echo "✅ ASYNC INFERENCE SUBMITTED!"
        echo "Response:"
        cat async_response.json | jq . 2>/dev/null || cat async_response.json
        
        # Extract output location
        OUTPUT_LOCATION=$(cat async_response.json | jq -r '.OutputLocation' 2>/dev/null)
        echo
        echo "Monitor this location for results:"
        echo "$OUTPUT_LOCATION"
        echo
        echo "Wait 2-5 minutes, then run:"
        echo "aws s3 ls s3://sagemaker-ap-south-1-732414292744/async-results/ --human-readable"
    else
        echo "❌ ASYNC INFERENCE FAILED"
    fi
    
else
    echo "❌ SYNC INFERENCE FAILED"
    echo "Response:"
    cat sync_result.json 2>/dev/null
fi

echo
echo "============================================================"