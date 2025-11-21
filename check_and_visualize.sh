#!/bin/bash
# Quick script to check async results and run visualization on SageMaker

echo "============================================================"
echo "StyGig V4 Async Results Checker & Visualizer"
echo "============================================================"
echo

# Step 1: Check for async results
echo "1. Checking for async inference results..."
aws s3 ls s3://sagemaker-ap-south-1-732414292744/async-results/ --recursive --human-readable

echo
echo "2. Listing most recent results with details..."

# Get the most recent result file
LATEST_RESULT=$(aws s3api list-objects-v2 \
    --bucket sagemaker-ap-south-1-732414292744 \
    --prefix async-results/ \
    --query 'sort_by(Contents, &LastModified)[-1].Key' \
    --output text)

if [ "$LATEST_RESULT" != "None" ] && [ "$LATEST_RESULT" != "" ]; then
    echo "✓ Found latest result: $LATEST_RESULT"
    
    # Check file size
    FILE_SIZE=$(aws s3api head-object \
        --bucket sagemaker-ap-south-1-732414292744 \
        --key "$LATEST_RESULT" \
        --query 'ContentLength' \
        --output text)
    
    echo "  File size: $FILE_SIZE bytes"
    
    if [ "$FILE_SIZE" -gt 0 ]; then
        echo
        echo "3. Running visualization..."
        
        # Install required packages
        pip install pillow boto3 --quiet
        
        # Run visualization
        python Generate_combined_image.py \
            --input_image_s3 s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \
            --json_s3 s3://sagemaker-ap-south-1-732414292744/$LATEST_RESULT \
            --output_file stygig_v4_comparison.png
        
        if [ $? -eq 0 ]; then
            echo
            echo "✅ Visualization completed successfully!"
            echo "📁 Output file: stygig_v4_comparison.png"
            echo
            ls -lh stygig_v4_comparison.png
            echo
            echo "To download to your local machine:"
            echo "Use SageMaker Studio's download feature or scp command"
        else
            echo "❌ Visualization failed"
        fi
    else
        echo "⚠️  Result file exists but is empty - async job may still be processing"
    fi
else
    echo "⚠️  No async results found yet"
    echo
    echo "Possible reasons:"
    echo "- Async inference is still processing (can take 5-15 minutes on CPU instances)"
    echo "- Async job failed"
    echo "- Results stored in different location"
    echo
    echo "Check SageMaker console for async inference job status"
fi

echo
echo "============================================================"