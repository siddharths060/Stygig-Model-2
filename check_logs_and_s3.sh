#!/bin/bash
# Check CloudWatch logs and async inference status

echo "============================================================"
echo "CloudWatch Logs Investigation"
echo "============================================================"
echo

# Check if log groups exist for the endpoint
echo "1. Looking for log groups..."
aws logs describe-log-groups --log-group-name-prefix '/aws/sagemaker/Endpoints/stygig-async-production' --query 'logGroups[].logGroupName' --output text

echo
echo "2. Checking recent log events (last 1 hour)..."

# Calculate timestamp for 1 hour ago (in milliseconds)
START_TIME=$(date -d '1 hour ago' +%s)000

LOG_GROUP="/aws/sagemaker/Endpoints/stygig-async-production"

# Check for log events
aws logs filter-log-events \
    --log-group-name $LOG_GROUP \
    --start-time $START_TIME \
    --query 'events[].[timestamp,message]' \
    --output text 2>/dev/null | head -20

echo
echo "3. If no logs above, checking for all log streams in the group..."
aws logs describe-log-streams \
    --log-group-name $LOG_GROUP \
    --order-by LastEventTime \
    --descending \
    --max-items 5 \
    --query 'logStreams[].[logStreamName,lastEventTime,lastIngestionTime]' \
    --output table 2>/dev/null

echo
echo "4. Checking S3 bucket with more detailed search..."
BUCKET="sagemaker-ap-south-1-732414292744"

echo "All objects in bucket:"
aws s3api list-objects-v2 --bucket $BUCKET --query 'Contents[].[Key,LastModified,Size]' --output table

echo
echo "5. Checking the original async inference from earlier..."
echo "Looking for any files with today's date:"
aws s3 ls s3://$BUCKET/ --recursive | grep "2025-11-21"

echo
echo "6. Let's try to find the original inference result..."
echo "The original inference ID was: 41744303-4f13-45cb-8207-ca3b38dbd486"
aws s3 ls s3://$BUCKET/ --recursive | grep "41744303"

echo
echo "============================================================"