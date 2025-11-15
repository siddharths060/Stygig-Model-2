# SNS Topic Setup for Async Inference

## Overview

The async deployment script now **requires** an SNS topic ARN instead of creating one automatically. This is the recommended approach for enterprise environments where SNS topics are centrally managed.

---

## What Changed

### Before (Auto-Creation):
```bash
# SNS topic was auto-created (not recommended for production)
python deploy_async_endpoint.py --model-uri s3://bucket/model.tar.gz
```

### After (Required Parameter):
```bash
# SNS topic ARN must be provided (recommended)
python deploy_async_endpoint.py \
    --model-uri s3://bucket/model.tar.gz \
    --sns-topic-arn arn:aws:sns:ap-south-1:123456789012:stygig-notifications
```

---

## Getting SNS Topic from Your Company

### What to Request

Ask your AWS administrator for:

1. **SNS Topic ARN** for async inference notifications
2. Format: `arn:aws:sns:REGION:ACCOUNT_ID:TOPIC_NAME`
3. Example: `arn:aws:sns:ap-south-1:123456789012:stygig-async-notifications`

### Required Permissions

Your IAM role/user needs these permissions for the SNS topic:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sns:GetTopicAttributes",
                "sns:Publish"
            ],
            "Resource": "arn:aws:sns:ap-south-1:123456789012:your-topic-name"
        }
    ]
}
```

---

## How the Script Uses SNS

### Verification Step

The script now **verifies** the SNS topic exists before deployment:

```python
# Automatically checks:
# 1. Topic exists
# 2. You have access to it
# 3. ARN format is correct

verify_sns_topic(sns_topic_arn, region)
```

### Notification Flow

```
Async Inference Request
        ↓
    Processing
        ↓
    Complete/Error
        ↓
    SNS Notification → Your Topic
        ↓
    Email/Lambda/SQS Subscribers
```

### What Gets Published

**Success Notification:**
```json
{
  "invocationStatus": "Completed",
  "requestId": "abc123...",
  "outputLocation": "s3://bucket/async-inference-results/abc123.out",
  "invocationTime": "2025-11-15T12:00:00Z",
  "completionTime": "2025-11-15T12:03:45Z"
}
```

**Error Notification:**
```json
{
  "invocationStatus": "Failed",
  "requestId": "abc123...",
  "failureLocation": "s3://bucket/async-inference-results/abc123.err",
  "invocationTime": "2025-11-15T12:00:00Z",
  "completionTime": "2025-11-15T12:02:30Z"
}
```

---

## Deployment Examples

### Basic Deployment
```bash
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz \
    --sns-topic-arn arn:aws:sns:ap-south-1:123456789012:stygig-notifications
```

### Full Configuration
```bash
python sagemaker/deploy_async_endpoint.py \
    --model-uri s3://stygig-ml-s3/model-artifacts/model.tar.gz \
    --sns-topic-arn arn:aws:sns:ap-south-1:123456789012:stygig-notifications \
    --endpoint-name stygig-async-prod \
    --instance-type ml.m5.large \
    --s3-output-path s3://stygig-ml-s3/async-results/ \
    --max-concurrent-invocations 10
```

---

## Error Handling

### Missing SNS Topic ARN
```
❌ Error: SNS topic ARN is required
Solution: Add --sns-topic-arn parameter
```

### Invalid SNS Topic ARN
```
❌ Error: SNS topic not found: arn:aws:sns:...
Solution: Verify the ARN with your AWS admin
```

### Access Denied
```
❌ Error: Access denied to SNS topic
Solution: Request SNS permissions from your AWS admin
```

---

## Subscribing to Notifications

### Email Subscription (Your AWS Admin Does This)
```bash
aws sns subscribe \
    --topic-arn arn:aws:sns:ap-south-1:123456789012:stygig-notifications \
    --protocol email \
    --notification-endpoint your-email@company.com
```

### Lambda Subscription (for automation)
```bash
aws sns subscribe \
    --topic-arn arn:aws:sns:ap-south-1:123456789012:stygig-notifications \
    --protocol lambda \
    --notification-endpoint arn:aws:lambda:ap-south-1:123456789012:function:process-results
```

---

## Benefits of Company-Managed SNS Topics

✅ **Centralized Management** - Single topic for all async endpoints  
✅ **Security** - Controlled access via IAM policies  
✅ **Monitoring** - Centralized logging and metrics  
✅ **Compliance** - Meets enterprise governance requirements  
✅ **Cost Tracking** - Better resource allocation and billing  

---

## Quick Reference

| Parameter | Required | Example |
|-----------|----------|---------|
| `--model-uri` | Yes | `s3://bucket/model.tar.gz` |
| `--sns-topic-arn` | **Yes** | `arn:aws:sns:region:account:topic` |
| `--s3-output-path` | No | `s3://bucket/results/` |
| `--endpoint-name` | No | `stygig-async-prod` |
| `--instance-type` | No | `ml.m5.large` |

---

## Contact

If you need an SNS topic created, contact:
- Your AWS Administrator
- DevOps Team
- Cloud Infrastructure Team

Provide them with:
- Project name: StyGig Fashion Recommendations
- Use case: Asynchronous inference notifications
- Region: ap-south-1
- Subscribers needed: [your email/Lambda function]
