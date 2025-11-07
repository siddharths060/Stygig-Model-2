# 🚀 StyGig SageMaker Deployment Guide for Beginners

This is a complete, beginner-friendly guide to train, deploy, and test your StyGig Fashion Recommendation System on AWS SageMaker.

---

## 📚 Table of Contents

1. [What is SageMaker?](#what-is-sagemaker)
2. [Prerequisites](#prerequisites)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Step 1: AWS Setup](#step-1-aws-setup)
5. [Step 2: Prepare Your Dataset](#step-2-prepare-your-dataset)
6. [Step 3: Upload Dataset to S3](#step-3-upload-dataset-to-s3)
7. [Step 4: Configure Your Environment](#step-4-configure-your-environment)
8. [Step 5: Train Your Model](#step-5-train-your-model)
9. [Step 6: Deploy the Model (Create Endpoint)](#step-6-deploy-the-model-create-endpoint)
10. [Step 7: Testing Your Model](#step-7-testing-your-model)
11. [Step 8: Managing Endpoints](#step-8-managing-endpoints)
12. [Cost Management](#cost-management)
13. [Troubleshooting](#troubleshooting)

---

## What is SageMaker?

**Amazon SageMaker** is AWS's machine learning service that helps you:
- **Train** ML models using powerful GPU/CPU instances
- **Deploy** models as API endpoints for real-time predictions
- **Scale** automatically based on traffic

**In Simple Terms:** Think of SageMaker as a cloud computer that:
1. Takes your fashion images as input
2. Trains a smart AI model to understand clothing
3. Creates an API endpoint that recommends matching outfits

---

## Prerequisites

Before you begin, you need:

### ✅ Required Accounts & Tools

1. **AWS Account** 
   - Sign up at: https://aws.amazon.com
   - You'll need a credit card (for verification)
   - AWS Free Tier available for testing

2. **AWS CLI (Command Line Interface)**
   - Download: https://aws.amazon.com/cli/
   - This lets you control AWS from your terminal

3. **Python 3.8 or higher**
   - Check version: `python --version`
   - Download from: https://www.python.org/downloads/

4. **Git** (already installed if you cloned this repo)
   - Check: `git --version`

5. **Basic Terminal Knowledge**
   - How to navigate folders (`cd`, `ls`)
   - How to run commands

### 💾 System Requirements

- **Storage:** At least 10GB free space
- **RAM:** 4GB minimum (8GB recommended)
- **Internet:** Stable connection for AWS operations

---

## Understanding the Architecture

### What Happens During Training?

```
Your Computer          →    AWS S3           →    SageMaker Training    →    Model Artifacts
(Fashion Images)            (Cloud Storage)       (Powerful Instance)         (Saved in S3)
    └─ Upload ───────────→ s3://bucket/train/ → Processes Images ─────→ s3://bucket/model.tar.gz
```

### What Happens During Inference?

```
Your Request    →    SageMaker Endpoint    →    Trained Model    →    Recommendations
(Image)              (API on m5.large)          (Loaded in RAM)       (Top 5 matches)
```

### Key Concepts

- **S3 Bucket:** Cloud storage for your images and models (like Dropbox)
- **Training Job:** Temporary computer that trains your model
- **Model Artifacts:** The "brain" of your AI (saved as model.tar.gz)
- **Endpoint:** A running API that accepts images and returns recommendations
- **Instance Type:** The computer size (e.g., ml.m5.large = 2 CPUs, 8GB RAM)

---

## Step 1: AWS Setup

### 1.1 Install AWS CLI

**For Windows:**
```bash
# Download installer from: https://aws.amazon.com/cli/
# Run the MSI installer
# Verify installation:
aws --version
```

**For Mac:**
```bash
brew install awscli
aws --version
```

**For Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

### 1.2 Create AWS Access Keys

1. **Log in to AWS Console:** https://console.aws.amazon.com
2. **Click your name** (top-right) → **Security Credentials**
3. **Scroll to "Access Keys"** → **Create New Access Key**
4. **Download the CSV file** (contains Access Key ID and Secret Access Key)
5. **IMPORTANT:** Keep this file secure! Never share these keys.

### 1.3 Configure AWS CLI

```bash
aws configure
```

**You'll be asked 4 questions:**

```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE          # Paste from CSV
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY  # Paste from CSV
Default region name [None]: ap-south-1                   # Or us-east-1
Default output format [None]: json                       # Press Enter
```

**Verify configuration:**
```bash
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDAI...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/yourname"
}
```

✅ **Success!** You're now connected to AWS.

---

## Step 2: Prepare Your Dataset

### 2.1 Dataset Structure

Your fashion images **must** be organized like this:

```
outfits_dataset/
└── train/
    ├── upperwear/
    │   ├── shirt/
    │   │   ├── image001.jpg
    │   │   ├── image002.jpg
    │   │   └── ...
    │   ├── tshirt/
    │   │   ├── image001.jpg
    │   │   └── ...
    │   └── jacket/
    │       └── ...
    ├── bottomwear/
    │   ├── pants/
    │   ├── shorts/
    │   └── skirt/
    ├── footwear/
    │   ├── shoes/
    │   ├── sneakers/
    │   ├── heels/
    │   └── flats/
    ├── accessories/
    │   ├── bag/
    │   └── hat/
    └── one-piece/
        └── dress/
```

### 2.2 Image Requirements

- **Format:** JPG, JPEG, or PNG
- **Size:** Any size (will be auto-resized)
- **Recommended:** 224x224 or larger
- **Minimum:** At least 10 images per category
- **Recommended:** 100+ images per category for better accuracy

### 2.3 Check Your Dataset

```bash
# Count total images
find outfits_dataset/train -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l

# List categories
ls -R outfits_dataset/train/
```

---

## Step 3: Upload Dataset to S3

### 3.1 Understand S3

**S3 (Simple Storage Service)** is like Dropbox for AWS. You need to:
1. Create a "bucket" (folder in the cloud)
2. Upload your images to this bucket
3. SageMaker will read images from this bucket

### 3.2 Create S3 Bucket (Optional - Already Created)

The project uses bucket: `stygig-ml-s3` (already created in region `ap-south-1`)

**If you want to create your own bucket:**

```bash
# Replace 'my-fashion-bucket' with your unique bucket name
aws s3 mb s3://my-fashion-bucket --region ap-south-1
```

**Note:** Bucket names must be globally unique across all AWS accounts.

### 3.3 Upload Your Dataset

```bash
# Navigate to your project directory
cd stygig_project

# Upload entire dataset to S3 (this will take 5-30 minutes depending on size)
aws s3 sync outfits_dataset/train/ s3://stygig-ml-s3/train/ --region ap-south-1

# You'll see output like:
# upload: outfits_dataset/train/upperwear/shirt/001.jpg to s3://stygig-ml-s3/train/upperwear/shirt/001.jpg
# upload: outfits_dataset/train/upperwear/shirt/002.jpg to s3://stygig-ml-s3/train/upperwear/shirt/002.jpg
# ... (continues for all images)
```

### 3.4 Verify Upload

```bash
# List uploaded files (first 20)
aws s3 ls s3://stygig-ml-s3/train/ --recursive | head -20

# Count total files uploaded
aws s3 ls s3://stygig-ml-s3/train/ --recursive | wc -l

# Check size of uploaded data
aws s3 ls s3://stygig-ml-s3/train/ --recursive --summarize --human-readable
```

**Expected output:**
```
Total Objects: 5432
   Total Size: 2.5 GB
```

✅ **Success!** Your dataset is now in the cloud.

---

## Step 4: Configure Your Environment

### 4.1 Make Scripts Executable

```bash
# Navigate to project directory
cd stygig_project

# Make all scripts executable
chmod +x scripts/set_permissions.sh
./scripts/set_permissions.sh
```

**Output:**
```
Making shell scripts executable...

✅ scripts/run_pipeline.sh
✅ scripts/deploy_model.sh
✅ scripts/set_permissions.sh
✅ scripts/testing/verify_structure.sh

✅ Done! Made 4 script(s) executable.
```

### 4.2 Verify Project Structure

```bash
./scripts/testing/verify_structure.sh
```

**Expected output:**
```
════════════════════════════════════════════════════════════════
   StyGig Project Structure Verification
════════════════════════════════════════════════════════════════

📁 Checking Core Directory Structure
────────────────────────────────────────────────────────────────
✅ src/
✅ src/stygig/
✅ sagemaker/
✅ config/
✅ scripts/
... (all checks should pass)

Total Checks: 35
Passed: 35
✅ Project structure is correct!
```

### 4.3 Set Environment Variables (Optional)

You can customize settings by exporting environment variables:

```bash
# S3 Configuration
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1

# Instance Types (Compute Power)
export TRAINING_INSTANCE_TYPE=ml.m5.large      # 2 CPUs, 8GB RAM (~$0.12/hour)
export INFERENCE_INSTANCE_TYPE=ml.m5.large     # For endpoint

# Cost Optimization
export USE_SPOT_INSTANCES=false                # Set true to save 70% (less reliable)

# Pipeline Behavior
export TEST_ENDPOINT=true                      # Automatically test after deployment
```

**Or edit the config file directly:**
```bash
# Open in your text editor
nano config/settings.py
# or
code config/settings.py
```

---

## Step 5: Train Your Model

### 5.1 What is Training?

**Training** is the process where SageMaker:
1. Downloads your images from S3
2. Uses a powerful instance (computer) to process images
3. Extracts visual features using CLIP (AI vision model)
4. Builds a FAISS index (fast search database)
5. Saves the trained model back to S3

**Duration:** 30-60 minutes (depends on dataset size and instance type)

### 5.2 Training Options

#### **Option A: Full Pipeline (Recommended for Beginners)**

This runs everything: training → deployment → testing

```bash
./scripts/run_pipeline.sh
```

**What happens:**
```
Step 1: Validating Environment ✓
Step 2: Loading Configuration ✓
Step 3: Verifying Dataset in S3 ✓
Step 4: Installing Dependencies ✓
Step 5: Checking SageMaker Permissions ✓
Step 6: Running SageMaker Training Job 🚀
   - Creating training job: stygig-training-1699123456
   - Downloading dataset from S3...
   - Processing images...
   - Training CLIP embeddings...
   - Building FAISS index...
   - Saving model to S3...
   ✓ Training completed in 45 minutes
Step 7: Deploying Model to Endpoint 🚀
   - Creating endpoint: stygig-endpoint-20251105-143022
   - Deploying model...
   ✓ Endpoint deployed in 8 minutes
Step 8: Testing Endpoint ✓
   ✓ Test completed successfully!
```

#### **Option B: Training Only (No Deployment)**

If you just want to train without deploying:

```bash
./scripts/run_pipeline.sh --skip-deployment
```

#### **Option C: Custom Training with Python**

For more control, use the Python script directly:

```bash
cd sagemaker
python run_sagemaker_pipeline.py
```

### 5.3 Monitor Training Progress

#### **Method 1: Watch the Terminal Output**

The script shows progress in your terminal. Look for:
```
📚 Building fashion index...
Processed 1/15 categories: upperwear/shirt (245 items)
Processed 2/15 categories: upperwear/tshirt (189 items)
...
✅ Index built in 42.5 seconds
```

#### **Method 2: Check AWS Console**

1. Open AWS Console: https://console.aws.amazon.com
2. Go to **Services → SageMaker → Training Jobs**
3. Find your job: `stygig-training-XXXXXXXXXX`
4. Click on it to see:
   - Status (InProgress, Completed, Failed)
   - Duration
   - Instance type
   - Logs

#### **Method 3: Monitor CloudWatch Logs**

In a **separate terminal**, run:

```bash
# Watch logs in real-time
aws logs tail /aws/sagemaker/TrainingJobs --follow --region ap-south-1
```

**You'll see detailed output:**
```
2025-11-05 14:30:22 Starting StyGig training pipeline...
2025-11-05 14:30:25 Loading CLIP model: ViT-B-32
2025-11-05 14:30:45 Processing category: upperwear/shirt
2025-11-05 14:31:12 Extracted features for 245 images
...
```

### 5.4 Training Outputs

Once training completes, you'll find:

1. **Model Artifacts in S3:**
   ```
   s3://stygig-ml-s3/model-artifacts/stygig-training-XXXXXXXXXX/output/model.tar.gz
   ```

2. **Training Logs in S3:**
   ```
   s3://stygig-ml-s3/logs/stygig-training-XXXXXXXXXX/
   ```

3. **Local Result File:**
   ```
   sagemaker/pipeline_results.json
   ```

**Check results:**
```bash
cat sagemaker/pipeline_results.json
```

**Example output:**
```json
{
  "training_job_name": "stygig-training-1699123456",
  "training_status": "Completed",
  "model_data": "s3://stygig-ml-s3/model-artifacts/.../model.tar.gz",
  "training_time_seconds": 2734,
  "billable_time_seconds": 2800
}
```

### 5.5 Understanding Training Costs

| Instance Type | vCPU | RAM | Cost/Hour | 1-Hour Training |
|---------------|------|-----|-----------|-----------------|
| ml.m5.large | 2 | 8GB | $0.115 | $0.12 |
| ml.m5.xlarge | 4 | 16GB | $0.23 | $0.23 |
| ml.m5.2xlarge | 8 | 32GB | $0.46 | $0.46 |

**Spot Instances:** Save ~70% but training might be interrupted.

```bash
# Enable spot instances (in config/settings.py or as env var)
export USE_SPOT_INSTANCES=true
```

✅ **Training Complete!** Your model is ready to deploy.

---

## Step 6: Deploy the Model (Create Endpoint)

### 6.1 What is an Endpoint?

An **endpoint** is like a web API that:
- Runs 24/7 on an AWS instance
- Accepts image inputs
- Returns fashion recommendations
- Scales automatically

**Think of it as:** A waiter at a restaurant who takes your order (image) and brings back recommendations.

### 6.2 Deployment Options

#### **Option A: Deploy After Training (Automatic)**

If you ran the full pipeline, deployment happens automatically after training.

#### **Option B: Deploy Existing Model**

If you already have a trained model:

```bash
# Method 1: Using training job name
./scripts/deploy_model.sh --training-job-name stygig-training-1699123456

# Method 2: Using direct S3 model URI
./scripts/deploy_model.sh --model-uri s3://stygig-ml-s3/model-artifacts/stygig-training-1699123456/output/model.tar.gz

# Method 3: With custom endpoint name
./scripts/deploy_model.sh \
    --training-job-name stygig-training-1699123456 \
    --endpoint-name my-fashion-api \
    --instance-type ml.m5.large
```

### 6.3 What Happens During Deployment?

```
Step 1: Validating inputs... ✓
Step 2: Creating endpoint: stygig-endpoint-20251105-143022
Step 3: Deploying model to SageMaker endpoint...
   - Downloading model from S3...
   - Initializing inference container...
   - Loading CLIP model (takes 1-2 minutes)...
   - Starting endpoint...
   ✓ Endpoint deployed successfully!
   
🎉 Your fashion recommendation API is now live!

📊 Endpoint Details:
   Name: stygig-endpoint-20251105-143022
   Region: us-east-1
   Instance: ml.m5.large
   Status: InService
```

### 6.4 Monitor Deployment

#### **Check Status via AWS Console:**
1. Go to: **SageMaker → Endpoints**
2. Find: `stygig-endpoint-XXXXXXXXXX`
3. Wait for **Status: InService** (takes 5-10 minutes)

#### **Check Status via CLI:**
```bash
aws sagemaker describe-endpoint --endpoint-name stygig-endpoint-20251105-143022 --region us-east-1
```

### 6.5 Endpoint Information

After deployment, endpoint details are saved to:
```
sagemaker/endpoint_info.json
```

**View endpoint info:**
```bash
cat sagemaker/endpoint_info.json
```

**Example output:**
```json
{
  "endpoint_name": "stygig-endpoint-20251105-143022",
  "model_uri": "s3://stygig-ml-s3/model-artifacts/.../model.tar.gz",
  "instance_type": "ml.m5.large",
  "region": "us-east-1",
  "deployment_time": "2025-11-05 14:30:22"
}
```

### 6.6 Understanding Endpoint Costs

| Instance Type | vCPU | RAM | Cost/Hour | Cost/Day | Cost/Month |
|---------------|------|-----|-----------|----------|------------|
| ml.m5.large | 2 | 8GB | $0.115 | $2.76 | $82.80 |
| ml.t2.medium | 2 | 4GB | $0.065 | $1.56 | $46.80 |
| ml.m5.xlarge | 4 | 16GB | $0.23 | $5.52 | $165.60 |

**💰 IMPORTANT:** Endpoints run 24/7 and cost money even when not in use!

**Always delete endpoints when not needed:**
```bash
python scripts/manage_endpoints.py delete-all
```

✅ **Endpoint Created!** Your API is ready to receive requests.

---

## Step 7: Testing Your Model

This is the most important section! Here you'll learn how to test your deployed model with specific images.

### 7.1 Understanding Testing

**Testing** means:
1. Sending an input image to your endpoint
2. Getting back fashion recommendations
3. Verifying the results make sense

### 7.2 Testing Methods Overview

| Method | Input Source | Output | Best For |
|--------|-------------|--------|----------|
| **Auto-test** | Random S3 image | Terminal + Visual | Quick check |
| **Specific S3 Image** | Exact S3 path | Terminal + Visual | Testing specific items |
| **Local Image** | Your computer | Terminal + Visual | Custom testing |
| **Python Script** | Any source | JSON + Code | Integration |

---

### 7.3 Method 1: Quick Auto-Test (Easiest)

This automatically picks a random image from S3 and tests it.

```bash
python scripts/testing/test_endpoint.py --save-visual
```

**What happens:**
1. Automatically detects your endpoint from `endpoint_info.json`
2. Picks a random shirt image from S3
3. Sends it to the endpoint
4. Displays recommendations in terminal
5. Saves visual output with images side-by-side

**Example output:**
```
Testing endpoint: stygig-endpoint-20251105-143022
Downloading from S3: s3://stygig-ml-s3/train/upperwear/shirt/0001.jpg
✓ Downloaded to: test_image.jpg
Input image: test_image.jpg (800x1200)
Sending inference request (top_k=5)...
⏳ Note: First request may take 1-2 minutes (cold start - loading model)

✅ Inference successful!
════════════════════════════════════════════════════════════════
RECOMMENDATION RESULTS
════════════════════════════════════════════════════════════════

📸 INPUT ITEM:
   Category: upperwear_shirt
   Gender: male
   Dominant Colors: blue, white

🎯 TOP 5 RECOMMENDATIONS:

1. bottomwear/pants/P001.jpg
   Category: bottomwear_pants
   Gender: male
   Score: 0.8925
   Color Match: 0.8750
   Category Match: 1.0000
   Gender Match: 1.0000
   Colors: navy, gray

2. footwear/shoes/S045.jpg
   Category: footwear_shoes
   Gender: male
   Score: 0.8623
   ...

📊 METADATA:
   Processing time: 156.45ms
   Total items searched: 5432
   Model version: ViT-B-32

📸 VISUAL OUTPUT:
   Image: scripts/testing/test_results/recommendations_20251105_143045.jpg
   JSON: scripts/testing/test_results/recommendations_20251105_143045.json
```

**View the visual output:**
- Open: `scripts/testing/test_results/recommendations_XXXXXX.jpg`
- You'll see your input image + 5 recommended items side-by-side with scores

---

### 7.4 Method 2: Test with Specific S3 Image

Test with a specific image from your S3 bucket.

#### **Step 1: Find Your Image in S3**

```bash
# List all available images
aws s3 ls s3://stygig-ml-s3/train/ --recursive

# Find specific category
aws s3 ls s3://stygig-ml-s3/train/upperwear/shirt/ --recursive

# Example output:
# 2025-11-05 14:22:15    245678 train/upperwear/shirt/shirt_001.jpg
# 2025-11-05 14:22:16    198234 train/upperwear/shirt/shirt_002.jpg
# 2025-11-05 14:22:17    234567 train/upperwear/shirt/shirt_003.jpg
```

#### **Step 2: Test with That Image**

```bash
# Format: s3://bucket-name/full/path/to/image.jpg
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/shirt_001.jpg \
    --save-visual

# Get more recommendations
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/shirt_001.jpg \
    --top-k 10 \
    --save-visual
```

#### **Step 3: Test Multiple Images**

Create a simple bash script to test multiple images:

```bash
# Create a test script
cat > test_multiple_images.sh << 'EOF'
#!/bin/bash

# List of images to test
images=(
    "s3://stygig-ml-s3/train/upperwear/shirt/shirt_001.jpg"
    "s3://stygig-ml-s3/train/upperwear/tshirt/tshirt_042.jpg"
    "s3://stygig-ml-s3/train/bottomwear/pants/pants_015.jpg"
    "s3://stygig-ml-s3/train/footwear/shoes/shoes_089.jpg"
)

# Test each image
for image in "${images[@]}"; do
    echo "Testing: $image"
    python scripts/testing/test_endpoint.py --s3-image "$image" --save-visual
    echo "---"
done

echo "All tests complete! Check scripts/testing/test_results/ for outputs"
EOF

# Make it executable
chmod +x test_multiple_images.sh

# Run it
./test_multiple_images.sh
```

---

### 7.5 Method 3: Test with Local Image

Test with an image on your computer (not in S3).

```bash
# Test with local image
python scripts/testing/test_endpoint.py \
    --image /path/to/your/image.jpg \
    --save-visual

# Example with relative path
python scripts/testing/test_endpoint.py \
    --image outfits_dataset/train/upperwear/shirt/my_shirt.jpg \
    --save-visual

# Get 10 recommendations
python scripts/testing/test_endpoint.py \
    --image my_test_image.jpg \
    --top-k 10 \
    --save-visual
```

---

### 7.6 Method 4: Custom Python Testing Script

Create your own testing script for advanced use cases.

#### **Example 1: Test and Save Results to CSV**

```python
# test_to_csv.py
import boto3
import json
import base64
import csv
from pathlib import Path

# Configuration
ENDPOINT_NAME = "stygig-endpoint-20251105-143022"
REGION = "us-east-1"
S3_BUCKET = "stygig-ml-s3"

# Initialize client
runtime = boto3.client('sagemaker-runtime', region_name=REGION)
s3_client = boto3.client('s3', region_name='ap-south-1')

def test_image(s3_key):
    """Test a single image and return results."""
    
    # Download image from S3
    local_path = f"/tmp/{Path(s3_key).name}"
    s3_client.download_file(S3_BUCKET, s3_key, local_path)
    
    # Convert to base64
    with open(local_path, 'rb') as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare payload
    payload = {
        'image': image_b64,
        'top_k': 5,
        'min_score': 0.5
    }
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse results
    result = json.loads(response['Body'].read().decode())
    return result

def main():
    # List of images to test
    test_images = [
        "train/upperwear/shirt/shirt_001.jpg",
        "train/upperwear/tshirt/tshirt_042.jpg",
        "train/bottomwear/pants/pants_015.jpg"
    ]
    
    # Open CSV file
    with open('test_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input Image', 'Rank', 'Recommended Item', 'Score', 
                        'Color Score', 'Category Score', 'Gender Score'])
        
        # Test each image
        for image_key in test_images:
            print(f"Testing: {image_key}")
            result = test_image(image_key)
            
            # Write recommendations to CSV
            for i, rec in enumerate(result['recommendations'], 1):
                writer.writerow([
                    image_key,
                    i,
                    rec['item_id'],
                    rec['score'],
                    rec['color_score'],
                    rec['category_score'],
                    rec['gender_score']
                ])
            
            print(f"  ✓ Got {len(result['recommendations'])} recommendations")
    
    print("\n✅ Results saved to: test_results.csv")

if __name__ == '__main__':
    main()
```

**Run it:**
```bash
python test_to_csv.py
```

#### **Example 2: Batch Test All Shirts**

```python
# batch_test_shirts.py
import boto3
import json
import base64
import time

ENDPOINT_NAME = "stygig-endpoint-20251105-143022"
REGION = "us-east-1"
S3_BUCKET = "stygig-ml-s3"
S3_PREFIX = "train/upperwear/shirt/"

runtime = boto3.client('sagemaker-runtime', region_name=REGION)
s3_client = boto3.client('s3', region_name='ap-south-1')

def get_all_shirt_images():
    """Get list of all shirt images from S3."""
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    
    images = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(obj['Key'])
    
    return images

def test_image(s3_key):
    """Test single image."""
    local_path = f"/tmp/{Path(s3_key).name}"
    s3_client.download_file(S3_BUCKET, s3_key, local_path)
    
    with open(local_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {'image': image_b64, 'top_k': 5}
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    return json.loads(response['Body'].read().decode())

def main():
    print("Finding all shirt images...")
    shirt_images = get_all_shirt_images()
    print(f"Found {len(shirt_images)} shirt images")
    
    results = []
    
    for i, image_key in enumerate(shirt_images, 1):
        print(f"Testing {i}/{len(shirt_images)}: {image_key}")
        
        try:
            result = test_image(image_key)
            results.append({
                'input': image_key,
                'success': True,
                'recommendations': len(result['recommendations']),
                'avg_score': sum(r['score'] for r in result['recommendations']) / len(result['recommendations'])
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'input': image_key,
                'success': False,
                'error': str(e)
            })
        
        # Sleep to avoid throttling
        time.sleep(0.5)
    
    # Save results
    with open('batch_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ Batch testing complete!")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Results saved to: batch_test_results.json")

if __name__ == '__main__':
    from pathlib import Path
    main()
```

---

### 7.7 Understanding Test Results

#### **Terminal Output Explained:**

```
📸 INPUT ITEM:
   Category: upperwear_shirt      # What type of clothing
   Gender: male                    # Detected gender
   Dominant Colors: blue, white    # Main colors in the image

🎯 TOP 5 RECOMMENDATIONS:

1. bottomwear/pants/P001.jpg
   Category: bottomwear_pants      # Recommended item type
   Gender: male                    # Compatible gender
   Score: 0.8925                   # Overall match score (0-1, higher is better)
   Color Match: 0.8750             # How well colors harmonize (0-1)
   Category Match: 1.0000          # Category compatibility (0-1)
   Gender Match: 1.0000            # Gender compatibility (0 or 1)
   Colors: navy, gray              # Colors in recommended item
```

#### **Score Interpretation:**

| Score Range | Meaning | Quality |
|-------------|---------|---------|
| 0.90 - 1.00 | Excellent match | Use these! |
| 0.75 - 0.89 | Good match | Acceptable |
| 0.60 - 0.74 | Fair match | May work |
| < 0.60 | Poor match | Avoid |

#### **What Makes a Good Recommendation:**

✅ **Good Signs:**
- High color harmony (> 0.75)
- Compatible categories (shirt → pants, not shirt → shirt)
- Matching gender
- Overall score > 0.80

❌ **Bad Signs:**
- Low color harmony (< 0.50)
- Same category recommendations
- Gender mismatch
- Overall score < 0.60

---

### 7.8 Visual Output Analysis

After running with `--save-visual`, check the output folder:

```bash
ls -la scripts/testing/test_results/

# Example output:
# recommendations_20251105_143045.jpg  <- Side-by-side image comparison
# recommendations_20251105_143045.json <- Raw JSON data
# rec_0_20251105_143045.jpg           <- Individual recommended items
# rec_1_20251105_143045.jpg
# ...
```

**Open the main image:**
```bash
# On Mac
open scripts/testing/test_results/recommendations_20251105_143045.jpg

# On Linux
xdg-open scripts/testing/test_results/recommendations_20251105_143045.jpg

# On Windows
start scripts/testing/test_results/recommendations_20251105_143045.jpg
```

**You'll see:**
- Input image on the left
- Top 5 recommendations on the right
- Scores displayed below each image
- Color information and categories

---

### 7.9 Testing Checklist

Use this checklist to verify your model works correctly:

- [ ] **Basic Test:** Run auto-test and get results
- [ ] **Specific Item:** Test a known good image from S3
- [ ] **Male Item:** Test with male clothing, verify no female recommendations
- [ ] **Female Item:** Test with female clothing, verify no male recommendations
- [ ] **Color Harmony:** Blue shirt should recommend navy/white pants (not red)
- [ ] **Category Diversity:** Shirt should NOT recommend other shirts
- [ ] **Performance:** First request < 2 minutes, subsequent < 2 seconds
- [ ] **Visual Output:** Images display correctly side-by-side
- [ ] **JSON Output:** Valid JSON with all required fields

---

### 7.10 Common Testing Scenarios

#### **Scenario 1: Test Color Matching**

```bash
# Test blue shirt - should recommend navy/white/gray items
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/blue_shirt_001.jpg \
    --save-visual

# Expected: High color scores for navy, white, gray items
# Not expected: Red or bright yellow items
```

#### **Scenario 2: Test Gender Filtering**

```bash
# Test male shirt - should only get male/unisex recommendations
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/mens_shirt.jpg \
    --save-visual

# Verify in output: All recommendations should show "Gender: male" or "Gender: unisex"
```

#### **Scenario 3: Test Category Diversity**

```bash
# Test shirt - should NOT recommend other shirts
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/shirt_001.jpg \
    --save-visual

# Verify: Recommendations include pants, shoes, accessories
# NOT: Other shirts or tshirts
```

---

## Step 8: Managing Endpoints

### 8.1 Why Manage Endpoints?

**Important:** Endpoints cost money 24/7, even when not in use!

| Action | When | Why |
|--------|------|-----|
| **List** | Anytime | See what's running |
| **Delete** | After testing | Save money |
| **Info** | Troubleshooting | Get details |

### 8.2 List All Endpoints

```bash
python scripts/manage_endpoints.py list
```

**Output:**
```
════════════════════════════════════════════════════════════════
Endpoint Name                                 Status         Instance Type   Created
════════════════════════════════════════════════════════════════
stygig-endpoint-20251105-143022              InService      ml.m5.large     2025-11-05 14:30:22
stygig-endpoint-20251104-091234              InService      ml.m5.large     2025-11-04 09:12:34
════════════════════════════════════════════════════════════════

Total: 2 endpoint(s)
```

### 8.3 Get Endpoint Details

```bash
python scripts/manage_endpoints.py info --endpoint-name stygig-endpoint-20251105-143022
```

**Output:**
```
════════════════════════════════════════════════════════════════
Endpoint Details: stygig-endpoint-20251105-143022
════════════════════════════════════════════════════════════════
Status: InService
ARN: arn:aws:sagemaker:us-east-1:123456789012:endpoint/stygig-endpoint-20251105-143022
Created: 2025-11-05 14:30:22
Last Modified: 2025-11-05 14:38:15
Config Name: stygig-endpoint-config-20251105-143022

Production Variants:
  - Instance Type: ml.m5.large
    Instance Count: 1
    Variant Name: AllTraffic
════════════════════════════════════════════════════════════════
```

### 8.4 Delete Single Endpoint

```bash
python scripts/manage_endpoints.py delete --endpoint-name stygig-endpoint-20251105-143022
```

**Output:**
```
Deleting endpoint: stygig-endpoint-20251105-143022
✅ Endpoint 'stygig-endpoint-20251105-143022' deleted successfully
✅ Endpoint config 'stygig-endpoint-config-20251105-143022' deleted successfully
```

### 8.5 Delete All Endpoints (Cost Savings!)

```bash
python scripts/manage_endpoints.py delete-all
```

**Output:**
```
════════════════════════════════════════════════════════════════
⚠️  WARNING: About to delete the following endpoints:
════════════════════════════════════════════════════════════════
1. stygig-endpoint-20251105-143022
   Status: InService
   Created: 2025-11-05 14:30:22

2. stygig-endpoint-20251104-091234
   Status: InService
   Created: 2025-11-04 09:12:34

Total: 2 endpoint(s)
════════════════════════════════════════════════════════════════

Are you sure you want to delete ALL these endpoints? (yes/no): yes

Deleting endpoint: stygig-endpoint-20251105-143022
✅ Endpoint 'stygig-endpoint-20251105-143022' deleted successfully

Deleting endpoint: stygig-endpoint-20251104-091234
✅ Endpoint 'stygig-endpoint-20251104-091234' deleted successfully

✅ Deleted 2/2 endpoint(s)

💰 Cost Savings: Deleting 2 endpoint(s) will save costs!
   Endpoints are billed per hour while running.
```

### 8.6 Endpoint Management Best Practices

✅ **DO:**
- Delete endpoints after testing
- Use one endpoint at a time
- Monitor costs in AWS Cost Explorer
- List endpoints regularly: `python scripts/manage_endpoints.py list`

❌ **DON'T:**
- Leave endpoints running overnight
- Create multiple endpoints unnecessarily
- Forget to delete test endpoints

---

## Cost Management

### Understanding AWS Costs

| Resource | Cost | Duration | Best Practice |
|----------|------|----------|---------------|
| **Training** | $0.12-0.46/hour | 30-60 min | One-time cost |
| **Endpoint** | $0.12-0.23/hour | 24/7 | Delete when not in use |
| **S3 Storage** | $0.023/GB/month | Always | Keep only needed data |

### Cost Calculation Example

**Scenario:** Train once, keep endpoint for 8 hours of testing

```
Training (1 hour):        $0.12
Endpoint (8 hours):       $0.96
S3 (5GB for 1 month):     $0.12
───────────────────────────────
Total:                    $1.20
```

### Cost Optimization Tips

1. **Use Spot Instances for Training:**
   ```bash
   export USE_SPOT_INSTANCES=true
   # Saves ~70% but may be interrupted
   ```

2. **Delete Endpoints When Not in Use:**
   ```bash
   python scripts/manage_endpoints.py delete-all
   ```

3. **Use Smaller Instances:**
   ```bash
   export TRAINING_INSTANCE_TYPE=ml.m5.large      # Instead of xl
   export INFERENCE_INSTANCE_TYPE=ml.m5.large     # Instead of xl
   ```

4. **Limit Dataset Size During Testing:**
   ```python
   # In config/settings.py
   'max-items-per-category': 100  # Instead of 500
   ```

5. **Monitor Costs:**
   - AWS Console → Billing Dashboard
   - Set up cost alerts
   - Review monthly bills

### Setting Cost Alerts

1. Go to: https://console.aws.amazon.com/billing/
2. Click: **Budgets**
3. Click: **Create budget**
4. Select: **Cost budget**
5. Set amount: $10 (or your limit)
6. Add email alerts at 80% and 100%

---

## Troubleshooting

### Issue 1: "AWS credentials not configured"

**Symptom:**
```
Unable to locate credentials. You can configure credentials by running "aws configure".
```

**Solution:**
```bash
aws configure
# Enter your Access Key ID and Secret Access Key
```

**Verify:**
```bash
aws sts get-caller-identity
```

---

### Issue 2: "S3 bucket not found"

**Symptom:**
```
❌ ERROR: Cannot access stygig-ml-s3 bucket
```

**Solution:**
```bash
# Check if bucket exists
aws s3 ls s3://stygig-ml-s3/ --region ap-south-1

# If not, create it
aws s3 mb s3://stygig-ml-s3 --region ap-south-1

# Upload dataset
aws s3 sync outfits_dataset/train/ s3://stygig-ml-s3/train/ --region ap-south-1
```

---

### Issue 3: "Training job failed"

**Symptom:**
```
Training job status: Failed
```

**Check logs:**
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow --region ap-south-1
```

**Common causes:**
- **Out of memory:** Use larger instance type
- **No images found:** Check dataset structure
- **Permission denied:** Check IAM role permissions

**Solution:**
```bash
# Use larger instance
export TRAINING_INSTANCE_TYPE=ml.m5.xlarge

# Verify dataset
aws s3 ls s3://stygig-ml-s3/train/ --recursive | head -20
```

---

### Issue 4: "Endpoint creation timeout"

**Symptom:**
```
Endpoint creation taking too long (>15 minutes)
```

**This is normal** for first deployment. The endpoint:
1. Downloads model from S3 (~2-3 min)
2. Loads CLIP model (~5-7 min)
3. Initializes FAISS index (~2-3 min)

**Wait patiently.** Check status:
```bash
aws sagemaker describe-endpoint --endpoint-name stygig-endpoint-XXXXXXXX
```

---

### Issue 5: "First inference request timeout"

**Symptom:**
```
First request taking 1-2 minutes
```

**This is normal!** This is called "cold start":
- CLIP model loads into memory (~60-90 seconds)
- Subsequent requests are fast (~1-2 seconds)

**Solution:** Just wait. The script handles this automatically with extended timeout (5 minutes).

**Important Note:** All deployment scripts in this project have been configured with proper timeouts to handle CLIP model cold starts:
- **Model Server Timeout:** 300s (5 minutes) - Set via `SAGEMAKER_MODEL_SERVER_TIMEOUT` environment variable
- **Container Startup Timeout:** 600s (10 minutes) - Set via `container_startup_health_check_timeout` parameter
- **Model Download Timeout:** 600s (10 minutes) - Set via `model_data_download_timeout` parameter

These settings are already applied in:
- `deploy_endpoint.py`
- `deploy_existing_model.py`
- `redeploy_endpoint.py`
- `redeploy_with_timeout.py`
- `run_sagemaker_pipeline.py`

If you see timeout errors during deployment or first inference, verify these settings are present in your deployment code.

---

### Issue 6: "ModelError: Your invocation timed out while waiting for a response from container"

**Symptom:**
```
ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation: 
Received server error (0) from primary with message "Your invocation timed out while 
waiting for a response from container primary. Review the latency metrics for each 
container in Amazon CloudWatch..."
```

**Root Cause:** 
This error occurs when the CLIP model takes 2-3 minutes to load in `model_fn()`, but the default SageMaker timeout is only 60 seconds.

**Solution:**
All deployment scripts in this project now include the **CRITICAL** timeout fix:

```python
# In model creation
model = PyTorchModel(
    model_data=model_uri,
    role=role,
    entry_point='sagemaker/inference.py',
    source_dir=project_root,
    framework_version='2.0.0',
    py_version='py310',
    sagemaker_session=session,
    model_server_workers=1,
    env={
        # CRITICAL: This is the key setting that fixes the timeout
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',  # 5 minutes for model_fn
        'MODEL_SERVER_TIMEOUT': '300',
        'TS_DEFAULT_RESPONSE_TIMEOUT': '300',
        # ... other settings
    }
)

# In deployment
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    # CRITICAL: Extended timeouts for cold start
    container_startup_health_check_timeout=600,  # 10 minutes
    model_data_download_timeout=600  # 10 minutes
)
```

**If you still see this error:**
1. Check that your deployment script includes `SAGEMAKER_MODEL_SERVER_TIMEOUT: '300'` in the `env` dictionary
2. Verify `container_startup_health_check_timeout=600` is set in `.deploy()`
3. Use `redeploy_with_timeout.py` as a reference implementation

---

### Issue 7: "Permission denied errors"

**Symptom:**
```
An error occurred (AccessDenied) when calling the CreateTrainingJob operation
```

**Solution:** Your IAM user needs these permissions:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

**How to add permissions:**
1. AWS Console → IAM → Users
2. Click your username
3. Click: **Add permissions**
4. Attach policies listed above

---

### Issue 8: "Out of memory during training"

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use smaller batch size:
   ```bash
   export BATCH_SIZE=16  # Instead of 32
   ```

2. Use larger instance:
   ```bash
   export TRAINING_INSTANCE_TYPE=ml.m5.xlarge
   ```

3. Limit items per category:
   ```python
   # In config/settings.py
   'max-items-per-category': 100
   ```

---

### Issue 9: "Module not found errors"

**Symptom:**
```
ModuleNotFoundError: No module named 'clip'
```

**Solution:**
```bash
cd sagemaker
pip install -r requirements.txt
```

---

### Issue 10: "Can't find endpoint_info.json"

**Symptom:**
```
No endpoint name provided. Use --endpoint-name or ensure endpoint_info.json exists
```

**Solution:**
```bash
# List your endpoints
python scripts/manage_endpoints.py list

# Test with specific endpoint name
python scripts/testing/test_endpoint.py --endpoint-name stygig-endpoint-XXXXXXXX
```

---

### Issue 11: "High AWS costs"

**Symptom:**
```
Unexpected $50+ bill from AWS
```

**Check what's running:**
```bash
# List all endpoints
python scripts/manage_endpoints.py list

# Delete all
python scripts/manage_endpoints.py delete-all

# Check for training jobs
aws sagemaker list-training-jobs --status-equals InProgress
```

**Prevention:**
- Always delete endpoints after testing
- Set up cost alerts
- Review costs weekly

---

## Quick Reference Commands

### Setup & Verification
```bash
# Install and configure AWS
aws configure
aws sts get-caller-identity

# Set up project
chmod +x scripts/set_permissions.sh
./scripts/set_permissions.sh
./scripts/testing/verify_structure.sh
```

### Dataset Upload
```bash
# Upload to S3
aws s3 sync outfits_dataset/train/ s3://stygig-ml-s3/train/ --region ap-south-1

# Verify upload
aws s3 ls s3://stygig-ml-s3/train/ --recursive | wc -l
```

### Training
```bash
# Full pipeline
./scripts/run_pipeline.sh

# Training only
./scripts/run_pipeline.sh --skip-deployment

# Monitor logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --region ap-south-1
```

### Deployment
```bash
# Deploy from training job
./scripts/deploy_model.sh --training-job-name stygig-training-XXXXXXXXXX

# Deploy from S3 model
./scripts/deploy_model.sh --model-uri s3://stygig-ml-s3/model-artifacts/.../model.tar.gz
```

### Testing
```bash
# Auto-test with random image
python scripts/testing/test_endpoint.py --save-visual

# Test specific S3 image
python scripts/testing/test_endpoint.py \
    --s3-image s3://stygig-ml-s3/train/upperwear/shirt/shirt_001.jpg \
    --save-visual

# Test local image
python scripts/testing/test_endpoint.py --image my_image.jpg --save-visual

# Get 10 recommendations
python scripts/testing/test_endpoint.py --top-k 10 --save-visual
```

### Endpoint Management
```bash
# List all endpoints
python scripts/manage_endpoints.py list

# Get endpoint details
python scripts/manage_endpoints.py info --endpoint-name stygig-endpoint-XXXXXXXX

# Delete one endpoint
python scripts/manage_endpoints.py delete --endpoint-name stygig-endpoint-XXXXXXXX

# Delete all endpoints (COST SAVINGS!)
python scripts/manage_endpoints.py delete-all
```

---

## Success Checklist

By the end of this guide, you should have:

- [x] AWS account configured with CLI
- [x] Dataset uploaded to S3 (`s3://stygig-ml-s3/train/`)
- [x] Training job completed successfully
- [x] Model deployed to endpoint
- [x] Tested endpoint with at least 3 different images
- [x] Verified recommendations make sense
- [x] Visual outputs generated and reviewed
- [x] Endpoint deleted to save costs

---

## What's Next?

After completing this guide:

1. **Integrate with Application:**
   - Use the endpoint from your web/mobile app
   - Send images via API calls
   - Display recommendations to users

2. **Improve Model:**
   - Add more training images
   - Retrain with better quality data
   - Fine-tune recommendation parameters

3. **Scale Up:**
   - Use auto-scaling for production
   - Set up monitoring and alerts
   - Implement caching for faster responses

4. **Optimize Costs:**
   - Use scheduled scaling (on during day, off at night)
   - Implement request batching
   - Consider serverless options for low traffic

---

## Support & Resources

- **Project Documentation:** `README.md`
- **Script Reference:** `scripts/README.md`
- **AWS SageMaker Docs:** https://docs.aws.amazon.com/sagemaker/
- **AWS Free Tier:** https://aws.amazon.com/free/
- **Cost Calculator:** https://calculator.aws/

---

**🎉 Congratulations!** You now know how to train, deploy, and test a machine learning model on AWS SageMaker!

**Remember:** Always delete endpoints when not in use to save costs! 💰

```bash
python scripts/manage_endpoints.py delete-all
```
