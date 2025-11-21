# Run Visualization on AWS SageMaker

## 1. First, check what async results are available:

```bash
# SSH into your SageMaker instance and run:
aws s3 ls s3://sagemaker-ap-south-1-732414292744/async-results/ --recursive --human-readable
```

## 2. If you see result files, use the most recent one with Generate_combined_image.py:

```bash
# Navigate to your project directory
cd ~/Stygig-Model-2

# Install required packages if not already installed
pip install pillow boto3

# Run the visualization (replace the json_s3 path with the actual result file)
python Generate_combined_image.py \
    --input_image_s3 s3://stygig-ml-s3/train/upperwear/tshirt/upperwear_tshirt100.png \
    --json_s3 s3://sagemaker-ap-south-1-732414292744/async-results/[ACTUAL_RESULT_FILE] \
    --output_file stygig_v4_comparison.png
```

## 3. Alternative: Check async job status first

```bash
# Check if the async inference job is still running or completed
python -c "
import boto3
sagemaker = boto3.client('sagemaker', region_name='ap-south-1')
s3 = boto3.client('s3', region_name='ap-south-1')

# List recent async results
try:
    response = s3.list_objects_v2(
        Bucket='sagemaker-ap-south-1-732414292744', 
        Prefix='async-results/',
        MaxKeys=10
    )
    
    if 'Contents' in response:
        print('Recent async results:')
        for obj in sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True):
            print(f'{obj[\"Key\"]} - {obj[\"LastModified\"]} - {obj[\"Size\"]} bytes')
    else:
        print('No async results found yet - job may still be processing')
        
except Exception as e:
    print(f'Error: {e}')
"
```

## 4. If no results exist yet, the async job might still be processing:

```bash
# Wait a bit more and check again, or check SageMaker console for job status
# Async inference can take several minutes, especially on CPU instances
```

## 5. Once you have the comparison image, download it to your local machine:

```bash
# From your local Windows machine, download the generated image:
scp -i your-key.pem ec2-user@your-sagemaker-ip:~/Stygig-Model-2/stygig_v4_comparison.png ./
```

Or use SageMaker Studio's file download feature if you're using Studio.