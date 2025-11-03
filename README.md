# StyGig Fashion Recommendation System
### Enterprise-Grade MVP

> **Professional fashion recommendation engine using computer vision, color theory, and intelligent matching algorithms**

---

## 🎯 Overview

StyGig is an enterprise-grade fashion recommendation system that provides intelligent outfit suggestions based on:

- **Advanced Color Harmony**: CIELAB color space with Itten color theory
- **Gender-Aware Filtering**: Hard gender compatibility rules
- **Category Intelligence**: Rule-based category compatibility matrix
- **Smart Diversity**: Configurable items per category (default: 2)
- **Professional Scoring**: Weighted algorithm (Color: 45%, Category: 25%, Gender: 30%)

### Key Features

✅ **Prevents Self-Matching**: Excludes input item from recommendations  
✅ **Color Harmony**: Advanced color matching beyond exact matches  
✅ **Gender Filtering**: Male users get male/unisex items only  
✅ **Category Rules**: No same-category recommendations (e.g., no shirt with shirt)  
✅ **AWS Ready**: Full SageMaker deployment pipeline included

---

## 📁 Project Structure

```
stygig_project/
├── src/
│   └── stygig/                      # Core Python package
│       ├── core/                    # Core recommendation logic
│       │   ├── recommendation_engine.py  # Main FashionEngine
│       │   ├── color_logic.py           # Color processing
│       │   ├── gender_logic.py          # Gender classification
│       │   └── rules/
│       │       └── category_compatibility.py  # Category rules
│       ├── api/                     # API endpoints (future)
│       └── utils/                   # Utility functions
│
├── sagemaker/                       # AWS SageMaker deployment
│   ├── train.py                     # Training script
│   ├── inference.py                 # Inference handler
│   ├── run_sagemaker_pipeline.py    # Pipeline orchestration
│   └── requirements.txt             # Production dependencies
│
├── testing/                         # Testing and validation
│   ├── integration_test.py          # End-to-end integration test
│   ├── test_engine.py               # Unit tests
│   └── local_train_test.py          # Local training validation
│
├── config/                          # Configuration management
│   ├── settings.py                  # SageMaker configuration
│   └── recommendation_config.py     # Recommendation parameters
│
├── outputs/                         # Local training outputs
├── outfits_dataset/                 # Fashion image dataset
├── requirements_local.txt           # Local development dependencies
├── run_project.sh                   # Main execution script
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **AWS Account** with SageMaker access
3. **AWS CLI** configured with credentials
4. **Fashion Dataset** uploaded to `s3://stygig-ml-s3/train/`

### Installation & Execution

```bash
# 1. Navigate to project directory
cd stygig_project

# 2. Make the execution script executable
chmod +x run_project.sh

# 3. Run the complete pipeline
./run_project.sh
```

The `run_project.sh` script will:
- ✅ Validate environment and AWS credentials
- ✅ Verify dataset availability in S3
- ✅ Install Python dependencies
- ✅ Launch SageMaker training job
- ✅ Deploy model to endpoint (optional)
- ✅ Test the deployed endpoint

---

## 🧪 Local Testing

For local development and testing without SageMaker:

```bash
# Install local dependencies
pip install -r requirements_local.txt

# Run integration test
cd testing
python integration_test.py ../outfits_dataset/train/upperwear_shirt/sample.jpg \
    --gender male \
    --items-per-category 2

# Run local training test
python local_train_test.py \
    --dataset-path ../outfits_dataset \
    --output-dir ../outputs
```

---

## 📊 Dataset Structure

Your fashion dataset should follow this structure:

```
outfits_dataset/
└── train/
    ├── upperwear/
    │   ├── shirt/
    │   ├── tshirt/
    │   └── jacket/
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

**Upload to S3:**
```bash
aws s3 sync outfits_dataset/train/ s3://stygig-ml-s3/train/ --region ap-south-1
```

---

## 🔧 Configuration

### Environment Variables

Configure the pipeline via environment variables (or edit `run_project.sh`):

```bash
export S3_BUCKET=stygig-ml-s3
export DATASET_S3_URI=s3://stygig-ml-s3/train/
export AWS_REGION=ap-south-1
export TRAINING_INSTANCE_TYPE=ml.m5.large
export INFERENCE_INSTANCE_TYPE=ml.m5.large
export USE_SPOT_INSTANCES=false
export TEST_ENDPOINT=true
export DEBUG_MODE=false
```

### Recommendation Parameters

Edit `config/recommendation_config.py` to customize:

```python
RecommendationConfig(
    items_per_category=2,              # Items per category
    min_similarity_threshold=0.5,      # Minimum match score
    enforce_category_diversity=True,   # Enable category rules
    faiss_search_multiplier=5          # Search space multiplier
)
```

---

## 🎨 How It Works

### 1. Color Harmony System

Uses **CIELAB color space** and **Itten color theory**:

| Input Color | Harmonious Colors | Score |
|-------------|-------------------|-------|
| **Neutrals** (black, white, gray) | All colors | 0.90 |
| **Warm** (red, orange, yellow) | Warm + complementary | 0.75-0.80 |
| **Cool** (blue, green, purple) | Cool + complementary | 0.75-0.80 |

### 2. Gender Compatibility

| User Gender | Compatible Items | Logic |
|-------------|-----------------|-------|
| **Male** | male, unisex | Hard filter: excludes female items |
| **Female** | female, unisex | Hard filter: excludes male items |
| **Unisex** | all | No filtering |

### 3. Category Rules

**Example**: Input = Shirt
- ✅ **Compatible**: pants, shorts, skirt, shoes, accessories
- ❌ **Avoided**: other shirts, t-shirts, jackets

### 4. Scoring Formula

```
Final Score = (0.45 × Color Harmony) 
            + (0.25 × Category Compatibility)
            + (0.30 × Gender Compatibility)
```

---

## 📖 API Reference

### Core Classes

#### `FashionEngine`

Main recommendation engine class.

```python
from stygig.core.recommendation_engine import FashionEngine

# Initialize
engine = FashionEngine(
    dataset_path='outfits_dataset',
    items_per_category=2,
    color_weight=0.45,
    category_weight=0.25,
    gender_weight=0.30
)

# Build index
engine.build_index()

# Get recommendations
results = engine.get_recommendations(
    image_path='path/to/item.jpg',
    user_gender='male',
    items_per_category=2
)
```

#### `ColorProcessor`

Handles color extraction and harmony calculation.

```python
from stygig.core.color_logic import ColorProcessor

processor = ColorProcessor(n_clusters=3)
colors = processor.extract_dominant_colors('image.jpg')
harmony_score = processor.calculate_color_harmony('blue', 'white')
```

#### `GenderClassifier`

Predicts gender and handles compatibility.

```python
from stygig.core.gender_logic import GenderClassifier

classifier = GenderClassifier()
gender, confidence = classifier.predict_gender('image.jpg', 'upperwear_shirt')
compatible = classifier.get_compatible_genders('male')  # ['male', 'unisex']
```

---

## 🔍 Troubleshooting

### AWS Credentials Issues
```bash
# Verify credentials
aws sts get-caller-identity

# Reconfigure if needed
aws configure
```

### S3 Access Issues
```bash
# Check bucket access
aws s3 ls s3://stygig-ml-s3/ --region ap-south-1

# Verify dataset
aws s3 ls s3://stygig-ml-s3/train/ --recursive | head -10
```

### SageMaker Permissions
Ensure your IAM role has:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scoped to stygig-ml-s3 bucket)

### Import Errors
```bash
# Ensure you're in the right directory
cd stygig_project

# Install dependencies
pip install -r sagemaker/requirements.txt  # For SageMaker
pip install -r requirements_local.txt      # For local testing
```

---

## 🏗️ Architecture Highlights

### Solved Problems

| Issue | Old Behavior | New Solution |
|-------|--------------|--------------|
| **Self-Matching** | Returns same item | Excludes input item by ID |
| **Color Matching** | Only exact matches | Advanced CIELAB harmony |
| **Gender Filtering** | No awareness | Hard gender rules |
| **Category Logic** | Same-category items | Compatibility matrix |
| **Diversity** | Random results | 2 items per category |

### Technology Stack

- **OpenCLIP**: Vision embeddings (ViT-B-32)
- **FAISS**: Fast similarity search
- **CIELAB**: Perceptually uniform color space
- **K-means**: Dominant color extraction
- **AWS SageMaker**: Training and deployment
- **Python 3.8+**: Core implementation

---

## 📝 Development

### Adding New Categories

1. Update `src/stygig/core/rules/category_compatibility.py`
2. Add category to `CATEGORY_COMPATIBILITY` dict
3. Define compatible and avoided categories
4. Rebuild index

### Customizing Scoring Weights

Edit weights in `FashionEngine` initialization:

```python
engine = FashionEngine(
    color_weight=0.50,      # Increase color importance
    category_weight=0.30,   # Increase category importance
    gender_weight=0.20      # Decrease gender importance
)
```

### Running Tests

```bash
cd testing

# Unit tests
python test_engine.py

# Integration test
python integration_test.py <image_path>

# Local training
python local_train_test.py --dataset-path ../outfits_dataset
```

---

## 📄 License

Copyright © 2025 StyGig Team. All rights reserved.

---

## 🤝 Contributing

This is an enterprise MVP. For contributions or inquiries, contact the development team.

---

## 📚 Additional Documentation

- **Training Pipeline**: See `sagemaker/train.py` docstring
- **Inference API**: See `sagemaker/inference.py` docstring
- **Configuration Guide**: See `config/settings.py`
- **Category Rules**: See `src/stygig/core/rules/category_compatibility.py`

---

**Built with ❤️ by the StyGig Team**
