"""
Test Script for Professional Fashion Recommendation Engine MVP

This script tests the new recommendation engine to ensure:
1. It doesn't return the same item as input (self-matching prevention)
2. Proper color harmony scoring works
3. Gender filtering works correctly
4. Category diversity is maintained
5. 2 items per category logic functions properly
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stygig.core.mvp_recommendation_engine import ProfessionalFashionRecommendationEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_recommendation_engine():
    """Test the professional recommendation engine."""
    
    print("🔍 Testing Professional Fashion Recommendation Engine MVP")
    print("=" * 60)
    
    # Initialize engine
    dataset_path = "outfits_dataset"  # Adjust path as needed
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    engine = ProfessionalFashionRecommendationEngine(
        dataset_path=dataset_path,
        items_per_category=2,
        color_weight=0.45,
        category_weight=0.25, 
        gender_weight=0.30
    )
    
    print("✅ Engine initialized with professional scoring weights")
    print(f"   Color: 45%, Category: 25%, Gender: 30%")
    
    # Build index
    print("\n🏗️ Building recommendation index...")
    try:
        engine.build_index()
        print("✅ Index built successfully")
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        return
    
    # Get statistics
    print("\n📊 Dataset Statistics:")
    stats = engine.get_statistics()
    print(f"   Total items: {stats['total_items']}")
    print(f"   Categories: {len(stats['categories'])}")
    print(f"   Gender distribution: {stats['gender_distribution']}")
    print(f"   Top colors: {dict(list(stats['color_distribution'].items())[:5])}")
    
    # Test with sample images
    test_cases = [
        {
            'path': 'outfits_dataset/train/upperwear_tshirt',
            'expected_categories': ['bottomwear_pants', 'bottomwear_shorts', 'footwear_sneakers'],
            'description': 'T-shirt should match with bottoms and footwear'
        },
        {
            'path': 'outfits_dataset/train/one-piece_dress', 
            'expected_categories': ['footwear_heels', 'accessories_bag', 'upperwear_jacket'],
            'description': 'Dress should match with shoes, bags, and jackets'
        },
        {
            'path': 'outfits_dataset/train/bottomwear_pants',
            'expected_categories': ['upperwear_shirt', 'upperwear_tshirt', 'footwear_shoes'],
            'description': 'Pants should match with tops and footwear'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print("-" * 50)
        
        # Find a sample image in the category
        category_path = Path(test_case['path'])
        if not category_path.exists():
            print(f"❌ Category path not found: {category_path}")
            continue
        
        # Get first image file
        image_files = list(category_path.glob('*.png')) + list(category_path.glob('*.jpg'))
        if not image_files:
            print(f"❌ No images found in {category_path}")
            continue
        
        test_image = str(image_files[0])
        print(f"   Input: {Path(test_image).name}")
        
        # Get recommendations
        try:
            result = engine.get_recommendations(
                image_path=test_image,
                items_per_category=2,
                user_gender='unisex'  # Test with unisex to see all options
            )
            
            if 'error' in result:
                print(f"❌ Recommendation failed: {result['error']}")
                continue
            
            query_item = result['query_item']
            recommendations = result['recommendations']
            processing_info = result['processing_info']
            
            print(f"   Query item: {query_item['category']} ({query_item['dominant_color']})")
            print(f"   Input excluded: {processing_info['input_excluded']}")
            print(f"   Total recommendations: {len(recommendations)}")
            print(f"   Categories found: {processing_info['categories_found']}")
            
            # Check that input item is not in recommendations
            input_id = query_item['id']
            self_matches = [r for r in recommendations if r['id'] == input_id]
            if self_matches:
                print(f"❌ FAILED: Input item found in recommendations!")
            else:
                print(f"✅ PASSED: Input item correctly excluded")
            
            # Check category diversity
            categories_found = set(r['category'] for r in recommendations)
            input_category = query_item['category']
            same_category_recs = [r for r in recommendations if r['category'] == input_category]
            
            if same_category_recs:
                print(f"❌ WARNING: {len(same_category_recs)} same-category recommendations found")
            else:
                print(f"✅ PASSED: No same-category recommendations")
            
            # Check items per category
            category_counts = {}
            for rec in recommendations:
                cat = rec['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            max_per_category = max(category_counts.values()) if category_counts else 0
            if max_per_category <= 2:
                print(f"✅ PASSED: Max {max_per_category} items per category (≤2)")
            else:
                print(f"❌ WARNING: {max_per_category} items in some category (>2)")
            
            # Show top recommendations
            print(f"   Top recommendations:")
            for j, rec in enumerate(recommendations[:6], 1):
                print(f"     {j}. {rec['category']} - {rec['color']} "
                      f"(score: {rec['score']:.3f}) - {rec['match_reason']}")
            
            # Check color harmony
            color_scores = [r['score_components']['color_harmony'] for r in recommendations]
            avg_color_score = sum(color_scores) / len(color_scores) if color_scores else 0
            print(f"   Average color harmony: {avg_color_score:.3f}")
            
            if avg_color_score >= 0.6:
                print(f"✅ PASSED: Good color harmony (≥0.6)")
            else:
                print(f"❌ WARNING: Low color harmony (<0.6)")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test gender filtering
    print(f"\n🚻 Testing Gender Filtering")
    print("-" * 30)
    
    # Find a dress (female item)
    dress_path = Path('outfits_dataset/train/one-piece_dress')
    if dress_path.exists():
        dress_files = list(dress_path.glob('*.png'))
        if dress_files:
            dress_image = str(dress_files[0])
            
            # Test with male user (should not get female-specific items)
            result_male = engine.get_recommendations(
                image_path=dress_image,
                user_gender='male',
                items_per_category=2
            )
            
            if 'error' not in result_male:
                male_recs = result_male['recommendations']
                female_items = [r for r in male_recs if r['gender'] == 'female']
                
                if female_items:
                    print(f"❌ FAILED: Male user got {len(female_items)} female items")
                else:
                    print(f"✅ PASSED: Male user got no female items")
                    
                unisex_items = [r for r in male_recs if r['gender'] == 'unisex']
                print(f"   Male user got {len(unisex_items)} unisex items (good)")
    
    print(f"\n🎉 Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_recommendation_engine()