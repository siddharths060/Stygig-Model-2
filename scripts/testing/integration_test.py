#!/usr/bin/env python3
"""
Fashion Recommendation Local Integration Test

This script tests the fashion recommendation engine locally without SageMaker.
It demonstrates the key improvements in the recommendation system.

Usage:
    python scripts/testing/integration_test.py <image_path> [OPTIONS]

Examples:
    python scripts/testing/integration_test.py outfits_dataset/train/upperwear_tshirt/sample.png --gender male
    python scripts/testing/integration_test.py path/to/image.jpg --items-per-category 3
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from stygig.core.recommendation_engine import FashionEngine


def main():
    parser = argparse.ArgumentParser(
        description='Fashion Recommendation Local Integration Test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image_path', help='Path to input fashion image')
    parser.add_argument(
        '--gender',
        choices=['male', 'female', 'unisex'], 
        help='User gender for filtering (if not provided, inferred from input)'
    )
    parser.add_argument(
        '--items-per-category',
        type=int,
        default=2,
        help='Number of items to return per category (default: 2)'
    )
    parser.add_argument(
        '--dataset',
        default='outfits_dataset',
        help='Path to fashion dataset (default: outfits_dataset)'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate input image
    if not Path(args.image_path).exists():
        print(f"❌ Error: Image not found at {args.image_path}")
        return 1
    
    # Validate dataset
    dataset_path = project_root / args.dataset
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return 1
    
    print("🎨 Professional Fashion Recommendation Engine - Local Test")
    print("=" * 80)
    print(f"Input: {Path(args.image_path).name}")
    print(f"User gender: {args.gender or 'auto-detect'}")
    print(f"Items per category: {args.items_per_category}")
    print()
    
    # Initialize engine
    print("🏗️ Initializing recommendation engine...")
    engine = FashionEngine(
        dataset_path=str(dataset_path),
        items_per_category=args.items_per_category,
        color_weight=0.45,    # 45% weight on color harmony
        category_weight=0.25, # 25% weight on category compatibility  
        gender_weight=0.30    # 30% weight on gender compatibility
    )
    
    # Build index
    print("📚 Building fashion index...")
    start_time = time.time()
    try:
        engine.build_index()
        build_time = time.time() - start_time
        print(f"✅ Index built in {build_time:.2f} seconds")
        
        # Show dataset stats
        stats = engine.get_statistics()
        print(f"   📊 Dataset: {stats['total_items']} items across {len(stats['categories'])} categories")
        
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get recommendations
    print("🔍 Generating recommendations...")
    start_time = time.time()
    
    try:
        result = engine.get_recommendations(
            image_path=args.image_path,
            user_gender=args.gender,
            items_per_category=args.items_per_category
        )
        
        rec_time = time.time() - start_time
        print(f"✅ Recommendations generated in {rec_time:.3f} seconds")
        
    except Exception as e:
        print(f"❌ Recommendation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check for errors
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return 1
    
    # Display results
    query_item = result['query_item']
    recommendations = result['recommendations']
    processing_info = result['processing_info']
    
    print("\n📋 Query Item Analysis:")
    print(f"   Category: {query_item['category']}")
    print(f"   Color: {query_item['dominant_color']}")
    print(f"   Gender: {query_item['gender']}")
    print(f"   ID: {query_item['id']}")
    
    print(f"\n🎯 Recommendation Results:")
    print(f"   Total recommendations: {len(recommendations)}")
    print(f"   Categories found: {processing_info['categories_found']}")
    print(f"   Input item excluded: {'✅ Yes' if processing_info['input_excluded'] else '❌ No'}")
    
    if not recommendations:
        print("   ⚠️ No recommendations found!")
        return 0
    
    # Group recommendations by category
    by_category = {}
    for rec in recommendations:
        cat = rec['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(rec)
    
    print(f"\n👗 Recommendations by Category:")
    for category, items in by_category.items():
        print(f"\n   {category.replace('_', ' ').title()} ({len(items)} items):")
        
        for i, item in enumerate(items, 1):
            score = item['score']
            color_score = item['score_components']['color_harmony']
            gender_score = item['score_components']['gender_compatibility']
            
            print(f"     {i}. {item['id']}")
            print(f"        Color: {item['color']} (harmony: {color_score:.3f})")
            print(f"        Gender: {item['gender']} (compat: {gender_score:.3f})")
            print(f"        Overall score: {score:.3f}")
            print(f"        Reason: {item['match_reason']}")
    
    # Show key improvements
    print(f"\n✨ Key Improvements Demonstrated:")
    
    # Check self-exclusion
    input_id = query_item['id']
    self_matches = [r for r in recommendations if r['id'] == input_id]
    if self_matches:
        print(f"   ❌ ISSUE: Input item found in recommendations")
    else:
        print(f"   ✅ Input item correctly excluded from recommendations")
    
    # Check category diversity
    input_category = query_item['category']
    same_category = [r for r in recommendations if r['category'] == input_category]
    if same_category:
        print(f"   ⚠️ WARNING: {len(same_category)} same-category recommendations")
    else:
        print(f"   ✅ No same-category recommendations (category diversity)")
    
    # Check color harmony
    color_scores = [r['score_components']['color_harmony'] for r in recommendations]
    avg_color_harmony = sum(color_scores) / len(color_scores)
    if avg_color_harmony >= 0.7:
        print(f"   ✅ Excellent color harmony (avg: {avg_color_harmony:.3f})")
    elif avg_color_harmony >= 0.5:
        print(f"   ✅ Good color harmony (avg: {avg_color_harmony:.3f})")
    else:
        print(f"   ⚠️ Fair color harmony (avg: {avg_color_harmony:.3f})")
    
    # Check gender filtering
    user_gender = query_item['gender']
    incompatible_genders = []
    for rec in recommendations:
        if user_gender == 'male' and rec['gender'] == 'female':
            incompatible_genders.append(rec)
        elif user_gender == 'female' and rec['gender'] == 'male':
            incompatible_genders.append(rec)
    
    if incompatible_genders:
        print(f"   ❌ ISSUE: {len(incompatible_genders)} gender-incompatible items")
    else:
        print(f"   ✅ All recommendations are gender-compatible")
    
    # Check items per category
    category_counts = {}
    for rec in recommendations:
        cat = rec['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    max_per_cat = max(category_counts.values()) if category_counts else 0
    if max_per_cat <= args.items_per_category:
        print(f"   ✅ Proper category distribution (max {max_per_cat} per category)")
    else:
        print(f"   ⚠️ Some categories exceed limit ({max_per_cat} > {args.items_per_category})")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n🎉 Recommendation test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
