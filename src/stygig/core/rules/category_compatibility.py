"""
Category Compatibility Rules for Fashion Recommendations

This module defines which fashion categories work well together
to create coherent and stylish outfit recommendations.

Key Principles:
1. No same-category recommendations (e.g., no shirt with another shirt)
2. Logical outfit building (shirts pair with pants, not with other tops)
3. Accessories complement main outfit items
"""

# Category compatibility matrix
# Defines which categories are compatible for recommendations
CATEGORY_COMPATIBILITY = {
    # Upperwear recommendations
    'upperwear_shirt': {
        'compatible': [
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt', 
            'footwear_shoes', 'footwear_sneakers', 'footwear_flats', 
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket']
    },
    'upperwear_tshirt': {
        'compatible': [
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'upperwear_jacket',  # Jackets can layer over t-shirts
            'footwear_shoes', 'footwear_sneakers', 'footwear_flats',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['upperwear_shirt', 'upperwear_tshirt']
    },
    'upperwear_jacket': {
        'compatible': [
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'footwear_shoes', 'footwear_sneakers', 'footwear_flats',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['upperwear_jacket']
    },
    
    # Bottomwear recommendations  
    'bottomwear_pants': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'footwear_shoes', 'footwear_sneakers', 'footwear_flats',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt', 'one-piece_dress']
    },
    'bottomwear_shorts': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'footwear_shoes', 'footwear_sneakers', 'footwear_flats',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt', 'one-piece_dress']
    },
    'bottomwear_skirt': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'footwear_shoes', 'footwear_heels', 'footwear_flats',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt', 'one-piece_dress']
    },
    
    # One-piece items (dresses)
    'one-piece_dress': {
        'compatible': [
            'upperwear_jacket',  # Can layer jacket over dress
            'footwear_heels', 'footwear_flats', 'footwear_shoes',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': [
            'upperwear_shirt', 'upperwear_tshirt',
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'one-piece_dress'
        ]
    },
    
    # Footwear recommendations
    'footwear_shoes': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'one-piece_dress', 'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['footwear_shoes', 'footwear_sneakers', 'footwear_heels', 'footwear_flats']
    },
    'footwear_sneakers': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_pants', 'bottomwear_shorts',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['footwear_shoes', 'footwear_sneakers', 'footwear_heels', 'footwear_flats']
    },
    'footwear_heels': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_skirt', 'one-piece_dress',
            'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['footwear_shoes', 'footwear_sneakers', 'footwear_heels', 'footwear_flats']
    },
    'footwear_flats': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'one-piece_dress', 'accessories_bag', 'accessories_hat'
        ],
        'avoid': ['footwear_shoes', 'footwear_sneakers', 'footwear_heels', 'footwear_flats']
    },
    
    # Accessories recommendations
    'accessories_bag': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'one-piece_dress', 'footwear_shoes', 'footwear_sneakers', 
            'footwear_heels', 'footwear_flats'
        ],
        'avoid': ['accessories_bag']
    },
    'accessories_hat': {
        'compatible': [
            'upperwear_shirt', 'upperwear_tshirt', 'upperwear_jacket',
            'bottomwear_pants', 'bottomwear_shorts', 'bottomwear_skirt',
            'one-piece_dress', 'footwear_shoes', 'footwear_sneakers',
            'footwear_heels', 'footwear_flats'
        ],
        'avoid': ['accessories_hat']
    }
}
