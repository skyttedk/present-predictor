"""
Shop Feature Resolver for ML Predictions

This module handles shop-level feature resolution with fallback strategies
for new shops without historical data, using branch codes to find similar shops.

IMPORTANT: During prediction, we don't have access to real shop_id values since
shop_id was just a bundling/batching parameter in the training dataset. The
resolver correctly handles None shop_id and falls back to branch-based resolution.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import os
import pickle

logger = logging.getLogger(__name__)

class ShopFeatureResolver:
    """
    Resolves shop-level features using pre-computed aggregates from the training data.
    
    Fallback strategy:
    1. Direct lookup for existing shop from snapshot.
    2. Average features from similar shops (same branch code) from snapshot.
    3. Global average defaults from snapshot.
    """
    
    def __init__(self, model_dir: str):
        self.shop_data: Dict[str, Dict] = {}
        self.global_defaults: Dict[str, any] = {}
        
        self.model_dir = model_dir
        self._load_shop_aggregates()

    def _load_shop_aggregates(self):
        """Load shop aggregates from the model directory."""
        # Load historical_features_snapshot (now shop_data)
        hist_features_path = os.path.join(self.model_dir, 'historical_features_snapshot.pkl')
        if os.path.exists(hist_features_path):
            try:
                with open(hist_features_path, 'rb') as f:
                    self.shop_data = pickle.load(f)
                logger.info(f"Loaded shop data for {len(self.shop_data)} shops.")
            except Exception as e:
                logger.error(f"Failed to load shop data: {e}")
                self.shop_data = {}
        else:
            logger.warning(f"Shop data not found in {self.model_dir}.")
            self.shop_data = {}

        # Load global defaults
        global_defaults_path = os.path.join(self.model_dir, 'global_defaults_snapshot.pkl')
        if os.path.exists(global_defaults_path):
            try:
                with open(global_defaults_path, 'rb') as f:
                    self.global_defaults = pickle.load(f)
                logger.info(f"Loaded global defaults.")
            except Exception as e:
                logger.error(f"Failed to load global defaults: {e}")
                self._set_default_features()
        else:
            logger.warning(f"Global defaults not found in {self.model_dir}.")
            self._set_default_features()

    
    def _set_default_features(self):
        """Set hardcoded default features when no historical data available"""
        self.global_defaults = {
            'main_category_diversity': 5,
            'brand_diversity': 8,
            'utility_type_diversity': 3,
            'sub_category_diversity': 6,
            'most_frequent_main_category': 'Home & Kitchen',
            'most_frequent_brand': 'NONE'
        }
        logger.info("Using hardcoded default features")
    
    def resolve_features(self, shop_id: str, main_category: str,
                        sub_category: str, brand: str) -> Dict[str, any]:
        """
        Simplified shop feature resolution - no branch fallback, no product relativity.
        
        Args:
            shop_id: Shop identifier (now equivalent to branch_code)
            main_category: Product main category
            sub_category: Product sub category
            brand: Product brand
            
        Returns:
            Dictionary of shop features only (diversity and most frequent items)
        """
        features = {}
        
        # Only keep diversity and most frequent features
        if shop_id in self.shop_data:
            shop_info = self.shop_data[shop_id]
            features.update({
                'shop_main_category_diversity_selected': shop_info.get('main_category_diversity', 0),
                'shop_sub_category_diversity_selected': shop_info.get('sub_category_diversity', 0),
                'shop_brand_diversity_selected': shop_info.get('brand_diversity', 0),
                'shop_utility_type_diversity_selected': shop_info.get('utility_type_diversity', 0),
                'shop_most_frequent_main_category_selected': shop_info.get('most_frequent_main_category', 'NONE'),
                'shop_most_frequent_brand_selected': shop_info.get('most_frequent_brand', 'NONE'),
            })
            
            # Binary features
            features['is_shop_most_frequent_main_category'] = int(
                main_category == features.get('shop_most_frequent_main_category_selected', '')
            )
            features['is_shop_most_frequent_brand'] = int(
                brand == features.get('shop_most_frequent_brand_selected', '')
            )
        else:
            # Use global defaults if shop not found
            features.update({
                'shop_main_category_diversity_selected': self.global_defaults.get('main_category_diversity', 5),
                'shop_sub_category_diversity_selected': self.global_defaults.get('sub_category_diversity', 6),
                'shop_brand_diversity_selected': self.global_defaults.get('brand_diversity', 8),
                'shop_utility_type_diversity_selected': self.global_defaults.get('utility_type_diversity', 3),
                'shop_most_frequent_main_category_selected': self.global_defaults.get('most_frequent_main_category', 'NONE'),
                'shop_most_frequent_brand_selected': self.global_defaults.get('most_frequent_brand', 'NONE'),
                'is_shop_most_frequent_main_category': 0,
                'is_shop_most_frequent_brand': 0
            })
        
        return features

    
    
    def get_available_shops(self) -> List[str]:
        """Get list of shops with historical data"""
        return list(self.shop_data.keys())