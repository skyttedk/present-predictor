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
        # Product relativity snapshot (share/rank numeric features)
        # Keyed by (shop_id, main_category, brand)
        self.product_relativity: Dict[tuple, Dict[str, float]] = {}
        
        self.model_dir = model_dir
        self._load_shop_aggregates()
        self._load_product_relativity()

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

    def _load_product_relativity(self):
        """
        Load product relativity snapshot containing share/rank numeric features:
        product_share_in_shop, brand_share_in_shop, product_rank_in_shop, brand_rank_in_shop
        The CSV is produced by the training pipeline and stored beside the model.
        """
        relativity_path = os.path.join(self.model_dir, "product_relativity_features.csv")
        if not os.path.exists(relativity_path):
            logger.warning(f"Product relativity snapshot not found at {relativity_path}. "
                           "Numeric share/rank features will fallback to defaults.")
            self.product_relativity = {}
            return

        try:
            df = pd.read_csv(relativity_path)
            required_cols = {
                "employee_shop",
                "product_main_category",
                "product_brand",
                "product_share_in_shop",
                "brand_share_in_shop",
                "product_rank_in_shop",
                "brand_rank_in_shop",
            }
            missing = required_cols - set(df.columns)
            if missing:
                logger.error(
                    f"Product relativity CSV missing columns {missing}. "
                    "Share/rank features will fallback to defaults."
                )
                self.product_relativity = {}
                return

            # Build fast lookup dictionary
            self.product_relativity = {
                (str(row["employee_shop"]), str(row["product_main_category"]), str(row["product_brand"])): {
                    "product_share_in_shop": float(row["product_share_in_shop"]),
                    "brand_share_in_shop": float(row["brand_share_in_shop"]),
                    "product_rank_in_shop": float(row["product_rank_in_shop"]),
                    "brand_rank_in_shop": float(row["brand_rank_in_shop"]),
                }
                for _, row in df.iterrows()
            }
            logger.info(
                f"Loaded product relativity snapshot with {len(self.product_relativity)} rows."
            )
        except Exception as e:
            logger.error(f"Failed to load product relativity snapshot: {e}")
            self.product_relativity = {}
    
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
        # C2 Fix: Handle both key formats (trainer saves with 'shop_' prefix)
        if shop_id in self.shop_data:
            shop_info = self.shop_data[shop_id]
            features.update({
                'shop_main_category_diversity_selected':
                    shop_info.get('shop_main_category_diversity_selected') or
                    shop_info.get('main_category_diversity', 0),
                'shop_sub_category_diversity_selected':
                    shop_info.get('shop_sub_category_diversity_selected') or
                    shop_info.get('sub_category_diversity', 0),
                'shop_brand_diversity_selected':
                    shop_info.get('shop_brand_diversity_selected') or
                    shop_info.get('brand_diversity', 0),
                'shop_utility_type_diversity_selected':
                    shop_info.get('shop_utility_type_diversity_selected') or
                    shop_info.get('utility_type_diversity', 0),
                'shop_most_frequent_main_category_selected':
                    shop_info.get('shop_most_frequent_main_category_selected') or
                    shop_info.get('most_frequent_main_category', 'NONE'),
                'shop_most_frequent_brand_selected':
                    shop_info.get('shop_most_frequent_brand_selected') or
                    shop_info.get('most_frequent_brand', 'NONE'),
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
        
        # ------------------------------------------------------------------
        # Inject product relativity numeric features (share / rank)
        # ------------------------------------------------------------------
        rel_key = (str(shop_id), str(main_category), str(brand))
        rel_data = self.product_relativity.get(rel_key, {})

        features['product_share_in_shop'] = rel_data.get('product_share_in_shop', 0.0)
        features['brand_share_in_shop'] = rel_data.get('brand_share_in_shop', 0.0)
        # Use 99.0 as default high rank (worst) when missing
        features['product_rank_in_shop'] = rel_data.get('product_rank_in_shop', 99.0)
        features['brand_rank_in_shop'] = rel_data.get('brand_rank_in_shop', 99.0)

        return features

    
    
    def get_available_shops(self) -> List[str]:
        """Get list of shops with historical data"""
        return list(self.shop_data.keys())