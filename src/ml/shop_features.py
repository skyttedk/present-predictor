"""
Shop Feature Resolver for ML Predictions

This module handles shop-level feature resolution with fallback strategies
for new shops without historical data, using branch codes to find similar shops.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class ShopFeatureResolver:
    """
    Resolves shop-level features using historical data or similar shop proxies.
    
    Fallback strategy:
    1. Direct lookup for existing shop
    2. Average features from similar shops (same branch code)
    3. Global average defaults
    """
    
    def __init__(self, historical_data_path: Optional[str] = None):
        self.historical_features = {}
        self.branch_mapping = {}
        self.global_defaults = {}
        
        if historical_data_path and os.path.exists(historical_data_path):
            self._load_historical_features(historical_data_path)
        else:
            logger.warning(f"Historical data not found at {historical_data_path}, using defaults")
            self._set_default_features()
    
    def _load_historical_features(self, data_path: str):
        """Load and compute shop features from historical selection data"""
        try:
            logger.info(f"Loading historical data from {data_path}")
            df = pd.read_csv(data_path, dtype=str)
            
            # Clean data
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.strip('"')
            
            df = df.fillna("NONE")
            
            # Normalize categorical columns
            categorical_cols_to_lower = [
                'employee_gender', 'product_target_gender',
                'product_utility_type', 'product_durability', 'product_type'
            ]
            for col in categorical_cols_to_lower:
                if col in df.columns:
                    df[col] = df[col].str.lower()
            
            # Compute shop diversity features
            shop_stats = df.groupby('employee_shop').agg({
                'product_main_category': 'nunique',
                'product_sub_category': 'nunique', 
                'product_brand': 'nunique',
                'product_utility_type': 'nunique'
            }).reset_index()
            
            shop_stats.columns = [
                'employee_shop',
                'shop_main_category_diversity_selected',
                'shop_sub_category_diversity_selected', 
                'shop_brand_diversity_selected',
                'shop_utility_type_diversity_selected'
            ]
            
            # Get most frequent categories and brands per shop
            shop_main_cats = df.groupby(['employee_shop', 'product_main_category']).size().reset_index(name='count')
            shop_main_cats = shop_main_cats.loc[shop_main_cats.groupby('employee_shop')['count'].idxmax()]
            
            shop_brands = df.groupby(['employee_shop', 'product_brand']).size().reset_index(name='count')
            shop_brands = shop_brands.loc[shop_brands.groupby('employee_shop')['count'].idxmax()]
            
            # Count unique product combinations per shop
            shop_combinations = df.groupby('employee_shop').size().reset_index(name='unique_product_combinations_in_shop')
            
            # Store features by shop
            for _, row in shop_stats.iterrows():
                shop_id = row['employee_shop']
                
                # Get most frequent items for this shop
                most_freq_cat = shop_main_cats[shop_main_cats['employee_shop'] == shop_id]['product_main_category'].iloc[0] if len(shop_main_cats[shop_main_cats['employee_shop'] == shop_id]) > 0 else 'NONE'
                most_freq_brand = shop_brands[shop_brands['employee_shop'] == shop_id]['product_brand'].iloc[0] if len(shop_brands[shop_brands['employee_shop'] == shop_id]) > 0 else 'NONE'
                unique_combinations = shop_combinations[shop_combinations['employee_shop'] == shop_id]['unique_product_combinations_in_shop'].iloc[0] if len(shop_combinations[shop_combinations['employee_shop'] == shop_id]) > 0 else 1
                
                self.historical_features[shop_id] = {
                    'shop_main_category_diversity_selected': int(row['shop_main_category_diversity_selected']),
                    'shop_brand_diversity_selected': int(row['shop_brand_diversity_selected']),
                    'shop_utility_type_diversity_selected': int(row['shop_utility_type_diversity_selected']),
                    'shop_sub_category_diversity_selected': int(row['shop_sub_category_diversity_selected']),
                    'shop_most_frequent_main_category_selected': most_freq_cat,
                    'shop_most_frequent_brand_selected': most_freq_brand,
                    'unique_product_combinations_in_shop': int(unique_combinations)
                }
            
            # Build branch to shop mapping
            branch_shop_map = df.groupby('employee_branch')['employee_shop'].unique().to_dict()
            for branch, shops in branch_shop_map.items():
                self.branch_mapping[branch] = list(shops)
            
            # Compute global defaults
            self._compute_global_defaults()
                
            logger.info(f"Loaded features for {len(self.historical_features)} shops")
            logger.info(f"Built branch mapping for {len(self.branch_mapping)} branches")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self._set_default_features()
    
    def _compute_global_defaults(self):
        """Compute global average features as ultimate fallback"""
        if not self.historical_features:
            self._set_default_features()
            return
        
        # Calculate averages for numeric features
        numeric_features = [
            'shop_main_category_diversity_selected',
            'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected',
            'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop'
        ]
        
        for feature in numeric_features:
            values = [shop_data[feature] for shop_data in self.historical_features.values()]
            self.global_defaults[feature] = int(np.mean(values)) if values else 5
        
        # Most common categorical features
        main_cats = [shop_data['shop_most_frequent_main_category_selected'] for shop_data in self.historical_features.values()]
        brands = [shop_data['shop_most_frequent_brand_selected'] for shop_data in self.historical_features.values()]
        
        self.global_defaults['shop_most_frequent_main_category_selected'] = max(set(main_cats), key=main_cats.count) if main_cats else 'Home & Kitchen'
        self.global_defaults['shop_most_frequent_brand_selected'] = max(set(brands), key=brands.count) if brands else 'NONE'
        
        logger.info(f"Computed global defaults: {self.global_defaults}")
    
    def _set_default_features(self):
        """Set hardcoded default features when no historical data available"""
        self.global_defaults = {
            'shop_main_category_diversity_selected': 5,
            'shop_brand_diversity_selected': 8,
            'shop_utility_type_diversity_selected': 3,
            'shop_sub_category_diversity_selected': 6,
            'shop_most_frequent_main_category_selected': 'Home & Kitchen',
            'shop_most_frequent_brand_selected': 'NONE',
            'unique_product_combinations_in_shop': 45
        }
        logger.info("Using hardcoded default features")
    
    def get_shop_features(self, shop_id: str, branch_code: str) -> Dict[str, any]:
        """
        Get shop features with intelligent fallback strategy.
        
        Args:
            shop_id: Shop identifier (e.g., "2960")
            branch_code: Branch/industry code (e.g., "621000")
            
        Returns:
            Dictionary of shop features
        """
        logger.debug(f"Resolving features for shop {shop_id}, branch {branch_code}")
        
        # Strategy 1: Direct lookup for existing shop
        if shop_id in self.historical_features:
            logger.debug(f"Found direct features for shop {shop_id}")
            return self.historical_features[shop_id].copy()
        
        # Strategy 2: Average features from similar shops (same branch)
        if branch_code in self.branch_mapping:
            similar_shops = [s for s in self.branch_mapping[branch_code] if s in self.historical_features]
            
            if similar_shops:
                logger.debug(f"Using features from {len(similar_shops)} similar shops in branch {branch_code}")
                return self._average_shop_features(similar_shops)
        
        # Strategy 3: Global defaults
        logger.debug(f"Using global default features for shop {shop_id}")
        return self.global_defaults.copy()
    
    def _average_shop_features(self, shop_ids: List[str]) -> Dict[str, any]:
        """Average features from multiple similar shops"""
        
        # Average numeric features
        numeric_features = [
            'shop_main_category_diversity_selected',
            'shop_brand_diversity_selected',
            'shop_utility_type_diversity_selected',
            'shop_sub_category_diversity_selected',
            'unique_product_combinations_in_shop'
        ]
        
        averaged_features = {}
        for feature in numeric_features:
            values = [self.historical_features[shop_id][feature] for shop_id in shop_ids]
            averaged_features[feature] = int(np.mean(values))
        
        # Most common categorical features
        main_cats = [self.historical_features[shop_id]['shop_most_frequent_main_category_selected'] for shop_id in shop_ids]
        brands = [self.historical_features[shop_id]['shop_most_frequent_brand_selected'] for shop_id in shop_ids]
        
        averaged_features['shop_most_frequent_main_category_selected'] = max(set(main_cats), key=main_cats.count)
        averaged_features['shop_most_frequent_brand_selected'] = max(set(brands), key=brands.count)
        
        return averaged_features
    
    def get_available_shops(self) -> List[str]:
        """Get list of shops with historical data"""
        return list(self.historical_features.keys())
    
    def get_available_branches(self) -> List[str]:
        """Get list of branches with shop mappings"""
        return list(self.branch_mapping.keys())
    
    def get_shop_count_by_branch(self, branch_code: str) -> int:
        """Get number of shops in a branch"""
        return len(self.branch_mapping.get(branch_code, []))