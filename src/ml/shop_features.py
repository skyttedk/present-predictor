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
        self.historical_features: Dict[str, Dict] = {}
        self.branch_mapping: Dict[str, List[str]] = {}
        self.global_defaults: Dict[str, any] = {}
        self.product_relativity_lookup: pd.DataFrame = pd.DataFrame()
        
        self.model_dir = model_dir
        self._load_snapshot_aggregates()

    def _load_snapshot_aggregates(self):
        """Loads all pre-computed aggregate snapshots from the model directory."""
        loaded_all = True

        # 1. Load historical_features_snapshot
        hist_features_path = os.path.join(self.model_dir, 'historical_features_snapshot.pkl')
        if os.path.exists(hist_features_path):
            try:
                with open(hist_features_path, 'rb') as f:
                    self.historical_features = pickle.load(f)
                logger.info(f"Loaded historical_features_snapshot.pkl for {len(self.historical_features)} shops.")
            except Exception as e:
                logger.error(f"Failed to load historical_features_snapshot.pkl: {e}")
                self.historical_features = {}
                loaded_all = False
        else:
            logger.warning(f"historical_features_snapshot.pkl not found in {self.model_dir}.")
            self.historical_features = {}
            loaded_all = False

        # 2. Load branch_mapping_snapshot
        branch_map_path = os.path.join(self.model_dir, 'branch_mapping_snapshot.pkl')
        if os.path.exists(branch_map_path):
            try:
                with open(branch_map_path, 'rb') as f:
                    self.branch_mapping = pickle.load(f)
                logger.info(f"Loaded branch_mapping_snapshot.pkl for {len(self.branch_mapping)} branches.")
            except Exception as e:
                logger.error(f"Failed to load branch_mapping_snapshot.pkl: {e}")
                self.branch_mapping = {}
                loaded_all = False
        else:
            logger.warning(f"branch_mapping_snapshot.pkl not found in {self.model_dir}.")
            self.branch_mapping = {}
            loaded_all = False

        # 3. Load global_defaults_snapshot
        global_defaults_path = os.path.join(self.model_dir, 'global_defaults_snapshot.pkl')
        if os.path.exists(global_defaults_path):
            try:
                with open(global_defaults_path, 'rb') as f:
                    self.global_defaults = pickle.load(f)
                logger.info(f"Loaded global_defaults_snapshot.pkl: {self.global_defaults}")
            except Exception as e:
                logger.error(f"Failed to load global_defaults_snapshot.pkl: {e}")
                self.global_defaults = {}
                loaded_all = False
        else:
            logger.warning(f"global_defaults_snapshot.pkl not found in {self.model_dir}.")
            self.global_defaults = {}
            loaded_all = False
            
        # 4. Load product_relativity_features.csv
        self._load_product_relativity_features_from_snapshot() # This method handles its own logging and fallbacks

        if not loaded_all or self.product_relativity_lookup.empty or not self.global_defaults:
            logger.warning("One or more snapshot aggregates failed to load. Falling back to hardcoded defaults for missing parts.")
            self._set_default_features_if_empty()

    def _load_product_relativity_features_from_snapshot(self):
        """Loads the pre-computed product relativity features lookup table from the model directory."""
        lookup_path = os.path.join(self.model_dir, "product_relativity_features.csv")

        if os.path.exists(lookup_path):
            try:
                self.product_relativity_lookup = pd.read_csv(lookup_path, dtype=str)
                # Convert numeric columns back to numeric, coercing errors
                for col in ['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']:
                    if col in self.product_relativity_lookup.columns:
                        self.product_relativity_lookup[col] = pd.to_numeric(self.product_relativity_lookup[col], errors='coerce')
                
                # Fill any NaNs that might have resulted from coercion
                self.product_relativity_lookup.fillna({
                    'product_share_in_shop': 0.0,
                    'brand_share_in_shop': 0.0,
                    'product_rank_in_shop': 99.0,
                    'brand_rank_in_shop': 99.0
                }, inplace=True)

                logger.info(f"Successfully loaded product_relativity_features.csv from snapshot with {len(self.product_relativity_lookup)} rows.")
            except Exception as e:
                logger.error(f"Failed to load or process product_relativity_features.csv from snapshot: {e}")
                self.product_relativity_lookup = pd.DataFrame()
        else:
            logger.warning(f"product_relativity_features.csv not found in {self.model_dir}. Product-specific features will use defaults.")
            self.product_relativity_lookup = pd.DataFrame()
    
    def _set_default_features_if_empty(self):
        """Sets hardcoded default features if any of the loaded snapshots are empty or missing."""
        if not self.global_defaults: # If global_defaults couldn't be loaded, set them.
            self.global_defaults = {
                'shop_main_category_diversity_selected': 5,
                'shop_brand_diversity_selected': 8,
                'shop_utility_type_diversity_selected': 3,
                'shop_sub_category_diversity_selected': 6,
                'shop_most_frequent_main_category_selected': 'Home & Kitchen',
                'shop_most_frequent_brand_selected': 'NONE',
                'unique_product_combinations_in_shop': 45
            }
            logger.info("Using hardcoded global default features as snapshot was missing/empty.")
        
        # Ensure historical_features and branch_mapping are at least empty dicts if loading failed
        if not self.historical_features:
            self.historical_features = {}
        if not self.branch_mapping:
            self.branch_mapping = {}
        if self.product_relativity_lookup.empty:
            # Ensure it's an empty DataFrame with expected columns for downstream logic if it failed to load
            self.product_relativity_lookup = pd.DataFrame(columns=[
                'employee_shop', 'employee_branch', 'product_main_category', 'product_brand',
                'product_share_in_shop', 'brand_share_in_shop',
                'product_rank_in_shop', 'brand_rank_in_shop'
            ])


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
    
    def get_shop_features(self, shop_id: Optional[str], branch_code: str, present_info: Dict) -> Dict[str, any]:
        """
        Get shop features with intelligent fallback, now including product-specific features.
        
        Args:
            shop_id: Shop identifier (e.g., "2960") or None if not available during prediction
            branch_code: Branch/industry code (e.g., "621000")
            present_info: Dictionary of the present's attributes to look up relativity features.
            
        Returns:
            Dictionary of shop and product-specific features.
        """
        logger.debug(f"Resolving features for shop {shop_id}, branch {branch_code}")
        
        # --- Step 1: Get base shop-level features ---
        base_shop_features = {}
        if shop_id and shop_id in self.historical_features:
            logger.debug(f"Found direct features for shop {shop_id}")
            base_shop_features = self.historical_features[shop_id].copy()
        elif branch_code in self.branch_mapping:
            similar_shops = [s for s in self.branch_mapping[branch_code] if s in self.historical_features]
            if similar_shops:
                logger.debug(f"Using features from {len(similar_shops)} similar shops in branch {branch_code}")
                base_shop_features = self._average_shop_features(similar_shops)
            else:
                logger.debug(f"Using global default features for shop {shop_id} as no similar shops found.")
                base_shop_features = self.global_defaults.copy()
        else:
            logger.debug(f"Using global default features for branch {branch_code}")
            base_shop_features = self.global_defaults.copy()

        # --- Step 2: Get product-specific relativity features ---
        product_relativity_features = self._get_product_relativity_features(shop_id, branch_code, present_info)
        
        # --- Step 3: Combine them ---
        final_features = {**base_shop_features, **product_relativity_features}
        
        return final_features

    def _get_product_relativity_features(self, shop_id: Optional[str], branch_code: str, present_info: Dict) -> Dict:
        """Looks up product-specific features from the loaded relativity table with improved fallback."""
        
        default_relativity = {
            'product_share_in_shop': 0.0,
            'brand_share_in_shop': 0.0,
            'product_rank_in_shop': 99.0,
            'brand_rank_in_shop': 99.0
        }

        if self.product_relativity_lookup.empty:
            logger.debug("Product relativity lookup is empty.")
            return default_relativity

        # Strategy 0: Handle "NONE" classification failures with random sampling FIRST
        if present_info.get('item_main_category', 'NONE') == 'NONE' and present_info.get('brand', 'NONE') == 'NONE':
            logger.debug(f"Product has NONE category and brand - using random sampling from popular products")
            
            # Sample from products with decent performance (not bottom quartile)
            popular_products = self.product_relativity_lookup[
                self.product_relativity_lookup['product_share_in_shop'] >
                self.product_relativity_lookup['product_share_in_shop'].quantile(0.25)
            ]
            
            if not popular_products.empty:
                # Take a random sample to add variation
                import random
                sample_size = min(100, len(popular_products))
                sampled_products = popular_products.sample(n=sample_size, random_state=hash(str(present_info.get('id', 'default'))) % 2**31)
                
                sampled_features = sampled_products[['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']].mean().to_dict()
                logger.debug(f"Used random sampling from {sample_size} popular products for NONE classification")
                return sampled_features

        # Strategy 1: Try exact shop + product match (only if shop_id is available)
        if shop_id:
            exact_shop_conditions = (
                (self.product_relativity_lookup['employee_shop'] == shop_id) &
                (self.product_relativity_lookup['product_main_category'] == present_info.get('item_main_category', 'NONE')) &
                (self.product_relativity_lookup['product_brand'] == present_info.get('brand', 'NONE'))
            )
            
            matched_rows = self.product_relativity_lookup[exact_shop_conditions]
            if not matched_rows.empty:
                avg_features = matched_rows[['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']].mean().to_dict()
                logger.debug(f"Found exact shop match: {len(matched_rows)} records for shop {shop_id}")
                return avg_features
        
        # Strategy 2: Try same branch + product match (similar shops)
        branch_conditions = (
            (self.product_relativity_lookup['employee_branch'] == branch_code) &
            (self.product_relativity_lookup['product_main_category'] == present_info.get('item_main_category', 'NONE')) &
            (self.product_relativity_lookup['product_brand'] == present_info.get('brand', 'NONE'))
        )
        
        matched_rows = self.product_relativity_lookup[branch_conditions]
        if not matched_rows.empty:
            avg_features = matched_rows[['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']].mean().to_dict()
            logger.debug(f"Found branch match: {len(matched_rows)} records for branch {branch_code}")
            return avg_features
        
        # Strategy 3: Try just product category + brand across all shops
        product_conditions = (
            (self.product_relativity_lookup['product_main_category'] == present_info.get('item_main_category', 'NONE')) &
            (self.product_relativity_lookup['product_brand'] == present_info.get('brand', 'NONE'))
        )
        
        matched_rows = self.product_relativity_lookup[product_conditions]
        if not matched_rows.empty:
            avg_features = matched_rows[['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']].mean().to_dict()
            logger.debug(f"Found product match: {len(matched_rows)} records across all shops")
            return avg_features
        
        # Strategy 4: Try just brand across all shops
        brand_conditions = (
            self.product_relativity_lookup['product_brand'] == present_info.get('brand', 'NONE')
        )
        
        matched_rows = self.product_relativity_lookup[brand_conditions]
        if not matched_rows.empty:
            # Only use brand-related features, keep product features as default
            brand_features = matched_rows[['brand_share_in_shop', 'brand_rank_in_shop']].mean().to_dict()
            logger.debug(f"Found brand match: {len(matched_rows)} records for brand {present_info.get('brand', 'NONE')}")
            return {
                'product_share_in_shop': 0.0,
                'brand_share_in_shop': brand_features.get('brand_share_in_shop', 0.0),
                'product_rank_in_shop': 99.0,
                'brand_rank_in_shop': brand_features.get('brand_rank_in_shop', 99.0)
            }
        
        # Strategy 5: Try just category across all shops
        category_conditions = (
            self.product_relativity_lookup['product_main_category'] == present_info.get('item_main_category', 'NONE')
        )
        
        matched_rows = self.product_relativity_lookup[category_conditions]
        if not matched_rows.empty:
            # Use average features for this category
            category_features = matched_rows[['product_share_in_shop', 'brand_share_in_shop', 'product_rank_in_shop', 'brand_rank_in_shop']].mean().to_dict()
            logger.debug(f"Found category match: {len(matched_rows)} records for category {present_info.get('item_main_category', 'NONE')}")
            return category_features
        
        logger.debug(f"No specific relativity record found for present. Using defaults.")
        return default_relativity
    
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