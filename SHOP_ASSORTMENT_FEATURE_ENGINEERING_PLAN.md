# Plan: Shop Assortment Feature Engineering

**Objective:** Enhance the existing XGBoost model's predictive power (improve R²) by engineering features that better represent the influence of the available gift assortment within each `employee_shop` on individual gift selection likelihood and quantity.

**Background:**
The current model achieves an R² of ~0.3. The `employee_shop` feature is ranked 4th in importance (0.1120), and `employee_branch` is 3rd (0.1195), indicating shop context is significant. However, the model currently sees `employee_shop` as a simple categorical label. This plan aims to create features that explicitly describe the characteristics of the gift assortment (proxied by selected items, as full historical assortment data is unavailable) within each shop.

**Data Source for EDA & Feature Engineering:**
*   The primary data will be the aggregated dataset (`agg_data`) as generated in the [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1) (lines 118-181), which contains one row per unique 11-feature combination (including all product attributes and employee demographics) along with the `selection_count`.
*   Original historical data: [`src/data/historical/present.selection.historic.csv`](src/data/historical/present.selection.historic.csv)

## Phase 1: Deeper Analysis of `employee_shop` Influence & Assortment Proxy Features

### Step 1.1: Analyze Current `employee_shop` Feature Importance
*   **Status:** COMPLETE.
*   **Finding:** `employee_shop` is the 4th most important feature (0.1120). This confirms its relevance and suggests potential for enhancement by adding more nuanced shop-related features.

### Step 1.2: Exploratory Data Analysis (EDA) Focused on Shop-Level Patterns
*   **Goal:** Identify if distinct "shop profiles" emerge based on selection patterns, using selected items as a proxy for the full assortment.
*   **Data:** Start with `agg_data` from the breakthrough notebook.
*   **Specific Analyses (Conceptual):**
    1.  **Load Cleaned, Aggregated Data:**
        *   Ensure `agg_data` (unique 11-feature combinations + `selection_count`) is available.
    2.  **Calculate Shop-Level Summaries:** For each `employee_shop`, aggregate:
        *   `total_shop_selections`: Sum of `selection_count`.
        *   `unique_product_combinations_in_shop`: Count of unique rows (product combinations) per shop.
        *   `distinct_main_categories_in_shop`: Number of unique `product_main_category` selected.
        *   `distinct_sub_categories_in_shop`: Number of unique `product_sub_category` selected.
        *   `distinct_brands_in_shop`: Number of unique `product_brand` selected.
        *   `distinct_utility_types_in_shop`: Number of unique `product_utility_type` selected.
        *   (Consider adding other distinct counts for relevant product attributes).
    3.  **Determine Most Frequent Items per Shop (based on `selection_count` sum):**
        *   For `product_main_category`: Identify the `product_main_category` with the highest total `selection_count` within each `employee_shop`. Handle ties if necessary (e.g., take the first).
        *   Repeat for `product_brand`.
    4.  **Visualizations:**
        *   Histograms or boxplots for shop-level summary statistics (e.g., `total_shop_selections`, `distinct_main_categories_in_shop`) to understand variability across shops.
        *   Bar charts showing the distribution of the "most frequent main category" or "most frequent brand" across shops.

### Step 1.3: Engineer Shop-Level Aggregate Features (Assortment Proxies)
*   **Goal:** Create features that describe the overall characteristics of selections within each `employee_shop`. These will be merged back to the main `agg_data`.
*   **Proposed Features:**
    *   `shop_total_selections`: Sum of `selection_count` for all unique product combinations within the shop. (From EDA)
    *   `shop_unique_product_combinations_selected`: Count of unique 11-feature combinations selected in that shop. (From EDA)
    *   `shop_avg_selections_per_product_combination`: `shop_total_selections` / `shop_unique_product_combinations_selected`.
    *   `shop_most_frequent_main_category_selected`: The `product_main_category` with the highest sum of `selection_count` in the shop. (From EDA; requires careful handling for merging, possibly as a categorical feature or one-hot encoded if vocabulary is small).
    *   `shop_most_frequent_brand_selected`: The `product_brand` with the highest sum of `selection_count` in the shop. (From EDA; similar handling as above).
    *   `shop_main_category_diversity_selected`: Number of unique `product_main_category` values with `selection_count > 0` in the shop. (From EDA's `distinct_main_categories_in_shop`).
    *   `shop_brand_diversity_selected`: Number of unique `product_brand` values with `selection_count > 0` in the shop. (From EDA's `distinct_brands_in_shop`).
    *   `shop_utility_type_diversity_selected`: Number of unique `product_utility_type` values with `selection_count > 0` in the shop. (From EDA's `distinct_utility_types_in_shop`).

### Step 1.4: Engineer Product-Relative-to-Shop-Proxy Features
*   **Goal:** For each unique product combination, create features that describe its standing or characteristics relative to other combinations *selected within the same `employee_shop`*.
*   **Proposed Features:**
    *   `is_shop_most_frequent_main_category`: Boolean. Does this product combination's `product_main_category` match the `shop_most_frequent_main_category_selected` for its shop?
    *   `is_shop_most_frequent_brand`: Boolean. Does this product combination's `product_brand` match the `shop_most_frequent_brand_selected` for its shop?
    *   `selection_count_rank_in_shop`: Rank of this product combination's `selection_count` within its `employee_shop` (e.g., using `groupby('employee_shop')['selection_count'].rank(method='dense', ascending=False)`).
    *   `selection_count_share_in_shop`: This product combination's `selection_count` / `shop_total_selections` for its `employee_shop`.
    *   `selection_count_vs_shop_avg`: This product combination's `selection_count` - `shop_avg_selections_per_product_combination` for its `employee_shop`.
    *   `product_main_category_selection_share_in_shop`: Sum of `selection_count` for this product's `product_main_category` in this shop / `shop_total_selections`.
    *   `product_brand_selection_share_in_shop`: Sum of `selection_count` for this product's `product_brand` in this shop / `shop_total_selections`.

## Phase 2: Model Retraining and Evaluation with New Features

*   **Step 2.1: Prepare Data and Retrain XGBoost Model.**
    *   Add the newly engineered features from Step 1.3 and 1.4 to the existing 11 features.
    *   Ensure all categorical features (including new ones like `shop_most_frequent_main_category_selected`) are appropriately encoded (e.g., LabelEncoding, or OneHotEncoding if cardinality is low and it proves beneficial).
    *   Retrain the XGBoost model using the same optimal configuration (hyperparameters, log-transformed target `y_log`) and cross-validation strategy (Stratified K-Fold by `y_strata` selection count bins) as established in [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1).
*   **Step 2.2: Evaluate Model Performance.**
    *   Compare the R² (and other relevant metrics like MAE, RMSE) of the new model with the baseline R² of ~0.3.
    *   Analyze the feature importances of all features, paying close attention to the newly engineered ones.
*   **Step 2.3: Iteration and Refinement.**
    *   Based on performance and feature importances, refine the engineered features or explore variations.

## Phase 3: Advanced Approaches (If R² Improvement is Still Modest)

*   **Step 3.1: Interaction Features.**
    *   Explicitly create interaction features between new shop-level proxy features and key product attributes (e.g., `shop_main_category_diversity_selected`_X_`product_main_category`).
*   **Step 3.2: Target Encoding for `employee_shop` (with caution).**
    *   Encode `employee_shop` based on the average `selection_count` (or log `selection_count`) observed for that shop, implemented carefully within a cross-validation framework to prevent target leakage.
*   **Step 3.3: Consider Two-Stage Modeling (More Complex).**
    *   Explore modeling (1) choice probability and (2) quantity given choice.
*   **Step 3.4: Explore "Popularity" or "Visibility" Proxies.**
    *   `product_global_selection_frequency`.
    *   `product_shop_lift` (relative performance in shop vs. global).

**Mermaid Diagram of Proposed Workflow Enhancement:**
```mermaid
graph TD
    A[Historical Data: present.selection.historic.csv] --> B{EDA & Shop-Level Analysis (Phase 1.2)};
    B --> C{Feature Engineering: Shop Aggregate & Relative Proxies (Phase 1.3, 1.4)};
    C --> D[Engineered Shop Assortment Features];
    A --> E[Original Product & Employee Features];
    D --> F[Combined Feature Set];
    E --> F;
    F --> G{XGBoost Model Training (Phase 2.1)};
    G --> H[Model Evaluation R² (Phase 2.2)];
    H --> I{Performance Improved Significantly?};
    I -- Yes --> J[Deploy/Further Refine (Phase 2.3)];
    I -- No/Modest --> K{Advanced Approaches (Phase 3)};
    K --> G;
```

**Next Steps after this plan is documented:**
1.  User approval of this documented plan.
2.  Switch to "Code" mode to implement Phase 1 (EDA and Feature Engineering).