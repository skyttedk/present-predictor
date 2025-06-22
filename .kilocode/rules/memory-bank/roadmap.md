# Implementation Roadmap

## Overview
This roadmap outlines the step-by-step implementation plan for the Predictive Gift Selection System, organized into logical phases with clear milestones and deliverables.

## Phase 1: Foundation Setup (Week 1)

### 1.1 Project Structure
**Timeline**: 1-2 days
**Deliverables**:
- [x] Create Python project directory structure
- [ ] Set up virtual environment (conda/venv)
- [ ] Initialize `requirements.txt` with core dependencies
- [ ] Create `requirements-dev.txt` for development tools
- [x] Set up basic `README.md` with project description

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ Full directory structure created (`src/`, `tests/` with all modules)
- ✅ All `__init__.py` files with proper documentation
- ✅ Comprehensive `README.md` with project overview, business context, and setup instructions

**Key Files to Create**:
```
src/
├── __init__.py
├── api/
│   └── __init__.py
├── data/
│   └── __init__.py
├── ml/
│   └── __init__.py
├── config/
│   └── __init__.py
└── utils/
    └── __init__.py
tests/
├── __init__.py
├── test_api/
├── test_data/
└── test_ml/
```

### 1.2 Development Environment
**Timeline**: 1 day
**Deliverables**:
- [x] Configure code formatting (Black)
- [x] Set up linting (Flake8)
- [x] Configure type checking (MyPy)
- [x] Initialize testing framework (Pytest)
- [x] Create development scripts

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`requirements.txt`](requirements.txt:1) - Core dependencies installed
- ✅ [`requirements-dev.txt`](requirements-dev.txt:1) - Development dependencies installed
- ✅ [`pyproject.toml`](pyproject.toml:1) - Tool configuration with Black, Flake8, MyPy, Pytest
- ✅ [`pytest.ini`](pytest.ini:1) - Test configuration with coverage
- ✅ [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1) - Git hooks configuration

**Configuration Files**:
- `pyproject.toml` - Tool configuration
- `.pre-commit-config.yaml` - Git hooks
- `pytest.ini` - Test configuration

### 1.3 Basic Configuration System
**Timeline**: 1 day
**Deliverables**:
- [x] Create `src/config/settings.py` - Application settings
- [x] Create `src/config/model_config.py` - ML model parameters
- [x] Set up environment variable handling
- [x] Create configuration validation

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`src/config/settings.py`](src/config/settings.py:1) - Comprehensive application settings with Pydantic
- ✅ [`src/config/model_config.py`](src/config/model_config.py:1) - XGBoost and ML pipeline configuration
- ✅ [`src/config/validation.py`](src/config/validation.py:1) - Configuration validation system
- ✅ [`.env.example`](.env.example:1) - Environment variable template
- ✅ Environment variable handling with pydantic-settings
- ✅ Configuration validation passes all checks

## Phase 2: Data Pipeline Development (Week 2-3)

### 2.1 Data Schemas and Models
**Timeline**: 2 days
**Deliverables**:
- [x] `src/data/schemas/data_models.py` - Core data structures
- [x] `src/api/schemas/requests.py` - API request models
- [x] `src/api/schemas/responses.py` - API response models
- [x] Pydantic models for all data structures

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`src/data/schemas/data_models.py`](src/data/schemas/data_models.py:1) - Complete data models with enums and validation
- ✅ [`src/api/schemas/requests.py`](src/api/schemas/requests.py:1) - API request models with validation
- ✅ [`src/api/schemas/responses.py`](src/api/schemas/responses.py:1) - API response models with examples
- ✅ Real data structure integration based on actual CSV and JSON schema files
- ✅ Field mapping between API schema and historical data format
- ✅ Comprehensive validation and error handling models

**Key Models**:
```python
# API Request Models
class GiftItem(BaseModel):
    product_id: str
    description: str

class Employee(BaseModel):
    name: str

class PredictionRequest(BaseModel):
    branch_no: str
    gifts: List[GiftItem]
    employees: List[Employee]

# Internal Data Models
class ClassifiedGift(BaseModel):
    item_main_category: str
    item_sub_category: str
    color: str
    brand: str
    target_demographics: str
    utility_type: str
    durability: str
    usage_type: str

class ProcessedEmployee(BaseModel):
    gender: str
```

### 2.2 Data Classification Components
**Timeline**: 3 days
**Deliverables**:
- [x] `src/data/classifier.py` - Gift categorization logic
- [x] OpenAI Assistant API integration for gift classification
- [x] Enhanced gender_guesser with Danish name support
- [x] Category mapping and validation
- [x] Unit tests for classification accuracy

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`src/data/openai_client.py`](src/data/openai_client.py:1) - OpenAI Assistant API integration
- ✅ [`src/data/gender_classifier.py`](src/data/gender_classifier.py:1) - Enhanced gender classification with Danish names
- ✅ [`src/data/classifier.py`](src/data/classifier.py:1) - Main classification orchestrator
- ✅ Complete three-step processing pipeline implementation
- ✅ Batch processing and error handling
- ✅ Classification validation and statistics

**OpenAI Assistant Implementation**:
- Assistant ID: `asst_BuFvA6iXF4xSyQ4px7Q5zjiN`
- API flow: CreateThread → AddMessage → Run → GetRunStatus → GetThreadMessage
- JSON schema validation for product attributes
- Error handling and fallback values

**Enhanced Gender Classification**:
- Enhanced gender_guesser with Danish names dictionary
- Support for compound names (hyphens, spaces)
- Nordic/European name patterns
- Fallback to Denmark country code
- Confidence scoring and uncertainty handling

### 2.3 Data Preprocessing Pipeline
**Timeline**: 3 days
**Deliverables**:
- [x] `src/data/preprocessor.py` - Data cleaning and transformation
- [x] Data aggregation by counting selection events
- [x] Data validation and quality checks
- [x] Historical data loading utilities
- [x] Feature extraction pipeline

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`src/data/preprocessor.py`](src/data/preprocessor.py:1) - Complete preprocessing pipeline
- ✅ Selection event aggregation (10 events → 9 unique combinations)
- ✅ Label encoding for categorical features
- ✅ Data summary and insights generation
- ✅ Integration with historical data structure

**Core Processing Logic**:
```python
# Data aggregation pattern from requirements
final_df = structured_data.groupby([
    'date', 'category', 'product_base', 
    'color', 'type', 'size'
]).agg({'qty_sold': 'sum'}).reset_index()
```

## Phase 3: Machine Learning Pipeline (Week 4-5)

### 3.1 Feature Engineering
**Timeline**: 3 days
**Deliverables**:
- [x] Categorical feature encoding (label encoding)
- [x] Feature importance analysis tools
- [x] Selection event aggregation features
- [x] Employee demographic features

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ Label encoding for all categorical variables
- ✅ Feature engineering integrated in preprocessing pipeline
- ✅ Feature importance analysis via XGBoost
- ✅ Employee demographic and product category features

### 3.2 Model Development
**Timeline**: 4 days
**Deliverables**:
- [x] `src/ml/model.py` - XGBoost model wrapper
- [x] Model training and prediction pipeline
- [x] Model evaluation and metrics
- [x] Model persistence (save/load)
- [x] Feature importance analysis

**Status**: ✅ **COMPLETED** - December 6, 2025
**Completed Items**:
- ✅ [`src/ml/model.py`](src/ml/model.py:1) - Complete XGBoost implementation
- ✅ Model training with train/validation split
- ✅ Cross-validation and metrics (MAE, RMSE, R²)
- ✅ Model persistence with metadata
- ✅ Feature importance and prediction explanations
- ✅ Successfully trained on historical data (9 combinations, 10 events)

**Model Implementation**:
```python
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DemandPredictor:
    def __init__(self):
        self.model = XGBRegressor()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        # Focus on metrics evaluation as per requirements
        predictions = self.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': mean_squared_error(y_test, predictions, squared=False),
            'r2': r2_score(y_test, predictions)
        }
```

### 3.3 Model Training and Validation
**Timeline**: 2 days
**Deliverables**:
- [x] Cross-validation framework
- [x] Model performance benchmarks
- [x] Training data preparation scripts
- [x] Model serialization/loading utilities
- [x] Performance monitoring tools

**Status**: ✅ **COMPLETED** - June 15, 2025 (Reflects baseline model before shop features)
**Critical Findings (Baseline Model R² = 0.2947)**:
- ✅ **Methodology Breakthrough**: Identified and resolved critical cross-validation issues.
- ✅ **Data Utilized**: Successfully processed 178,736 events into 98,741 unique combinations.
- ✅ **Performance Achieved (Baseline)**: Stratified CV R² (log target) = **0.2947** with minimal overfitting (-0.0051).
  - Model: XGBoost with log target transformation.
  - CV: StratifiedKFold by selection_count bins.
- ✅ **Production Viability (Baseline)**: Model demonstrates moderate predictive power, suitable for initial business testing and inventory guidance.
- ✅ **Notebook Reference**: Initial breakthrough documented in optimization scripts and precursor to [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1).

**Business Impact Assessment (Baseline Model R² = 0.2947)**:
- Provides significant improvement over manual estimation.
- Establishes a solid foundation for further enhancements (like shop assortment features).
- Sufficient for initial production integration and testing.
## Phase 3.4: Model Enhancement with Shop Assortment Features (Week 6 - Current)

### 3.4.1 Shop Assortment Feature Engineering
**Timeline**: 2-3 days
**Deliverables**:
- [x] EDA for shop-level selection patterns.
- [x] Engineering of non-leaky shop assortment features:
  - `shop_main_category_diversity_selected`
  - `shop_brand_diversity_selected`
  - `shop_utility_type_diversity_selected`
  - `shop_sub_category_diversity_selected`
  - `shop_most_frequent_main_category_selected`
  - `shop_most_frequent_brand_selected`
  - `is_shop_most_frequent_main_category`
  - `is_shop_most_frequent_brand`
- [x] Integration of new features into the modeling pipeline.

**Status**: ✅ **COMPLETED** - June 15, 2025
**Completed Items**:
- ✅ Shop assortment features successfully engineered and tested as per [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1).
- ✅ Features confirmed to be non-leaky and provide additional predictive signals.

### 3.4.2 Model Retraining and Validation with Shop Features
**Timeline**: 2 days
**Deliverables**:
- [x] Retrain XGBoost model with the original 11 features + new shop assortment features.
- [x] Validate using StratifiedKFold (5 splits) by selection_count bins.
- [x] Evaluate performance on log-transformed target and original scale.
- [x] Update feature importance analysis.

**Status**: ✅ **COMPLETED** - June 15, 2025
**Critical Findings (Enhanced Model R² = 0.3112)**:
- ✅ **Performance Improvement**: Stratified CV R² (log target) increased to **0.3112** (from 0.2947 baseline).
  - Example Overfitting (log target): ~ -0.0062 (Validation R² ~0.3050 - CV R² 0.3112).
- ✅ **Model Configuration**: XGBoost with log target, stratified CV, and combined feature set.
- ✅ **Production Viability (Enhanced)**: Model demonstrates improved moderate predictive power, further enhancing suitability for business testing and inventory guidance.
- ✅ **Notebook Reference**: Enhancements documented in [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1).

**Business Impact Assessment (Enhanced Model R² = 0.3112)**:
- Provides further improvement over the baseline model and manual estimation.
- Increases confidence in using the model for operational inventory planning.

## Phase 4: Re-architecture (Expert-Guided)

### 4.1 Training Pipeline Re-engineering
**Timeline**: 3-4 days
**Priority**: CRITICAL
**Deliverables**:
- [ ] **Modify Data Source**: Update data loading to include `total_employees_in_group` for each historical record.
- [ ] **New Target Variable**: Create `selection_rate` = `selection_count` / `total_employees_in_group`.
- [ ] **Update `catboost_trainer.py`**:
    - Change target to `selection_rate`.
    - Change loss function from `Poisson` to `RMSE` or another standard regression loss.
    - Remove `log1p` transformation.
- [ ] **Retrain Model**: Train the CatBoost model on the new rate-based target.
- [ ] **Validate Performance**: Evaluate the new model's performance on predicting rates.

### 4.2 Prediction Pipeline Correction
**Timeline**: 2 days
**Priority**: CRITICAL
**Deliverables**:
- [ ] **Update `predictor.py`**:
    - Modify `_aggregate_predictions` to correctly scale the predicted rate.
    - New logic: `expected_qty = predicted_rate * num_employees_in_subgroup`.
- [ ] **Update `predict` method**: Ensure the overall logic sums the scaled predictions from subgroups.
- [ ] **Unit & Integration Tests**: Create new tests to validate the rate-based prediction logic.

### 4.3 API and Schema Updates
**Timeline**: 1 day
**Priority**: MEDIUM
**Deliverables**:
- [ ] Review API schemas in `src/api/schemas/` to ensure they align with any changes.
- [ ] Update API documentation if necessary.

## Phase 5: API Development & Integration (Post-Re-architecture)

### 5.1 FastAPI Application Setup
**Timeline**: 2 days
**Deliverables**:
- [ ] `src/api/main.py` - FastAPI application entry point
- [ ] Basic API structure and middleware
- [ ] Request/response logging
- [ ] Error handling middleware
- [ ] CORS configuration

**Status**: ⏳ **IN PROGRESS / NEXT UP**

### 5.2 Prediction Endpoints
**Timeline**: 3 days (after 5.1)
**Deliverables**:
- [ ] `src/api/endpoints/predictions.py` - Prediction endpoints
- [ ] Integration of the re-architected prediction pipeline.
- [ ] Request validation and sanitization
- [ ] API documentation (OpenAPI/Swagger)

### 5.3 API Testing and Documentation
**Timeline**: 2 days
**Deliverables**:
- [ ] Comprehensive API tests for the new logic.
- [ ] Integration tests for the full, corrected pipeline.
- [ ] API documentation improvements.

## Success Criteria and Milestones

### Technical Milestones
- [x] **Week 1**: Project foundation complete ✅
- [x] **Week 3**: Data pipeline functional ✅
- [x] **Week 6**: Initial model with shop features (CV R² ≈ 0.3112) ✅
- [ ] **Week 7-8**: Re-architecture complete (rate-based model) 🚀 **NEXT**
- [ ] **Week 9-10**: API integration and testing of new model.

### Performance Targets
- [ ] **Prediction Accuracy**: Achieve a stable, calibrated model that provides meaningful discrimination between gifts.
- [ ] **Business Logic**: `sum(expected_qty)` should be a reasonable reflection of `total_employees`, without artificial normalization.

This roadmap now reflects the critical re-architecture plan based on expert feedback.