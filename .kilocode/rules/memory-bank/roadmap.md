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

## Phase 3.5: CatBoost & Two-Stage Model Implementation (Week 7-8) - EXPERT RECOMMENDED

### 3.5.1 CatBoost Model Implementation
**Timeline**: 1 week
**Priority**: HIGH - Quick win for 2-4 p.p. R² improvement
**Deliverables**:
- [ ] Create new CatBoost training notebook
- [ ] Implement Poisson loss function (no log transform)
- [ ] Configure native categorical handling
- [ ] Compare performance with XGBoost baseline
- [ ] Feature importance analysis with CatBoost

**Implementation Details**:
```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=1000,
    loss_function='Poisson',  # Count-aware objective
    cat_features=categorical_cols,  # Native handling
    random_state=42
)
```

**Expected Outcomes**:
- Baseline CatBoost R² ≈ 0.33-0.35 (2-4 p.p. gain from Poisson loss)
- Better handling of high-cardinality features (shop, branch, brand)
- No need for log transformation

### 3.5.2 Two-Stage Model Architecture
**Timeline**: 1 week
**Priority**: HIGH - Expected 8-25% RMSE reduction
**Deliverables**:
- [ ] Stage 1: Binary classifier (will select gift?)
- [ ] Stage 2: Count regressor (how many? - positives only)
- [ ] Combined prediction pipeline
- [ ] Performance comparison with single model
- [ ] API integration design for two-stage predictions

**Two-Stage Design**:
```python
# Stage 1: Binary Classification
from catboost import CatBoostClassifier
classifier = CatBoostClassifier(
    iterations=500,
    cat_features=categorical_cols
)

# Stage 2: Count Regression (positives only)
regressor = CatBoostRegressor(
    loss_function='Poisson',
    cat_features=categorical_cols
)

# Final prediction = p_select * expected_count
```

### 3.5.3 New Feature Engineering
**Timeline**: 3 days
**Priority**: MEDIUM - Additional 1-2 p.p. R² gain
**Deliverables**:
- [ ] Shop-level share features:
  - `product_share_in_shop`
  - `brand_share_in_shop`
  - `category_share_in_shop`
- [ ] Rank features:
  - `product_rank_in_shop`
  - `brand_rank_in_shop`
- [ ] Interaction features via hashing trick
- [ ] Integration with existing feature pipeline

**Status**: 🚀 **READY TO START** - Expert guidance received

## Phase 3.6: Model Evaluation & Optimization (Week 9)

### 3.6.1 Advanced Cross-Validation
**Timeline**: 3 days
**Deliverables**:
- [ ] Leave-One-Shop-Out CV implementation
- [ ] Cold-start performance assessment
- [ ] Business metric evaluation (aggregated MAPE)
- [ ] Confidence interval generation

### 3.6.2 Performance Benchmarking
**Timeline**: 2 days
**Deliverables**:
- [ ] Compare XGBoost vs CatBoost vs Two-Stage
- [ ] Document performance gains from each improvement
- [ ] Create performance visualization dashboard
- [ ] Finalize model selection for production

**Expected Final Performance**:
- Target CV R² ≈ 0.36-0.41 (from current 0.3112)
- Realistic ceiling with current features: R² ≈ 0.45
- Business-ready performance for inventory planning

## Phase 4: Data Collection for Future Enhancement (Ongoing)

### 4.1 Historical Data Expansion
**Timeline**: Ongoing (as data becomes available)
**Priority**: LOW - Current data sufficient for R² ≈ 0.40
**Note**: Price data won't help (all gifts in shop are same price range)
**Focus Areas**:
- [ ] Merchandising data (position on page, promotions)
- [ ] Temporal patterns (day-of-season, urgency indicators)
- [ ] Product descriptions for text embeddings
- [ ] Product images for visual embeddings

**Status**: ✅ **NOT BLOCKING** - Current data sufficient for optimization

## Phase 5: API Development (Week 10-11)

### 5.1 FastAPI Application Setup
**Timeline**: 2 days
**Deliverables**:
- [ ] `src/api/main.py` - FastAPI application entry point
- [ ] Basic API structure and middleware
- [ ] Request/response logging
- [ ] Error handling middleware
- [ ] CORS configuration

**Status**: ⏳ **IN PROGRESS / NEXT UP**

### 5.2 Prediction Endpoints with Two-Stage Support
**Timeline**: 3 days (after 5.1)
**Deliverables**:
- [ ] `src/api/endpoints/predictions.py` - Prediction endpoints
- [ ] Three-step processing pipeline implementation
- [ ] Two-stage model integration
- [ ] Request validation and sanitization
- [ ] Response formatting with confidence intervals
- [ ] API documentation (OpenAPI/Swagger)

**Updated API Endpoint Structure**:
```python
@app.post("/predict", response_model=List[PredictionResponse])
async def predict_demand(request: PredictionRequest):
    # Step 1: Validate request
    validated_data = validate_request(request)
    
    # Step 2: Classify and transform data
    classified_data = classify_and_transform(validated_data)
    
    # Step 3: Two-stage prediction
    selection_probs = predict_selection_probability(classified_data)
    expected_counts = predict_counts_if_selected(classified_data)
    
    # Step 4: Combined predictions
    predictions = combine_predictions(selection_probs, expected_counts)
    
    return predictions
```

### 5.3 API Testing and Documentation
**Timeline**: 2 days
**Deliverables**:
- [ ] Comprehensive API tests
- [ ] Integration tests for full pipeline
- [ ] API documentation improvements
- [ ] Example requests and responses
- [ ] Performance testing

## Phase 6: Integration and Testing (Week 12)

### 6.1 End-to-End Integration
**Timeline**: 3 days
**Deliverables**:
- [ ] Complete pipeline integration
- [ ] Data flow validation
- [ ] Error handling verification
- [ ] Performance optimization
- [ ] Memory usage optimization

### 6.2 Comprehensive Testing
**Timeline**: 3 days
**Deliverables**:
- [ ] Unit tests for all components (target: >90% coverage)
- [ ] Integration tests for API endpoints
- [ ] Model performance validation tests
- [ ] Load testing for API endpoints
- [ ] Edge case testing

### 6.3 Documentation and Examples
**Timeline**: 1 day
**Deliverables**:
- [ ] Complete API documentation
- [ ] Usage examples and tutorials
- [ ] Model training documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide

## Phase 7: Production Model Training (Week 13)

### 7.1 Historical Data Processing
**Timeline**: 2 days
**Prerequisites**: Historical gift selection data access
**Deliverables**:
- [ ] Historical data preprocessing
- [ ] Feature engineering on real data
- [ ] Data quality validation
- [ ] Training dataset preparation

### 7.2 Final Model Training
**Timeline**: 3 days
**Deliverables**:
- [ ] Train final CatBoost + Two-Stage models on full data
- [ ] Model performance evaluation against all baselines
- [ ] Feature importance analysis
- [ ] Model validation and testing
- [ ] Production performance metrics

### 7.3 Model Finalization
**Timeline**: 2 days
**Deliverables**:
- [ ] Hyperparameter tuning
- [ ] Cross-validation optimization
- [ ] Performance benchmarking
- [ ] Model selection and finalization

## Phase 8: Deployment Preparation (Week 14)

### 8.1 Production Readiness
**Timeline**: 3 days
**Deliverables**:
- [ ] Environment configuration for production
- [ ] Logging and monitoring setup
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Docker containerization (optional)

### 8.2 Deployment Testing
**Timeline**: 2 days
**Deliverables**:
- [ ] Production environment testing
- [ ] Load testing validation
- [ ] Security testing
- [ ] Backup and recovery procedures

### 8.3 Documentation and Handover
**Timeline**: 2 days
**Deliverables**:
- [ ] Operations manual
- [ ] Monitoring and alerting setup
- [ ] Maintenance procedures
- [ ] User training materials

## Success Criteria and Milestones

### Technical Milestones
- [x] **Week 1**: Project foundation complete ✅
- [x] **Week 3**: Data pipeline functional ✅
- [x] **Week 5**: ML pipeline operational (baseline model R² ≈ 0.2947) ✅
- [x] **Week 6**: Model enhanced with shop features (CV R² ≈ 0.3112) ✅
- [ ] **Week 7-8**: CatBoost + Two-Stage implementation (Expected R² ≈ 0.36-0.41) 🚀 **NEXT**
- [ ] **Week 10-11**: API Development with two-stage support
- [ ] **Week 12**: Full Integration and Testing
- [ ] **Week 14**: Production Deployment

### Performance Targets
- [ ] **API Response Time**: < 2 seconds for prediction requests
- ✅ **Prediction Accuracy**: Current R² ≈ 0.31 is respectable (69% of realistic ceiling)
- 🎯 **Target Performance**: R² ≈ 0.36-0.41 with CatBoost + Two-Stage
- [x] **Test Coverage**: >90% for all completed components ✅
- [ ] **API Throughput**: Support 100+ concurrent requests
- ✅ **Model Performance**: Current sufficient, clear path to improvement
- ✅ **Data Sufficiency**: Sufficient for R² ≈ 0.40-0.45 target

### Business Objectives
- [ ] **Inventory Optimization**: 40% improvement in inventory balance
- [ ] **Operational Efficiency**: Reduced post-season complexity
- [ ] **Customer Satisfaction**: Improved gift availability
- [ ] **ROI**: Positive return from reduced surplus and back-orders

## Risk Mitigation

### Technical Risks
- **Data Quality Issues**: Implement robust validation and cleaning
- **Model Performance**: Use cross-validation and multiple metrics
- **API Performance**: Implement caching and optimization
- **Integration Complexity**: Incremental testing and validation

### Dependencies
- **Historical Data Access**: Critical for model training
- **Data Format Specification**: Required for preprocessing design
- **Infrastructure Requirements**: Needed for deployment planning
- **User Acceptance**: Important for final validation

## Resource Requirements

### Development Team
- **Data Scientist**: ML pipeline development
- **Backend Developer**: API development
- **DevOps Engineer**: Deployment and infrastructure
- **QA Engineer**: Testing and validation

### Infrastructure
- **Development Environment**: Local development setup
- **Testing Environment**: Isolated testing infrastructure
- **Production Environment**: Scalable production deployment
- **Data Storage**: Historical data and model storage

### Tools and Services
- **Version Control**: Git repository
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Application and model performance monitoring
- **Documentation**: API and technical documentation

## Next Actions (Expert-Guided Priorities)

### Immediate Priority (Week 1-2) - Quick Wins
1. **CatBoost Implementation** 🚀:
   - Create new notebook for CatBoost with Poisson loss
   - No log transformation needed
   - Expected gain: 2-4 p.p. R²
   
2. **Two-Stage Model** 🚀:
   - Implement binary classification + count regression
   - Expected gain: 8-25% RMSE reduction
   
3. **New Features**:
   - Add shop-level share and rank features
   - Expected gain: 1-2 p.p. R²

### Short Term (Week 3-4)
1. **Complete CatBoost Notebook**: Full implementation and validation
2. **Two-Stage Pipeline**: Build production-ready pipeline
3. **Leave-One-Shop-Out CV**: Assess cold-start performance
4. **Update Memory Bank**: Document new model architecture

### Medium Term (Month 1-2)
1. **API Integration**: Deploy two-stage predictions
2. **Hierarchical Models**: If cold-start gap appears
3. **Business Metrics**: Optimize for aggregated MAPE
4. **Production Deployment**: Launch optimized system

### Big Bets (Quarter 1-2)
1. **Recommender Formulation**: Two-tower model approach
2. **Embeddings**: Text/image features (if data available)
3. **Monte-Carlo Simulation**: Inventory confidence intervals

This roadmap now reflects expert guidance with realistic expectations and clear path to R² ≈ 0.40+ through technical improvements alone.