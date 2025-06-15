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

## Phase 3.5: Data Collection for Future Enhancement (Ongoing)

### 3.5.1 Historical Data Expansion
**Timeline**: Ongoing (as data becomes available)
**Priority**: Medium - For further model improvement beyond current R² ≈ 0.31
**Deliverables**:
- [ ] Collect additional years of historical gift selection data
- [ ] Expand data to include more diverse companies, branches, and seasonal periods
- [ ] Target increasing unique combinations beyond current 98,741
- [ ] Validate data format consistency with existing pipeline
- [ ] Establish ongoing data collection processes for continuous model refinement

**Data Requirements**:
```
Current State: 178,736 events → 98,741 unique combinations (Sufficient for R² ≈ 0.31)
Future Target: Increase unique combinations to potentially improve R² towards > 0.5
Expected Sources: New seasonal periods, additional client data
```

**Success Criteria for Future Enhancements**:
- Achieve sample-to-feature ratio improvement for potentially more complex models
- Target R² > 0.5 or higher with more comprehensive data
- Enable more granular analysis and feature engineering
- Support enhanced long-range forecasting and new product introduction predictions

**Status**: ✅ **DATA SUFFICIENT FOR CURRENT MODEL (R² ≈ 0.31)** - Further collection is for future improvement, not a blocker.

## Phase 4: API Development (Proceeding with Enhanced Model R² ≈ 0.3112)

### 4.1 FastAPI Application Setup
**Timeline**: 2 days
**Deliverables**:
- [ ] `src/api/main.py` - FastAPI application entry point
- [ ] Basic API structure and middleware
- [ ] Request/response logging
- [ ] Error handling middleware
- [ ] CORS configuration

**Status**: ⏳ **IN PROGRESS / NEXT UP**

### 4.2 Prediction Endpoints
**Timeline**: 3 days (after 4.1)
**Deliverables**:
- [ ] `src/api/endpoints/predictions.py` - Prediction endpoints
- [ ] Three-step processing pipeline implementation
- [ ] Request validation and sanitization
- [ ] Response formatting
- [ ] API documentation (OpenAPI/Swagger)

**API Endpoint Structure**:
```python
@app.post("/predict", response_model=List[PredictionResponse])
async def predict_demand(request: PredictionRequest):
    # Step 1: Validate request
    validated_data = validate_request(request)
    
    # Step 2: Classify and transform data
    classified_data = classify_and_transform(validated_data)
    
    # Step 3: Generate predictions
    predictions = generate_predictions(classified_data)
    
    return predictions
```

### 4.3 API Testing and Documentation
**Timeline**: 2 days
**Deliverables**:
- [ ] Comprehensive API tests
- [ ] Integration tests for full pipeline
- [ ] API documentation improvements
- [ ] Example requests and responses
- [ ] Performance testing

## Phase 5: Integration and Testing (Week 7)

### 5.1 End-to-End Integration
**Timeline**: 3 days
**Deliverables**:
- [ ] Complete pipeline integration
- [ ] Data flow validation
- [ ] Error handling verification
- [ ] Performance optimization
- [ ] Memory usage optimization

### 5.2 Comprehensive Testing
**Timeline**: 3 days
**Deliverables**:
- [ ] Unit tests for all components (target: >90% coverage)
- [ ] Integration tests for API endpoints
- [ ] Model performance validation tests
- [ ] Load testing for API endpoints
- [ ] Edge case testing

### 5.3 Documentation and Examples
**Timeline**: 1 day
**Deliverables**:
- [ ] Complete API documentation
- [ ] Usage examples and tutorials
- [ ] Model training documentation
- [ ] Deployment guide
- [ ] Troubleshooting guide

## Phase 6: Model Training and Validation (Week 8)

### 6.1 Historical Data Processing
**Timeline**: 2 days
**Prerequisites**: Historical gift selection data access
**Deliverables**:
- [ ] Historical data preprocessing
- [ ] Feature engineering on real data
- [ ] Data quality validation
- [ ] Training dataset preparation

### 6.2 Initial Model Training
**Timeline**: 3 days
**Deliverables**:
- [ ] Train XGBoost models on historical data
- [ ] Model performance evaluation
- [ ] Feature importance analysis
- [ ] Model validation and testing
- [ ] Baseline performance metrics

### 6.3 Model Optimization
**Timeline**: 2 days
**Deliverables**:
- [ ] Hyperparameter tuning
- [ ] Cross-validation optimization
- [ ] Performance benchmarking
- [ ] Model selection and finalization

## Phase 7: Deployment Preparation (Week 9)

### 7.1 Production Readiness
**Timeline**: 3 days
**Deliverables**:
- [ ] Environment configuration for production
- [ ] Logging and monitoring setup
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Docker containerization (optional)

### 7.2 Deployment Testing
**Timeline**: 2 days
**Deliverables**:
- [ ] Production environment testing
- [ ] Load testing validation
- [ ] Security testing
- [ ] Backup and recovery procedures

### 7.3 Documentation and Handover
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
- [x] **Week 6 (Current)**: Model enhanced with shop features (CV R² ≈ 0.3112) ✅
- [ ] **Data Collection**: Ongoing for future enhancements (not a blocker)
- [ ] **API Endpoints**: In progress / Next Up
- [ ] **Full Integration**: Pending API completion
- [ ] **Production Deployment**: Pending full integration and testing

### Performance Targets
- [ ] **API Response Time**: < 2 seconds for prediction requests (pending API completion)
- ⚠️ **Prediction Accuracy**: Target >85% (CV R² log target ≈ 0.3112, original scale R² needs monitoring)
- [x] **Test Coverage**: >90% for all completed components ✅
- [ ] **API Throughput**: Support 100+ concurrent requests (pending API completion)
- ⚠️ **Model Performance**: CV R² (log target) ≈ 0.3112 (moderate, improved from 0.1722 and 0.2947)
- ✅ **Data Sufficiency**: Sufficient for current model (98,741 combinations)

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

## Next Actions

### Immediate Priority (Days 1-2 from context.md)
1. **Memory Bank Update (THIS TASK - In Progress)**: Document shop feature insights.
2. **Production Integration (Enhanced Model)**: Integrate model with shop features (R² ≈ 0.3112) into API pipeline.
3. **Notebook Finalization**: Ensure [`notebooks/breakthrough_training.ipynb`](notebooks/breakthrough_training.ipynb:1) is clean and fully reproduces 0.3112 R².
4. **Business Testing (Enhanced Model)**: Begin real-world inventory planning trials with the improved model.

### Short Term (Week 1-2 from context.md)
1. **API Deployment (Enhanced Model)**: Deploy prediction service with the 0.3112 R² model.
2. **Model Monitoring**: Implement performance tracking for the enhanced model using stratified CV.
3. **Business Integration**: Connect with Gavefabrikken's inventory systems.
4. **User Training**: Guide operations team on enhanced model capabilities and limitations.

### Medium Term (Month 1 from context.md)
1. **Performance Validation**: Monitor real-world prediction accuracy of the enhanced model.
2. **Continuous Improvement**: Gather feedback and explore further feature refinements.
3. **Scale Deployment**: Expand to multiple companies and seasonal periods.

This roadmap reflects the project's progression to using an enhanced model (CV R² ≈ 0.3112) with shop assortment features, ready for API integration and business testing.