# System Architecture

## Overview
The Predictive Gift Selection System follows a modular architecture with clear separation of concerns between data processing, machine learning, and API layers. The current primary focus is on resolving a mismatch between the ML model's training target and its interpretation during prediction.

## High-Level Architecture

```mermaid
graph TD
    A[API Gateway (FastAPI)] --> B{Request Processing & Validation};
    B --> C{Feature Engineering & Context Resolution};
    C --> D[Data Storage / Cache (Present Attributes)];
    C --> E{ML Prediction (CatBoost Regressor)};
    E --> F{Prediction Aggregation & Formatting};
    F --> A;
    subgraph DataSources
        G[Historical Data (CSV)]
        H[Present Attributes Schema (JSON)]
        I[Real-time Request Data]
    end
    G --> J[Model Training Pipeline (catboost_trainer.py)];
    H --> C;
    I --> B;
    J --> K[Trained CatBoost Model];
    K --> E;
    D --> C;
```

## Core Components

### 1. API Layer
**Location**: [`src/api/`](src/api/:1)
- **Framework**: FastAPI
- **Responsibilities**:
  - Request validation and parsing (using Pydantic schemas).
  - API endpoint management (e.g., `/predict`).
  - Response formatting.
  - Orchestration of feature processing and prediction.
- **Key Files**:
  - [`src/api/main.py`](src/api/main.py:1): FastAPI application entry point, `/predict` endpoint.
  - [`src/api/schemas/requests.py`](src/api/schemas/requests.py:1): API request models.
  - [`src/api/schemas/responses.py`](src/api/schemas/responses.py:1): API response models.

### 2. Data Processing & Feature Engineering Layer
**Location**: [`src/data/`](src/data/:1), [`src/ml/shop_features.py`](src/ml/shop_features.py:1)
- **Frameworks**: Pandas
- **Responsibilities**:
  - **Historical Data Preprocessing** (in [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)): Cleaning, aggregation to create `selection_count`.
  - **Real-time Feature Engineering** (in [`src/ml/predictor.py`](src/ml/predictor.py:1), [`src/ml/shop_features.py`](src/ml/shop_features.py:1)):
    - Classification of gift attributes (potentially using OpenAI, as per `OpenAISettings` in [`src/config/settings.py`](src/config/settings.py:94)).
    - Employee gender classification (e.g., `gender_guesser`).
    - Resolution of shop-specific features.
    - Creation of feature vectors for the ML model.
- **Key Files**:
  - [`src/data/classifier.py`](src/data/classifier.py:1): Gift categorization logic (if used).
  - [`src/data/gender_classifier.py`](src/data/gender_classifier.py:1): Gender classification.
  - [`src/ml/shop_features.py`](src/ml/shop_features.py:1): `ShopFeatureResolver` for shop-specific context.
  - [`src/ml/predictor.py`](src/ml/predictor.py:1): Orchestrates feature creation for prediction.

### 3. Machine Learning Layer
**Location**: [`src/ml/`](src/ml/:1)
- **Framework**: CatBoost
- **Responsibilities**:
  - **Model Training** ([`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)):
    - Trains a `CatBoostRegressor` model with a `Poisson` loss function.
    - Target variable: `selection_count` (total occurrences of a feature combination).
    - Includes feature engineering (shop assortment, interaction features).
    - Hyperparameter tuning with Optuna.
  - **Prediction** ([`src/ml/predictor.py`](src/ml/predictor.py:1)):
    - Loads the trained CatBoost model.
    - Makes predictions based on engineered features.
    - **Prediction Aggregation**: The `_aggregate_predictions()` method now correctly weights predictions by gender ratios and normalizes the final output to ensure the total quantity matches the number of employees.
- **Key Files**:
  - [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1): Training pipeline.
  - [`src/ml/predictor.py`](src/ml/predictor.py:1): Prediction service.
  - Model artifacts stored in `models/catboost_poisson_model/`.

### 4. Configuration Layer
**Location**: [`src/config/`](src/config/:1)
- **Responsibilities**: Manages application, API, data, model, and external service (e.g., OpenAI) configurations using Pydantic.
- **Key Files**:
  - [`src/config/settings.py`](src/config/settings.py:1): Main application settings.
  - [`src/config/model_config.py`](src/config/model_config.py:1): (Potentially outdated) XGBoost-focused model configurations. The CatBoost settings are primarily in `settings.py`.

## Data Flow Architecture (Focus on Prediction)

**Input Data for Prediction:**
- API Request: Branch code, list of presents (with attributes), list of employees (with gender).
- Historical Data: Used by `ShopFeatureResolver` and for model training context (medians, etc.).
- Present Attributes: Cached or classified attributes for gifts.

**Prediction Pipeline (`predictor.py`):**
1.  **Request Input**: Branch, presents, employees.
2.  **Context Resolution**:
    *   Calculate employee gender statistics (ratios).
    *   Resolve shop-specific features using `ShopFeatureResolver`.
3.  **Feature Vector Creation**: For each present, create feature vectors per gender group.
4.  **Model Prediction (Raw Scores)**: The CatBoost model predicts a "popularity score" for each present. This is calculated by weighting the gender-specific model outputs by the corresponding gender ratio and number of employees.
    *   `raw_score = sum(model_output_gender * ratio_gender * total_employees)`
5.  **Prediction Normalization**: The raw scores for all presents are summed up. Each individual score is then normalized against this total to ensure the final sum of `expected_qty` equals `total_employees`.
    *   `final_qty = (raw_score / total_raw_scores) * total_employees`
6.  **Response Generation**: Formats the normalized predictions into the final API response.

## Key Technical Decisions & Design Patterns

-   **FastAPI for API**: Leverages Pydantic for data validation and auto-documentation.
-   **Pydantic for Configuration**: Centralized and typed settings management.
-   **CatBoost for ML**: Chosen for its handling of categorical features and performance. Poisson loss for count data.
-   **Modular Structure**: Separation of API, data processing, and ML concerns.
-   **Singleton Pattern for Predictor**: [`get_predictor()`](src/ml/predictor.py:403) ensures efficient model loading.
-   **Feature Hashing**: Used for interaction features to manage dimensionality.

## Source Code Structure
(As outlined in `README.md` and confirmed by file tree)
```
src/
├── api/                 # FastAPI application and endpoints
├── data/               # Data processing and classification helpers
├── ml/                 # Machine learning models, training, prediction, features
├── config/             # Application and model configuration
└── utils/              # Logging, exceptions (if any)
```

## Critical Implementation Paths

1.  **Model Training Target Definition** ([`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:108)): `selection_count = cleaned_data.groupby(grouping_cols).size().reset_index(name='selection_count')`. This defines what the model learns.
2.  **Prediction Aggregation Logic** ([`src/ml/predictor.py`](src/ml/predictor.py:351-372)): `_aggregate_predictions()`. This is where the misinterpretation occurs and needs correction as per Option A.
3.  **Feature Engineering Consistency**: Ensuring features created at inference time in [`src/ml/predictor.py`](src/ml/predictor.py:1) precisely match those used in training ([`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)), including handling of missing values, defaults, and data types. The use of `numeric_medians` loaded from training artifacts is a good step here.

## Scalability Considerations (Future)
- Current focus is on correcting core logic.
- Future considerations might include more robust data storage than CSVs, caching strategies for features/predictions, and asynchronous processing for heavy tasks.