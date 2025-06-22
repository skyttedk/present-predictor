# System Architecture

## Overview
The Predictive Gift Selection System follows a modular architecture. **Following expert review, the system is undergoing a critical re-architecture to address a fundamental mismatch between the ML model's training target and its prediction application.**

## High-Level Architecture (Revised)

```mermaid
graph TD
    A[API Gateway (FastAPI)] --> B{Request Processing & Validation};
    B --> C{Feature Engineering & Context Resolution};
    C --> D[Data Storage / Cache (Present Attributes)];
    C --> E{ML Prediction (CatBoost Regressor)};
    E --> F{Prediction Aggregation & Scaling};
    F --> A;
    subgraph DataSources
        G[Historical Data (CSV) + Employee Counts]
        H[Present Attributes Schema (JSON)]
        I[Real-time Request Data]
    end
    G --> J[Model Training Pipeline (catboost_trainer.py)];
    H --> C;
    I --> B;
    J --> K[Trained CatBoost Model (Predicts Selection Rate)];
    K --> E;
    D --> C;
```

## Core Components

### 1. API Layer
**Location**: [`src/api/`](src/api/:1)
- **Responsibilities**: Request validation, API endpoint management, orchestration of feature processing and prediction.
- **Key Files**: [`src/api/main.py`](src/api/main.py:1), [`src/api/schemas/requests.py`](src/api/schemas/requests.py:1), [`src/api/schemas/responses.py`](src/api/schemas/responses.py:1).

### 2. Data Processing & Feature Engineering Layer
**Location**: [`src/data/`](src/data/:1), [`src/ml/shop_features.py`](src/ml/shop_features.py:1)
- **Responsibilities**:
  - **Historical Data Preprocessing** (in [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)): **Crucially, this now involves calculating `selection_rate` by dividing `selection_count` by the total number of employees in that historical group.**
  - **Real-time Feature Engineering** (in [`src/ml/predictor.py`](src/ml/predictor.py:1)): Classification of gift attributes, gender classification, and resolution of shop-specific features.
- **Key Files**: [`src/data/classifier.py`](src/data/classifier.py:1), [`src/data/gender_classifier.py`](src/data/gender_classifier.py:1), [`src/ml/shop_features.py`](src/ml/shop_features.py:1).

### 3. Machine Learning Layer (Revised)
**Location**: [`src/ml/`](src/ml/:1)
- **Framework**: CatBoost
- **Responsibilities**:
  - **Model Training** ([`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1)):
    - Trains a `CatBoostRegressor` model.
    - **New Target Variable**: `selection_rate` (a float, not a count).
    - **New Loss Function**: A standard regression loss (e.g., RMSE), **not** Poisson.
  - **Prediction** ([`src/ml/predictor.py`](src/ml/predictor.py:1)):
    - Loads the retrained CatBoost model.
    - Predicts the `selection_rate` for each gift and gender combination.
    - **Prediction Aggregation**: The `_aggregate_predictions()` method will now correctly scale the predicted rate to an expected quantity.
- **Key Files**: [`src/ml/catboost_trainer.py`](src/ml/catboost_trainer.py:1), [`src/ml/predictor.py`](src/ml/predictor.py:1).

### 4. Configuration Layer
**Location**: [`src/config/`](src/config/:1)
- **Responsibilities**: Manages application, API, data, and model configurations.
- **Key Files**: [`src/config/settings.py`](src/config/settings.py:1).

## Data Flow Architecture (Revised Prediction Pipeline)

**Input Data for Prediction:**
- API Request: Branch code, list of presents, list of employees.
- Historical Data: Used by `ShopFeatureResolver` and for model training context.

**Prediction Pipeline (`predictor.py`):**
1.  **Request Input**: Branch, presents, employees.
2.  **Context Resolution**: Calculate employee gender statistics (ratios and counts).
3.  **Feature Vector Creation**: For each present, create feature vectors per gender group.
4.  **Model Prediction (Selection Rate)**: The CatBoost model predicts the `selection_rate` for each present-gender combination.
5.  **Prediction Scaling & Aggregation**: The predicted rate is scaled by the number of employees in that gender subgroup to get an expected count. These counts are then summed.
    *   `expected_count_subgroup = predicted_rate * num_employees_in_subgroup`
    *   `total_expected_qty = sum(expected_count_subgroup for each subgroup)`
6.  **Response Generation**: Formats the final expected quantities into the API response. Post-prediction normalization is **not** applied.

## Key Technical Decisions & Design Patterns (Revised)

-   **Rate-Based ML Target**: The model now predicts a `selection_rate`, not a raw count. This is the most critical architectural change.
-   **Standard Regression Loss**: Using RMSE or a similar loss function appropriate for rate prediction.
-   **Principled Aggregation**: Scaling predictions based on rates and employee counts, which is mathematically sound.
-   **FastAPI for API**: Leverages Pydantic for data validation.
-   **Pydantic for Configuration**: Centralized and typed settings management.
-   **Modular Structure**: Separation of API, data processing, and ML concerns.

## Source Code Structure
(No changes to the file structure itself)
```
src/
├── api/
├── data/
├── ml/
├── config/
└── utils/
```

## Critical Implementation Paths (New Focus)

1.  **Data Pipeline Modification**: The absolute priority is to modify the training data pipeline to correctly calculate `selection_rate`. This requires access to historical employee counts for each group.
2.  **Model Retraining**: Retrain the `CatBoostRegressor` on the new `selection_rate` target with an appropriate regression loss function.
3.  **Prediction Logic Update**: Rewrite the `_aggregate_predictions()` method in [`src/ml/predictor.py`](src/ml/predictor.py:1) to implement the new rate-scaling logic.