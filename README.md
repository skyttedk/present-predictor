# Predictive Gift Selection System

A machine learning-powered demand prediction system for Gavefabrikken's B2B gift distribution service.

## Overview

This system transforms Gavefabrikken's manual inventory estimation process into a data-driven prediction engine using historical gift selection data to forecast demand quantities for seasonal periods, particularly Christmas campaigns.

## Business Problem

Gavefabrikken operates B2B gift distribution where companies provide curated gift selections to employees through dedicated portals. The current manual forecasting approach results in:

- Inventory imbalances (overstocking and stockouts)
- Significant post-season operational overhead
- Increased costs from back-orders and surplus returns
- Suboptimal customer satisfaction due to unavailable selections

## Solution

ML-powered demand prediction system that:

1. **Processes Historical Data**: Analyzes multi-year gift selection records
2. **Classifies Inputs**: Automatically categorizes gifts and employee demographics
3. **Generates Predictions**: Uses XGBoost regression to forecast product-specific demand
4. **Provides API**: RESTful service for real-time prediction requests

## Technology Stack

- **Backend**: FastAPI for RESTful API
- **Data Processing**: Pandas for aggregation and transformation
- **Machine Learning**: XGBoost for demand regression
- **Validation**: Pydantic for schema validation
- **Testing**: Pytest for comprehensive testing

## Project Structure

```
src/
├── api/                 # FastAPI application and endpoints
├── data/               # Data processing and classification
├── ml/                 # Machine learning models and features
├── config/             # Application and model configuration
└── utils/              # Logging, exceptions, and utilities

tests/
├── test_api/           # API endpoint tests
├── test_data/          # Data processing tests
└── test_ml/            # ML model tests
```

## API Flow

```
Step 1: Request Input
{
  "branch_no": "123",
  "gifts": [{"product_id": "ABC", "description": "Red sweater"}],
  "employees": [{"name": "John Doe"}]
}

Step 2: Internal Classification
{
  "branch_no": "123", 
  "gifts": [{"Item Main Category": "Clothing", "Color": "Red", ...}],
  "employees": [{"gender": "Male"}]
}

Step 3: Prediction Response
[
  {"product_id": "ABC", "expected_qty": 15}
]
```

## Getting Started

### Prerequisites

- Python 3.9+
- Virtual environment (venv or conda)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd predict-presents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Development Setup

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Development Status

**Current Phase**: Foundation Setup (Phase 1.1)
- ✅ Project structure created
- ⏳ Virtual environment setup
- ⏳ Dependencies configuration
- ⏳ Basic configuration system

**Next**: Data pipeline development and API design

## Success Metrics

- **Prediction Accuracy**: >85% within ±20% of actual demand
- **API Response Time**: <2 seconds
- **Inventory Improvement**: 40% reduction in imbalances
- **Test Coverage**: >90%

## License

Internal Gavefabrikken project - All rights reserved

## Contact

Development Team - Gavefabrikken