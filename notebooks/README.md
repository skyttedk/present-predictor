# Notebooks Directory

This directory contains Jupyter notebooks for interactive analysis and experimentation with the Gavefabrikken Demand Prediction system.

## Available Notebook

### 📊 `model_training_analysis.ipynb`
**The complete ML training and analysis notebook** including:
- **Data Loading & Exploration**: Load and visualize historical gift selection data (583 events)
- **Data Aggregation**: Convert selection events to training features
- **Feature Engineering**: Label encoding and feature preparation
- **Model Training**: Train XGBoost regression model with real data
- **Evaluation**: Model metrics, feature importance, and validation
- **Predictions**: Make and visualize predictions
- **Educational Content**: Understanding ML with real vs small datasets
- **Production Insights**: Business-ready feature importance analysis

## Getting Started

### 1. Install Dependencies
```bash
# From the notebooks directory
pip install -r requirements.txt

# Or from project root
pip install -r requirements.txt
pip install jupyter matplotlib seaborn
```

### 2. Launch Jupyter
```bash
# From the notebooks directory
jupyter notebook

# Or from project root
jupyter notebook notebooks/
```

### 3. Run the Training Notebook
1. Open `model_training_analysis.ipynb`
2. Run all cells sequentially
3. Explore the interactive sections

## Notebook Features

### 📈 Data Visualization
- Historical data distribution plots
- Feature importance charts
- Prediction vs actual comparisons
- Residual analysis plots

### 🔧 Interactive Features
- Custom prediction interface
- Feature encoder mappings
- Model performance metrics
- Real-time training statistics

### 💾 Model Persistence
- Save trained models with metadata
- Load and verify saved models
- Export training statistics

## Key Outputs

The notebook will generate:
- **Trained Model**: `../models/demand_predictor_notebook.pkl`
- **Training Metrics**: Validation scores and cross-validation results
- **Feature Analysis**: Importance rankings and encoder mappings
- **Visualizations**: Plots and charts for analysis

## Usage Examples

### Training a New Model
```python
# In the notebook
preprocessor = DataPreprocessor()
model = DemandPredictor()
X, y = preprocessor.create_training_features()
training_stats = model.train(X, y)
```

### Making Custom Predictions
```python
# Create custom feature dictionary
custom_features = {
    'employee_gender': 0,  # 0=female, 1=male
    'product_main_category': 1,  # See encoder mappings
    'product_utility_type': 0,  # practical=0, aesthetic=1, etc.
    # ... other features
}

prediction = model.predict_single(custom_features)
```

### Analyzing Feature Importance
```python
importance = model.get_feature_importance()
explanation = model.get_prediction_explanation(features)
```

## Data Requirements

The notebook expects:
- **Historical Data**: `../src/data/historical/present.selection.historic.csv`
- **Schema**: `../src/data/product.attributes.schema.json`
- **Source Modules**: Access to `../src/` directory

## Troubleshooting

### Common Issues
1. **Module Import Errors**: Ensure the project root is in Python path
2. **Data File Not Found**: Check historical data file location
3. **Memory Issues**: Use smaller datasets for testing

### Solutions
```python
# Fix import path
import sys
sys.path.append('..')

# Check data files
from pathlib import Path
print(Path('../src/data/historical/').exists())

# Reduce memory usage
pd.set_option('display.max_rows', 50)
```

## Next Steps

After running the notebook:
1. **Analyze Results**: Review model performance metrics
2. **Tune Parameters**: Adjust XGBoost hyperparameters if needed
3. **Collect More Data**: Add historical data to improve accuracy
4. **Deploy Model**: Use trained model in the API pipeline
5. **Monitor Performance**: Track predictions vs actual results

## Integration with Main System

The notebook-trained models are compatible with the main system:
```python
# In your API code
from src.ml.model import DemandPredictor
model = DemandPredictor()
model.load_model('models/demand_predictor_notebook.pkl')
```

This allows you to train models interactively and deploy them in production.