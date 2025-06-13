# Training Notebooks

This folder contains all training-related notebooks and code.

## Files:

### `model_training_analysis.ipynb` 
**Main training notebook** - Your primary workspace for:
- Loading 178K historical data
- Data preprocessing and aggregation
- XGBoost model training
- Performance analysis

### `enhanced_preprocessing_cells.py`
**Enhanced preprocessing code** - Copy these optimized cells into your main notebook:
- Cell 2: Memory-optimized data loading
- Cell 3: Efficient cleaning for 178K records  
- Cell 4: Performance-tracked aggregation
- Cell 5: Validated feature engineering

## Usage:

1. Open `model_training_analysis.ipynb`
2. Replace cells 2-5 with the enhanced versions from `enhanced_preprocessing_cells.py`
3. Run to process 178K dataset efficiently
4. Expected result: R² > 0.7 (vs previous 0.17)

## Output:
- Trained model saved to `../models/`
- Label encoders saved for API use
- Performance analytics and insights