# Files to Send to ML Expert for Review

## Core Implementation Files
1. `src/ml/predictor.py` - Production prediction service with aggregation logic
2. `src/ml/catboost_trainer.py` - Model training script  
3. `src/api/main.py` - API endpoint implementation (focus on lines 145-329)
4. `src/ml/shop_features.py` - Shop-level feature resolution

## Supporting Files
5. `src/data/schemas/data_models.py` - Data structure definitions
6. `src/data/preprocessor.py` - Data preprocessing pipeline
7. `src/config/model_config.py` - Model configuration

## Documentation & Context
8. `.kilocode/rules/memory-bank/architecture.md` - System architecture
9. `.kilocode/rules/memory-bank/expert-feedback-catboost-strategy.md` - Previous expert recommendations
10. `.kilocode/rules/memory-bank/roadmap.md` - Implementation roadmap showing two-stage plan
11. `.kilocode/rules/memory-bank/data-handling.md` - Data structure documentation

## Example Output & Test Data
12. `test_production_predict.json` - Example of problematic predictions
13. `src/data/historical/present.selection.historic.csv` - Sample of training data (if shareable)

## Recent Changes
14. This technical letter: `ml_expert_technical_letter.md`
15. Any recent git commits showing the changes we've made

## Optional (if needed for deeper investigation)
16. `notebooks/catboost_implementation.ipynb` - Original model development notebook
17. `models/catboost_poisson_model/model_metadata.pkl` - Model metadata including feature importance
18. Recent API logs showing prediction patterns

## Key Points for Expert
- Focus on the prediction aggregation logic in `predictor.py` lines 341-362
- Note the missing `src/ml/two_stage.py` file referenced in documentation
- Review the hardcoded `scaling_factor = 0.15` (now changed to 1.0)
- Consider the single-stage vs documented two-stage architecture mismatch