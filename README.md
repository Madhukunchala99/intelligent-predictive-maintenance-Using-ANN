# Intelligent Predictive Maintenance & Machine Failure Risk Management

## Project Overview
A machine learning project that predicts industrial machine failure risk using sensor readings and supports risk-based maintenance inspection decisions.

## Dataset
- Source: AI4I 2020 Predictive Maintenance Dataset (UCI)
- Records: 10,000
- Target: Machine failure (0 = No Failure, 1 = Failure)
- Failure rate: 3.39%
- Features: Machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

Identifiers and failure-mode indicators were excluded from model inputs to prevent data leakage.

## Methodology
1. Business understanding and data quality assessment
2. Exploratory data analysis and preprocessing
3. Stratified train/validation/test split (70/15/15)
4. Logistic Regression, Random Forest, and ANN development
5. Class weighting and random oversampling comparisons
6. ROC-AUC, PR-AUC, threshold and hypothetical cost analysis
7. Controlled ANN learning-rate experiment
8. Business interpretation and independent test evaluation

## Final Selected Model
Oversampled Random Forest

Selected validation threshold: 0.0580690315

### Independent Test Results
- Precision: 0.2682
- Recall: 0.9412
- F1-score: 0.4174
- ROC-AUC: 0.9593
- PR-AUC: 0.7440
- True Positives: 48
- False Positives: 131
- False Negatives: 3
- True Negatives: 1318

## Hypothetical Maintenance Cost
- False alarm: INR 2,000
- Missed failure: INR 50,000
- Validation error cost: INR 320,000
- Independent test error cost: INR 412,000

These are illustrative assumptions, not actual factory expenses.

## Controlled ANN Experiment
Compared Adam learning rates of 0.001 and 0.0005 using three random seeds.

Reducing the learning rate did not improve mean validation PR-AUC, although it reduced variation across runs.

## Project Files
- `main.ipynb` — Complete project notebook
- `results/model_comparison.csv` — Model comparison
- `results/experiment_results.csv` — Controlled experiment
- `models/final_predictive_maintenance_rf.joblib` — Saved model and preprocessing
- `reports/Final_Technical_Report.pdf` — Technical report
- `requirements.txt` — Python dependencies

## How to Run
1. Install dependencies using `pip install -r requirements.txt`.
2. Download the AI4I 2020 dataset from UCI if not already available.
3. Place `ai4i2020.csv` in the project folder.
4. Open `main.ipynb` and run the cells in order.

## Limitations
- Synthetic dataset without machine-level time histories
- Potential Random Forest overfitting
- Uncalibrated risk scores
- Hypothetical maintenance costs and validation-selected threshold
- Further evaluation on real industrial data is required before deployment

## Author
Madhu Kunchala

GitHub: https://github.com/Madhukunchala99
