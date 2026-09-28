# Sales Prediction using Python

Predicting product sales from advertising spend (TV, Radio, Newspaper) with machine learning regression models.

## Dataset
The public **Advertising** dataset (ISLR): 200 rows, 3 features (`TV`, `Radio`, `Newspaper` spend) and the target `Sales`. No missing values or duplicates.

## Approach
1. Load and inspect the data
2. Exploratory analysis (correlation heatmap, spend vs sales plots)
3. 80/20 train-test split
4. Train and compare Linear Regression, Random Forest and Gradient Boosting
5. Evaluate with R², MAE, RMSE and 5-fold cross-validation
6. Predict sales for new advertising budgets

## Results

| Model | R² (test) | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.899 | 1.46 | 1.78 |
| Random Forest | 0.982 | 0.63 | 0.76 |
| Gradient Boosting | 0.983 | 0.62 | 0.73 |

**Best model:** Gradient Boosting.

## Key findings
- **TV** is the strongest driver of sales (correlation 0.78, feature importance about 0.63).
- **Radio** is second (correlation 0.58, importance about 0.36).
- **Newspaper** has almost no effect (importance about 0.01), so budget spent there adds little.
- Tree-based models beat linear regression clearly, suggesting the channels interact rather than adding up independently.

## How to run
pip install pandas scikit-learn matplotlib seaborn
python Sales_prediction.py

Plots are saved to the `outputs/` folder.

## Files
- `Sales_prediction.py`: full pipeline
- `index.html`: project website
- `outputs/`: generated plots
