"""
Task 4: Sales Prediction using Python
-------------------------------------
Predicts product sales from advertising spend (TV, Radio, Newspaper)
using machine learning regression models.

Dataset: the public "Advertising" dataset (ISLR), 200 rows.
The script loads it from a local 'Advertising.csv' if present,
otherwise downloads it from GitHub.

Run:  python sales_prediction.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save plots to files; remove this line to show windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_URL = (
    "https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv"
)
LOCAL_FILE = "Advertising.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    source = LOCAL_FILE if os.path.exists(LOCAL_FILE) else DATA_URL
    df = pd.read_csv(source)
    # The original file has an unnamed index column - drop it
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded data from: {source}")
    return df


df = load_data()

print("\n=== First 5 rows ===")
print(df.head())
print("\n=== Shape ===", df.shape)
print("\n=== Info ===")
df.info()
print("\n=== Summary statistics ===")
print(df.describe().round(2))
print("\n=== Missing values ===")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


# ----------------------------------------------------------------------
# 2. Exploratory data analysis
# ----------------------------------------------------------------------
print("\n=== Correlation with Sales ===")
corr = df.corr()
print(corr["Sales"].sort_values(ascending=False).round(3))

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.regplot(x=col, y="Sales", data=df, ax=ax, line_kws={"color": "red"})
    ax.set_title(f"{col} spend vs Sales")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/spend_vs_sales.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 3. Train / test split
# ----------------------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# ----------------------------------------------------------------------
# 4. Train and compare models
# ----------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    results.append(
        {
            "Model": name,
            "R2 (test)": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 (5-fold CV)": cv_r2,
        }
    )

results_df = pd.DataFrame(results).set_index("Model").round(3)
print("\n=== Model comparison ===")
print(results_df)

best_name = results_df["R2 (test)"].idxmax()
best_model = models[best_name]
print(f"\nBest model on test set: {best_name}")


# ----------------------------------------------------------------------
# 5. Interpret the results
# ----------------------------------------------------------------------
lin = models["Linear Regression"]
print("\n=== Linear Regression coefficients (extra sales per unit of spend) ===")
for feat, coef in zip(X.columns, lin.coef_):
    print(f"  {feat:<10}: {coef:.4f}")
print(f"  Intercept : {lin.intercept_:.4f}")

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
print("\n=== Random Forest feature importances ===")
print(importances.sort_values(ascending=False).round(3))

plt.figure(figsize=(6, 4))
importances.plot(kind="barh", color="#5B3A8E")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()

# Actual vs predicted for the best model
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions[best_name], alpha=0.8)
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/actual_vs_predicted.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 6. Predict sales for new advertising budgets
# ----------------------------------------------------------------------
def predict_sales(tv: float, radio: float, newspaper: float) -> float:
    """Predict sales for a given advertising budget using the best model."""
    new = pd.DataFrame([[tv, radio, newspaper]], columns=X.columns)
    return float(best_model.predict(new)[0])


print("\n=== Example predictions ===")
scenarios = [
    (200, 40, 20),   # heavy TV + radio
    (100, 20, 50),   # moderate spend
    (50, 5, 80),     # newspaper-heavy
]
for tv, radio, news in scenarios:
    print(
        f"TV={tv:>4}, Radio={radio:>3}, Newspaper={news:>3} "
        f"-> predicted sales: {predict_sales(tv, radio, news):.2f}"
    )

print(f"\nPlots saved to ./{OUT_DIR}/")