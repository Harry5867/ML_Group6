import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ===================== CUSTOM SCALER =====================
class CustomStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ===================== GRADIENT BOOSTING (GĐ5) =====================
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        y_pred = np.full(len(y), self.initial_prediction)

        for i in range(self.n_estimators):
            residuals = y - y_pred

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

            print(f"Tree {i+1}/{self.n_estimators} trained")

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred


# ===================== LOAD DATA =====================
# Nếu có file:
# df = pd.read_excel("data.xlsx")
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# Demo data
X, y = make_regression(n_samples=1000, n_features=6, noise=15, random_state=42)


# ===================== SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===================== SCALE =====================
scaler = CustomStandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===================== MODELS =====================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
}


# ===================== TRAIN + EVALUATE =====================
results = []

for name, model in models.items():
    print(f"\n===== Training {name} =====")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ===================== RESULTS =====================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False)

print("\n===== FINAL RESULTS =====")
print(results_df)


# ===================== VISUALIZATION =====================
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=30)
plt.ylabel("R2 Score")
plt.show()
