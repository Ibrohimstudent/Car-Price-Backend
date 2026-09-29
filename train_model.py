"""Modelni o'qitadi va backend uchun kerakli fayllarni saqlaydi."""
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("CarPrice_Assignment.csv")

CAT_FEATURES = ["fueltype", "carbody", "drivewheel", "enginetype", "cylindernumber"]
NUM_FEATURES = [
    "wheelbase", "carlength", "carwidth", "curbweight",
    "enginesize", "horsepower", "citympg", "highwaympg",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

X = df[FEATURES]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CatBoost (asosiy model, saqlanadi) ---
model = CatBoostRegressor(
    depth=4, learning_rate=0.05, iterations=500,
    loss_function="RMSE", verbose=0, random_seed=42,
)
t0 = time.perf_counter()
model.fit(X_train, y_train, cat_features=CAT_FEATURES)
cat_time = time.perf_counter() - t0
cat_pred = model.predict(X_test)
cat_rmse = float(np.sqrt(mean_squared_error(y_test, cat_pred)))
cat_r2 = float(r2_score(y_test, cat_pred))

model.save_model("car_model.cbm")

# --- Solishtirish uchun: LightGBM va XGBoost (encoding bilan) ---
X_encoded = pd.get_dummies(X, columns=CAT_FEATURES, drop_first=True)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

lgbm = LGBMRegressor(random_state=42)
t0 = time.perf_counter()
lgbm.fit(X_train2, y_train2)
lgbm_time = time.perf_counter() - t0
lgbm_pred = lgbm.predict(X_test2)
lgbm_rmse = float(np.sqrt(mean_squared_error(y_test2, lgbm_pred)))
lgbm_r2 = float(r2_score(y_test2, lgbm_pred))

xgb = XGBRegressor(random_state=42)
t0 = time.perf_counter()
xgb.fit(X_train2, y_train2)
xgb_time = time.perf_counter() - t0
xgb_pred = xgb.predict(X_test2)
xgb_rmse = float(np.sqrt(mean_squared_error(y_test2, xgb_pred)))
xgb_r2 = float(r2_score(y_test2, xgb_pred))

print(f"CatBoost  RMSE={cat_rmse:.2f} R2={cat_r2:.3f}")
print(f"LightGBM  RMSE={lgbm_rmse:.2f} R2={lgbm_r2:.3f}")
print(f"XGBoost   RMSE={xgb_rmse:.2f} R2={xgb_r2:.3f}")

# --- Statistika sahifasi uchun JSON ---
importance = model.get_feature_importance()
importance_df = pd.DataFrame({"feature": FEATURES, "importance": importance})
importance_df = importance_df.sort_values("importance", ascending=False).head(10)

stats = {
    "model_comparison": [
        {"model": "CatBoost", "rmse": round(cat_rmse, 2), "r2": round(cat_r2, 3), "time": round(cat_time, 2)},
        {"model": "LightGBM", "rmse": round(lgbm_rmse, 2), "r2": round(lgbm_r2, 3), "time": round(lgbm_time, 2)},
        {"model": "XGBoost", "rmse": round(xgb_rmse, 2), "r2": round(xgb_r2, 3), "time": round(xgb_time, 2)},
    ],
    "feature_importance": [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 2)}
        for _, row in importance_df.iterrows()
    ],
    "dataset_size": len(df),
    "price_min": float(df["price"].min()),
    "price_max": float(df["price"].max()),
    "price_avg": float(df["price"].mean()),
}
with open("stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# --- Interfeys uchun meta (dropdown/min-max) ---
meta = {
    "features": FEATURES,
    "cat": {c: sorted(df[c].unique().tolist()) for c in CAT_FEATURES},
    "num": {
        c: {
            "min": float(df[c].min()), "max": float(df[c].max()),
            "median": float(df[c].median()),
            "is_int": bool(pd.api.types.is_integer_dtype(df[c])),
        }
        for c in NUM_FEATURES
    },
}
with open("meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saqlandi: car_model.cbm, meta.json, stats.json")
