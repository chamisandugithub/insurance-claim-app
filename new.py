# %%
## Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# %%
## Load the dataset
df = pd.read_csv("insurance.csv")
print(df.head())

# %%
pd.set_option('display.float_format', '{:.2f}'.format)
print(df.info())
print(df.shape)

# %%
# Fill missing values instead of dropping
df['age'].fillna(df['age'].median(), inplace=True)
df['region'].fillna(df['region'].mode()[0], inplace=True)

# %%
print("Missing values after imputation:\n", df.isnull().sum())

# %%
# EDA: distributions of numeric columns
sns.set_theme(style="white")
numeric_columns = ["age", "bmi", "bloodpressure", "children", "claim"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")
for j in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# %%
# Separate predictors and target
X = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'region']]
y = df['claim']

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Preprocessing
categorical_cols = ["gender", "diabetic", "smoker", "region"]
numeric_cols = ["age", "bmi", "bloodpressure", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# %%
# Build Model Pipelines
linreg_pipeline = Pipeline([("preprocessor", preprocessor),
                            ("model", LinearRegression())])

rf_pipeline = Pipeline([("preprocessor", preprocessor),
                        ("model", RandomForestRegressor(n_estimators=300, random_state=42))])

xgb_pipeline = Pipeline([("preprocessor", preprocessor),
                         ("model", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                                subsample=0.8, colsample_bytree=0.8, random_state=42))])

# %%
# Train models
linreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

# Predictions
y_pred_lin = linreg_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)

# %%
# Evaluation function
def evaluate(y_true, y_pred, name):
    print(f"{name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print("-"*30)

evaluate(y_test, y_pred_lin, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")
evaluate(y_test, y_pred_xgb, "XGBoost")

# %%
# Hyperparameter tuning Random Forest
param_grid_rf = {
    "model__n_estimators": [200, 500, 800],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=3, scoring="r2", n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

evaluate(y_test, y_pred_best_rf, "Tuned Random Forest")

# %%
# Hyperparameter tuning XGBoost
param_dist_xgb = {
    "model__n_estimators": [300, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [4, 6, 8],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}
random_xgb = RandomizedSearchCV(xgb_pipeline, param_distributions=param_dist_xgb,
                                n_iter=30, cv=3, scoring="r2", n_jobs=-1, random_state=42)
random_xgb.fit(X_train, y_train)
best_xgb = random_xgb.best_estimator_
y_pred_best_xgb = best_xgb.predict(X_test)

evaluate(y_test, y_pred_best_xgb, "Tuned XGBoost")

# %%
# Compare all models
model_metrics = {
    "Linear Regression": [mean_absolute_error(y_test, y_pred_lin),
                          np.sqrt(mean_squared_error(y_test, y_pred_lin)),
                          r2_score(y_test, y_pred_lin)],
    "Random Forest": [mean_absolute_error(y_test, y_pred_rf),
                      np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                      r2_score(y_test, y_pred_rf)],
    "XGBoost": [mean_absolute_error(y_test, y_pred_xgb),
                np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
                r2_score(y_test, y_pred_xgb)],
    "Tuned Random Forest": [mean_absolute_error(y_test, y_pred_best_rf),
                            np.sqrt(mean_squared_error(y_test, y_pred_best_rf)),
                            r2_score(y_test, y_pred_best_rf)],
    "Tuned XGBoost": [mean_absolute_error(y_test, y_pred_best_xgb),
                      np.sqrt(mean_squared_error(y_test, y_pred_best_xgb)),
                      r2_score(y_test, y_pred_best_xgb)]
}

metrics_df = pd.DataFrame(model_metrics, index=["MAE","RMSE","R2"]).T.sort_values(by="R2", ascending=False)
print(metrics_df)

best_model_name = metrics_df.index[0]
print(f"\n🏆 Best Model: {best_model_name}")

# %%
# Save final predictions using best model
final_model = best_xgb if best_model_name=="Tuned XGBoost" else best_rf
y_pred_final = final_model.predict(X_test)
final_df = X_test.copy()
final_df["Actual_Claim"] = y_test.values
final_df["Predicted_Claim"] = y_pred_final.round(2)
final_df.to_csv("insurance_claim_predictions.csv", index=False)
print("✅ Predictions saved to 'insurance_claim_predictions.csv'")

# %%
# Feature importance (only for tree-based models)
if best_model_name in ["Tuned XGBoost","Tuned Random Forest"]:
    model_obj = final_model.named_steps["model"]
    importance = model_obj.feature_importances_
    cat_columns = final_model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
    all_features = numeric_cols + list(cat_columns)
    feature_importance_df = pd.DataFrame({"Feature": all_features, "Importance": importance}).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# %%
# Save the best model
joblib.dump(final_model, "best_model.pkl")
print(f"✅ {best_model_name} saved as 'best_model.pkl'")
