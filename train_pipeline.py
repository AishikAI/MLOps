import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc
)

# Set the tracking URI
mlflow.set_tracking_uri("file:///C:/Users/rajes/OneDrive/Desktop/Praxis/Predictive_maintenance/mlruns")

# Load Updated Data for Retraining
df = pd.read_csv("updated_training_data.csv")  # Use the latest combined dataset

# Define Features and Target
FEATURES = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
            "Torque [Nm]", "Tool wear [min]", "Type"]
TARGET = "Machine failure"

X = df[FEATURES].copy()
y = df[TARGET]

# Fix for XGBoost & MLflow: Remove special characters from feature names
X.columns = [col.replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]

# Identify Numerical & Categorical Features
num_features = ["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"]
cat_features = ["Type"]

# Handle Missing Values Separately for Numeric & Categorical Features
X.loc[:, num_features] = X[num_features].apply(pd.to_numeric, errors="coerce")
X.loc[:, num_features] = X[num_features].fillna(X[num_features].median())
X.loc[:, cat_features] = X[cat_features].fillna("Unknown")

# Create Preprocessing Pipeline
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Initialize Models with Pipelines
models = {
    "RandomForest": Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
    "DecisionTree": Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(random_state=42))]),
    "CatBoost": Pipeline([("preprocessor", preprocessor), ("model", CatBoostClassifier(verbose=0))]),
    "LogisticRegression": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())]),
    "XGBoost": Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))]),
    "GradientBoosting": Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    "SVM": Pipeline([("preprocessor", preprocessor), ("model", SVC(probability=True))])
}

# Define Model Evaluation Function
best_model = None
best_auc = 0
best_model_name = ""

def evaluate_and_log_model(model_name, model):
    """Train, evaluate, and log model performance in MLflow."""
    global best_model, best_auc, best_model_name
    
    with mlflow.start_run(run_name=model_name):  # Create a separate run for each model
        print(f"🚀 Training {model_name}...")
        
        # Train Model
        model.fit(X_train, y_train)

        # Make Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["model"], "predict_proba") else None

        # Compute Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=1),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        }
        if y_prob is not None:
            auc_score = roc_auc_score(y_test, y_prob)
            metrics["AUC"] = auc_score

            # Update Best Model If Applicable
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_model_name = model_name

        # Log Metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log Precision-Recall Curve
        if y_prob is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_vals, precision_vals)
            mlflow.log_metric("PR AUC", pr_auc)

            plt.figure(figsize=(6, 6))
            plt.plot(recall_vals, precision_vals, marker='.', label=f"PR Curve (AUC={pr_auc:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.legend()
            plt.grid()

            pr_curve_path = f"pr_curve_{model_name}.png"
            plt.savefig(pr_curve_path)
            plt.close()

            mlflow.log_artifact(pr_curve_path)

        # Log Model
        model_filename = f"{model_name}_model.pkl"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)

        print(f"✅ {model_name} logged successfully in MLflow!\n")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set Experiment Name
mlflow.set_experiment("My_Predictive_Maintenance_Experiment")

# Train and Log All Models
for model_name, model in models.items():
    evaluate_and_log_model(model_name, model)

# Register the Best Model in MLflow
if best_model:
    print(f"🏆 Best Model: {best_model_name} with AUC: {best_auc:.4f}")
    with mlflow.start_run(run_name="Best_Model_Registration"):
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
        print(f"📋 Registered Best Model: {best_model_name} in MLflow")
