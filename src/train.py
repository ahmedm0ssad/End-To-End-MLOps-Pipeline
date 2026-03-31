# ============================================================
# train.py — ML Training Script with MLflow Tracking
# Trains a RandomForest on Iris dataset
# Logs params, metrics, and artifacts to DagsHub MLflow
# ============================================================

import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ── DagsHub Credentials ──────────────────────────────────────
# Read from environment variables (set as GitHub Secrets in CI/CD)
# Locally: set these in your terminal or .env file
# These are used by MLflow to authenticate with DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

# ── MLflow Tracking URI ───────────────────────────────────────
# Points to DagsHub remote MLflow server
# Falls back to DagsHub URL if env variable not set
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/ahmedm0ssad/MLOps-Devlopment.mlflow",
    )
)

# ── Load Data ─────────────────────────────────────────────────
# Iris dataset: 150 samples, 4 features, 3 classes
X, y = load_iris(return_X_y=True)

# ── Split Data ────────────────────────────────────────────────
# 80% training, 20% testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Create Output Folder ──────────────────────────────────────
os.makedirs("outputs", exist_ok=True)


# ── MLflow Tracking ───────────────────────────────────────────
# Set experiment name — groups all runs under one project
mlflow.set_experiment("Development")

# Start a new run — everything inside is tracked
with mlflow.start_run(run_name="RandomForest_Iris_v1"):

    # ① Tags — searchable metadata about this run
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("developer", "Mossad")

    # ② Params — input config logged ONCE per run
    mlflow.log_param("n_estimators", 100)  # number of trees
    mlflow.log_param("test_size", 0.2)     # train/test split ratio
    mlflow.log_param("random_state", 42)   # reproducibility seed

    # ③ Train — fit the model on training data
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # ④ Evaluate — run predictions on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # ⑤ Metrics — output results logged to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # ⑥ Confusion Matrix — visualize prediction performance
    # Saved as PNG then uploaded to MLflow as artifact
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("outputs/confusion_matrix.png")

    # ⑦ Log Model — save model in MLflow standard format (flavor)
    # mlflow.sklearn handles serialization automatically
    # replaces joblib.dump() — more portable across platforms
    mlflow.sklearn.log_model(model, "model")

    # ── Save Model URI ────────────────────────────────────────
    # Get the MLflow artifact URI for this run's model
    # Save to model_info.txt so the CI/CD deploy job can use it
    model_uri = mlflow.get_artifact_uri("model")
    with open("model_info.txt", "w") as f:
        f.write(model_uri)

print("Training + MLflow logging completed successfully.")