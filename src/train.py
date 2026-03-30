import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Load data
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow tracking 
mlflow.set_experiment("Development")

with mlflow.start_run(run_name="RandomForest_Iris_v1"):

    # ① Tag
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "Iris")
    mlflow.set_tag("developer", "Mossad")

    # ② Params
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # ③ Train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # ④ Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # ⑤ Metrics
    mlflow.log_metric("accuracy", accuracy)

    # ⑥ Confusion Matrix → artifact
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("outputs/confusion_matrix.png")

    # ⑦ Log model 
    mlflow.sklearn.log_model(model, "model")

print("Training + MLflow logging completed successfully.")