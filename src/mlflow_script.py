import mlflow
from train import *

#  MLflow tracking 
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
