# Finds the best model and saves its URI to a file
import mlflow

client = mlflow.tracking.MlflowClient()

exp = client.get_experiment_by_name('Development')
runs = client.search_runs(
    exp.experiment_id,
    order_by=['metrics.accuracy DESC'],
    max_results=1)
with open('best_model_uri.txt', 'w') as f:
    f.write(runs[0].info.artifact_uri + '/model')   
