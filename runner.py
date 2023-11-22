import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback

#==================================================
# Parameters
#==================================================
generation_params = {
    "seed": 123,
    "interval": 0.033,
    "num_samples": 1,
    "num_frames": 2,
    "exposure_time": 0.033,
    "Nm": [100, 100, 100],
    "Dm": [0.222e-12, 0.032e-12, 0.008e-12],
    "transmat": [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 1.0, 0.0]]
}

analysis_params = {
    "max_sigma": 4,
    "min_sigma": 1,
    "threshold": 50.0,
    "overlap": 0.5
}

evaluation_params = {
    "max_distance": 5.0
}

exposure_time=0
num_samples=1

trial_number=0



mlflc = MLflowCallback(
    #tracking_uri="http://127.0.0.1:5000",
    tracking_uri="http://127.0.0.1:8888",
    metric_name="sum of square x_mean and y_mean",
)

@mlflc.track_in_mlflow()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)

    # The parameters get from trial.sugget_XXX(XXX=float, int, ...) are 
    # automatically logged as parameter.
    # Simirally, the return value of the objective() function is also automatically
    # logged with the name of metric_name set in the MLflowCallback. 
    mlflow.log_param("power", 2)
    mlflow.log_metric("base of metric", x-2)
    return (x-2) ** 2

if __name__ == '__main__':
    study = optuna.create_study(
            storage="sqlite:///example2.10.1.db", 
            study_name="test_x_y_mean_storage_7_2.10.1", 
            load_if_exists=True, 
            sampler=optuna.samplers.CmaEsSampler())

    study.optimize(objective, n_trials=5, callbacks=[mlflc])
