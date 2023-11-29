import optuna
import mlflow
from pathlib import Path
from optuna.integration.mlflow import MLflowCallback
from user_functions import generation1, analysis1, evaluation1

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
    tracking_uri="http://127.0.0.1:7777",
    metric_name="sum of square x_mean and y_mean",
)

def generate_objective_function(generation_output_path):
    # This function returns the closure that hold the generation_output_path.

    @mlflc.track_in_mlflow()
    def _objective(trial):
        generation_output = generation_output_path  
        mlflow.log_param("generation_output_path", generation_output)

        # The parameters get from trial.suggest_XXX(XXX=float, int, ...) are automatically changed to optimize.
        analysis_params_mod = analysis_params.copy()
        analysis_params_mod["threshold"] = trial.suggest_float("threshold", 10, 100)
        analysis_params_mod["overlap"] = trial.suggest_float("overlap", 0.1, 1.0)

        analysis_output=Path('./outputs_analysis_run/'+str(trial_number))
        analysis_output.mkdir(parents=True, exist_ok=True)
        a,b = analysis1([generation_output], analysis_output, analysis_params_mod)

        evaluation_output=Path('./outputs_evaluation_run/'+str(trial_number))
        evaluation_output.mkdir(parents=True, exist_ok=True)

        c,d = evaluation1([generation_output,analysis_output], evaluation_output, evaluation_params)

        x_mean = d["x_mean"]
        y_mean = d["y_mean"]
        mlflow.log_metric("x_mean", x_mean)
        mlflow.log_metric("y_mean", y_mean)
        result = (x_mean)**2+(y_mean)**2
        return result

    return _objective


if __name__ == '__main__':
    skip_generation = False
    generation_output = Path('./test2/abcd')
    if skip_generation != True:
        if not generation_output.exists():
            generation_output.mkdir(parents = True, exist_ok = True)
        artifacts,metrics = generation1([], generation_output, generation_params)


    study = optuna.create_study(
            storage="sqlite:///example2.10.1.db", 
            study_name="test_x_y_mean_storage_7_2.10.1", 
            load_if_exists=True, 
            sampler=optuna.samplers.CmaEsSampler())

    objective = generate_objective_function(generation_output)
    study.optimize(objective, n_trials=10, callbacks=[mlflc])




