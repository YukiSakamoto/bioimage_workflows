#!/usr/bin/python
# Setup MLflow server
#  cd work/bioimage_workflows/
#  source ~/venv-ecell/bin/activate
#  mlflow server --host 0.0.0.0
# Start Optuna
#  cd work/bioimage_workflows/
#  source ~/venv-ecell/bin/activate
# Run Sample
#  python generation_analysis.py
# 
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
from user_functions import generation1, analysis1, evaluation1
from pathlib import Path

mlflc = MLflowCallback(
    tracking_uri="http://127.0.0.1:5000",
    metric_name="sum of square x_mean and y_mean",
)

# generation data pass as global variable
# Add generation to mlflow


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

@mlflc.track_in_mlflow()
def objective(trial):
    # variables for analysis.
    global analysis_params, evaluation_params
    # print()
    # print(dir(trial))
    # exit()
    # call analysis
    # input
    generation_output=Path('./outputs_generation')
    # analysis output
    # create new dir, use trial.number
    analysis_output=Path('./outputs_analysis_run/'+str(trial.number))
    analysis_output.mkdir(parents=True, exist_ok=True)
    overlap = trial.suggest_float("overlap", 0, 1)
    threshold = trial.suggest_float("threshold", 30, 70)

    with mlflow.start_run(nested=True, run_name="analysis_"+str(trial.number)) as run_analysis:
        # set param
        analysis_params["overlap"]=overlap
        analysis_params["threshold"]=threshold
        
        a,b = analysis1([generation_output], analysis_output, analysis_params)
        num_spots = b["num_spots"]
        # Set param
        mlflow.log_param("overlap", overlap)
        mlflow.log_param("threshold", threshold)

        # Set metric
        mlflow.log_metric("num_spots", num_spots)
        # End analysis run
        # mlflow.end_run()
    
    ## call generation
    # output
    evaluation_output=Path('./outputs_evaluation_run/'+str(trial.number))
    evaluation_output.mkdir(parents=True, exist_ok=True)

    # max_distance = trial.suggest_float("max_distance", 0, 1)
    max_distance = evaluation_params["max_distance"]

    mlflow.log_param("overlap", overlap)
    mlflow.log_param("threshold", threshold)

    mlflow.log_param("max_distance", max_distance)
    
    evaluation_params["max_distance"] = max_distance
    c,d = evaluation1([generation_output,analysis_output], evaluation_output, evaluation_params)

    x_mean = d["x_mean"]
    y_mean = d["y_mean"]

    mlflow.log_metric("x_mean", x_mean)
    mlflow.log_metric("y_mean", y_mean)

    # TODO: not hard code 600
    result = (x_mean)**2+(y_mean)**2
    return result



def main():
    global generation_params,analysis_params
    # Setup for Optuna MLFlow
    # generation_output=Path('./outputs_generation')
    # generation1([], generation_output, generation_params)

    # analysis_output=Path('./outputs_analysis')
    
    # a,b = analysis1([generation_output], analysis_output, analysis_params)
    # print(a)
    # print(b)
    # Execute Optuna MLFlow
    study = optuna.create_study(study_name="test_x_y_mean_4")
    study.optimize(objective, n_trials=10, callbacks=[mlflc])
    print(study.best_params)

if __name__ == "__main__":
    main()