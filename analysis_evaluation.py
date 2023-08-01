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
from user_functions import generation1, analysis1, evaluation1, analysis2
from pathlib import Path
from mlflow import MlflowClient
from bioimage_workflow.utils import check_if_already_ran
import sys

## hydra
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils

mlflc = MLflowCallback(
    #tracking_uri="http://127.0.0.1:5000",
    tracking_uri="http://127.0.0.1:8888",
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

exposure_time=0
num_samples=1

trial_number=0
# @mlflc.track_in_mlflow()
# def objective(trial):
@hydra.main(config_path="bioimage_workflow/conf", config_name="config")
def objective(cfg: DictConfig):
    global trial_number
    trial_number = trial_number + 1
    # variables for analysis.
    # global generation_params, analysis_params, evaluation_params, exposure_time, num_samples
    
    # generation_params["exposure_time"] = exposure_time
    # generation_params["num_samples"] = num_samples

    #client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri="http://127.0.0.1:8888")
    # print("--- cleint")
    # print(dir(client))
    # from mlflow.tracking.fluent import _get_experiment_id
    # experiment_id = _get_experiment_id()
    # print(experiment_id)
    # all_run_infos = reversed(client.list_run_infos(experiment_id))
    # print(all_run_infos)
    # for ru in all_run_infos:
    #     print(ru)


    ## NOTE: ignore_tags is False, so name is not checked, it checks only params
    local_artifacts_dir="./artifacts/"
    # generation_params["exposure_time"] = cfg.experiment.generation.params.exposure_time
    # generation_params["num_samples"] = cfg.experiment.generation.params.num_samples
    # run=check_if_already_ran(client,"", generation_params)
    run=check_if_already_ran(client,"", cfg.experiment.generation.params)
    # print("hydra param")
    # for key, value in cfg.experiment.generation.params.items():
    #     print(key, value)
    # print("generation_params param")
    # for key, value in generation_params.items():
    #     print(key, value)

    # sys.exit()

    if run is None:
        # Exec generation
        with mlflow.start_run(nested=True, run_name="generation_"+str(trial_number)) as run_analysis:
            generation_output=Path(local_artifacts_dir+'/'+run_analysis.info.run_id)
            print(str(generation_output))
            if not generation_output.exists():
                generation_output.mkdir(parents=True, exist_ok=True)

            artifacts,metrics = generation1([], generation_output, cfg.experiment.generation.params)
            for key, value in cfg.experiment.generation.params.items():
                mlflow.log_param(key, value)
            #
            mlflow.log_artifacts(str(generation_output))
    else:
        #
        generation_output=Path(local_artifacts_dir+'/'+run.info.run_id)
        if not generation_output.exists():
            generation_output.mkdir(parents=True, exist_ok=True)
            client.download_artifacts(run_id=run.info.run_id, path="", dst_path=str(generation_output))

    # sys.exit()

    # print()
    # print(dir(trial))
    # exit()
    # call analysis
    # input
    # generation_output=Path('./outputs_generation')
    # analysis output
    # create new dir, use trial_number
    analysis_output=Path('./outputs_analysis_run/'+str(trial_number))
    analysis_output.mkdir(parents=True, exist_ok=True)
    # overlap = trial.suggest_float("overlap", 0, 1)
    # threshold = trial.suggest_float("threshold", 30, 70)

    with mlflow.start_run(nested=True, run_name="evaluation_"+str(trial_number)) as run_evaluation:

        with mlflow.start_run(nested=True, run_name="analysis_"+str(trial_number)) as run_analysis:
            # set param
            # analysis_params["overlap"]=overlap
            # analysis_params["threshold"]=threshold
            
            a,b = analysis1([generation_output], analysis_output, cfg.experiment.analysis.params)
            analysis2_artifacts, analysis2_metrics = analysis2([generation_output], analysis_output, cfg.experiment.analysis.params)
            num_spots = b["num_spots"]
            # Set param
            mlflow.log_param("overlap", cfg.experiment.analysis.params.overlap)
            mlflow.log_param("threshold", cfg.experiment.analysis.params.threshold)

            # Set metric
            mlflow.log_metric("num_spots", num_spots)
            # End analysis run
            # mlflow.end_run()
            mlflow.log_metric("observation_count", analysis2_metrics["observation_count"])
            mlflow.log_metric("observation_length", analysis2_metrics["observation_length"])
            mlflow.log_artifacts(str(analysis_output))
    
        ## call evaluation
        # output
        evaluation_output=Path('./outputs_evaluation_run/'+str(trial_number))
        evaluation_output.mkdir(parents=True, exist_ok=True)

        # max_distance = trial.suggest_float("max_distance", 0, 1)
        max_distance = evaluation_params["max_distance"]

        mlflow.log_param("overlap", cfg.experiment.analysis.params.overlap)
        mlflow.log_param("threshold", cfg.experiment.analysis.params.threshold)

        mlflow.log_param("max_distance", cfg.experiment.evaluation.params.max_distance)
        
        # evaluation_params["max_distance"] = max_distance
        c,d = evaluation1([generation_output,analysis_output], evaluation_output, cfg.experiment.evaluation.params)

        x_mean = d["x_mean"]
        y_mean = d["y_mean"]

        mlflow.log_metric("x_mean", x_mean)
        mlflow.log_metric("y_mean", y_mean)

        # TODO: not hard code 600
        result = (x_mean)**2+(y_mean)**2
    return result



def main():
    global generation_params,analysis_params, exposure_time, num_samples
    # Setup for Optuna MLFlow
    # generation_output=Path('./outputs_generation')
    # generation1([], generation_output, generation_params)

    # analysis_output=Path('./outputs_analysis')
    
    # a,b = analysis1([generation_output], analysis_output, analysis_params)
    # print(a)
    # print(b)
    # Execute Optuna MLFlow
    # study = optuna.create_study(storage="sqlite:///example2.10.1.db", study_name="test_x_y_mean_storage_7_2.10.1", load_if_exists=True, sampler=optuna.samplers.CmaEsSampler())
    objective()

    # print("--- study")
    # print(dir(study))
    # for j in range(1,6):
    #     num_samples = j
    #     exposure_time=0
    #     for i in range(5):
    #         exposure_time=exposure_time+0.033
    #         study.optimize(objective, n_trials=5, callbacks=[mlflc])
    #         print(study.best_params)

    # ここで別のexperimentにoptuna記録するという方法もある。

# print("--- mlflc")
# print(dir(mlflc))

if __name__ == "__main__":
    main()
