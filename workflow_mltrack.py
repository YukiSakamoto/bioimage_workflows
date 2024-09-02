from pathlib import Path
import user_functions
import pathlib
#import mlflow
#from optuna.integration.mlflow import MLflowCallback
from optuna.integration import MLflowCallback
import task_mlflow_wrapper

#from prefect.runtime import flow_run, task_run
import os, shutil
import optuna
import random

import sys
import json

from typeguard import typechecked

generation_params = {
    "seed": 123,
    "interval": 0.033,
    "num_samples": 1,
    "num_frames": 1,
    "exposure_time": 0.033,
    "Nm": [100, 100, 100],
    "Dm": [0.222e-12, 0.032e-12, 0.008e-12],    # m^2 / sec
    "transmat": [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 1.0, 0.0]]
}

analysis_params = {
    "max_sigma": 4,
    "min_sigma": 1,
    "threshold": 50.0,
    "overlap": 0.5,
    "cutoff_distance": 2,
    "interval": 0.033,
}

evaluation_params = {
    "max_distance": 5.0,
    "transmat": [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 1.0, 0.0]
    ]
}

work_dir_root = "./optimization_task1/"

def generate_task_name():
    flow_name = flow_run.flow_name
    task_name = task_run.task_name

    parameters = task_run.parameters
    interval = parameters['param']['interval']
    Nm = parameters['param']['Nm']
    return f"{flow_name}_interval-{interval}_Nm-{Nm}"

#@task(name = "generate_image_series",task_run_name = generate_task_name)
@task_mlflow_wrapper.task_with_mlflow(arg_name_artifact_dir_after_exec = "image_dir")
def generate_image_series(param: dict, image_dir: Path):

    # Check if the image has already generated with same parameter
    json_path = image_dir / 'params.json'
    if os.path.exists(image_dir) and os.path.exists(json_path):
        with open(json_path, 'r', encoding = 'utf-8') as file:
            load_param = json.load(file)
            if load_param == param:
                print("generate image in {} skipped, since already generated with same parameters".format(image_dir), file = sys.stderr)
                return image_dir

    print("Will generate image in {}".format(image_dir), file = sys.stderr)
    os.makedirs(image_dir, exist_ok = True)
    artifacts, metrics = user_functions.generation1([], image_dir, param)
    with open(json_path, 'w', encoding = 'utf-8') as file:
        json.dump(param, file, ensure_ascii = False, indent = 4)
    print("generate image in {} done".format(image_dir), file = sys.stderr)
    return image_dir

#@task(name = "generate_multiple_image_series", task_run_name = generate_task_name)
@task_mlflow_wrapper.task_with_mlflow()
def generate_multiple_image_series(param: dict, image_root_dir: Path, num_series = 10):
    ret = []
    os.makedirs(image_root_dir, exist_ok = True)
    for i in range(num_series):
        image_dir = image_root_dir / f"series_{i}"
        json_path = image_dir / 'params.json'
        ret.append(image_dir)

        # Check if the image has already generated with same parameter
        if os.path.exists(image_dir) and os.path.exists(json_path):
            with open(json_path, 'r', encoding = 'utf-8') as file:
                load_param = json.load(file)
                if load_param == param:
                    print("generate {} in {} skipped, since already generated with same parameters".format(i, image_dir), file = sys.stderr)
                    continue

        print("Will generate {} in {}".format(i, image_dir), file = sys.stderr)
        os.makedirs(image_dir, exist_ok = True)
        artifacts, metrics = user_functions.generation1([], image_dir, param)

        with open(json_path, 'w', encoding = 'utf-8') as file:
            json.dump(param, file, ensure_ascii = False, indent = 4)
        print("generate {} in {} done".format(i, image_dir), file = sys.stderr)
    return ret

#@task(name = "evaluation_single_image", log_prints = True)
@task_mlflow_wrapper.task_with_mlflow(arg_name_artifact_dir_before_exec="image_dir_list")
def evaluation_single_image(image_dir_list: list[Path], optimized_params: dict):
    mean_norm_sum = 0.0
    for image_dir in image_dir_list:
        analysis1_dir = Path("./evalution_dir")
        if os.path.isdir(analysis1_dir):
            shutil.rmtree(analysis1_dir)
        os.makedirs(analysis1_dir, exist_ok = True)

        trial_analysis_params = analysis_params.copy()
        trial_analysis_params.update(optimized_params)
        a, analysis1_metrics = user_functions.analysis1([image_dir], analysis1_dir, trial_analysis_params)
        
        # Do Evaluation
        evaluation1_dir = Path("./evaluation1_dir/")
        if os.path.isdir(evaluation1_dir):
            shutil.rmtree(evaluation1_dir)
        os.makedirs(evaluation1_dir, exist_ok = True)
        c, evaluation1_metrics = user_functions.evaluation1([image_dir, analysis1_dir], evaluation1_dir, evaluation_params)

        x_mean = evaluation1_metrics["x_mean"]
        y_mean = evaluation1_metrics["y_mean"]
        mean_norm1 = (x_mean)**2 + (y_mean)**2
        print(mean_norm1, file=sys.stderr)
        
        mean_norm_sum += mean_norm1

    n_images = len(image_dir_list)
    result = mean_norm_sum / n_images
    print("Evaluation: mean_norm_sum/n_images = {}".format(mean_norm_sum / n_images), file=sys.stderr)
    return result


#@task(name = "opt_single_image", log_prints = True)
@task_mlflow_wrapper.task_with_mlflow(
        #arg_name_artifact_dir_before_exec="image_dir_list", 
        #pathobj_log_artifacts = True, 
        #dirname_of_artifacts_after_exec="ok_after", 
        #dirname_of_artifacts_before_exec="ok_before"
)
@typechecked
def optimize_single_image(image_dir_list: list[Path], analysis_param: dict, n_trials: int = 10):
    artifact_dir2 = Path('hoge')

    mlflc = MLflowCallback(tracking_uri = "10.5.1.218", metric_name = "optimize_single_image", mlflow_kwargs={"nested": True})

    @mlflc.track_in_mlflow()
    def _objective(trial):
        # 1. Prepare the Parameter
        trial_analysis_params = analysis_params.copy()
        trial_analysis_params["threshold"] = trial.suggest_float("threshold", 48, 52)
        trial_analysis_params["overlap"] = trial.suggest_float("overlap", 0.4, 0.6)
        trial_analysis_params["max_sigma"] = trial.suggest_float("max_sigma", 3, 5)
        trial_analysis_params["min_sigma"] = trial.suggest_float("min_sigma", 0, 2)

        mean_norm_sum = 0.
        
        # 2. Do Analysis1
        for image_dir in image_dir_list:
            analysis1_dir = Path("./opt_analysis1_dir/")
            if os.path.isdir(analysis1_dir):
                shutil.rmtree(analysis1_dir)
            os.makedirs(analysis1_dir, exist_ok = True)
            a, analysis1_metrics = user_functions.analysis1([image_dir], analysis1_dir, trial_analysis_params)
            
            # 3. Do Evaluation
            evaluation1_dir = Path("./opt_evaluation1_dir/")
            if os.path.isdir(evaluation1_dir):
                shutil.rmtree(evaluation1_dir)
            os.makedirs(evaluation1_dir, exist_ok = True)
            c, evaluation1_metrics = user_functions.evaluation1([image_dir, analysis1_dir], evaluation1_dir, evaluation_params)

            x_mean = evaluation1_metrics["x_mean"]
            y_mean = evaluation1_metrics["y_mean"]
            mean_norm1 = (x_mean)**2 + (y_mean)**2
            
            mean_norm_sum += mean_norm1

        n_images = len(image_dir_list)
        print("Trial #{}: mean_norm_sum/n_images = {}".format(trial.number, mean_norm_sum / n_images), file=sys.stderr)
        return mean_norm_sum / n_images

    study = optuna.create_study(load_if_exists=True, sampler=optuna.samplers.CmaEsSampler(), study_name = "optimize_single_image")
    study.optimize(_objective, n_trials = n_trials, callbacks = [mlflc])
    print("best params: {}".format(study.best_params), file = sys.stderr)
    print("best objective: {}".format(study.best_value), file = sys.stderr)
    return study.best_params

@task_mlflow_wrapper.flow
def run_flow():
    image_dir = Path("./save_image_dir_aaa/")
    task_mlflow_wrapper.set_mlflow_server_uri("10.5.1.218")
    task_mlflow_wrapper.set_mlflow_server_port("7777")
    print(task_mlflow_wrapper.get_mlflow_server_uri())

    #test_image_dir_list = generate_multiple_image_series(generation_params, image_dir / "test", 3)
    #train_image_dir_list = generate_multiple_image_series(generation_params, image_dir / "train", 7)
    test_image_dir_list = []
    train_image_dir_list = []
    for i in range(7):
        param = generation_params.copy()
        param['seed'] = i
        d = generate_image_series(param, image_dir/"train"/f"series_{i}")
        train_image_dir_list.append(d)

    for i in range(3):
        param = generation_params.copy()
        param['seed'] = i
        d = generate_image_series(param, image_dir/"test"/f"series_{i}")
        test_image_dir_list.append(d)

    optimized_params = optimize_single_image(train_image_dir_list,analysis_params, 1)
    
    # Evaluation
    eval_result = evaluation_single_image(test_image_dir_list, optimized_params)
    print("Evaluation Result: {}".format(eval_result) )

    regenerate_image_dir = Path("./regenerate_image_dir/")

    #merge parameters
    generation_param2 = generation_params.copy()
    generation_param2.update(optimized_params)
    generate_image_series(generation_param2, regenerate_image_dir)

if __name__ == "__main__":
    run_flow()
