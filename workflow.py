from pathlib import Path
#from user_functions import generation1, analysis1, evaluation1, analysis2, evaluation2
#from user_functions import generation1, analysis1, evaluation1
import user_functions
import pathlib

from prefect import task, flow
from prefect.runtime import flow_run, task_run
import os, shutil
import optuna

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

@task(name = "generate_image",task_run_name = generate_task_name)
def generate_image(param: dict):
    image_dir = Path("./save_image_dir/")
    os.makedirs(image_dir, exist_ok = True)
    artifacts, metrics = generation1([], image_dir, param)
    return image_dir


@task(name = "opt_single_image", )
def optimize_single_image(image_dir: Path, analysis_param: dict, n_trials: int = 10):
    artifact_dir2 = Path('hoge')
    def _objective(trial):
        # 1. Prepare the Parameter
        trial_analysis_params = analysis_params.copy()
        trial_analysis_params["threshold"] = trial.suggest_float("threshold", 10, 100)
        trial_analysis_params["overlap"] = trial.suggest_float("overlap", 0.1, 1.0)
        trial_analysis_params["max_sigma"] = trial.suggest_float("max_sigma", 4, 10)
        trial_analysis_params["min_sigma"] = trial.suggest_float("min_sigma", 0, 2)
        
        # 2. Do Analysis1
        analysis1_dir = Path("./analysis1_dir/")
        if os.path.isdir(analysis1_dir):
            shutil.rmtree(analysis1_dir)
        os.makedirs(analysis1_dir, exist_ok = True)
        a, analysis1_metrics = user_functions.analysis1([image_dir], analysis1_dir, trial_analysis_params)
        
        # 3. Do Evaluation
        evaluation1_dir = Path("./evaluation1_dir/")
        if os.path.isdir(evaluation1_dir):
            shutil.rmtree(evaluation1_dir)
        os.makedirs(evaluation1_dir, exist_ok = True)
        c, evaluation1_metrics = user_functions.evaluation1([image_dir, analysis1_dir], evaluation1_dir, evaluation_params)

        x_mean = evaluation1_metrics["x_mean"]
        y_mean = evaluation1_metrics["y_mean"]
        mean_norm1 = (x_mean)**2 + (y_mean)**2
        return mean_norm1

    study = optuna.create_study(load_if_exists=True, sampler=optuna.samplers.CmaEsSampler())
    study.optimize(_objective, n_trials = n_trials)
    return study.best_params

@flow
def run_flow():
    #image_dir = generate_image(generation_params)
    image_dir = Path("./save_image_dir/")
    params = optimize_single_image(image_dir,analysis_params, 2)
    print(params)

if __name__ == "__main__":
    run_flow()
