import optuna
import mlflow
from pathlib import Path
from optuna.integration.mlflow import MLflowCallback
from user_functions import generation1, analysis1, evaluation1, analysis2, evaluation2
import sys
import numpy as np
import shutil
import argparse
import json
import itertools

storage="sqlite:///example2.10.1.db" 
study_name = "parameter_optimization"
study_name2 = "parameter_optimization2"
image_dir = Path("./test_image")
#artifact_dir = Path("./outputs/")

#==================================================
# Parameters
#==================================================
generation_params = {
    "seed": 123,
    "interval": 0.033,
    "num_samples": 1,
    "num_frames": 10,
    "exposure_time": 0.033,
    "Nm": [100, 100, 100],
    "Dm": [0.222e-12, 0.032e-12, 0.008e-12],    # m^2 / sec
    "transmat": [
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.2],
        [0.0, 1.0, 0.0]]
}

#analysis_params = {
#    "max_sigma": 4,
#    "min_sigma": 1,
#    "threshold": 50.0,
#    "overlap": 0.5,
#    "cutoff_distance": 2,
#    "interval": 0.033,
#}

analysis_params = {
    "max_sigma": 0,
    "min_sigma": 0,
    "threshold": 0.0,
    "overlap": 0.0,
    "cutoff_distance": 0,
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

exposure_time=0
num_samples=1

trial_number=0



mlflc = MLflowCallback(
    #tracking_uri="http://127.0.0.1:5000",
    #tracking_uri="http://127.0.0.1:7777",
    #tracking_uri="http://10.5.1.218:7777",
    metric_name="analysis1_and_2",
)

mlflc2 = MLflowCallback(
    metric_name="analysis1",
)

# This function is for the serialization of numpy object to json.
def default(o):
    print(f"{type(o)=}")
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.bool_):
        return bool(o)
    elif isinstance(o, np.ndarray):
        return list(o)
    raise TypeError(repr(o) + " is not JSON serializable")

def generate_objective_function(generation_output_path, generation_params, artifact_dir, eval_weight = {"transmat_rss": 0.5, "startprob": 0.25, "D": 0.25}):
    # This function returns the closure that hold the generation_output_path.

    @mlflc.track_in_mlflow()
    def _objective(trial):
        print("enter objective")
        generation_output = generation_output_path  
        mlflow.log_param("generation_output_path", generation_output)   #XXX should be use log_artifact ???
        mlflow.log_param("generation_params", generation_params)

        # The parameters get from trial.suggest_XXX(XXX=float, int, ...) are automatically changed to optimize.
        analysis_params_mod = analysis_params.copy()
        analysis_params_mod["threshold"] = trial.suggest_float("threshold", 10, 100)
        analysis_params_mod["overlap"] = trial.suggest_float("overlap", 0.1, 1.0)
        analysis_params_mod["max_sigma"] = trial.suggest_float("max_sigma", 4, 10)
        analysis_params_mod["min_sigma"] = trial.suggest_float("min_sigma", 0, 2)
        analysis_params_mod["cutoff_distance"] = trial.suggest_float("cutoff_distance", 1, 10)  #XXX obj2

        all_log_dict = dict()
        all_log_dict["generation_params"] = generation_params

        analysis1_output = artifact_dir / str(trial.number) / 'analysis1/'
        analysis1_output.mkdir(parents=True, exist_ok=True)
        a, analysis1_metrics = analysis1([generation_output], analysis1_output, analysis_params_mod)
        all_log_dict["analysis1_metrics"] = analysis1_metrics

        evaluation_output = artifact_dir / str(trial.number) / 'evaluation1/'
        evaluation_output.mkdir(parents=True, exist_ok=True)
        c, evaluation1_metrics = evaluation1([generation_output,analysis1_output], evaluation_output, evaluation_params)

        x_mean = evaluation1_metrics["x_mean"]
        y_mean = evaluation1_metrics["y_mean"]
        mean_norm2 = (x_mean)**2+(y_mean)**2
        mlflow.log_metric("x_mean", x_mean)
        mlflow.log_metric("y_mean", y_mean)
        mlflow.log_metric("mean_norm2", mean_norm2)
        all_log_dict["evaluation1_metrics"] = evaluation1_metrics

        # analysis2
        analysis2_output = artifact_dir / str(trial.number) / 'analysis2/'
        analysis2_output.mkdir(parents=True, exist_ok=True)
        analysis2_artifacts, analysis2_metrics = analysis2([generation_output], analysis1_output, analysis2_output, analysis_params_mod )
        all_log_dict["analysis2_metrics"] = analysis2_metrics

        evaluation2_output = artifact_dir / str(trial.number) / 'evaluation2/'
        evaluation2_output.mkdir(parents=True, exist_ok=True)
        _, evaluation2_metrics = evaluation2([generation_output, analysis2_output], evaluation_output, evaluation_params)
        all_log_dict["evaluation2_metrics"] = evaluation2_metrics

        # log all the metrics
        jsonpath = artifact_dir / str(trial.number) / 'metrics.json'
        with open(jsonpath, "w") as f:
            json.dump(all_log_dict, f, default=default)
        mlflow.log_artifact(jsonpath)

        start_ratio = np.array(generation_params["Nm"]) / sum(generation_params["Nm"])
        startprob_rss = np.sum(np.square(analysis2_metrics["startprob"] - start_ratio) )
        analysis_Dm = np.sort(np.array(list(map(lambda x: x[0], analysis2_metrics["D"]) ) ))
        generation_Dm = np.sort(np.array(generation_params["Dm"]))
        dm_rss = np.sum(np.square(analysis_Dm - generation_Dm) ) 
        transmat_rss = evaluation2_metrics["transmat_rss"]

        objective_parameters = {
            "startprob": startprob_rss,
            "D": dm_rss,
            "transmat_rss": transmat_rss
        }
        #result = 0.
        #for k in eval_weight.keys():
        #    print("{} : {} : {}".format(k, eval_weight[k], objective_parameters[k]))
        #    result += objective_parameters[k] * eval_weight[k]
        #mlflow.log_artifacts(artifact_dir / str(trial.number))

        r = all_log_dict["evaluation1_metrics"]["r"]
        miss_count = all_log_dict["evaluation1_metrics"]["miss_count"]
        missing = all_log_dict["evaluation1_metrics"]["missing"]
        result = abs(1-r) + miss_count + missing
        return result

    return _objective

def generate_objective_function2(generation_output_path, generation_params, artifact_dir, eval_weight = {"transmat_rss": 0.5, "startprob": 0.25, "D": 0.25}):
    # This function returns the closure that hold the generation_output_path.

    @mlflc2.track_in_mlflow()
    def _objective(trial):
        print("enter objective")
        generation_output = generation_output_path  
        mlflow.log_param("generation_output_path", generation_output)   #XXX should be use log_artifact ???
        mlflow.log_param("generation_params", generation_params)

        # The parameters get from trial.suggest_XXX(XXX=float, int, ...) are automatically changed to optimize.
        analysis_params_mod = analysis_params.copy()
        analysis_params_mod["threshold"] = trial.suggest_float("threshold", 10, 100)
        analysis_params_mod["overlap"] = trial.suggest_float("overlap", 0.1, 1.0)
        analysis_params_mod["max_sigma"] = trial.suggest_float("max_sigma", 4, 10)
        analysis_params_mod["min_sigma"] = trial.suggest_float("min_sigma", 0, 2)
        #analysis_params_mod["cutoff_distance"] = trial.suggest_float("cutoff_distance", 1, 10)  #XXX obj2

        all_log_dict = dict()
        all_log_dict["generation_params"] = generation_params

        analysis1_output = artifact_dir / str(trial.number) / 'analysis1/'
        analysis1_output.mkdir(parents=True, exist_ok=True)
        a, analysis1_metrics = analysis1([generation_output], analysis1_output, analysis_params_mod)
        all_log_dict["analysis1_metrics"] = analysis1_metrics

        evaluation_output = artifact_dir / str(trial.number) / 'evaluation1/'
        evaluation_output.mkdir(parents=True, exist_ok=True)
        c, evaluation1_metrics = evaluation1([generation_output,analysis1_output], evaluation_output, evaluation_params)

        x_mean = evaluation1_metrics["x_mean"]
        y_mean = evaluation1_metrics["y_mean"]
        mean_norm2 = (x_mean)**2+(y_mean)**2
        mlflow.log_metric("x_mean", x_mean)
        mlflow.log_metric("y_mean", y_mean)
        mlflow.log_metric("mean_norm2", mean_norm2)
        all_log_dict["evaluation1_metrics"] = evaluation1_metrics

        # log all the metrics
        jsonpath = artifact_dir / str(trial.number) / 'metrics.json'
        with open(jsonpath, "w") as f:
            json.dump(all_log_dict, f, default=default)
        mlflow.log_artifact(jsonpath)

        r = all_log_dict["evaluation1_metrics"]["r"]
        miss_count = all_log_dict["evaluation1_metrics"]["miss_count"]
        missing = all_log_dict["evaluation1_metrics"]["missing"]
        result = abs(1-r) + miss_count + missing
        return result

    return _objective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', type =int, default = 10, help = 'the number of the optimization steps')
    parser.add_argument('-n2', type =int, default = 10, help = 'the number of the optimization steps')
    parser.add_argument('--clear', action = "store_true", default = False, help = 'Clear the previous optimization logs')

    args = parser.parse_args()

    if args.clear == True:
        optuna.delete_study(storage = storage, study_name = study_name)
        shutil.rmtree(artifact_dir)
        shutil.rmtree("mlruns")
        exit()

    nsteps1 = args.n1
    nsteps2 = args.n2
    #========================================
    # Generation Image
    #========================================
    generation_output = image_dir
    if generation_output.exists() and (generation_output / "config.yaml").exists():
        print("Generation skip",file = sys.stderr )
    else:
        if not generation_output.exists():
            generation_output.mkdir(parents = True, exist_ok = True)
        print("Generation start",file = sys.stderr )
        artifacts,metrics = generation1([], generation_output, generation_params)
        print("Generation done",file = sys.stderr )


    #========================================
    # Optimize with analysis1 + analysis2
    #========================================
    study = optuna.create_study(
            storage = storage,  study_name = study_name, 
            load_if_exists=True, sampler=optuna.samplers.CmaEsSampler())

    artifact_dir = Path("./outputs/")
    objective = generate_objective_function(generation_output, generation_params, artifact_dir)
    study.optimize(objective, n_trials = nsteps1, callbacks=[mlflc])

    analysis_best_trial_number = study.best_trial.number
    print("analysis1+2 optimization done. {} is the best trial number".format(analysis_best_trial_number))
    print(study.best_params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html('study.html')

    #========================================
    # Optimize with ONLY analysis1 
    #========================================
    study2 = optuna.create_study(
            storage = storage, study_name = study_name2, 
            load_if_exists=True, sampler=optuna.samplers.CmaEsSampler())
    artifact_dir2 = Path("./outputs2/")
    objective2 = generate_objective_function2(generation_output, generation_params, artifact_dir2)
    study2.optimize(objective2, n_trials = nsteps2, callbacks=[mlflc2])
    analysis_best_trial_number = study2.best_trial.number
    print("analysis1 optimization done. {} is the best trial number".format(analysis_best_trial_number))
    print(study2.best_params)

    fig = optuna.visualization.plot_optimization_history(study2)
    fig.write_html('study2.html')


    print("Re-generate: use trial #{} parameters.".format(study.best_trial.number))
    generation_params_merge = generation_params.copy()
    metrics_filename = artifact_dir / "{}".format(study.best_trial.number) / "metrics.json"
    with open(metrics_filename, 'r') as f:
        metrics = json.load(f)

    #generation_params_merge["transmat"] = np.load(artifact_dir / "{}".format(study.best_trial.number) / "analysis2/transmat.npy" )
    generation_params_merge["transmat"] = metrics["analysis2_metrics"]["state_transition_matrix"]
    print(generation_params_merge["transmat"])
    diffusivities_flatten = list(itertools.chain.from_iterable(metrics["analysis2_metrics"]["diffusivities"]))
    generation_params_merge["Dm"] = list(map(lambda x: x*1e-12,diffusivities_flatten))
    nm_initial = metrics["analysis1_metrics"]["num_spots_each_frame"][0]
    startprob = metrics["analysis2_metrics"]["startprob"]
    #for i in range(len(startprob)):
    #    generation_params_merge["Nm"][i] = int(nm_initial * startprob[i])
    generation_params_merge["Nm"] = list(map(lambda x: int(nm_initial * x), startprob))

    print("Re-Generate with optimized parameter as follows.")
    print(generation_params_merge)

    print("cf. Default generation parameter")
    print(generation_params)

    generation_output2 = Path("./re-generation/")
    if not generation_output2.exists():
        generation_output2.mkdir(parents = True, exist_ok = True)
    artifacts, metrics = generation1([], generation_output2, generation_params_merge)

