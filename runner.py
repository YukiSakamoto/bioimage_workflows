import optuna
import mlflow
from pathlib import Path
from optuna.integration.mlflow import MLflowCallback
from user_functions import generation1, analysis1, evaluation1, analysis2, evaluation2
import sys
import numpy as np

#==================================================
# Parameters
#==================================================
generation_params = {
    "seed": 123,
    "interval": 0.033,
    "num_samples": 1,
    "num_frames": 5,
    "exposure_time": 0.033,
    "Nm": [100, 100, 100],
    "Dm": [0.222e-12, 0.032e-12, 0.008e-12], 
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
    tracking_uri="http://10.5.1.218:7777",
    metric_name="sum of square x_mean and y_mean",
)

def generate_objective_function(generation_output_path, eval_weight = {"transmat_rss": 0.5, "startprob": 0.5, "D": 0.0}):
    # This function returns the closure that hold the generation_output_path.

    @mlflc.track_in_mlflow()
    def _objective(trial):
        generation_output = generation_output_path  
        mlflow.log_param("generation_output_path", generation_output)   #XXX should be use log_artifact ???

        # The parameters get from trial.suggest_XXX(XXX=float, int, ...) are automatically changed to optimize.
        analysis_params_mod = analysis_params.copy()
        analysis_params_mod["threshold"] = trial.suggest_float("threshold", 10, 100)
        analysis_params_mod["overlap"] = trial.suggest_float("overlap", 0.1, 1.0)
        analysis_params_mod["max_sigma"] = trial.suggest_float("max_sigma", 4, 10)
        analysis_params_mod["min_sigma"] = trial.suggest_float("min_sigma", 0, 2)
        analysis_params_mod["cutoff_distance"] = trial.suggest_float("cutoff_distance", 1, 10)  #XXX obj2

        analysis1_output=Path('./outputs_analysis_run/'+str(trial.number) + '/analysis1/')
        analysis1_output.mkdir(parents=True, exist_ok=True)
        a, b = analysis1([generation_output], analysis1_output, analysis_params_mod)

        evaluation_output=Path('./outputs_analysis_run/'+str(trial_number) + '/evaluation1/')
        evaluation_output.mkdir(parents=True, exist_ok=True)

        c,d = evaluation1([generation_output,analysis_output], evaluation_output, evaluation_params)

        x_mean = d["x_mean"]
        y_mean = d["y_mean"]
        mlflow.log_metric("x_mean", x_mean)
        mlflow.log_metric("y_mean", y_mean)
        mean_norm2 = (x_mean)**2+(y_mean)**2
        mlflow.log_metric("mean_norm2", mean_norm2)

        # analysis2
        analysis2_output = Path('./outputs_analysis_run/'+str(trial.number) + '/analysis2/')
        analysis2_output.mkdir(parents=True, exist_ok=True)
        analysis2_artifacts, analysis2_metrics = analysis2([generation_output], analysis1_output, analysis2_output, analysis_params_mod )

        evaluation_output=Path('./outputs_analyais_run/'+str(trial_number) + '/evaluation2/')
        evaluation_output.mkdir(parents=True, exist_ok=True)
        _, metrics = evaluation2([generation_output, analysis2_output], evaluation_output, evaluation_params)

        start_ratio = np.array(generation_params["Nm"]) / sum(generation_params["Nm"])
        startprob_rss = np.sum(np.square(analysis2_metrics["startprob"] - start_ratio) )
        analysis_Dm = np.sort(np.array(list(map(lambda x: x[0], analysis2_metrics["D"]) ) ))
        generation_Dm = np.sort(np.array(generation_params["Dm"]))
        dm_rss = np.sum(np.square(analysis_Dm - generation_Dm) ) 

        objective_parameters = {
            "startprob": startprob_rss,
            "D": dm_rss,
            "transmat_rss": metrics["transmat_rss"]
        }
        result = 0.
        for k in eval_weight.keys():
            print("{} : {} : {}".format(k, eval_weight[k], objective_parameters[k]))
            result += objective_parameters[k] * eval_weight[k]
        mlflow.log_artifacts(Path('./outputs_analysis_run/'+str(trial.number) ))
        return result

    return _objective


def generate_objective2_function(generation_output_path, analysis1_output_path, eval_weight = {"transmat_rss": 0.5, "startprob": 0.5, "D": 0.0}):
    @mlflc.track_in_mlflow()
    def _objective2(trial):
        generation_output = generation_output_path  
        analysis2_output=Path('./outputs_analysis2_run/'+str(trial.number))
        analysis2_output.mkdir(parents=True, exist_ok=True)
        analysis_params_mod = analysis_params.copy()
        analysis_params_mod["cutoff_distance"] = trial.suggest_float("cutoff_distance", 1, 10)
        #analysis_params_mod["interval"] = trial.suggest_float("interval", 0, 0.15)


        analysis2_artifacts, analysis2_metrics = analysis2([generation_output], analysis1_output_path, analysis2_output, analysis_params_mod )
        mlflow.log_param("generation_output_path", generation_output)

        evaluation_output=Path('./outputs_evaluation_run/'+str(1))
        evaluation_output.mkdir(parents=True, exist_ok=True)
        _, metrics = evaluation2([generation_output, analysis2_output], evaluation_output, evaluation_params)
        #result = metrics["transmat_rss"]

        # calculate the difference of startprob
        start_ratio = np.array(generation_params["Nm"]) / sum(generation_params["Nm"])
        startprob_rss = np.sum(np.square(analysis2_metrics["startprob"] - start_ratio) )
        analysis_Dm = np.sort(np.array(list(map(lambda x: x[0], analysis2_metrics["D"]) ) ))
        generation_Dm = np.sort(np.array(generation_params["Dm"]))
        dm_rss = np.sum(np.square(analysis_Dm - generation_Dm) ) 

        objective_parameters = {
            "startprob": startprob_rss,
            "D": dm_rss,
            "transmat_rss": metrics["transmat_rss"]
        }
        result = 0.
        for k in eval_weight.keys():
            print("{} : {} : {}".format(k, eval_weight[k], objective_parameters[k]))
            result += objective_parameters[k] * eval_weight[k]

        return result

    return _objective2

def objective2(generation_output_path, best_analysis1_trial_number ):
    generaion_output = generation_output_path
    analysis1_output=Path('./outputs_analysis_run/'+str(best_analysis1_trial_number))
    analysis2_output=Path('./outputs_analysis2_run/'+str(1))
    analysis2_output.mkdir(parents=True, exist_ok=True)
    analysis_params_mod = analysis_params.copy()
    analysis2_artifacts, analysis2_metrics = analysis2([generation_output], analysis1_output, analysis2_output, analysis_params_mod )
    print(analysis2_artifacts)
    print(analysis2_metrics)

    evaluation_output=Path('./outputs_evaluation_run/'+str(1))
    evaluation_output.mkdir(parents=True, exist_ok=True)
    evaluation2([generation_output, analysis2_output], evaluation_output, evaluation_params)
    return 


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        optuna.delete_study(
            storage="sqlite:///example2.10.1.db", 
            study_name="test_x_y_mean_storage2_7_2.10.1")
        optuna.delete_study(
            storage="sqlite:///example2.10.1.db", 
            study_name="test_x_y_mean_storage2_7_2.10.1")
        exit()

    generation_output = Path('./test_8frame/')
    if generation_output.exists() and (generation_output / "config.yaml").exists():
        print("Generation skip",file = sys.stderr )
    else:
        if not generation_output.exists():
            generation_output.mkdir(parents = True, exist_ok = True)
        print("Generation start",file = sys.stderr )
        artifacts,metrics = generation1([], generation_output, generation_params)
        print("Generation done",file = sys.stderr )

    study = optuna.create_study(
            storage="sqlite:///example2.10.1.db", 
            study_name="test_x_y_mean_storage_7_2.10.1", 
            load_if_exists=True, 
            sampler=optuna.samplers.CmaEsSampler())

    objective = generate_objective_function(generation_output)
    study.optimize(objective, n_trials=2, callbacks=[mlflc])

    analysis1_best_trial_number = study.best_trial.number
    print("#==================================================#")
    print("analysis1 optimization done")
    print(analysis1_best_trial_number)
    print(study.best_params)
    print("#==================================================#")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html('study.html')

    #objective2(generation_output, analysis1_best_trial_number)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html('study.html')

