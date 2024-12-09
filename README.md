# Setup (cloning this repo)
```
cd ~
git clone https://github.com/ecell/bioimage_workflows
cd bioimage_workflows
git checkout -t origin/hydra_optuna_mlflow_ishii
```

# Setup (pip install for Ubuntu 20.04)
```
sudo apt update
sudo apt install python3-venv
cd ~
python3 -m venv venv-ecell
source ~/venv-ecell/bin/activate
pip install wheel
pip install mlflow==1.30.0 hmmlearn==0.2.6 hydra-core scikit-image plotly kaleido azure-storage-blob
pip install git+https://github.com/ecell/scopyon.git@99436fbfd34bb684966846eba75b206c2806f69c
pip install optuna==3.0.5 protobuf==3.20.3 optuna-dashboard==0.8.1
pip install hydra-optuna-sweeper==1.2.0
```

Sakamoto
```
pip install -no-deps -r requirements.lock2
```

# Running mlflow tracking server

## Command Line
```
cd ~/bioimage_workflows
mlflow server --host 0.0.0.0
```

## Docker

If you want to use NAS or other storage, change `$HOME/mlflowbackend` to your strorage path.

```
mkdir $HOME/mlflowbackend
docker run --rm --user $UID:1000 -p 5000:5000  -v $HOME/mlflowbackend:/backend ghcr.io/mlflow/mlflow:v1.30.0 mlflow server --host 0.0.0.0 --backend-store-uri sqlite:////backend/tracking.db --artifacts-destination file:///backend/artifacts --serve-artifacts
```

# Running experiment
```
cd ~/bioimage_workflows
python analysis_evaluation.py
```

# Running experiment with hydra

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 time python analysis_evaluation.py experiment.evaluation.params.max_distance="range(4,8)" experiment.analysis.params.overlap="range(0.1,1,0.4)" experiment.analysis.params.threshold=40,50  --multirun
```

# Browsing mlflow tracking server
Open `http://THE_TRACKING_SERVER_IP:5000`

# Browsing optuna-dashboard

```
optuna-dashboard sqlite:///example2.db --port 5050 --host 0.0.0.0
```

