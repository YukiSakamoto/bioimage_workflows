# Setup (remotehost)
```
sudo python3 -m pip install mlflow azure-storage-blob
export AZURE_STORAGE_CONNECTION_STRING="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
mlflow server --host 0.0.0.0 --default-artifact-root wasbs://XXXXX@YYYYY.blob.core.windows.net/
```

# Setup (localhost)
```
sudo apt install python3-venv
cd ~
python3 -m venv venv-ecell
source venv-ecell/bin/activate
pip install -U pip
pip install mlflow toml hydra-core scikit-image plotly kaleido azure-storage-blob
pip install git+git://github.com/ecell/scopyon.git@99436fbfd34bb684966846eba75b206c2806f69c
export AZURE_STORAGE_CONNECTION_STRING="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" #when using remote data storage
export MLFLOW_TRACKING_URI="http://**.**.**.**:5000/" #this can be replaced to "http://localhost:5000/"
```

# Running experiment (localhost)
```
git clone https://github.com/ecell/bioimage_workflows
cd bioimage_workflows
git checkout -t origin/hydra
python -m bioimage_workflow
```
