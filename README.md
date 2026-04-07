# P1B3-Drugd-Response-Predictor-XAI
Implementation of two XAI methods on a P1B3-style deep MLP:  SHAP (GradientExplainer) and Integrated Gradients (manual TensorFlow implementation).

This project uses the official ```p1b3_baseline_keras2.py``` [repository](https://github.com/CBIIT/NCI-DOE-Collab-Pilot1-Single-Drug-Response-Predictor/tree/master) alogn with the CANDLE (Cancer Distributed Learning Environment) framework to use real cancer data and predict NCI-60 cancer drug responses.

**1. Environment set up**
Copy this repository and install the requirements. Ensure you have Miniconda or Anaconda installed, the official P1B3 conda environment activated and the XAI libraries/dependencies: 
```bash
git clone <repo_url> # copy this repository 

# activate the benchmark environment
conda activate P1B3
# project requirements
pip install -r requirements.txt
# install XAI libraries and visualization dependencies
pip install shap matplotlib
```

**2. Downloading the data**
Because of the file sizes (>1GB), the datasets cannot be included in this repository, however when you run it automatically. If after running the ```p1b3_baseline_keras2.py --feature_subsample 500 --train_steps 100 --epochs 1``` the data has been downloaded as a JSON file, to download the data you need to follow this steps:

1. Create a free account in [NCI MoDaC](https://modac.cancer.gov/).
2. In said website, search for "Pilot 1 Cancer Drug Response Prediction Dataset" (https://modac.cancer.gov/assetDetails?assetIdentifier=cancer_drug_response_prediction_dataset&&returnToSearch=true) and download these files:
- NCI60_dose_response_with_missing_z5_avg.csv (Target values)
- P1B3_cellline_expressions.tsv (Cell line features)
- NCI60_dragon7_descriptors.tsv (Drug chemical features)
3. Create a folder named data in the repository and move these files into it.

**3. Integration of the XAI methods**
Make sure that xai_utils.py in the same directory as the main script. Also make sure that the "XAI Hook" code block (in this repository or in the end of this README file) is added to the bottom of p1b3_baseline_keras2.py.

**4. Running the script**
Run the model with these optimized parameters (for 15-minute of training, not very accurate given the dimensionality):

```bash
python p1b3_baseline_keras2.py --feature_subsample 500 -e 1 --batch_size 100
```


## Dataset note
The P1B3 dataset (NCI-60 cell lines, 26K gene expressions, 4K drug descriptors) is hosted at MoDaC. You'll need your MoDaC credentials; the script will prompt for them to authenticate the download. If you've already downloaded the files manually, the script will skip the authentification and load from the local directory.
