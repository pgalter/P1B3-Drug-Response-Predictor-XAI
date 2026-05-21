# P1B3-Drugd-Response-Predictor-XAI
Implementation of two XAI methods on a P1B3-style deep MLP:  SHAP (GradientExplainer) and Integrated Gradients (manual TensorFlow implementation).

This project uses the official ```p1b3_baseline_keras2.py``` [repository](https://github.com/CBIIT/NCI-DOE-Collab-Pilot1-Single-Drug-Response-Predictor/tree/master) alogn with the CANDLE (Cancer Distributed Learning Environment) framework to use real cancer data and predict NCI-60 cancer drug responses. Then, we use different XAI methods (SHAP and IG) to test this predictions and we analyse these metrics with a fidelity score.

**1. Environment set up**
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Copy this repository 
3. Create the environment as shown:

```bash
# copy this repository 
git clone <repo_url> 

cd Pilot1/P1B3
conda env create -f environment.yml -n P1B3 python=3.6.13

# activate the benchmark environment
conda activate P1B3

# install XAI libraries and visualization dependencies
pip install shap matplotlib
# install the CANDLE library
pip install git+https://github.com/ECP-CANDLE/candle_lib.git --no-deps
```
**2. Downloading the data**
Because of the file sizes (>1GB), the datasets cannot be included in this repository, however when you run it automatically. If after running the ```p1b3_baseline_keras2.py --feature_subsample 500 --train_steps 100 --epochs 1``` the data has been downloaded as a JSON file, to download the data you need to follow this steps:

1. Create a free account in [NCI MoDaC](https://modac.cancer.gov/).
2. In said website, search for "Pilot 1 Cancer Drug Response Prediction Dataset" (https://modac.cancer.gov/assetDetails?assetIdentifier=cancer_drug_response_prediction_dataset&&returnToSearch=true) and download these files:
- NCI60_dose_response_with_missing_z5_avg.csv (Target values)
- P1B3_cellline_expressions.tsv (Cell line features)
- NCI60_dragon7_descriptors.tsv (Drug chemical features)
3. If that doesn't work, try downloading the files from here https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/.
4. Create a folder named data in the repository and move these files into it.

**3. Integration of the XAI methods**
Make sure that xai_utils.py in the same directory as the main script. Also make sure that the "XAI Hook" code block (in this repository or in the end of this README file) is added to the bottom of p1b3_baseline_keras2.py. This should be automatic as you cloned the repository.

**4. Running the script**
Run the model with these optimized parameters. For 15-minute of training, which won't result in an accurate model given the dimensionality of the data, run the ```p1b3_baseline_keras.py``` file.

## Dataset note
The P1B3 dataset (NCI-60 cell lines, 26K gene expressions, 4K drug descriptors) is hosted at MoDaC. You'll need your MoDaC credentials; the script will prompt for them to authenticate the download. If you've already downloaded the files manually, the script will skip the authentification and load from the local directory.

## Project Structure Overview
### Main scripts
* **`p1b3_baseline_keras.py`**: Model building, training, and the final XAI evaluation.
* **`p1b3.py`**: Data staging, downloading biological datasets from CANDLE servers, and feature preprocessing.
* **`xai_utils.py`**: Implementations for Integrated Gradients and SHAP summary logic.
* **`p1b3_infer.py`**: Script for running predictions (inference) using a previously trained model.

### Configuration files
* **`p1b3_default_model.txt`**: Where the model architecture (layers, neurons, activation functions).
* **`environment.yml`**: Configuration file to recreate the specific Conda environment required for the project.
* **`requirements.txt`**: List of Python dependencies for pip-based installations.
* **`run.RUN000.json`**: A log file generated after execution containing the parameters used for that run.

### Directories
* **`common/`**: Stores the local source code for the CANDLE library framework.
* **`xai_outputs/`**: Stores generated visualization plots (e.g., SHAP summary plots and IG importance charts).

### Control
* **`.gitignore`**: Prevents large data files, model weights, and temporary caches from being uploaded to Git.
* **`.gitattributes`**: Technical settings for how Git tracks certain file types.
