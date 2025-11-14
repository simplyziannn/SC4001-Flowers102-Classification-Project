# Environment Setup
This project requires **Python 3.13.5**. 

We have intentionally not installed packages using pip install -r requirements.txt because certain dependencies especially the PyTorch CUDA wheels which cannot be resolved correctly through the default pip index. This leads to installation failures when attempting to build a virtual environment directly from ```requirements.txt``` (e.g., in VS Code’s “Create Virtual Environment from Requirements.txt” feature).

For this project, the ```requirements.txt``` file is included only as a reference to show the full list of dependencies used.
To ensure reliable installation, **please follow the manual installation steps below** instead of installing everything from the requirements file.

Instead, create and install dependencies manually using the terminal.

### 1. Create a python virtual environment
```python -m venv .venv```
### 2. Activate python virtual environment once created
```.\.venv\Scripts\activate```

### 3. Check your CUDA version (Do note you need an Nvidia GPU!) 
Run following command 
```nvidia-smi```

Then look at the line labeled CUDA Version in the output. 
```| NVIDIA-SMI 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.6     |``` 
In this example, CUDA version is 12.6 

### 4. Install PyTorch according to your CUDA version **before everything else** using the official CUDA wheel index:
For example: 
**CUDA 12.6**
```pip install torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \--index-url https://download.pytorch.org/whl/cu126``` 

**CUDA 12.4**
```pip install torch==2.9.0+cu124 torchvision==0.24.0+cu124 torchaudio==2.9.0+cu124 \--index-url https://download.pytorch.org/whl/cu124```

**CUDA 12.1**
```pip install torch==2.9.0+cu121 torchvision==0.24.0+cu121 torchaudio==2.9.0+cu121 \--index-url https://download.pytorch.org/whl/cu121```

**CUDA 11.8**
```pip install torch==2.9.0+cu118 torchvision==0.24.0+cu118 torchaudio==2.9.0+cu118 \--index-url https://download.pytorch.org/whl/cu118`` 

If your version is not above you can refer to https://download.pytorch.org to find whoich pytorch wheels to download. 

### 5. Install the rest of the packages
```pip install matplotlib numpy optuna pandas Pillow scikit-learn scipy tqdm joblib networkx```

# Notebook Overview
## 1. Baseline+fewshot.ipynb — Baseline Model & Few-Shot Metric Learning

This notebook establishes the foundations of the project:

**Key components**

Baseline classifier using a ResNet backbone.

Deterministic data augmentation pipeline for reproducible experiments.

Few-shot setups (1-shot and 5-shot) using:

Siamese networks with Contrastive Loss

Triplet networks with Triplet Loss

**Purpose:**

Establish a baseline accuracy to compare other approaches.

Compare baseline with Data augmentation and with no data augmentation

Evaluate how metric learning performs with extremely limited data.


## 2. Hyperparameter Finetuning.ipynb — Greedy Search vs Optuna Optimization

This notebook focuses on systematically tuning the model.

Two hyperparameter optimization approaches

**(1) Greedy Hyperparameter Search:**

Sequentially optimizes each parameter.

Fast and simple.

Provides interpretable tuning trajectories.

**(2) Optuna Optimization:**

Explores the hyperparameter space more efficiently.

Produces better minima on most trials.

**Purpose:**

Compare how Greedy vs Optuna differ in:

performance and convergence behavior

Select the best performing parameter sets to be applied later in the ensemble phase.


## 3. Ensemble_Experiments.ipynb — Ensemble Models (k = 1 to 15)

This notebook conducts all experiments involving k-fold ensembling.

What is done here

Build ensembles ranging from k = 1 to k = 15.

For each k:

Train k models, each on different folds.

Evaluate ensemble accuracy, macro-F1, and weighted-F1.

Identify the optimal ensemble size.

**Key findings:**

k = 12 provides the best test accuracy results.

Both greedy-tuned and Optuna-tuned hyperparameters are applied to 12-fold ensembles.

12-fold Optuna-tuned model produced the best resuts compared to greedy and baseline 12 fold models. 

The 12-fold Optuna-tuned model becomes the final reference model.

**Large-train configuration:**

Combine train + validation into a single large dataset.

Re-train the 12 models with fold-based validation inside the loop.

This creates the strongest possible ensemble prior to final testing. 

12-fold Optuna-tuned model trained on large-train configuration produced the best results overall for all supervised classification ensemble models in this experiment. 

**Metric Learning vs Ensembling:**

Finally, in this notebook:

Test 12-fold 5-shot Siamese (Contrastive Loss) performance

Compare directly against the 12-fold Optuna ensemble

This answers the key research question:
Can we combine metric learning methods with the earlier ensemble strategies to improve performance?


## 4. Visualization.ipynb — Analysis & Plotting

This notebook is meant to be run last.

**Provides visualizations for:**

Hyperparameter trajectories (Greedy vs Optuna)

Ensemble accuracy curves for k = 1 to 15

Distribution of folds and performance variance


**Purpose**

Summarize insights from the entire research pipeline.

Produce clean figures for reports, publications, and presentations.


# Experimental Pipeline

1. Train baseline model

2. Run Siamese / Triplet few-shot experiments

3. Perform greedy + Optuna hyperparameter tuning

4. Run ensemble experiments k = 1 to 15

5. Select best k (found to be k = 12)

6. Apply best hyperparameters to 12-fold models

7. Train large-train 12-fold ensembles

8. Combine Siamese metric learning method with 12-fold model and assess performance

9. Visualize all results

