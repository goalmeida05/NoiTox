# NoiTox: A machine learning strategy to predict peptide toxicity

**NoiTox** is a machine learning pipeline developed to predict whether a given peptide sequence has citotoxic properties.  
This repository contains all code, datasets, models, and outputs necessary to reproduce the training process, make new predictions, and evaluate model performance.

---

## Repository Structure


├── CD08/                 # Datasets used for training and testing and the respective model

├── CD09/                 # Datasets used for training and testing and the respective model

├── Independent/          # The independent dataset

├── NoiTox.py           # Main script: feature extraction, model training, and prediction

├── utils.py               # Helper functions for feature extraction

└── README.md              # This file




---

## Dependencies

To run this project, you need to have **Python 3.8 or higher** installed.

We recommend creating a virtual environment to avoid conflicts.  
You can set up your environment by running:

```bash
python -m venv hecate-env
source hecate-env/bin/activate  # On Windows use: hecate-env\Scripts\activate
```
```bash
pip install -r requirements.txt
```
## How to Run NoiTox

### 1. Predicting New Sequences

Execute the main script:

```bash
python NoiTox.py
```

You will see the following menu:
      1 - TRAINING MODEL

      2 - TESTING MODEL

Choose option 2 - TESTING MODEL if you want to classify new sequences using the pretrained model.

You will be asked to enter the file path to your dataset.

Accepted input formats:

* FASTA: Standard FASTA format.
* CSV: CSV file containing a single column with peptide sequences.

Example:
Data path: IndependentDataset/independent.fasta

The prediction results will be saved into the main directory as Results.csv, containing:
* seq: the original peptide sequence
* prob: probability score of being a CPP
* Classification: predicted class (1 = CPP, 0 = non-CPP)
Also, a cpps_test_matrix.csv was save to, with all sequences and the respective descriptors


### 2. Training a New Model (Optional)

If you wish to retrain the model using your own datasets, select option 1 - TRAINING MODEL after executing NoiTox.py.

You will be prompted to provide:
* The path to your positive dataset (sequences labeled as CPPs)
* The path to your negative dataset (sequences labeled as non-CPPs)

```bash
Positives path: DATASETS/positives.fasta
Negatives path: DATASETS/negatives.fasta
```
The pipeline will:

* Extract features
* Generate the training matrix
* Perform repeated 10x10 cross-validation
* Display and save performance metrics
* Save the new trained model as model.pkl


## Notes on Feature Extraction
Several handcrafted features are calculated, including:

* Amino acid composition
* Dipeptide and tripeptide frequencies
* Physicochemical properties (molecular weight, isoelectric point, net charge, hydropathy)
* CKSAAGP features (K-spaced amino acid group pairs with k=1)
* Atomic Composition 

Sequences containing ambiguous amino acids (X, B, Z, J, O, U, *, -) are automatically filtered.
