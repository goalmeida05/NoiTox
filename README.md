# NoiTox

**A Multi-Model Strategy for Peptide Toxicity Prediction Reveals Label Noise in Benchmark Datasets**

NoiTox is a peptide toxicity prediction framework built around three complementary models:

- **NoiToxML** — an Extremely Randomized Trees (ExtraTrees) classifier, available in two variants:
  - `full`: trained on the complete set of ~8,833 physicochemical, compositional, and atomic descriptors.
  - `Bo`: trained on a compact, interpretable subset of 44 descriptors confirmed by the Boruta feature selection algorithm.
- **NoiToxDP** — a deep learning model using ESM-2 protein language model embeddings and a multilayer perceptron (MLP) classifier.
- **NoiToxGP** — a graph neural network (GATv2) that combines ESM-2 residue embeddings with peptide graph topology and the same 44 Boruta-selected descriptors.

Beyond predictive performance, this work identifies and quantifies **label noise by omission** in standard peptide toxicity benchmarks: sequences labeled non-toxic solely because no toxicity annotation exists for them, some of which show experimentally confirmed toxic activity in independent databases.

If you use this repository, please cite the associated paper (see [Citation](#citation)).

---

## Repository structure

```
NoiTox/
├── NoiToxML.py            # NoiToxML (full / Bo) — feature extraction, training, and inference
├── NoiToxDP.py             # NoiToxDP — ESM-2 embedding extraction and MLP inference
├── NoiToxGP.py             # NoiToxGP — ESM-2 + GATv2 graph construction, model, and inference
├── NoiTox.py               # Unified orchestrator: runs all four model variants on one input file
├── train_ml_models.py      # Trains NoiToxML (full) and NoiToxML (Bo) from labeled FASTA files
├── requirements.txt        # Pinned Python dependencies
├── LICENSE
├── models/                 # Trained model checkpoints (see "Trained models" below)
│   ├── extratrees_full_08.pkl
│   ├── extratrees_boruta_08.pkl
│   ├── extratrees_full_09.pkl
│   ├── extratrees_boruta_09.pkl
│   ├── noitoxdp_08.pth
│   ├── noitoxdp_09.pth
│   ├── noitoxgp_08.pt
│   └── noitoxgp_09.pt
└── data/                   # Benchmark datasets (FASTA + label CSVs)
    ├── test_08.fasta
    ├── test_09.fasta
    ├── independent.fasta
    └── labels_*.csv
```

> **Note:** file and folder names above reflect the recommended layout. Adjust the paths in the example commands below to match your local structure.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/goalmeida05/NoiTox.git
cd NoiTox
pip install -r requirements.txt
```

`torch` and `torch-geometric` wheels are CUDA-version-specific. If you have a GPU, install the build matching your CUDA version by following the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) and the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) **before** running `pip install -r requirements.txt`, or CPU-only builds will be installed by default.

---

## Quick start: predicting toxicity for new sequences

The recommended entry point is `NoiTox.py`, which runs all four model variants (NoiToxML full, NoiToxML Bo, NoiToxDP, NoiToxGP) on the same input FASTA file and reports side-by-side metrics.

**With ground-truth labels** (for benchmarking against a labeled dataset):

```bash
python3 NoiTox.py \
    --input data/independent.fasta \
    --labels_csv data/labels_independent.csv \
    --threshold_tag 09 \
    --ml_full_model models/extratrees_full_09.pkl \
    --ml_boruta_model models/extratrees_boruta_09.pkl \
    --dp_model models/noitoxdp_09.pth \
    --gp_model models/noitoxgp_09.pt \
    --outdir results_09
```

**Without ground-truth labels** (prediction on new, unannotated sequences):

```bash
python3 NoiTox.py \
    --input my_new_peptides.fasta \
    --threshold_tag custom \
    --ml_full_model models/extratrees_full_09.pkl \
    --ml_boruta_model models/extratrees_boruta_09.pkl \
    --dp_model models/noitoxdp_09.pth \
    --gp_model models/noitoxgp_09.pt \
    --outdir my_results
```

Omitting `--labels_csv` runs the pipeline in prediction-only mode: sequence-level predictions are still saved for every model, but ACC/SN/SP/MCC are not computed.

### Output

For each model, `NoiTox.py` saves a CSV with columns `Sequence, ML_Score, Prediction` (and `real_label` if labels were provided) to `--outdir`, plus a `summary_noitox_<threshold_tag>.csv` consolidating ACC/SN/SP/MCC across all four models.

### Input format

- **FASTA**: standard format, one header (`>id`) per sequence.
- **Labels CSV** (optional): must contain a `sequence` column and a `real_label` column (`1` = toxic, `0` = non-toxic). Column names are case-insensitive.

### Choosing which model checkpoint to use

Models trained at CD-HIT 0.8 vs. 0.9 (see [Model variants](#model-variants) below) differ only in the training partition's redundancy-reduction threshold. If you are not reproducing a specific benchmark comparison from the paper, either threshold's checkpoint is a reasonable default; the 0.9 models were trained on a larger, less redundant partition.

---

## Model variants

| Model | Architecture | Input features |
|---|---|---|
| NoiToxML (full) | ExtraTrees | ~8,833 physicochemical/compositional/atomic descriptors |
| NoiToxML (Bo) | ExtraTrees | 44 Boruta-selected descriptors (see Supplementary Table S2 of the paper) |
| NoiToxDP | ESM-2 (`esm2_t12_35M_UR50D`) + MLP | Mean-pooled per-residue ESM-2 embeddings |
| NoiToxGP | ESM-2 + GATv2 (3 layers) + attention pooling | Per-residue ESM-2 embeddings (graph nodes) + 44 global descriptors |

Full hyperparameters for every model are listed in the paper's Supplementary Table S3.

Both `full`/`Bo` and the deep-learning models (`NoiToxDP`, `NoiToxGP`) are provided as two independently trained checkpoints, corresponding to the two CD-HIT sequence-identity thresholds (0.8 and 0.9) used to construct the training/test partitions described in the paper.

---

## Running individual models

Each model can also be run standalone via its own module:

```bash
# NoiToxML (full or Boruta-selected features)
python3 -c "
from NoiToxML import build_feature_dataframe
import pickle
df = build_feature_dataframe('my_sequences.fasta')
model = pickle.load(open('models/extratrees_boruta_09.pkl', 'rb'))
"

# NoiToxDP and NoiToxGP expose similar importable functions;
# see the docstrings at the top of NoiToxDP.py and NoiToxGP.py
# for their public API (read_fasta, load_esm, get_embeddings,
# load_mlp, predict_probs, and run_modelo_grafo, respectively).
```

We recommend using `NoiTox.py` unless you specifically need to customize a single model's pipeline.

---

## Training NoiToxML from scratch

`train_ml_models.py` reproduces NoiToxML (full) and NoiToxML (Bo) from labeled positive/negative FASTA files, using the same descriptor set and Extremely Randomized Trees hyperparameters reported in the paper:

```bash
python3 train_ml_models.py
```

Edit the `positives_path` / `negatives_path` arguments at the bottom of the script to point to your own training FASTA files. The Boruta-selected feature list is fixed to the 44 descriptors identified in the original analysis (see `BORUTA_44_FEATURES` in `NoiToxML.py`) rather than re-run, since Boruta's feature selection is stochastic and re-running it is not guaranteed to reproduce the same subset.

Training scripts for NoiToxDP and NoiToxGP are not included in this quick-start guide; see `NoiToxDP.py` and `NoiToxGP.py` for the model architectures, and adapt your own training loop using the same ESM-2 embedding extraction and hyperparameters listed in Supplementary Table S3.

---

## Trained models

Pretrained checkpoints for all four model variants (both CD-HIT 0.8 and 0.9 versions) are available at: **[TODO: add download link — e.g. Zenodo/Figshare DOI, or a `models/` release attached to this repository]**.

Checkpoint files are not tracked directly in this Git repository due to file size.

---

## Data

The benchmark datasets used in this work (test sets at CD-HIT 0.8/0.9 and the independent validation set) originate from ToxiPep (Guan et al., 2025) and are available at **[TODO: add dataset source/link]**. Ground-truth label files (`sequence,real_label`) used to reproduce the paper's tables are included in `data/`.

---

## Citation

If you use NoiTox in your research, please cite:

```
[TODO: add full citation once the manuscript is published, including DOI]
```

---

## License

This project is licensed under the **[TODO: specify license, e.g. MIT]** — see [LICENSE](LICENSE) for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author at gustavo.o.almeida@ufv.br.
