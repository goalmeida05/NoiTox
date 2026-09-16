"""
NoiTox.py — Unified orchestrator
=========================================
Runs all four NoiTox models (NoiToxML Bo, NoiToxML Full, NoiToxDP,
NoiToxGP) on the same FASTA file, merges predictions with a CSV of
ground-truth labels (sequence,real_label), and reports ACC/SN/SP/MCC
for each model side by side.

Reuses functions already defined in:
  - NoiToxML.py  (build_feature_dataframe, BORUTA_44_FEATURES)
  - NoiToxDP.py  (read_fasta, load_esm, get_embeddings, load_mlp, predict_probs)
  - NoiToxGP.py  (run_modelo_grafo)

ADJUST the imports below if your actual file names differ.

Usage:
    python3 NoiTox.py \
        --input independent.fasta \
        --labels_csv labels_independent.csv \
        --threshold_tag 09 \
        --ml_full_model models/extratrees_full_09.pkl \
        --ml_boruta_model models/extratrees_boruta_09.pkl \
        --dp_model models/noitoxdp_09.pth \
        --gp_model models/noitoxgp_09.pt \
        --outdir results_09 \
        --gp_hidden 128 --gp_layers 3 --gp_heads 4 --gp_attn_dim 64 --gp_dropout 0.2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

# ── Existing modules (adjust names if necessary) ─────────────────────────────
from NoiToxML import build_feature_dataframe, BORUTA_44_FEATURES  # noqa: E402
from NoiToxDP import read_fasta, load_esm, get_embeddings, load_mlp, predict_probs  # noqa: E402
from NoiToxGP import run_modelo_grafo  # noqa: E402


# ── Shared metrics utility ────────────────────────────────────────────────────

def merge_with_labels(df_pred, labels_csv, seq_col="Sequence"):
    if labels_csv is None:
        merged = df_pred.copy()
        merged["real_label"] = np.nan
        return merged

    df_labels = pd.read_csv(labels_csv)
    df_labels.columns = [c.strip().lower() for c in df_labels.columns]
    merged = pd.merge(df_pred, df_labels, left_on=seq_col, right_on="sequence", how="left")
    if "sequence" in merged.columns and seq_col != "sequence":
        merged = merged.drop(columns=["sequence"])
    return merged


def compute_metrics(merged, pred_col="Prediction", label_col="real_label"):
    if merged[label_col].isna().all():
        print("  No ground-truth labels provided; skipping metric computation.")
        return {
            "n_sequences": len(merged), "TP": None, "TN": None, "FP": None, "FN": None,
            "ACC": None, "SN": None, "SP": None, "MCC": None,
        }

    n_missing = merged[label_col].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} sequence(s) had no match in the labels file.",
              file=sys.stderr)

    matched = merged.dropna(subset=[label_col]).copy()
    matched[label_col] = matched[label_col].astype(int)
    matched[pred_col] = matched[pred_col].astype(int)

    tp = int(((matched[pred_col] == 1) & (matched[label_col] == 1)).sum())
    tn = int(((matched[pred_col] == 0) & (matched[label_col] == 0)).sum())
    fp = int(((matched[pred_col] == 1) & (matched[label_col] == 0)).sum())
    fn = int(((matched[pred_col] == 0) & (matched[label_col] == 1)).sum())
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total > 0 else float("nan")
    sn = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    sp = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else float("nan")

    return {
        "n_sequences": total, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "ACC": round(acc, 4), "SN": round(sn, 4), "SP": round(sp, 4), "MCC": round(mcc, 4),
    }


# ── NoiToxML (Bo / Full) ─────────────────────────────────────────────────────

def run_ml(input_fasta, labels_csv, model_path, model_type, threshold, outdir, tag):
    print(f"\n{'='*60}\nNoiToxML ({model_type}) — {tag}\n{'='*60}")

    df_features = build_feature_dataframe(input_fasta)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if model_type == "boruta":
        missing = [c for c in BORUTA_44_FEATURES if c not in df_features.columns]
        if missing:
            raise ValueError(f"Missing Boruta features: {missing}")
        X = df_features[BORUTA_44_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        feature_cols = [c for c in df_features.columns if c != "sequence"]
        X = df_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "Sequence": df_features["sequence"].values,
        "ML_Score": np.round(probs, 4),
        "Prediction": preds,
    })

    merged = merge_with_labels(result, labels_csv)
    out_path = os.path.join(outdir, f"result_ml_{model_type}_{tag}_with_labels.csv")
    merged.to_csv(out_path, index=False)

    metrics = compute_metrics(merged)
    print(f"  {metrics}")
    return metrics


# ── NoiToxDP ─────────────────────────────────────────────────────────────────

def run_dp(input_fasta, labels_csv, model_path, threshold, outdir, tag):
    print(f"\n{'='*60}\nNoiToxDP — {tag}\n{'='*60}")

    sequences = read_fasta(input_fasta)

    model_esm, batch_converter = load_esm()
    embeddings = get_embeddings(sequences, model_esm, batch_converter)
    del model_esm, batch_converter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X = torch.tensor(embeddings, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)

    mlp = load_mlp(model_path, X.shape[1])
    probs = predict_probs(mlp, X)
    preds = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "Sequence": sequences,
        "ML_Score": np.round(probs, 4),
        "Prediction": preds,
    })

    merged = merge_with_labels(result, labels_csv)
    out_path = os.path.join(outdir, f"result_dp_{tag}_with_labels.csv")
    merged.to_csv(out_path, index=False)

    metrics = compute_metrics(merged)
    print(f"  {metrics}")
    return metrics


# ── NoiToxGP ─────────────────────────────────────────────────────────────────

def run_gp(input_fasta, labels_csv, model_path, threshold, outdir, tag,
           hidden_dim, num_layers, num_heads, attn_dim, dropout):
    print(f"\n{'='*60}\nNoiToxGP — {tag}\n{'='*60}")

    df = run_modelo_grafo(
        input_fasta,
        caminho_modelo=model_path,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        attn_dim=attn_dim,
        dropout=dropout,
    )
    # run_modelo_grafo returns: sequence, probabilidade, label_previsto (sorted by descending prob)
    result = df.rename(columns={
        "sequence": "Sequence",
        "probabilidade": "ML_Score",
        "label_previsto": "Prediction",
    })[["Sequence", "ML_Score", "Prediction"]].copy()

    if threshold != 0.5:
        result["Prediction"] = (result["ML_Score"] >= threshold).astype(int)

    merged = merge_with_labels(result, labels_csv)
    out_path = os.path.join(outdir, f"result_gp_{tag}_with_labels.csv")
    merged.to_csv(out_path, index=False)

    metrics = compute_metrics(merged)
    print(f"  {metrics}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--labels_csv", default=None,
                         help="Optional CSV with columns sequence,real_label. "
                              "If omitted, predictions are saved without metric computation.")
    parser.add_argument("--threshold_tag", required=True, help="CD-HIT threshold label (e.g. 08, 09), used only in output file names")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--outdir", default=".", help="Output directory")

    parser.add_argument("--ml_full_model", required=True)
    parser.add_argument("--ml_boruta_model", required=True)
    parser.add_argument("--dp_model", required=True)
    parser.add_argument("--gp_model", required=True)

    parser.add_argument("--gp_hidden", type=int, default=128)
    parser.add_argument("--gp_layers", type=int, default=3)
    parser.add_argument("--gp_heads", type=int, default=4)
    parser.add_argument("--gp_attn_dim", type=int, default=64)
    parser.add_argument("--gp_dropout", type=float, default=0.2)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.labels_csv is None:
        print("No --labels_csv provided: running in prediction-only mode "
              "(no ACC/SN/SP/MCC will be computed).\n")

    all_metrics = []

    m = run_ml(args.input, args.labels_csv, args.ml_full_model, "full",
               args.threshold, args.outdir, args.threshold_tag)
    all_metrics.append({"model": "NoiToxML (full)", "threshold_tag": args.threshold_tag, **m})

    m = run_ml(args.input, args.labels_csv, args.ml_boruta_model, "boruta",
               args.threshold, args.outdir, args.threshold_tag)
    all_metrics.append({"model": "NoiToxML (Bo)", "threshold_tag": args.threshold_tag, **m})

    m = run_dp(args.input, args.labels_csv, args.dp_model,
               args.threshold, args.outdir, args.threshold_tag)
    all_metrics.append({"model": "NoiToxDP", "threshold_tag": args.threshold_tag, **m})

    m = run_gp(args.input, args.labels_csv, args.gp_model,
               args.threshold, args.outdir, args.threshold_tag,
               args.gp_hidden, args.gp_layers, args.gp_heads, args.gp_attn_dim, args.gp_dropout)
    all_metrics.append({"model": "NoiToxGP", "threshold_tag": args.threshold_tag, **m})

    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(args.outdir, f"summary_noitox_{args.threshold_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n\n{'='*60}\nSUMMARY — CD-HIT {args.threshold_tag}\n{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()