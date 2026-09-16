import argparse
import pickle
import sys

import numpy as np
import pandas as pd


from NoiToxML_BO_FULL import calc_properties_test, cabecalho 


BORUTA_44_FEATURES = [
    "Norm_mass", "Norm_isoelectric_point", "Norm_net_charge",
    "Norm_gravy", "Norm_hidro/total", "len",
    "normnC", "normnH", "normnN", "normnO", "normnS", "total_elements",
    "norm_hydrophobic_negatively_charged_CKSAAGP",
    "norm_hydrophobic_polar_non_charged_CKSAAGP",
    "norm_hydrophobic_positively_charged_CKSAAGP",
    "norm_negatively_charged_polar_non_charged_CKSAAGP",
    "norm_polar_non_charged_hydrophobic_CKSAAGP",
    "norm_polar_non_charged_polar_non_charged_CKSAAGP",
    "norm_polar_non_charged_positively_charged_CKSAAGP",
    "norm_positively_charged_hydrophobic_CKSAAGP",
    "norm_positively_charged_polar_non_charged_CKSAAGP",
    "normA", "normC", "normI", "normK", "normL",
    "normM", "normT", "normV", "normW",
    "normCC", "CCS", "normCG", "normCK", "normCP",
    "normCS", "normCY", "normDC", "normEC", "normGC",
    "normKK", "normPC", "normTC", "normWC",
]


def build_feature_dataframe(input_path):
    """
    Usa calc_properties_test do pipeline original para gerar a matriz
    completa de descritores (sem 'label', já que é dado novo/inferência).
    """
    matrix = calc_properties_test(input_path)

    columns = [c for c in cabecalho if c != "label"]

    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"Número de colunas da matriz ({matrix.shape[1]}) não bate com "
            f"o cabecalho esperado ({len(columns)}). Verifique se 'cabecalho' "
            f"está no mesmo estado usado durante o treino."
        )

    df = pd.DataFrame(matrix, columns=columns)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Caminho do .pkl treinado")
    parser.add_argument("--model_type", required=True, choices=["full", "boruta"],
                         help="'full' usa todos os descritores; 'boruta' filtra para as 44 fixas")
    parser.add_argument("--input", required=True, help="FASTA ou CSV de sequências novas")
    parser.add_argument("--labels_csv", default=None,
                         help="CSV opcional com colunas sequence,real_label para merge e cálculo de métricas")
    parser.add_argument("--output", required=True, help="CSV de saída")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Threshold de decisão sobre a probabilidade (default: 0.5)")
    args = parser.parse_args()

    print(f"Extraindo descritores de '{args.input}'...")
    df_features = build_feature_dataframe(args.input)
    print(f"  {len(df_features)} sequências, {df_features.shape[1] - 1} descritores calculados")

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    if args.model_type == "boruta":
        missing = [c for c in BORUTA_44_FEATURES if c not in df_features.columns]
        if missing:
            raise ValueError(f"Features Boruta ausentes na matriz calculada: {missing}")
        X = df_features[BORUTA_44_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        feature_cols = [c for c in df_features.columns if c != "sequence"]
        X = df_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"Rodando predição ({args.model_type}, {X.shape[1]} features)...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    result = pd.DataFrame({
        "Sequence": df_features["sequence"].values,
        "ML_Score": np.round(probs, 4),
        "Prediction": preds,
    })

    if args.labels_csv:
        df_labels = pd.read_csv(args.labels_csv)
        df_labels.columns = [c.strip().lower() for c in df_labels.columns]
        result = pd.merge(result, df_labels, left_on="Sequence", right_on="sequence", how="left")
        result = result.drop(columns=["sequence"])

        n_missing = result["real_label"].isna().sum()
        if n_missing > 0:
            print(f"Aviso: {n_missing} sequência(s) sem correspondência no arquivo de rótulos.",
                  file=sys.stderr)

        matched = result.dropna(subset=["real_label"]).copy()
        matched["real_label"] = matched["real_label"].astype(int)

        tp = int(((matched["Prediction"] == 1) & (matched["real_label"] == 1)).sum())
        tn = int(((matched["Prediction"] == 0) & (matched["real_label"] == 0)).sum())
        fp = int(((matched["Prediction"] == 1) & (matched["real_label"] == 0)).sum())
        fn = int(((matched["Prediction"] == 0) & (matched["real_label"] == 1)).sum())
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else float("nan")
        sn = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        sp = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else float("nan")

        print(f"\n=== Confusion matrix ===")
        print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"ACC={acc:.4f} SN={sn:.4f} SP={sp:.4f} MCC={mcc:.4f}")

    result.to_csv(args.output, index=False)
    print(f"\nResultado salvo em: {args.output}")


if __name__ == "__main__":
    main()