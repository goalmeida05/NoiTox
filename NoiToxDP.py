import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import esm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
)

# ── Configurações ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_LAYER = 12          # camada de representação do ESM2
BATCH_SIZE = 64
THRESHOLD  = 0.5


# ── Arquitetura do MLP (deve ser idêntica à usada no treino) ─────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Leitura de FASTA ─────────────────────────────────────────────────────────

def read_fasta(fasta_path: str) -> list[str]:
    """Lê um arquivo FASTA e retorna lista de sequências."""
    sequences = []
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

    if current_seq:
        sequences.append("".join(current_seq))

    print(f"[FASTA] {len(sequences)} sequências carregadas de '{fasta_path}'")
    return sequences


# ── Extração de embeddings ESM2 ───────────────────────────────────────────────

def load_esm():
    """Carrega o modelo ESM2 e o batch converter."""
    print("[ESM2] Carregando modelo...")
    model_esm, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model_esm = model_esm.to(DEVICE)
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"[ESM2] Pronto — device: {next(model_esm.parameters()).device}")
    return model_esm, batch_converter


def get_embeddings(sequences: list[str], model_esm, batch_converter) -> np.ndarray:
    """Extrai embeddings mean-pooled da camada ESM_LAYER para cada sequência."""
    embeddings = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i : i + BATCH_SIZE]
        data  = [(str(j), seq) for j, seq in enumerate(batch)]

        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast(device_type=DEVICE.type):
                out = model_esm(tokens, repr_layers=[ESM_LAYER])

        reps = out["representations"][ESM_LAYER]

        for j, seq in enumerate(batch):
            emb = reps[j, 1 : len(seq) + 1].mean(0)
            embeddings.append(emb.cpu().numpy())

        print(f"  embeddings: {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)}")

    return np.array(embeddings)


# ── Carregamento dos modelos MLP ─────────────────────────────────────────────

def load_mlp(path: str, input_dim: int) -> MLP:
    """Instancia e carrega os pesos de um MLP salvo em .pth."""
    model = MLP(input_dim).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[MLP] Modelo carregado: {path}")
    return model


# ── Inferência ────────────────────────────────────────────────────────────────

def predict_probs(model: MLP, X: torch.Tensor) -> np.ndarray:
    """Retorna probabilidades sigmoid para cada amostra."""
    with torch.no_grad():
        logits = model(X).squeeze()
        return torch.sigmoid(logits).cpu().numpy()


# ── Função principal ──────────────────────────────────────────────────────────

def run_single(
    csv_path: str,
    model_path: str,
    model_name: str = "model",
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Roda um único modelo MLP em um arquivo CSV com colunas sequence, label.

    Parâmetros
    ----------
    csv_path    : caminho para o CSV de entrada (colunas: sequence, label)
    model_path  : caminho para o .pth do modelo
    model_name  : nome do modelo para exibição no log (ex: "CD08", "CD09")
    output_path : se informado, salva o CSV neste caminho

    Retorna
    -------
    DataFrame com colunas: sequence, label, prob, predicted_label
    """
    # 1. Leitura do CSV
    df_input    = pd.read_csv(csv_path, sep=",")
    sequences   = df_input["sequence"].tolist()
    true_labels = df_input["label"].values.astype(int)

    # 2. Embeddings ESM2
    model_esm, batch_converter = load_esm()
    embeddings = get_embeddings(sequences, model_esm, batch_converter)
    del model_esm, batch_converter
    torch.cuda.empty_cache()

    # 3. Tensor de entrada
    X         = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    input_dim = X.shape[1]

    # 4. Carrega o modelo
    mlp = load_mlp(model_path, input_dim)

    # 5. Probabilidades e predições
    probs           = predict_probs(mlp, X)
    predicted_label = (probs >= THRESHOLD).astype(int)

    # 6. Métricas
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_label).ravel()

    acc    = accuracy_score(true_labels, predicted_label)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc    = matthews_corrcoef(true_labels, predicted_label)
    f1     = f1_score(true_labels, predicted_label)
    pr     = precision_score(true_labels, predicted_label)
    auc    = roc_auc_score(true_labels, probs)
    aucpr  = average_precision_score(true_labels, probs)

    print("\n" + "=" * 45)
    print(f"  MÉTRICAS — {model_name:^33}")
    print("=" * 45)
    print(f"  ACC    : {acc:.4f}")
    print(f"  RECALL : {recall:.4f}   (Sensitivity)")
    print(f"  SP     : {sp:.4f}   (Specificity)")
    print(f"  MCC    : {mcc:.4f}")
    print(f"  AUC    : {auc:.4f}")
    print(f"  AUCPR  : {aucpr:.4f}")
    print(f"  F1     : {f1:.4f}")
    print(f"  PR     : {pr:.4f}   (Precision)")
    print("=" * 45)
    print(f"  Positivos preditos : {predicted_label.sum()} / {len(predicted_label)}")
    print("=" * 45 + "\n")

    # 7. Monta o DataFrame de saída
    result_df = pd.DataFrame({
        "sequence":        sequences,
        "label":           true_labels,
        "prob":            probs.round(4),
        "predicted_label": predicted_label,
    })

    # 8. Salva CSV se solicitado
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"[CSV] Resultado salvo em: {output_path}")

    return result_df


run_single('Datasets/Independent Dataset/independent_corrected.csv','bruno/model_cd08_only.pth', 'model_08', 'bruno/results/results_08_independent.csv')