import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from step1_data_loader   import validar_sequencia
from step2_embeddings    import (extrair_todos_embeddings,
                                 embeddings_simulados,
                                 MODELO_PADRAO)
from step3_graph_builder import sequencia_para_grafo
from step4_model         import PeptideGNN

from torch_geometric.loader import DataLoader as PyGDataLoader


# ── Configuração padrão do modelo ─────────────────────────────────────────────
# Deve bater com os hiperparâmetros usados no treino.
# Altere aqui ou passe via kwargs para run_modelo_grafo().

CONFIG_PADRAO = dict(
    caminho_modelo = "output/melhor_modelo.pt",
    hidden_dim     = 128,
    num_layers     = 3,
    num_heads      = 4,
    attn_dim       = 64,
    dropout        = 0.2,
    esm_modelo     = MODELO_PADRAO,
    usar_hf        = False,
    simular        = False,
    batch_size     = 32,
    device         = None,
)


# ── Leitura de arquivos ───────────────────────────────────────────────────────

def _ler_fasta(caminho: Path) -> pd.DataFrame:
    """
    Lê um arquivo .fasta e retorna DataFrame com colunas:
        sequence         — sequência de aminoácidos
        label_real       — ausente (None), arquivo FASTA não tem labels

    Suporta sequências multi-linha.
    """
    sequencias = []
    seq_atual  = []

    with open(caminho, "r") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith(">"):
                if seq_atual:
                    sequencias.append("".join(seq_atual).upper())
                    seq_atual = []
            else:
                seq_atual.append(linha)
        if seq_atual:
            sequencias.append("".join(seq_atual).upper())

    print(f"  {len(sequencias)} sequências lidas do FASTA")
    return pd.DataFrame({"sequence": sequencias})


def _ler_csv(caminho: Path) -> tuple[pd.DataFrame, bool]:
    """
    Lê um .csv com coluna obrigatória 'sequence' e opcional 'label'.
    Retorna (DataFrame, tem_label).
    """
    df = pd.read_csv(caminho)
    if "sequence" not in df.columns:
        raise ValueError(f"CSV sem coluna 'sequence'. "
                         f"Colunas encontradas: {list(df.columns)}")
    tem_label = "label" in df.columns
    print(f"  {len(df)} sequências lidas do CSV "
          f"({'com' if tem_label else 'sem'} labels)")
    return df, tem_label


def _carregar_arquivo(caminho: str) -> tuple[pd.DataFrame, bool]:
    """
    Detecta o formato pelo sufixo e delega para o leitor correto.
    Valida as sequências e remove as inválidas.

    Retorna:
        df        — DataFrame com 'sequence' (e 'label' se existir)
        tem_label — bool
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    sufixo = caminho.suffix.lower()

    if sufixo in (".fasta", ".fa", ".faa"):
        df_raw  = _ler_fasta(caminho)
        tem_label = False
    elif sufixo == ".csv":
        df_raw, tem_label = _ler_csv(caminho)
    else:
        raise ValueError(f"Formato não suportado: '{sufixo}'. "
                         f"Use .fasta, .fa, .faa ou .csv")

    # Validar sequências
    validas   = []
    removidas = 0
    for _, row in df_raw.iterrows():
        seq = str(row["sequence"]).strip().upper() if pd.notna(row["sequence"]) else ""
        ok, motivo = validar_sequencia(seq)
        if ok:
            entrada = {"sequence": seq}
            if tem_label:
                entrada["label_real"] = int(row["label"])
            validas.append(entrada)
        else:
            print(f"  [REMOVIDA] '{seq[:25]}' → {motivo}")
            removidas += 1

    df = pd.DataFrame(validas)
    if removidas:
        print(f"  {removidas} sequências removidas por validação")
    print(f"  {len(df)} sequências válidas para inferência")

    return df, tem_label


# ── Modelo ────────────────────────────────────────────────────────────────────

def _carregar_modelo(caminho_pt: str, embed_dim: int,
                     hidden_dim: int, num_layers: int,
                     num_heads: int, attn_dim: int,
                     dropout: float, device: str) -> nn.Module:
    caminho_pt = Path(caminho_pt)
    if not caminho_pt.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_pt}")

    model = PeptideGNN(
        in_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, num_heads=num_heads,
        attn_dim=attn_dim, dropout=dropout,
        num_global_features=44,  # <- faltava isso
    )
    model.load_state_dict(torch.load(caminho_pt, map_location=device))
    model.eval().to(device)

    print(f"  Modelo carregado: {caminho_pt.name} "
          f"({sum(p.numel() for p in model.parameters()):,} parâmetros)")
    return model


# ── Inferência ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _inferir(model, sequencias, embeddings, device, batch_size):
    grafos = [
        sequencia_para_grafo(seq, embeddings[seq], label=0)
        for seq in sequencias
    ]
    loader    = PyGDataLoader(grafos, batch_size=batch_size, shuffle=False)
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        all_probs.extend(probs.tolist())

    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    return all_probs, all_preds


# ── Métricas ──────────────────────────────────────────────────────────────────

def _calcular_metricas(labels, probs, preds):
    from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                                 precision_score, recall_score,
                                 confusion_matrix, matthews_corrcoef)
    print("\n── Métricas no dataset independente ──")
    try:
        print(f"  AUROC    : {roc_auc_score(labels, probs):.4f}")
    except ValueError:
        print("  AUROC    : N/A (só uma classe presente)")
    print(f"  MCC      : {matthews_corrcoef(labels, preds):.4f}")
    print(f"  Acurácia : {accuracy_score(labels, preds):.4f}")
    print(f"  F1       : {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"  Precisão : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(labels, preds, zero_division=0):.4f}")
    cm = confusion_matrix(labels, preds)
    print(f"\n  Matriz de confusão:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


# ── Função pública principal ──────────────────────────────────────────────────

def run_modelo_grafo(
    caminho_arquivo : str,
    caminho_saida   : str  = None,
    caminho_modelo  : str  = CONFIG_PADRAO["caminho_modelo"],
    hidden_dim      : int  = CONFIG_PADRAO["hidden_dim"],
    num_layers      : int  = CONFIG_PADRAO["num_layers"],
    num_heads       : int  = CONFIG_PADRAO["num_heads"],
    attn_dim        : int  = CONFIG_PADRAO["attn_dim"],
    dropout         : float = CONFIG_PADRAO["dropout"],
    esm_modelo      : str  = CONFIG_PADRAO["esm_modelo"],
    usar_hf         : bool = CONFIG_PADRAO["usar_hf"],
    simular         : bool = CONFIG_PADRAO["simular"],
    batch_size      : int  = CONFIG_PADRAO["batch_size"],
    device          : str  = CONFIG_PADRAO["device"],
) -> pd.DataFrame:
    """
    Roda o modelo GNN treinado em um arquivo .fasta ou .csv.

    Parâmetros
    ----------
    caminho_arquivo : str
        Caminho para o arquivo de entrada (.fasta, .fa, .faa ou .csv).
        CSV deve ter coluna 'sequence' (e opcionalmente 'label').
        FASTA não precisa de labels.

    caminho_saida : str, opcional
        Se fornecido, salva o DataFrame de resultados nesse caminho (.csv).
        Se None, não salva — só retorna o DataFrame.

    caminho_modelo : str
        Caminho para o arquivo .pt com os pesos treinados.

    hidden_dim, num_layers, num_heads, attn_dim, dropout : int/float
        Hiperparâmetros da arquitetura — devem ser os mesmos do treino.

    esm_modelo : str
        Variante do ESM-2 usada para extrair embeddings.

    usar_hf : bool
        Se True, usa HuggingFace Transformers em vez de fair-esm.

    simular : bool
        Se True, usa embeddings aleatórios (apenas para testes).

    batch_size : int
        Sequências por batch na inferência.

    device : str ou None
        'cpu', 'cuda', 'mps'. Se None, detecta automaticamente.

    Retorna
    -------
    pd.DataFrame com colunas:
        sequence        — sequência de aminoácidos
        probabilidade   — probabilidade de ser tóxico (0 a 1)
        label_previsto  — 0 ou 1 (threshold = 0.5)
        label_real      — label verdadeiro (só se o arquivo tiver labels)

    Ordenado por probabilidade decrescente.
    """
    print("=" * 50)
    print("run_modelo_grafo — Inferência GNN")
    print("=" * 50)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Arquivo : {caminho_arquivo}")
    print(f"Device  : {device}\n")

    # 1. Carregar arquivo (fasta ou csv)
    df, tem_label = _carregar_arquivo(caminho_arquivo)
    sequencias    = df["sequence"].tolist()

    # 2. Embeddings ESM-2
    if simular:
        dim_map = {"esm2_t6_8M_UR50D": 320, "esm2_t12_35M_UR50D": 480,
                   "esm2_t30_150M_UR50D": 640, "esm2_t33_650M_UR50D": 1280}
        embed_dim  = dim_map.get(esm_modelo, 480)
        embeddings = embeddings_simulados(sequencias, embed_dim=embed_dim)
        print(f"\nEmbeddings simulados (D={embed_dim})")
    else:
        df_dummy   = pd.DataFrame({"sequence": sequencias,
                                   "label": [0] * len(sequencias)})
        print("\nExtraindo embeddings ESM-2...")
        embeddings = extrair_todos_embeddings(
            df_dummy, df_dummy,
            nome_modelo=esm_modelo,
            usar_hf=usar_hf,
            device=device,
        )

    embed_dim = next(iter(embeddings.values())).shape[1]

    # 3. Carregar modelo
    print("\nCarregando modelo treinado...")
    model = _carregar_modelo(
        caminho_modelo, embed_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, attn_dim=attn_dim,
        dropout=dropout, device=device,
    )

    # 4. Inferência
    print("\nRodando inferência...")
    probs, preds = _inferir(model, sequencias, embeddings, device, batch_size)
    print(f"  {len(sequencias)} sequências processadas")

    # 5. Montar DataFrame de resultado
    resultado = pd.DataFrame({
        "sequence"      : sequencias,
        "probabilidade" : np.round(probs, 6),
        "label_previsto": preds,
    })
    if tem_label:
        resultado["label_real"] = df["label_real"].values

    resultado = resultado.sort_values(
        "probabilidade", ascending=False
    ).reset_index(drop=True)

    # 6. Métricas (se houver labels reais)
    if tem_label:
        _calcular_metricas(
            resultado["label_real"].values,
            resultado["probabilidade"].values,
            resultado["label_previsto"].values,
        )

    # 7. Salvar CSV (opcional)
    if caminho_saida:
        Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(caminho_saida, index=False)
        print(f"\n  Salvo em: {caminho_saida}")

    print("\n[run_modelo_grafo CONCLUÍDO]")
    return resultado



run_modelo_grafo('Datasets/Independent Dataset/independent_corrected.csv', 'matheus/results/results_08_independent.csv', 'matheus/melhor_modelo08.pt')