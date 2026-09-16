"""
PASSO 3 — Construção dos Grafos (PyTorch Geometric)
====================================================
Responsabilidade:
  - Receber CSV com colunas 'sequence' e 'label' apenas
  - Calcular as 44 features globais selecionadas pelo Boruta diretamente
    da sequência (sem precisar de colunas pré-calculadas no CSV)
  - Construir grafo completo (todos os resíduos conectados entre si)
  - Empacotar num objeto PyG Data com campo global_features
  - Aplicar resampling estratificado por label × Cys/Met (opcional)
  - Retornar DataLoaders prontos para treino/teste

Campos do objeto Data por peptídeo:
  x               : (L, D)    embeddings dos resíduos (nós) via ESM-2
  edge_index      : (2, E)    arestas do grafo completo, sem self-loops
  y               : (1,)      label binário (0 ou 1)
  seq             : str       sequência original (para CSV de saída)
  global_features : (1, 44)   features físico-químicas globais do Boruta
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler


# ── Constantes de normalização ────────────────────────────────────────────────

_NORM_MASS_MIN   = 516.6314
_NORM_MASS_MAX   = 5121.012000000002
_NORM_PI_MIN     = 4.0500284194946286
_NORM_PI_MAX     = 12.999967765808105
_NORM_CHARGE_MIN = -8.173321821201688
_NORM_CHARGE_MAX = 23.75392760999528
_NORM_GRAVY_MIN  = -4.5
_NORM_GRAVY_MAX  = 2.7111111111111112

# Grupos de aminoácidos para CKSAAGP
_AA_GROUPS = {
    'hydrophobic':        list('AVLIMFWC'),
    'polar_non_charged':  list('STNQ'),
    'positively_charged': list('KRH'),
    'negatively_charged': list('DE'),
}
_AA_TO_GROUP = {aa: grp for grp, aas in _AA_GROUPS.items() for aa in aas}

# Composição atômica por aminoácido
_ATOM_COMP = {
    'A': {'nC': 3,  'nH': 7,  'nN': 1, 'nO': 2, 'nS': 0},
    'R': {'nC': 6,  'nH': 14, 'nN': 4, 'nO': 2, 'nS': 0},
    'N': {'nC': 4,  'nH': 8,  'nN': 2, 'nO': 3, 'nS': 0},
    'D': {'nC': 4,  'nH': 7,  'nN': 1, 'nO': 4, 'nS': 0},
    'C': {'nC': 3,  'nH': 7,  'nN': 1, 'nO': 2, 'nS': 1},
    'E': {'nC': 5,  'nH': 9,  'nN': 1, 'nO': 4, 'nS': 0},
    'Q': {'nC': 5,  'nH': 10, 'nN': 2, 'nO': 3, 'nS': 0},
    'G': {'nC': 2,  'nH': 5,  'nN': 1, 'nO': 2, 'nS': 0},
    'H': {'nC': 6,  'nH': 9,  'nN': 3, 'nO': 2, 'nS': 0},
    'I': {'nC': 6,  'nH': 13, 'nN': 1, 'nO': 2, 'nS': 0},
    'L': {'nC': 6,  'nH': 13, 'nN': 1, 'nO': 2, 'nS': 0},
    'K': {'nC': 6,  'nH': 14, 'nN': 2, 'nO': 2, 'nS': 0},
    'M': {'nC': 5,  'nH': 11, 'nN': 1, 'nO': 2, 'nS': 1},
    'F': {'nC': 9,  'nH': 11, 'nN': 1, 'nO': 2, 'nS': 0},
    'P': {'nC': 5,  'nH': 9,  'nN': 1, 'nO': 2, 'nS': 0},
    'S': {'nC': 3,  'nH': 7,  'nN': 1, 'nO': 3, 'nS': 0},
    'T': {'nC': 4,  'nH': 9,  'nN': 1, 'nO': 3, 'nS': 0},
    'W': {'nC': 11, 'nH': 12, 'nN': 2, 'nO': 2, 'nS': 0},
    'Y': {'nC': 9,  'nH': 11, 'nN': 1, 'nO': 3, 'nS': 0},
    'V': {'nC': 5,  'nH': 11, 'nN': 1, 'nO': 2, 'nS': 0},
}

_KSAAGP_PAIRS = [
    f"{g1}-{g2}"
    for g1 in _AA_GROUPS
    for g2 in _AA_GROUPS
]

# Features selecionadas pelo Boruta (44 no total) — ordem fixa
GLOBAL_FEATURE_COLS = [
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
    "normCC", "normCCS", "normCG", "normCK", "normCP",
    "normCS", "normCY", "normDC", "normEC", "normGC",
    "normKK", "normPC", "normTC", "normWC",
]

NUM_GLOBAL_FEATURES = len(GLOBAL_FEATURE_COLS)  # 44

# Resíduos sulfurados para o resampling
RESIDUOS_SULFURADOS = {"C", "M"}


# ── Cálculo das features globais ──────────────────────────────────────────────

def calc_global_features(seq: str) -> np.ndarray:
    """
    Calcula as 44 features globais do Boruta diretamente da sequência.
    Retorna np.ndarray de shape (44,).
    """
    seq = str(seq).upper()
    L   = len(seq)
    pa  = ProteinAnalysis(seq)
    pa.count_amino_acids()
    scale = ProtParamData.gravy_scales.get('KyteDoolitle')

    feats = {}

    # Globais simples
    mass   = pa.molecular_weight()
    pi     = pa.isoelectric_point()
    charge = pa.charge_at_pH(7.0)
    gravy  = pa.gravy()
    hydrophilic = sum(1 for a in seq if scale.get(a, 0) < 0)

    feats["Norm_mass"]              = (mass   - _NORM_MASS_MIN)   / (_NORM_MASS_MAX   - _NORM_MASS_MIN)
    feats["Norm_isoelectric_point"] = (pi     - _NORM_PI_MIN)     / (_NORM_PI_MAX     - _NORM_PI_MIN)
    feats["Norm_net_charge"]        = (charge - _NORM_CHARGE_MIN) / (_NORM_CHARGE_MAX - _NORM_CHARGE_MIN)
    feats["Norm_gravy"]             = (gravy  - _NORM_GRAVY_MIN)  / (_NORM_GRAVY_MAX  - _NORM_GRAVY_MIN)
    feats["Norm_hidro/total"]       = hydrophilic / L
    feats["len"]                    = float(L)

    # Composição atômica
    nC = nH = nN = nO = nS = 0
    for aa, cnt in pa.amino_acids_content.items():
        comp = _ATOM_COMP.get(aa, {})
        nC  += comp.get('nC', 0) * cnt
        nH  += comp.get('nH', 0) * cnt
        nN  += comp.get('nN', 0) * cnt
        nO  += comp.get('nO', 0) * cnt
        nS  += comp.get('nS', 0) * cnt

    nH -= 2 * (L - 1)
    nO -= (L - 1)

    total = nC + nH + nN + nO + nS
    feats["normnC"]         = nC / total if total > 0 else 0.0
    feats["normnH"]         = nH / total if total > 0 else 0.0
    feats["normnN"]         = nN / total if total > 0 else 0.0
    feats["normnO"]         = nO / total if total > 0 else 0.0
    feats["normnS"]         = nS / total if total > 0 else 0.0
    feats["total_elements"] = float(total)

    # CKSAAGP (k=1)
    ksaagp = {p: 0.0 for p in _KSAAGP_PAIRS}
    for i in range(L - 2):
        g1 = _AA_TO_GROUP.get(seq[i])
        g2 = _AA_TO_GROUP.get(seq[i + 2])
        if g1 and g2:
            ksaagp[f"{g1}-{g2}"] += 1.0
    for p in ksaagp:
        ksaagp[p] /= L

    pair_to_col = {
        "hydrophobic-negatively_charged":       "norm_hydrophobic_negatively_charged_CKSAAGP",
        "hydrophobic-polar_non_charged":         "norm_hydrophobic_polar_non_charged_CKSAAGP",
        "hydrophobic-positively_charged":        "norm_hydrophobic_positively_charged_CKSAAGP",
        "negatively_charged-polar_non_charged":  "norm_negatively_charged_polar_non_charged_CKSAAGP",
        "polar_non_charged-hydrophobic":         "norm_polar_non_charged_hydrophobic_CKSAAGP",
        "polar_non_charged-polar_non_charged":   "norm_polar_non_charged_polar_non_charged_CKSAAGP",
        "polar_non_charged-positively_charged":  "norm_polar_non_charged_positively_charged_CKSAAGP",
        "positively_charged-hydrophobic":        "norm_positively_charged_hydrophobic_CKSAAGP",
        "positively_charged-polar_non_charged":  "norm_positively_charged_polar_non_charged_CKSAAGP",
    }
    for pair, col in pair_to_col.items():
        feats[col] = ksaagp.get(pair, 0.0)

    # AAC
    for aa in ['A', 'C', 'I', 'K', 'L', 'M', 'T', 'V', 'W']:
        feats[f"norm{aa}"] = pa.amino_acids_content.get(aa, 0) / L

    # Dipeptídeos
    dipep_targets = {
        "normCC": ("C","C"), "normCG": ("C","G"), "normCK": ("C","K"),
        "normCP": ("C","P"), "normCS": ("C","S"), "normCY": ("C","Y"),
        "normDC": ("D","C"), "normEC": ("E","C"), "normGC": ("G","C"),
        "normKK": ("K","K"), "normPC": ("P","C"), "normTC": ("T","C"),
        "normWC": ("W","C"),
    }
    den = L - 1 if L > 1 else 1
    for col, (a1, a2) in dipep_targets.items():
        count = sum(1 for i in range(L - 1) if seq[i] == a1 and seq[i+1] == a2)
        feats[col] = count / den

    # Tripeptídeo CCS
    den3 = L - 2 if L > 2 else 1
    count_ccs = sum(
        1 for i in range(L - 2)
        if seq[i] == 'C' and seq[i+1] == 'C' and seq[i+2] == 'S'
    )
    feats["normCCS"] = count_ccs / den3

    return np.array([feats[col] for col in GLOBAL_FEATURE_COLS], dtype=np.float32)


# ── Construção de arestas ─────────────────────────────────────────────────────

def build_grafo_completo(L: int) -> torch.Tensor:
    indices = torch.arange(L)
    pares   = torch.combinations(indices, r=2)
    src = torch.cat([pares[:, 0], pares[:, 1]])
    dst = torch.cat([pares[:, 1], pares[:, 0]])
    return torch.stack([src, dst], dim=0)


# ── Construção de um grafo único ──────────────────────────────────────────────

def sequencia_para_grafo(
    seq: str,
    embedding: torch.Tensor,
    label: int,
    use_global: bool = True,
) -> Data:
    L = embedding.shape[0]
    edge_index = build_grafo_completo(L)

    data = Data(
        x          = embedding.float(),
        edge_index = edge_index,
        y          = torch.tensor([label], dtype=torch.float),
    )
    data.seq       = seq
    data.num_nodes = L

    if use_global:
        try:
            vals = calc_global_features(seq)
            data.global_features = torch.tensor(vals, dtype=torch.float).unsqueeze(0)
        except Exception as e:
            print(f"[AVISO] Erro ao calcular features de '{seq[:20]}': {e}")
            data.global_features = torch.zeros(1, NUM_GLOBAL_FEATURES)

    return data


# ── Dataset ───────────────────────────────────────────────────────────────────

class PeptideDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, torch.Tensor],
        use_global: bool = True,
    ):
        super().__init__()
        self.df         = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.use_global = use_global
        self.num_global_features = NUM_GLOBAL_FEATURES if use_global else 0

    def len(self):
        return len(self.df)

    def get(self, idx):
        row   = self.df.iloc[idx]
        seq   = str(row["sequence"]).upper()
        label = int(row["label"])
        emb   = self.embeddings[seq]
        return sequencia_para_grafo(seq, emb, label, self.use_global)


# ── Resampling estratificado ──────────────────────────────────────────────────

def tem_residuo_sulfurado(seq: str) -> bool:
    return any(aa in RESIDUOS_SULFURADOS for aa in seq.upper())


def calcular_pesos_resampling(df: pd.DataFrame) -> torch.Tensor:
    sulfurado = df["sequence"].apply(tem_residuo_sulfurado)
    label     = df["label"].astype(int)
    grupo     = label * 2 + sulfurado.astype(int)
    freq      = grupo.value_counts()

    print("\n  Resampling estratificado por label × Cys/Met:")
    nomes = {
        0: "label=0, sem Cys/Met",
        1: "label=0, com Cys/Met",
        2: "label=1, sem Cys/Met  [minoritário]",
        3: "label=1, com Cys/Met  [dominante]",
    }
    for g in sorted(freq.index):
        print(f"    grupo {g} ({nomes.get(g, g)}): {freq[g]} seqs, peso={1/freq[g]:.5f}")

    pesos = grupo.map(lambda g: 1.0 / freq[g])
    return torch.tensor(pesos.values, dtype=torch.float)


# ── Função principal ──────────────────────────────────────────────────────────

def construir_dataloaders(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    embeddings: Dict[str, torch.Tensor],
    batch_size: int = 32,
    use_global: bool = True,
    num_workers: int = 0,
    resampling: bool = False,
):
    """
    Cria DataLoaders de treino e teste.

    Parâmetros:
        use_global  : calcula e anexa as 44 features físico-químicas do Boruta
        resampling  : WeightedRandomSampler estratificado por label × Cys/Met

    Retorna:
        loader_treino, loader_teste, embed_dim, num_global_features
    """
    print("=" * 50)
    print("PASSO 3 — Construção dos grafos")
    print("=" * 50)
    print(f"Batch size      : {batch_size}")
    print(f"Features globais: {'ativadas (44 Boruta)' if use_global else 'desativadas'}")
    print(f"Resampling      : {'ativado (label × Cys/Met)' if resampling else 'desativado'}")

    ds_treino = PeptideDataset(df_treino, embeddings, use_global=use_global)
    ds_teste  = PeptideDataset(df_teste,  embeddings, use_global=use_global)

    # Sampler de treino
    if resampling:
        pesos   = calcular_pesos_resampling(df_treino)
        sampler = WeightedRandomSampler(
            weights=pesos,
            num_samples=len(ds_treino),
            replacement=True,
        )
        loader_treino = DataLoader(
            ds_treino, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers,
        )
    else:
        loader_treino = DataLoader(
            ds_treino, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
        )

    loader_teste = DataLoader(
        ds_teste, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    seq_ex    = df_treino["sequence"].iloc[0]
    embed_dim = embeddings[seq_ex].shape[1]
    n_global  = ds_treino.num_global_features

    print(f"\nResumo:")
    print(f"  Treino          : {len(ds_treino)} peptídeos")
    print(f"  Teste           : {len(ds_teste)} peptídeos")
    print(f"  Dim. embedding  : {embed_dim}")
    print(f"  Features globais: {n_global}")
    print(f"\n[PASSO 3 CONCLUÍDO]\n")

    return loader_treino, loader_teste, embed_dim, n_global
