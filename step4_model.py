"""
PASSO 4 — Modelo GNN (GAT + Attention Pooling + MLP)
=====================================================
Responsabilidade:
  - Definir a arquitetura do modelo completo
  - GAT: propaga e atualiza embeddings dos nós com atenção
  - Attention Pooling: agrega nós → vetor do grafo
  - [NOVO] Features globais: concatenadas ao vetor do grafo após pooling
  - MLP: classifica o vetor combinado em 0 ou 1

Fluxo por batch:
  x (N_total, D)        →  GAT camadas  →  x' (N_total, H)
  x' + batch_idx        →  Attention Pooling  →  h_graph (B, H)
  global_feats (B, G)   →  Projeção linear    →  h_global (B, H//2)  [opcional]
  concat([h_graph, h_global])                 →  h_combined (B, H + H//2)
  h_combined            →  MLP  →  logit (B,)
  logit                 →  sigmoid  →  prob (B,)  [só na inferência]

Por que concatenar após o pooling?
  As features globais descrevem o peptídeo inteiro (massa, carga, etc.),
  não resíduos individuais. O ponto natural de injeção é após o pooling,
  onde o grafo já foi reduzido a um único vetor representando o peptídeo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax as pyg_softmax


# ── Attention Pooling ─────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Agrega os embeddings dos nós de cada grafo num único vetor,
    usando pesos de atenção aprendidos.

    Para cada nó i com embedding h_i:
      score_i  = w^T · tanh(W · h_i + b)   ← MLP de 1 camada
      alpha_i  = softmax(score_i) dentro do grafo
      h_grafo  = Σ alpha_i * h_i            ← soma ponderada

    O vetor 'w' e a matriz 'W' são aprendidos durante o treino.
    Nós mais "informativos para a classificação" recebem alpha maior.
    """

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        # Projeta cada nó para o espaço de atenção
        self.W = nn.Linear(hidden_dim, attn_dim)
        # Vetor que pondera a importância
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x     : (N_total, hidden_dim) — embeddings de todos os nós do batch
        batch : (N_total,) — índice do grafo de cada nó (0, 0, ..., 1, 1, ...)
        Retorna: (B, hidden_dim) — um vetor por grafo
        """
        # score escalar por nó: (N_total, 1)
        score = self.w(torch.tanh(self.W(x)))

        # softmax dentro de cada grafo (usando pyg_softmax que respeita o batch)
        alpha = pyg_softmax(score, batch)    # (N_total, 1)

        # soma ponderada: (B, hidden_dim)
        B = int(batch.max().item()) + 1
        out = torch.zeros(B, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), alpha * x)

        return out


# ── Modelo Principal ──────────────────────────────────────────────────────────

class PeptideGNN(nn.Module):
    """
    Modelo GNN para classificação de peptídeos.

    Arquitetura:
      [Projeção inicial]
        Linear(in_dim → hidden_dim) + LayerNorm + ELU

      [GAT layers × num_layers]
        Cada camada GATv2Conv(hidden_dim → hidden_dim, heads=num_heads)
        com multi-head attention. As heads são concatenadas →
        dimensão temporária = hidden_dim * num_heads, depois projetada
        de volta para hidden_dim com uma Linear.
        Após cada camada: LayerNorm + ELU + Dropout + residual

      [Attention Pooling]
        (N_total, hidden_dim) → (B, hidden_dim)

      [Projeção de features globais]  ← NOVO
        (B, num_global_features) → Linear → LayerNorm → ELU → (B, hidden_dim//2)
        Concatenado com o vetor do grafo: (B, hidden_dim + hidden_dim//2)

      [MLP Classifier]
        (hidden_dim [+ hidden_dim//2]) → hidden_dim//2 → 1
        Com LayerNorm e Dropout no meio.

      [Saída] logit escalar por grafo, shape (B,)
    """

    def __init__(
        self,
        in_dim: int,                    # D — dimensão do embedding ESM-2
        hidden_dim: int = 128,          # dimensão interna das camadas GAT
        num_layers: int = 3,            # número de camadas GAT
        num_heads: int = 4,             # cabeças de atenção por camada GAT
        attn_dim: int = 64,             # dimensão interna do attention pooling
        dropout: float = 0.2,           # dropout após cada camada
        num_global_features: int = 0,   # número de features globais (Boruta)
    ):
        super().__init__()

        self.dropout_p = dropout

        # ── Projeção inicial ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # ── Camadas GAT ───────────────────────────────────────────────────────
        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.projs      = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            # Projeta de volta de (hidden_dim * num_heads) → hidden_dim
            proj = nn.Linear(hidden_dim * num_heads, hidden_dim, bias=False)
            norm = nn.LayerNorm(hidden_dim)

            self.gat_layers.append(conv)
            self.projs.append(proj)
            self.norms.append(norm)

        # ── Attention Pooling ─────────────────────────────────────────────────
        self.pooling = AttentionPooling(hidden_dim, attn_dim)

        # ── Projeção de features globais ──────────────────────────────────────
        # Só é criada se num_global_features > 0
        self.use_global = num_global_features > 0
        if self.use_global:
            self.global_proj = nn.Sequential(
                nn.Linear(num_global_features, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ELU(),
            )
            classifier_in_dim = hidden_dim + hidden_dim // 2
        else:
            classifier_in_dim = hidden_dim

        # ── Classificador MLP ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """
        data : batch PyG com atributos:
               - x            (N_total, D)
               - edge_index   (2, E_total)
               - batch        (N_total,)
               - global_features (B, G)  ← NOVO, opcional
        Retorna logit (B,) — um escalar por grafo
        """
        x          = data.x
        edge_index = data.edge_index
        batch      = data.batch

        # 1. Projeção inicial: D → hidden_dim
        x = self.input_proj(x)

        # 2. Camadas GAT com conexão residual
        for conv, proj, norm in zip(self.gat_layers, self.projs, self.norms):
            residual = x

            # GAT: (N, hidden_dim) → (N, hidden_dim * num_heads)
            x_new = conv(x, edge_index)

            # Projeta de volta para hidden_dim
            x_new = proj(x_new)

            # Residual + norm + ativação + dropout
            x = norm(x_new + residual)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3. Attention Pooling: (N_total, hidden_dim) → (B, hidden_dim)
        h_graph = self.pooling(x, batch)

        # 4. Concatenar features globais (se fornecidas)
        if self.use_global and hasattr(data, 'global_features'):
            # global_features: (B, G)
            g = data.global_features.to(h_graph.device)
            h_global = self.global_proj(g)           # (B, hidden_dim // 2)
            h_graph = torch.cat([h_graph, h_global], dim=-1)  # (B, hidden_dim + hidden_dim//2)

        # 5. Classificador: → logit (B,)
        return self.classifier(h_graph).squeeze(-1)


# ── Teste rápido ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch

    print("Testando PeptideGNN com e sem features globais...\n")

    D = 480    # dimensão ESM-2 35M
    N1, N2 = 8, 12   # tamanhos dos dois peptídeos de teste
    G = 65     # número de features globais (Boruta)

    def grafo_aleatorio(n_nos, embed_dim, n_global, label):
        """Cria um grafo completo aleatório para teste."""
        # Grafo completo: todas as arestas entre n_nos nós
        src = torch.arange(n_nos).repeat_interleave(n_nos)
        dst = torch.arange(n_nos).repeat(n_nos)
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=0)

        return Data(
            x=torch.randn(n_nos, embed_dim),
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.float),
            global_features=torch.randn(1, n_global),
        )

    g1 = grafo_aleatorio(N1, D, G, label=1)
    g2 = grafo_aleatorio(N2, D, G, label=0)
    batch = Batch.from_data_list([g1, g2])

    # ── Teste 1: sem features globais ────────────────────────────────────────
    model_base = PeptideGNN(
        in_dim=D,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        attn_dim=64,
        dropout=0.2,
        num_global_features=0,   # sem features globais
    )
    model_base.eval()
    with torch.no_grad():
        logits = model_base(batch)
    print(f"[Sem features globais]")
    print(f"  Logits shape : {logits.shape}")   # esperado: (2,)
    print(f"  Logits       : {logits}")
    print(f"  Probs        : {torch.sigmoid(logits)}\n")

    # ── Teste 2: com features globais ────────────────────────────────────────
    model_global = PeptideGNN(
        in_dim=D,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        attn_dim=64,
        dropout=0.2,
        num_global_features=G,   # 65 features do Boruta
    )
    model_global.eval()
    with torch.no_grad():
        logits = model_global(batch)
    print(f"[Com {G} features globais]")
    print(f"  Logits shape : {logits.shape}")   # esperado: (2,)
    print(f"  Logits       : {logits}")
    print(f"  Probs        : {torch.sigmoid(logits)}\n")

    # Contagem de parâmetros
    n_base   = sum(p.numel() for p in model_base.parameters())
    n_global = sum(p.numel() for p in model_global.parameters())
    print(f"Parâmetros (sem global)  : {n_base:,}")
    print(f"Parâmetros (com global)  : {n_global:,}")
    print(f"Diferença                : +{n_global - n_base:,} params")
    print("\nTeste concluído com sucesso.")
