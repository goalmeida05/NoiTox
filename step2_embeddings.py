"""
PASSO 2 — Extração de embeddings com ESM-2
==========================================
Responsabilidade:
  - Carregar o modelo ESM-2 pré-treinado (Meta AI)
  - Para cada sequência, extrair o tensor de embeddings por resíduo
  - Processar em lotes (batches) para eficiência de memória
  - Retornar dict: sequência → tensor (L, D)

Entrada:  lista de sequências (strings)
Saída:    dict[str, torch.Tensor]  — shape por sequência: (L, D)
          onde L = comprimento da sequência, D = 480 ou 1280

Modelos disponíveis (do mais leve ao mais pesado):
  esm2_t6_8M_UR50D    →  D=320,  8M params   (muito rápido, qualidade menor)
  esm2_t12_35M_UR50D  →  D=480,  35M params  (recomendado para peptídeos)
  esm2_t30_150M_UR50D →  D=640,  150M params (bom equilíbrio)
  esm2_t33_650M_UR50D →  D=1280, 650M params (melhor qualidade, pesado)

Para peptídeos curtos (< 50 resíduos), o 35M é suficiente na maioria dos casos.
Use o 650M se tiver GPU e quiser máxima qualidade.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


# ── Constantes ──────────────────────────────────────────────────────────────

# Modelo padrão: bom equilíbrio entre velocidade e qualidade para peptídeos
MODELO_PADRAO = "esm2_t12_35M_UR50D"

# Tamanho do batch: quantas sequências processar de uma vez.
# Reduza se tiver OOM (Out of Memory). Aumente se tiver GPU grande.
BATCH_SIZE_PADRAO = 16


# ── Carregamento do modelo ───────────────────────────────────────────────────

def carregar_modelo_esm(nome_modelo: str = MODELO_PADRAO, device: str = None):
    """
    Carrega o modelo ESM-2 e o tokenizador (batch_converter).

    O ESM-2 usa a biblioteca 'fair-esm' da Meta:
        pip install fair-esm

    Alternativa via HuggingFace (mais fácil de instalar):
        pip install transformers
        → use carregar_modelo_esm_hf() abaixo

    Retorna:
        model       — o modelo ESM-2 em modo eval
        alphabet    — objeto com regras de tokenização
        batch_converter — função que converte (nome, seq) → tokens
        device      — 'cuda' ou 'cpu'
        embed_dim   — D (dimensão dos embeddings)
    """
    try:
        import esm as esm_lib
    except ImportError:
        raise ImportError(
            "Biblioteca ESM não encontrada.\n"
            "Instale com: pip install fair-esm\n"
            "Ou use a versão HuggingFace: pip install transformers"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ESM-2] Carregando modelo: {nome_modelo}")
    print(f"[ESM-2] Device: {device}")

    # Carrega modelo e alphabet diretamente do hub da Meta
    model, alphabet = esm_lib.pretrained.load_model_and_alphabet(nome_modelo)
    model = model.eval().to(device)

    batch_converter = alphabet.get_batch_converter()

    # Descobrir a dimensão do embedding consultando o modelo
    embed_dim = model.embed_dim
    print(f"[ESM-2] Dimensão do embedding (D): {embed_dim}")
    print(f"[ESM-2] Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    return model, alphabet, batch_converter, device, embed_dim


def carregar_modelo_esm_hf(nome_modelo: str = "facebook/esm2_t12_35M_UR50D",
                            device: str = None):
    """
    Alternativa usando HuggingFace Transformers.
    Mais fácil de instalar, mesmos pesos.

    pip install transformers

    Mapeamento de nomes:
      esm2_t6_8M_UR50D     → facebook/esm2_t6_8M_UR50D
      esm2_t12_35M_UR50D   → facebook/esm2_t12_35M_UR50D
      esm2_t30_150M_UR50D  → facebook/esm2_t30_150M_UR50D
      esm2_t33_650M_UR50D  → facebook/esm2_t33_650M_UR50D
    """
    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError:
        raise ImportError("Instale com: pip install transformers")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ESM-2/HF] Carregando: {nome_modelo}")
    print(f"[ESM-2/HF] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = EsmModel.from_pretrained(nome_modelo).eval().to(device)

    embed_dim = model.config.hidden_size
    print(f"[ESM-2/HF] Dimensão do embedding (D): {embed_dim}")

    return model, tokenizer, device, embed_dim


# ── Extração via fair-esm ────────────────────────────────────────────────────

@torch.no_grad()
def extrair_embeddings_esm(
    sequencias: List[str],
    model,
    alphabet,
    batch_converter,
    device: str,
    batch_size: int = BATCH_SIZE_PADRAO,
) -> Dict[str, torch.Tensor]:
    """
    Extrai embeddings por resíduo para cada sequência.

    O ESM-2 funciona como BERT para proteínas:
      - Tokeniza a sequência: [CLS] A C D E F [EOS]
      - Passa pela rede (atenção entre todos os tokens)
      - Extrai as representações da última camada
      - Remove os tokens especiais [CLS] e [EOS]
      - Resultado: tensor (L, D) — um vetor por resíduo

    Retorna:
        dict: sequência → tensor (L, D) em CPU
    """
    embeddings = {}
    n = len(sequencias)

    # Processar em batches para não explodir a memória
    for inicio in range(0, n, batch_size):
        fim = min(inicio + batch_size, n)
        lote = sequencias[inicio:fim]

        print(f"  Processando sequências {inicio+1}–{fim} de {n}...")

        # ESM espera lista de (nome, sequência)
        dados_batch = [(f"seq_{i}", seq) for i, seq in enumerate(lote)]

        # batch_converter faz o padding e tokenização
        _, _, tokens = batch_converter(dados_batch)
        tokens = tokens.to(device)

        # Forward pass — extraímos a última camada de representação
        saida = model(tokens, repr_layers=[model.num_layers],
                      return_contacts=False)

        # representations["token_representations"][camada] → (B, L+2, D)
        # +2 porque inclui [CLS] no início e [EOS] no fim
        representacoes = saida["representations"][model.num_layers]

        for j, seq in enumerate(lote):
            L = len(seq)
            # Fatia [1 : L+1] remove [CLS] (pos 0) e [EOS] (pos L+1)
            emb = representacoes[j, 1:L+1, :]  # (L, D)
            embeddings[seq] = emb.cpu()

    return embeddings


# ── Extração via HuggingFace ─────────────────────────────────────────────────

@torch.no_grad()
def extrair_embeddings_hf(
    sequencias: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = BATCH_SIZE_PADRAO,
) -> Dict[str, torch.Tensor]:
    """
    Mesma lógica, usando o modelo HuggingFace.
    """
    embeddings = {}
    n = len(sequencias)

    for inicio in range(0, n, batch_size):
        fim = min(inicio + batch_size, n)
        lote = sequencias[inicio:fim]

        print(f"  Processando sequências {inicio+1}–{fim} de {n}...")

        # Tokenizar — padding automático para o maior do lote
        inputs = tokenizer(
            lote,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        saida = model(**inputs)

        # last_hidden_state: (B, L+2, D)
        representacoes = saida.last_hidden_state

        for j, seq in enumerate(lote):
            L = len(seq)
            emb = representacoes[j, 1:L+1, :]  # remove [CLS] e [EOS]
            embeddings[seq] = emb.cpu()

    return embeddings


# ── Função principal do passo 2 ──────────────────────────────────────────────

def extrair_todos_embeddings(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    nome_modelo: str = MODELO_PADRAO,
    batch_size: int = BATCH_SIZE_PADRAO,
    usar_hf: bool = False,
    device: str = None,
) -> Dict[str, torch.Tensor]:
    """
    Função principal do Passo 2.

    Combina treino + teste em uma única passagem para evitar
    reprocessar sequências duplicadas entre os splits.

    Retorna:
        embeddings: dict[sequência → tensor (L, D)]
    """
    print("=" * 50)
    print("PASSO 2 — Extração de embeddings ESM-2")
    print("=" * 50)

    # Unir todas as sequências únicas dos dois splits
    todas_seqs = list(set(df_treino["sequence"].tolist() +
                          df_teste["sequence"].tolist()))
    print(f"Total de sequências únicas: {len(todas_seqs)}")

    if usar_hf:
        nome_hf = f"facebook/{nome_modelo}" if not nome_modelo.startswith("facebook") else nome_modelo
        model, tokenizer, device, embed_dim = carregar_modelo_esm_hf(nome_hf, device)
        print(f"\nExtraindo embeddings...")
        embeddings = extrair_embeddings_hf(todas_seqs, model, tokenizer,
                                           device, batch_size)
    else:
        model, alphabet, batch_converter, device, embed_dim = carregar_modelo_esm(
            nome_modelo, device)
        print(f"\nExtraindo embeddings...")
        embeddings = extrair_embeddings_esm(todas_seqs, model, alphabet,
                                            batch_converter, device, batch_size)

    # Verificação de sanidade
    for seq, emb in list(embeddings.items())[:3]:
        assert emb.shape == (len(seq), embed_dim), \
            f"Shape inesperado: {emb.shape} para seq de comprimento {len(seq)}"

    print(f"\n[PASSO 2 CONCLUÍDO]")
    print(f"  {len(embeddings)} embeddings extraídos")
    print(f"  Shape exemplo: seq '{todas_seqs[0]}' → {embeddings[todas_seqs[0]].shape}")
    print()

    return embeddings


# ── Versão simulada para testes sem GPU / sem ESM instalado ─────────────────

def embeddings_simulados(
    sequencias: List[str],
    embed_dim: int = 480,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Gera embeddings aleatórios para testes do pipeline sem precisar
    do ESM-2 instalado. NÃO use para treino real.

    Útil para:
      - Testar a construção do grafo e o modelo GNN
      - Verificar shapes e o fluxo de dados
      - Rodar em ambiente sem GPU/internet
    """
    torch.manual_seed(seed)
    return {
        seq: torch.randn(len(seq), embed_dim)
        for seq in sequencias
    }


# ── Teste rápido (sem ESM real) ──────────────────────────────────────────────
if __name__ == "__main__":
    sequencias_teste = ["ACDEF", "GHIKLM", "PEPTIDE", "ACDE"]

    print("Testando com embeddings simulados (D=480)...")
    embs = embeddings_simulados(sequencias_teste, embed_dim=480)

    for seq, emb in embs.items():
        print(f"  {seq:12s} → shape: {tuple(emb.shape)}  "
              f"(esperado: ({len(seq)}, 480))")

    print("\nVerificação de shapes: OK")
    print()
    print("Para usar ESM-2 real:")
    print("  pip install fair-esm          # opção 1 (Meta)")
    print("  pip install transformers      # opção 2 (HuggingFace)")
    print()
    print("Então substitua embeddings_simulados() por extrair_todos_embeddings()")
