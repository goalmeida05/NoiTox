"""
PASSO 1 — Carregamento e validação dos dados
=============================================
Responsabilidade:
  - Ler os CSVs de treino e teste
  - Validar cada sequência (apenas aminoácidos canônicos)
  - Reportar sequências inválidas sem quebrar o pipeline
  - Retornar DataFrames limpos e prontos para o próximo passo

Entrada:  train.csv, test.csv  (colunas: sequence, label)
Saída:    dois DataFrames pandas com sequências válidas
"""

import pandas as pd
import re
from pathlib import Path


# Os 20 aminoácidos canônicos. Qualquer outro caractere é inválido.
AMINOACIDOS_CANONICOS = set("ACDEFGHIKLMNPQRSTVWY")

# Comprimentos razoáveis para peptídeos.
# Sequências muito longas (> 50) tornam o grafo completo caro (quadrático).
# Sequências muito curtas (< 2) não formam grafo.
MIN_LEN = 2
MAX_LEN = 50


def carregar_csv(caminho: str) -> pd.DataFrame:
    """
    Lê o CSV e faz verificações básicas de estrutura.
    Retorna DataFrame com colunas 'sequence' e 'label'.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho, sep=",")

    # Verificar colunas obrigatórias
    colunas_esperadas = {"sequence", "label"}
    colunas_faltando = colunas_esperadas - set(df.columns)
    if colunas_faltando:
        raise ValueError(
            f"Colunas faltando em {caminho.name}: {colunas_faltando}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    print(f"[OK] {caminho.name}: {len(df)} linhas carregadas")
    return df


def validar_sequencia(seq: str) -> tuple[bool, str]:
    """
    Valida uma sequência de aminoácidos.
    Retorna (é_válida, motivo_se_inválida).
    """
    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return False, "sequência vazia ou nula"

    seq = seq.strip().upper()

    if len(seq) < MIN_LEN:
        return False, f"muito curta ({len(seq)} < {MIN_LEN})"

    if len(seq) > MAX_LEN:
        return False, f"muito longa ({len(seq)} > {MAX_LEN})"

    # Detectar caracteres inválidos
    invalidos = set(seq) - AMINOACIDOS_CANONICOS
    if invalidos:
        return False, f"caracteres inválidos: {sorted(invalidos)}"

    return True, ""


def validar_label(label) -> tuple[bool, str]:
    """
    Valida que o label é 0 ou 1.
    """
    try:
        val = int(label)
        if val not in (0, 1):
            return False, f"label deve ser 0 ou 1, encontrado: {val}"
        return True, ""
    except (ValueError, TypeError):
        return False, f"label não é inteiro: {repr(label)}"


def limpar_dataframe(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """
    Aplica todas as validações linha a linha.
    Remove linhas inválidas e reporta um resumo.
    """
    resultados = []

    for idx, row in df.iterrows():
        seq = str(row["sequence"]).strip().upper() if pd.notna(row["sequence"]) else ""
        label = row["label"]

        seq_ok, motivo_seq = validar_sequencia(seq)
        label_ok, motivo_label = validar_label(label)

        if seq_ok and label_ok:
            resultados.append({
                "sequence": seq,
                "label": int(label),
                "comprimento": len(seq),
            })
        else:
            motivo = motivo_seq or motivo_label
            print(f"  [REMOVIDA] linha {idx}: '{seq[:20]}...' → {motivo}")

    df_limpo = pd.DataFrame(resultados)

    # Relatório
    removidas = len(df) - len(df_limpo)
    print(f"\n--- Relatório: {nome} ---")
    print(f"  Total original : {len(df)}")
    print(f"  Válidas        : {len(df_limpo)}")
    print(f"  Removidas      : {removidas}")

    if len(df_limpo) > 0:
        n0 = (df_limpo["label"] == 0).sum()
        n1 = (df_limpo["label"] == 1).sum()
        print(f"  Classe 0       : {n0} ({100*n0/len(df_limpo):.1f}%)")
        print(f"  Classe 1       : {n1} ({100*n1/len(df_limpo):.1f}%)")
        print(f"  Comp. médio    : {df_limpo['comprimento'].mean():.1f} resíduos")
        print(f"  Comp. min/max  : {df_limpo['comprimento'].min()} / {df_limpo['comprimento'].max()}")

        # Alerta de desbalanceamento
        ratio = max(n0, n1) / max(min(n0, n1), 1)
        if ratio > 3:
            print(f"  [AVISO] Dataset desbalanceado (ratio {ratio:.1f}x).")
            print(f"          Use pos_weight no BCEWithLogitsLoss.")

    return df_limpo


def carregar_dados(caminho_treino: str, caminho_teste: str):
    """
    Função principal do passo 1.
    Carrega, valida e retorna os dois DataFrames prontos.

    Retorna:
        df_treino, df_teste  — DataFrames com colunas:
                               'sequence' (str), 'label' (int), 'comprimento' (int)
    """
    print("=" * 50)
    print("PASSO 1 — Carregamento e validação dos dados")
    print("=" * 50)

    df_treino_raw = carregar_csv(caminho_treino)
    df_teste_raw  = carregar_csv(caminho_teste)

    print("\nValidando treino...")
    df_treino = limpar_dataframe(df_treino_raw, "treino")

    print("\nValidando teste...")
    df_teste  = limpar_dataframe(df_teste_raw, "teste")

    if len(df_treino) == 0:
        raise ValueError("Nenhuma sequência válida no treino após validação.")
    if len(df_teste) == 0:
        raise ValueError("Nenhuma sequência válida no teste após validação.")

    print("\n[PASSO 1 CONCLUÍDO]")
    print(f"  df_treino: {len(df_treino)} sequências")
    print(f"  df_teste : {len(df_teste)} sequências")
    print()

    return df_treino, df_teste


# ── Exemplo de uso e teste rápido ──────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # Criar CSVs de exemplo para testar
    treino_conteudo = """sequence,label
ACDEFGHIK,1
LMNPQRSTVW,0
ACDEFGHIKLM,1
ZZINVALID,0
,1
ACDE,0
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST,1
PEPTIDE,1
"""
    teste_conteudo = """sequence,label
GHIKLM,1
ACDE,0
LMNP,1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(treino_conteudo)
        path_treino = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(teste_conteudo)
        path_teste = f.name

    df_treino, df_teste = carregar_dados(path_treino, path_teste)

    print("Primeiras linhas do treino:")
    print(df_treino.head())

    os.unlink(path_treino)
    os.unlink(path_teste)
