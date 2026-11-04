"""Limpeza da base NSL-KDD modificada (Etapa 2).

Uso no notebook do grupo:

    from limpeza import limpar
    treino = limpar(pd.read_csv("BANCO_DE_DADOS/treino.csv"), eh_treino=True)
    teste = limpar(pd.read_csv("BANCO_DE_DADOS/teste_sem_rotulo.csv"), eh_treino=False)

Só regras fixas, nada é aprendido dos dados. Aqui NÃO tem imputação,
codificação nem normalização: isso fica no pipeline de cada pessoa.
"""
import pandas as pd

ALVO = "class"

# colunas de texto e como cada uma é padronizada
TEXTO_MINUSCULA = ["protocol_type", "service"]
TEXTO_MAIUSCULA = ["flag"]

# limites que vêm da definição dos atributos no NSL-KDD
LIMITES = {"count": 511, "srv_count": 511, "dst_host_count": 255, "dst_host_srv_count": 255}
NAO_NEGATIVAS = ["duration", "src_bytes", "dst_bytes"]

# colunas que saem
COPIAS = ["src_bytes_kb", "dst_bytes_kb", "total_bytes", "same_srv_pct", "protocolo_cod"]
CONSTANTE = ["num_outbound_cmds"]
VAZAMENTO = ["tempo_resposta_analista_min", "tipo_alerta_soc"]
REMOVER = COPIAS + CONSTANTE + VAZAMENTO
# score_ids_legado fica; o grupo ainda vai decidir se usa


def _taxas(df):
    return [c for c in df.columns if c.endswith("_rate")]


def padronizar_texto(df):
    """Passo 1: strip + minúsculas (protocol_type, service) e strip + maiúsculas (flag)."""
    df = df.copy()
    for c in TEXTO_MINUSCULA:
        df[c] = df[c].str.strip().str.lower()
    for c in TEXTO_MAIUSCULA:
        df[c] = df[c].str.strip().str.upper()
    return df


def impossiveis_para_nan(df):
    """Passo 2: valores impossíveis viram NaN (a imputação fica pro pipeline)."""
    df = df.copy()
    for c in NAO_NEGATIVAS:
        df[c] = df[c].mask(df[c] < 0)
    for c in _taxas(df):
        df[c] = df[c].mask(df[c] > 1)
    for c, maximo in LIMITES.items():
        df[c] = df[c].mask(df[c] > maximo)
    return df


def corrigir_su_attempted(df):
    """Passo 3: su_attempted = 2 vira 1."""
    df = df.copy()
    df["su_attempted"] = df["su_attempted"].replace(2, 1)
    return df


def remover_colunas(df):
    """Passo 4: cópias, constante e vazamento. score_ids_legado é mantida."""
    return df.drop(columns=REMOVER)


def remover_duplicatas(df):
    """Passo 5a (só treino): remove linhas exatamente iguais, com a mesma classe."""
    return df.drop_duplicates().reset_index(drop=True)


def remover_contraditorias(df):
    """Passo 5b (só treino): remove TODOS os grupos de linhas com atributos
    idênticos e classes diferentes (não dá pra saber qual rótulo está certo).
    Deve rodar depois de remover_duplicatas."""
    atributos = [c for c in df.columns if c != ALVO]
    n_classes = df.groupby(atributos, dropna=False)[ALVO].transform("nunique")
    return df[n_classes == 1].reset_index(drop=True)


def limpar(df, eh_treino):
    """Aplica toda a limpeza. No teste (eh_treino=False) nenhuma linha é removida
    e a coluna id é mantida."""
    df = padronizar_texto(df)
    df = impossiveis_para_nan(df)
    df = corrigir_su_attempted(df)
    df = remover_colunas(df)
    if eh_treino:
        df = remover_duplicatas(df)
        df = remover_contraditorias(df)
    return df
