"""Transformação numérica e redução de dimensionalidade (Etapa 2, Pessoa 3).

Uso no notebook do grupo:

    from transformacao import montar_preprocessador, COLS_LOG_CONTAGENS
    pre = montar_preprocessador(cols_log=COLS_LOG_CONTAGENS, escalador="standard")
    pipe = Pipeline([("pre", pre), ("modelo", LogisticRegression(max_iter=2000))])

Tudo aqui é um transformador do sklearn, então só é ajustado no treino de cada
fold. A imputação (mediana / mais frequente) e a codificação (one-hot) são
provisórias: a escolha final é da Pessoa 2 e entra no lugar delas na integração.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

ALVO = "class"
CATEGORICAS = ["protocol_type", "service", "flag"]
# fora do modelo nesta comparação; o grupo ainda decide se usa (ver Pessoa 1)
FORA = ["score_ids_legado"]

# o que o enunciado pede: bytes e duração
COLS_LOG_BYTES = ["duration", "src_bytes", "dst_bytes"]
# variante: todas as contagens (colunas que não são taxa nem 0/1)
COLS_LOG_CONTAGENS = COLS_LOG_BYTES + [
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "num_compromised",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
]

ESCALADORES = {"nenhum": None, "standard": StandardScaler, "robust": RobustScaler}


def separar(df):
    """Devolve X (sem alvo nem colunas fora) e y (1 = anomaly)."""
    X = df.drop(columns=[ALVO] + FORA, errors="ignore")
    y = (df[ALVO] == "anomaly").astype(int)
    return X, y


def colunas_numericas(X):
    return [c for c in X.columns if c not in CATEGORICAS + ["id"]]


def bloco_numerico(log, escalador):
    """Imputa pela mediana, aplica log1p (opcional) e escala (opcional).
    A mediana vem antes do log1p: como o log é crescente, dá no mesmo."""
    passos = [("imputar", SimpleImputer(strategy="median"))]
    if log:
        passos.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if ESCALADORES[escalador] is not None:
        passos.append(("escalar", ESCALADORES[escalador]()))
    return Pipeline(passos)


def bloco_categorico():
    """Provisório (Pessoa 2): mais frequente + one-hot que ignora categoria nova."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def montar_preprocessador(cols_log=(), escalador="standard"):
    """ColumnTransformer com três grupos: colunas com log1p, demais numéricas e
    categóricas. O escalador vale para todas as numéricas; o one-hot fica 0/1."""
    cols_log = list(cols_log)

    def _cols_resto(X):   # as numéricas que não levam log1p, lidas do próprio X
        return [c for c in colunas_numericas(X) if c not in cols_log]

    return ColumnTransformer([
        ("log", bloco_numerico(True, escalador), cols_log),
        ("num", bloco_numerico(False, escalador), _cols_resto),
        ("cat", bloco_categorico(), CATEGORICAS),
    ])


def com_pca(preprocessador, n_componentes, random_state=42):
    """Pré-processamento seguido de PCA (n_componentes inteiro ou fração da variância)."""
    return Pipeline([
        ("pre", preprocessador),
        ("pca", PCA(n_components=n_componentes, random_state=random_state)),
    ])
