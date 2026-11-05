"""Ausentes e codificação (Etapa 2, Pessoa 2).

Monta a parte do pipeline que imputa os ausentes e codifica os categóricos.
Tudo aqui é transformador do sklearn, então quando entra num Pipeline o
ajuste (mediana, frequências, médias por categoria...) é feito só com o
treino de cada fold.

Uso:

    from ausentes_codificacao import montar_preprocessador
    pre = montar_preprocessador(num="mediana", cat="moda", codificacao="onehot")
    pipe = Pipeline([("pre", pre), ("modelo", LogisticRegression())])
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (libera o IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

CATEGORICAS = ["protocol_type", "service", "flag"]
ALTA_CARDINALIDADE = "service"        # 65 categorias no treino limpo
BAIXA_CARDINALIDADE = ["protocol_type", "flag"]   # 3 e 11 categorias: sempre one-hot
COL_AUSENCIA_INFORMATIVA = "dst_host_serror_rate"  # ausente bem mais nas anomalias (Etapa 1)
MIN_FREQ_RARAS = 50                   # service com menos de 50 linhas vira "rara"


class CodificadorFrequencia(BaseEstimator, TransformerMixin):
    """Troca cada categoria pela proporção de linhas em que ela aparece no treino.

    Categoria que não apareceu no fit (ex.: um serviço que só existe no teste) vira 0.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.colunas_ = list(X.columns)
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.colunas_)
        saida = {c: X[c].map(self.freq_[c]).astype(float).fillna(0.0) for c in self.colunas_}
        return pd.DataFrame(saida, index=X.index).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.colunas_], dtype=object)


class IndicadorAusencia(BaseEstimator, TransformerMixin):
    """Coluna 0/1 dizendo se o valor estava ausente (para uma coluna só)."""

    def fit(self, X, y=None):
        self.colunas_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X).isna().astype(float).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_ausente" for c in self.colunas_], dtype=object)


def _imputador_numerico(num):
    if num == "mediana":
        return SimpleImputer(strategy="median")
    if num == "media":
        return SimpleImputer(strategy="mean")
    if num == "mediana+indicadores":
        # uma coluna 0/1 para cada atributo que tinha ausente no treino do fold
        return SimpleImputer(strategy="median", add_indicator=True)
    if num == "mediana+indicador_serror":
        return SimpleImputer(strategy="median")   # o indicador entra à parte (ver abaixo)
    if num == "iterativa":
        # cada coluna com ausente é prevista pelas outras (regressão), em rodadas
        return IterativeImputer(max_iter=10, random_state=42)
    raise ValueError(f"imputação numérica desconhecida: {num}")


def _imputador_categorico(cat):
    if cat == "moda":
        return SimpleImputer(strategy="most_frequent")
    if cat == "constante":
        return SimpleImputer(strategy="constant", fill_value="ausente")
    raise ValueError(f"imputação categórica desconhecida: {cat}")


def _target_encoder():
    """TargetEncoder com sorteio fixo. A partir do sklearn 1.9 a semente vai num KFold
    passado em cv; antes disso vai direto em random_state."""
    versao = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
    if versao >= (1, 9):
        return TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=42))
    return TargetEncoder(target_type="binary", random_state=42)


def _codificador_service(codificacao):
    if codificacao == "onehot":
        return OneHotEncoder(handle_unknown="ignore")
    if codificacao == "onehot_raras":
        # serviços com menos de MIN_FREQ_RARAS linhas no treino viram uma coluna só;
        # serviço desconhecido no teste cai nessa mesma coluna
        return OneHotEncoder(min_frequency=MIN_FREQ_RARAS, handle_unknown="infrequent_if_exist")
    if codificacao == "frequencia":
        return make_pipeline(CodificadorFrequencia(), StandardScaler())
    if codificacao == "alvo":
        # média da classe por categoria; no fit usa validação cruzada interna, então
        # a linha não "vê" o próprio rótulo
        return make_pipeline(_target_encoder(), StandardScaler())
    if codificacao == "ordinal":
        return make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), StandardScaler())
    raise ValueError(f"codificação desconhecida: {codificacao}")


def montar_preprocessador(num="mediana", cat="moda", codificacao="onehot", colunas_numericas=None):
    """Imputação + codificação (+ StandardScaler nos números, fixo, só pro modelo linear funcionar).

    num: "mediana", "media", "mediana+indicadores", "mediana+indicador_serror" ou "iterativa"
    cat: "moda" ou "constante" (categoria nova "ausente")
    codificacao (só do service; protocol_type e flag são sempre one-hot):
        "onehot", "onehot_raras", "frequencia", "alvo" ou "ordinal"
    colunas_numericas: lista das numéricas; se None, usa tudo que não é categórico
    (resolvido no fit, pelo nome das colunas do DataFrame).
    """
    if colunas_numericas is None:
        def colunas_numericas(df):
            return [c for c in df.columns if c not in CATEGORICAS]

    partes = [
        ("num", make_pipeline(_imputador_numerico(num), StandardScaler()), colunas_numericas),
        ("cat_baixa", make_pipeline(_imputador_categorico(cat), OneHotEncoder(handle_unknown="ignore")),
         BAIXA_CARDINALIDADE),
        ("service", make_pipeline(_imputador_categorico(cat), _codificador_service(codificacao)),
         [ALTA_CARDINALIDADE]),
    ]
    if num == "mediana+indicador_serror":
        partes.append(("ind_serror", IndicadorAusencia(), [COL_AUSENCIA_INFORMATIVA]))
    return ColumnTransformer(partes, verbose_feature_names_out=False)
