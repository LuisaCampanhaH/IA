"""Avaliação oficial do grupo e testes extras da Etapa 2 (Pessoa 1).

Uso no notebook do grupo:

    from avaliacao import avaliar, pipeline_etapa2
    r = avaliar(pipeline_etapa2(modelo), X, y)
    r["f1_macro"], r["desvio"], r["folds"], r["precisao"], r["revocacao"]

Os folds são os mesmos do grupo (5 x 2, semente 42). Tudo que aprende com os
dados fica dentro do fit, então é ajustado só no treino de cada fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import limpeza as L
import transformacao as T

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
METRICAS = {"f1_macro": "f1_macro", "precisao": "precision", "revocacao": "recall"}   # precisão e revocação de anomaly
MIN_FOLDS = 8   # a técnica passa se ganhar da referência em pelo menos 8 dos 10 folds

# os três algoritmos do grupo
MODELOS_BASE = {
    "Árvore de decisão": DecisionTreeClassifier(random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300, random_state=42),
}


def avaliar(pipeline, X, y, extras=None):
    """F1-macro (média, desvio e cada fold) e precisão e revocação da classe anomaly.
    extras = {nome: função(estimador, X, y)} guarda mais uma medida por fold."""
    r = cross_validate(pipeline, X, y, cv=CV, scoring={**METRICAS, **(extras or {})}, n_jobs=-1)
    saida = {"f1_macro": r["test_f1_macro"].mean(), "desvio": r["test_f1_macro"].std(),
             "folds": r["test_f1_macro"], "precisao": r["test_precisao"].mean(),
             "revocacao": r["test_revocacao"].mean()}
    for nome in extras or {}:
        saida[nome] = r[f"test_{nome}"]
    return saida


def regra(candidata, referencia):
    """Compara fold a fold (os folds são os mesmos). Passa com pelo menos MIN_FOLDS vitórias."""
    d = candidata["folds"] - referencia["folds"]
    return {"dif": d.mean(), "ganhou": int((d > 0).sum()), "passa": bool((d > 0).sum() >= MIN_FOLDS)}


def pipeline_etapa2(modelo, cols_log=T.COLS_LOG_CONTAGENS, escalador="standard"):
    """O pipeline final da Etapa 2: o pré-processamento das Pessoas 2 e 3 + o modelo."""
    return Pipeline([("pre", T.montar_preprocessador(cols_log, escalador)), ("modelo", clone(modelo))])


class Cortar(BaseEstimator, TransformerMixin):
    """Winsorização: no fit, guarda os percentis de cada coluna; no transform, corta nesses limites.
    Se os dois limites são iguais no treino do fold, cortar deixaria a coluna constante, então
    ela fica sem corte. cortadas_ marca as colunas em que o corte mudou algum valor do treino."""

    def __init__(self, inf=1, sup=99):
        self.inf = inf
        self.sup = sup

    def fit(self, X, y=None):
        inf, sup = np.percentile(X, self.inf, axis=0), np.percentile(X, self.sup, axis=0)
        self.sem_corte_ = inf == sup
        self.limite_inf_ = np.where(self.sem_corte_, -np.inf, inf)
        self.limite_sup_ = np.where(self.sem_corte_, np.inf, sup)
        self.cortadas_ = ((X < self.limite_inf_) | (X > self.limite_sup_)).any(axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.limite_inf_, self.limite_sup_)

    def get_feature_names_out(self, input_features=None):
        return input_features


def _numericas_sem_log(X):
    return [c for c in T.colunas_numericas(X) if c not in T.COLS_LOG_CONTAGENS]


def _bloco_com_corte(log, inf, sup):
    """Mediana -> log1p (se tiver) -> corte -> StandardScaler."""
    return Pipeline(T.bloco_numerico(log, "nenhum").steps
                    + [("cortar", Cortar(inf, sup)), ("escalar", StandardScaler())])


def pipeline_corte(modelo, inf, sup):
    """Igual ao pipeline_etapa2, com o corte depois do log1p e antes do StandardScaler."""
    pre = ColumnTransformer([("log", _bloco_com_corte(True, inf, sup), T.COLS_LOG_CONTAGENS),
                             ("num", _bloco_com_corte(False, inf, sup), _numericas_sem_log)]
                            + T.blocos_categoricos(),
                            sparse_threshold=0, verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("modelo", clone(modelo))])


def ler_cortadas(estimador, X, y):
    """Quantas colunas o corte mudou de fato no treino deste fold."""
    blocos = estimador.named_steps["pre"].named_transformers_
    return int(blocos["log"].named_steps["cortar"].cortadas_.sum() + blocos["num"].named_steps["cortar"].cortadas_.sum())


def pares_contraditorios(df_bruto):
    """Um representante de cada par que remover_contraditorias tira da base: mesmos
    passos do limpar() até as duplicatas, ficando com o que limpar() joga fora."""
    df = L.padronizar_texto(df_bruto)
    df = L.impossiveis_para_nan(df)
    df = L.corrigir_su_attempted(df)
    df = L.remover_colunas(df)
    df = L.remover_duplicatas(df)
    atributos = [c for c in df.columns if c != L.ALVO]
    n_classes = df.groupby(atributos, dropna=False)[L.ALVO].transform("nunique")
    return df[n_classes > 1].drop_duplicates(atributos)[atributos].reset_index(drop=True)


class ComParesReetiquetados(ClassifierMixin, BaseEstimator):
    """Treina no treino do fold, prevê o rótulo dos pares, junta ao treino e treina de novo.
    A validação não muda: o predict é só do modelo final, nas mesmas linhas limpas."""

    def __init__(self, pipeline, pares):
        self.pipeline = pipeline
        self.pares = pares

    def fit(self, X, y):
        pares = self.pares[X.columns]
        rotulos = clone(self.pipeline).fit(X, y).predict(pares)
        X_novo = pd.concat([X, pares], ignore_index=True)
        y_novo = np.concatenate([np.asarray(y), rotulos])
        self.modelo_ = clone(self.pipeline).fit(X_novo, y_novo)
        self.classes_ = self.modelo_.classes_
        self.pares_anomalia_ = int(rotulos.sum())
        return self

    def predict(self, X):
        return self.modelo_.predict(X)


def ler_pares_anomalia(estimador, X, y):
    return estimador.pares_anomalia_


def com_limiar_no_fold(pipeline):
    """Escolhe o limiar (0,01 a 0,95) maximizando o F1-macro em folds internos,
    só com o treino do fold externo. A validação externa não participa."""
    return TunedThresholdClassifierCV(pipeline, scoring="f1_macro", thresholds=np.arange(0.01, 0.955, 0.01),
                                      cv=StratifiedKFold(5, shuffle=True, random_state=42))


def ler_limiar(estimador, X, y):
    return estimador.best_threshold_
