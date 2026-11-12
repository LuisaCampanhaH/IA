"""Experimentos da Etapa 3 (importado pelo etapa3_classificadores.ipynb).

Mesmos folds do grupo (RepeatedStratifiedKFold 5x2, random_state=42), pré-processamento
configurável e ajustado só no treino de cada fold, limiar escolhido fora do fold avaliado
e comparação fold a fold com o teste t corrigido de Bouckaert & Frank (2004).
[IA] Código escrito com apoio do Claude (Anthropic) e revisado pelo grupo."""
import hashlib, json, os, pickle, sys, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
GRUPO = os.path.dirname(os.path.abspath(__file__))   # pasta codigo/, onde fica limpeza.py
AQUI = os.path.dirname(GRUPO)                         # pasta principal do projeto (com BANCO_DE_DADOS/)
RESULTADOS = os.path.join(AQUI, "resultados_experimentos")
sys.path.insert(0, GRUPO)
import limpeza as L  # noqa: E402

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

TREINO_LIMPO = os.path.join(AQUI, "BANCO_DE_DADOS", "treino_limpo.csv")
TREINO_BRUTO = os.path.join(AQUI, "BANCO_DE_DADOS", "treino.csv")
CACHE = os.path.join(RESULTADOS, "cache_experimentos"); os.makedirs(CACHE, exist_ok=True)
LOG_RES = os.path.join(RESULTADOS, "resultados_experimentos.jsonl")
CAT = ["protocol_type", "service", "flag"]
ALVO = "class"
COLS_LOG = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "num_compromised", "num_root", "num_file_creations", "num_shells", "num_access_files",
            "count", "srv_count", "dst_host_count", "dst_host_srv_count"]   # = T.COLS_LOG_CONTAGENS do grupo
N_JOBS = 2


# ------------------------------------------------------------------ dados
def carregar():
    tr = pd.read_csv(TREINO_LIMPO)
    bruto = pd.read_csv(TREINO_BRUTO)
    d = L.remover_duplicatas(L.remover_colunas(L.corrigir_su_attempted(L.impossiveis_para_nan(L.padronizar_texto(bruto)))))
    atrib = [c for c in d.columns if c != ALVO]
    n = d.groupby(atrib, dropna=False)[ALVO].transform("nunique")
    pares = d[n > 1].drop_duplicates(subset=atrib).drop(columns=ALVO).reset_index(drop=True)
    X = tr.drop(columns=[ALVO])
    y = (tr[ALVO] == "anomaly").astype(int).to_numpy()
    return X, y, pares[X.columns]


X, Y, PARES = carregar()
TESTE_BRUTO = os.path.join(AQUI, "BANCO_DE_DADOS", "teste_sem_rotulo.csv")
_te = L.limpar(pd.read_csv(TESTE_BRUTO), eh_treino=False)
IDS_TESTE = _te["id"].to_numpy()
TESTE = _te[X.columns].reset_index(drop=True)


def _partes(f):
    """Treino, validação e pares de um fold; f == 'final' usa o treino inteiro e o teste."""
    if f == "final":
        return {"tr": X.copy(), "va": TESTE.copy(), "pa": PARES.copy()}
    itr, iva = FOLDS[f]
    return {"tr": X.iloc[itr].reset_index(drop=True), "va": X.iloc[iva].reset_index(drop=True), "pa": PARES.copy()}


def _y_tr(f):
    return Y if f == "final" else Y[FOLDS[f][0]]
FOLDS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, Y))
NUM_TODAS = [c for c in X.columns if c not in CAT]                 # inclui score_ids_legado
NUM_SEM_SCORE = [c for c in NUM_TODAS if c != "score_ids_legado"]


# ------------------------------------------------------------------ imputações caras (cache por fold)
def _base_cat(df, alvo, colunas_ref=None):
    outras = [c for c in CAT if c != alvo]
    Z = pd.get_dummies(df[NUM_SEM_SCORE + outras], columns=outras, dummy_na=True)
    if colunas_ref is not None:
        Z = Z.reindex(columns=colunas_ref, fill_value=0)
    return Z


def cache_fold(f):
    """Para o fold f: service/flag imputados por Random Forest e numéricas por MissForest,
    sempre ajustados só no treino do fold e aplicados ao treino, à validação e aos pares."""
    arq = os.path.join(CACHE, f"fold{f}.pkl")
    if os.path.exists(arq):
        return arq
    partes = _partes(f)
    saida = {k: {} for k in partes}
    # categorias por floresta
    for alvo in ["service", "flag"]:
        Ztr = _base_cat(partes["tr"], alvo)
        imp = SimpleImputer(strategy="median").fit(Ztr)
        m = partes["tr"][alvo].notna().to_numpy()
        rf = RandomForestClassifier(200, n_jobs=1, random_state=42).fit(imp.transform(Ztr)[m], partes["tr"].loc[m, alvo])
        for k, df in partes.items():
            col = df[alvo].copy(); falta = col.isna().to_numpy()
            if falta.any():
                Z = _base_cat(df.loc[falta], alvo, Ztr.columns)
                col.loc[falta] = rf.predict(imp.transform(Z))
            saida[k][alvo] = col.to_numpy()
    # números por MissForest (só as numéricas, sem o score)
    it = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, max_depth=12, n_jobs=1, random_state=42),
                          max_iter=5, random_state=42).fit(partes["tr"][NUM_SEM_SCORE])
    for k, df in partes.items():
        saida[k]["missforest"] = it.transform(df[NUM_SEM_SCORE])
    pickle.dump(saida, open(arq, "wb"))
    return arq


def ler_cache(f):
    return pickle.load(open(os.path.join(CACHE, f"fold{f}.pkl"), "rb"))


# ------------------------------------------------------------------ pré-processamento configurável
PADRAO = dict(score=False, cat="moda", num="mediana", log=True, clip=None, nzv=False, fe=(), escala=True)

RELACOES = [  # (coluna a preencher, função que estima pela definição)
    ("serror_rate", lambda d: d["srv_serror_rate"]),
    ("srv_serror_rate", lambda d: d["serror_rate"]),
    ("rerror_rate", lambda d: d["srv_rerror_rate"]),
    ("srv_rerror_rate", lambda d: d["rerror_rate"]),
    ("same_srv_rate", lambda d: 1 - d["diff_srv_rate"]),
    ("diff_srv_rate", lambda d: 1 - d["same_srv_rate"]),
    ("dst_host_same_srv_rate", lambda d: 1 - d["dst_host_diff_srv_rate"]),
    ("dst_host_diff_srv_rate", lambda d: 1 - d["dst_host_same_srv_rate"]),
    ("srv_count", lambda d: (d["same_srv_rate"] * d["count"]).round()),
    ("dst_host_srv_count", lambda d: (d["dst_host_same_srv_rate"] * d["dst_host_count"]).round()),
]


def _fe(df, fe):
    novas = {}
    if "zeros" in fe:
        novas["src_bytes_zero"] = (df["src_bytes"] == 0).astype(float)
        novas["dst_bytes_zero"] = (df["dst_bytes"] == 0).astype(float)
    if "razoes" in fe:
        novas["log_razao_bytes"] = np.log1p(df["src_bytes"].clip(lower=0)) - np.log1p(df["dst_bytes"].clip(lower=0))
        novas["log_bytes_por_seg"] = np.log1p((df["src_bytes"] + df["dst_bytes"]).clip(lower=0) / (df["duration"].clip(lower=0) + 1))
        novas["srv_sobre_count"] = df["srv_count"] / (df["count"] + 1)
    if "erros" in fe:
        novas["erro_total"] = df["serror_rate"] + df["rerror_rate"]
        novas["dst_host_erro_total"] = df["dst_host_serror_rate"] + df["dst_host_rerror_rate"]
    return pd.concat([df, pd.DataFrame(novas, index=df.index)], axis=1) if novas else df


def preprocessar(f, cfg):
    """Devolve (A_tr, A_va, A_pares, nomes) para o fold f, tudo ajustado só no treino do fold."""
    cfg = {**PADRAO, **cfg}
    chave = hashlib.md5(json.dumps([f, cfg], sort_keys=True, default=str).encode()).hexdigest()[:16]
    arq = os.path.join(CACHE, f"pre_{chave}.npz")
    if os.path.exists(arq):
        z = np.load(arq, allow_pickle=True)
        return z["tr"], z["va"], z["pa"], list(z["nomes"])
    c = ler_cache(f) if (cfg["cat"] == "rf" or cfg["num"] == "missforest") else None
    num = NUM_TODAS if cfg["score"] else NUM_SEM_SCORE
    dfs = {k: v[CAT + num].copy() for k, v in _partes(f).items()}
    # 1) categorias
    if cfg["cat"] == "rf":
        for k in dfs:
            for alvo in ["service", "flag"]:
                dfs[k][alvo] = c[k][alvo]
    for alvo in CAT:
        moda = dfs["tr"][alvo].mode()[0]
        for k in dfs:
            dfs[k][alvo] = dfs[k][alvo].fillna(moda)
    # 2) números
    if cfg["num"] == "missforest":
        for k in dfs:
            dfs[k][NUM_SEM_SCORE] = c[k]["missforest"]
    if cfg["num"] == "definicao":
        for k in dfs:
            d = dfs[k]
            for col, est in RELACOES:
                falta = d[col].isna()
                if falta.any():
                    d.loc[falta, col] = est(d.loc[falta])
            for col in [x for x in num if x.endswith("_rate")]:
                d[col] = d[col].clip(0, 1)
    if cfg["num"] == "servico":
        med_g = dfs["tr"].groupby("service")[num].median()
        for k in dfs:
            g = med_g.reindex(dfs[k]["service"]).reset_index(drop=True)
            dfs[k][num] = dfs[k][num].fillna(g)
    med = dfs["tr"][num].median()
    for k in dfs:
        dfs[k][num] = dfs[k][num].fillna(med)
    # 3) atributos novos
    for k in dfs:
        dfs[k] = _fe(dfs[k], cfg["fe"])
    numc = [x for x in dfs["tr"].columns if x not in CAT]
    # 4) log1p nas contagens
    if cfg["log"]:
        for k in dfs:
            for col in COLS_LOG:
                dfs[k][col] = np.log1p(dfs[k][col].clip(lower=0))
    # 5) filtro de variância quase zero (Kuhn & Johnson: razão de frequência > 19 e < 10% de valores distintos)
    if cfg["nzv"]:
        tira = []
        for col in numc:
            vc = dfs["tr"][col].value_counts()
            razao = vc.iloc[0] / vc.iloc[1] if len(vc) > 1 else np.inf
            if razao > 19 and len(vc) / len(dfs["tr"]) < 0.10:
                tira.append(col)
        numc = [x for x in numc if x not in tira]
    # 6) corte dos extremos por percentis do treino (pula colunas que ficariam constantes)
    if cfg["clip"]:
        lo_p, hi_p = cfg["clip"]
        for col in numc:
            lo, hi = np.percentile(dfs["tr"][col], [lo_p, hi_p])
            if hi > lo:
                for k in dfs:
                    dfs[k][col] = dfs[k][col].clip(lo, hi)
    # 7) escala
    A = {k: dfs[k][numc].to_numpy(float) for k in dfs}
    if cfg["escala"]:
        sc = StandardScaler().fit(A["tr"])
        A = {k: sc.transform(v) for k, v in A.items()}
    # 8) codificação (igual à escolha da Pessoa 2)
    oh_baixa = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(dfs["tr"][["protocol_type", "flag"]])
    oh_serv = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False).fit(dfs["tr"][["service"]])
    for k in dfs:
        A[k] = np.hstack([A[k], oh_baixa.transform(dfs[k][["protocol_type", "flag"]]), oh_serv.transform(dfs[k][["service"]])])
    nomes = numc + list(oh_baixa.get_feature_names_out()) + list(oh_serv.get_feature_names_out())
    np.savez(arq, tr=A["tr"], va=A["va"], pa=A["pa"], nomes=np.array(nomes, dtype=object))
    return A["tr"], A["va"], A["pa"], nomes


# ------------------------------------------------------------------ modelos
class MediaSementes:
    """Várias redes iguais, cada uma com uma semente; a probabilidade é a média delas."""
    def __init__(self, modelos): self.modelos = modelos
    def fit(self, A, y):
        for m in self.modelos:
            m.fit(A, y)
        self.classes_ = self.modelos[0].classes_
        return self
    def predict_proba(self, A):
        return np.mean([m.predict_proba(A) for m in self.modelos], axis=0)


def criar_modelo(nome, params=None):
    params = params or {}
    if nome == "arvore":
        return DecisionTreeClassifier(random_state=42, **params)
    if nome == "rf":
        p = dict(n_estimators=100, n_jobs=1, random_state=42); p.update(params)
        return RandomForestClassifier(**p)
    if nome == "mlp":
        p = dict(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300, random_state=42); p.update(params)
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        k = p.pop("sementes", 1)
        if k > 1:
            return MediaSementes([MLPClassifier(**{**p, "random_state": 42 + i}) for i in range(k)])
        return MLPClassifier(**p)
    raise ValueError(nome)


def _ros(A, y, seed=42):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    extra = rng.choice(pos, len(neg) - len(pos), replace=True)
    idx = rng.permutation(np.concatenate([neg, pos, extra]))
    return A[idx], y[idx]


def _treinar(nome, params, A, y, balanc):
    if balanc == "ros":
        A, y = _ros(A, y)
    return criar_modelo(nome, params).fit(A, y)


def rodar_fold(f, cfg, nome, params, balanc, pares):
    Atr, Ava, Apa, _ = preprocessar(f, cfg)
    ytr = _y_tr(f)
    m = _treinar(nome, params, Atr, ytr, balanc)
    if pares:   # reetiqueta os pares contraditórios com o próprio modelo do fold e treina de novo
        yp = (m.predict_proba(Apa)[:, 1] >= 0.5).astype(int)
        m = _treinar(nome, params, np.vstack([Atr, Apa]), np.concatenate([ytr, yp]), balanc)
    return m.predict_proba(Ava)[:, 1]


GRADE = np.round(np.arange(0.02, 0.99, 0.01), 2)


def _f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average="macro")


def avaliar(rotulo, cfg=None, nome="rf", params=None, balanc="none", pares=False, folds=range(10), salvar=True):
    """F1-macro por fold com limiar 0,5 e com limiar escolhido nos OUTROS folds da mesma repetição."""
    cfg = cfg or {}
    t0 = time.time()
    folds = list(folds)
    probs = Parallel(n_jobs=N_JOBS)(delayed(rodar_fold)(f, cfg, nome, params, balanc, pares) for f in folds)
    P = {f: p for f, p in zip(folds, probs)}
    chave_p = hashlib.md5(json.dumps([cfg, nome, params, balanc, pares], sort_keys=True, default=str).encode()).hexdigest()[:16]
    pickle.dump(P, open(os.path.join(CACHE, f"probs_{chave_p}.pkl"), "wb"))
    f05, ft, lims = {}, {}, {}
    for f in folds:
        _, iva = FOLDS[f]
        f05[f] = _f1(Y[iva], P[f], 0.5)
        rep = f // 5
        outros = [g for g in folds if g // 5 == rep and g != f]
        if outros:
            yo = np.concatenate([Y[FOLDS[g][1]] for g in outros]); po = np.concatenate([P[g] for g in outros])
            t = max(GRADE, key=lambda t: _f1(yo, po, t))
        else:
            t = 0.5
        lims[f] = float(t); ft[f] = _f1(Y[iva], P[f], t)
    res = dict(rotulo=rotulo, chave_probs=chave_p, cfg={**PADRAO, **cfg}, modelo=nome, params=params or {}, balanc=balanc, pares=pares,
               folds=folds, f1_05=[f05[f] for f in folds], f1_lim=[ft[f] for f in folds], limiares=[lims[f] for f in folds],
               media_05=float(np.mean(list(f05.values()))), media_lim=float(np.mean(list(ft.values()))),
               desvio_lim=float(np.std(list(ft.values()), ddof=1)) if len(folds) > 1 else 0.0, segundos=round(time.time() - t0))
    if salvar:
        with open(LOG_RES, "a") as fh:
            fh.write(json.dumps(res) + "\n")
    print(f"{rotulo:62s} F1 lim 0,5: {res['media_05']:.4f} | lim ajustado: {res['media_lim']:.4f} ± {res['desvio_lim']:.4f} "
          f"(lim ~{np.median(res['limiares']):.2f}) [{res['segundos']}s]", flush=True)
    return res


def avaliar_probs(rotulo, P, salvar=True):
    """Mesmo cálculo do avaliar(), a partir de probabilidades já prontas (ex.: média de dois modelos)."""
    folds = sorted(P)
    f05, ft, lims = [], [], []
    for f in folds:
        _, iva = FOLDS[f]
        outros = [g for g in folds if g // 5 == f // 5 and g != f]
        yo = np.concatenate([Y[FOLDS[g][1]] for g in outros]); po = np.concatenate([P[g] for g in outros])
        t = max(GRADE, key=lambda t: _f1(yo, po, t))
        f05.append(_f1(Y[iva], P[f], 0.5)); ft.append(_f1(Y[iva], P[f], t)); lims.append(float(t))
    res = dict(rotulo=rotulo, folds=folds, f1_05=f05, f1_lim=ft, limiares=lims, media_05=float(np.mean(f05)),
               media_lim=float(np.mean(ft)), desvio_lim=float(np.std(ft, ddof=1)))
    if salvar:
        with open(LOG_RES, "a") as fh:
            fh.write(json.dumps(res) + "\n")
    print(f"{rotulo:62s} F1 lim 0,5: {res['media_05']:.4f} | lim ajustado: {res['media_lim']:.4f} ± {res['desvio_lim']:.4f} "
          f"(lim ~{np.median(lims):.2f})", flush=True)
    return res


def ler_probs(res):
    return pickle.load(open(os.path.join(CACHE, f"probs_{res['chave_probs']}.pkl"), "rb"))


def comparar(novo, ref, chave="f1_lim"):
    """Diferença média, vitórias fold a fold e teste t corrigido (Bouckaert & Frank 2004) para 5x2."""
    from scipy import stats
    d = np.array(novo[chave]) - np.array(ref[chave])
    k = len(d)
    vit = int(np.sum(d > 0))
    var = np.var(d, ddof=1)
    if var == 0:
        return float(d.mean()), vit, k, 1.0
    t = d.mean() / np.sqrt((1 / k + 0.25) * var)    # n_teste/n_treino = 1/4 com 5 folds
    p = 2 * stats.t.sf(abs(t), k - 1)
    return float(d.mean()), vit, k, float(p)


# ------------------------------------------------------------------ apoio ao notebook da Etapa 3
RODAR_TUDO = False   # True: refaz todos os experimentos (≈ 2–3 h em 2 núcleos); False: reaproveita os já salvos


def _todos():
    return [json.loads(l) for l in open(LOG_RES)] if os.path.exists(LOG_RES) else []


def buscar(cfg=None, nome="rf", params=None, balanc="none", pares=False, folds=range(10)):
    """Procura no registro um experimento com exatamente esta configuração (o mais recente)."""
    alvo = json.loads(json.dumps(dict(cfg={**PADRAO, **(cfg or {})}, modelo=nome, params=params or {},
                                      balanc=balanc, pares=pares, folds=list(folds)), default=str))
    achados = [r for r in _todos() if all(r.get(k) == v for k, v in alvo.items())]
    return achados[-1] if achados else None


def rodar(rotulo, cfg=None, nome="rf", params=None, balanc="none", pares=False, folds=range(10)):
    """Reaproveita o resultado salvo desta configuração; se não houver (ou RODAR_TUDO), roda de novo."""
    r = None if RODAR_TUDO else buscar(cfg, nome, params, balanc, pares, folds)
    if r is None:
        return avaliar(rotulo, cfg, nome, params, balanc, pares, folds)
    r = dict(r, rotulo=rotulo)
    print(f"{rotulo:62s} F1 lim 0,5: {r['media_05']:.4f} | lim ajustado: {r['media_lim']:.4f} ± {r['desvio_lim']:.4f} "
          f"(lim ~{np.median(r['limiares']):.2f}) [salvo]", flush=True)
    return r


def metricas_por_classe(res):
    """Precisão, revocação e F1 de cada classe, somando as matrizes de confusão dos folds
    (cada fold com o limiar escolhido fora dele)."""
    P = ler_probs(res)
    vp = fp = fn = vn = 0
    for f, t in zip(res["folds"], res["limiares"]):
        y = Y[FOLDS[f][1]]; pred = (P[f] >= t).astype(int)
        vp += int(((y == 1) & (pred == 1)).sum()); fp += int(((y == 0) & (pred == 1)).sum())
        fn += int(((y == 1) & (pred == 0)).sum()); vn += int(((y == 0) & (pred == 0)).sum())
    prec_a, rec_a = vp / (vp + fp), vp / (vp + fn)
    prec_n, rec_n = vn / (vn + fn), vn / (vn + fp)
    f1 = lambda p, r: 2 * p * r / (p + r)
    return dict(VP=vp, FP=fp, FN=fn, VN=vn, precisao_anomaly=prec_a, revocacao_anomaly=rec_a, f1_anomaly=f1(prec_a, rec_a),
                precisao_normal=prec_n, revocacao_normal=rec_n, f1_normal=f1(prec_n, rec_n))
