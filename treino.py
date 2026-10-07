"""
Treino e validação de um classificador escolhido, reutilizando a lógica do
pacote_padrao (Abordagem A + seleção por F1-macro).

O usuário escolhe:
  - qual algoritmo (dos classificadores/)
  - quantas linhas do dataset de referência usar
  - qual abordagem de pré-processamento (A ou B)

Fluxo:
  1. Carrega o dataset e corta para N linhas.
  2. Converte para a abordagem escolhida:
       A: usa dataset_final_numerico.csv diretamente (9 features: 1/-1/0).
       B: usa dataset_final.csv e deriva 7 features (qtd_x, qtd_o,
          posições_ocupadas, linhas_2x, linhas_2o, casas_vazias, jogador_da_vez).
  3. Divide em treino (60%) / validação (20%) / teste (20%), estratificado.
  4. Seleciona hiperparâmetros pela F1-macro na validação (hold-out).
  5. Retreina em treino+validação e avalia UMA vez no teste.

Também expõe traduzir_tabuleiro(), usado a cada jogada para montar a entrada
do classificador no mesmo formato da abordagem escolhida.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "dataset_final.csv"
DATASET_NUMERICO = RAIZ / "dataset_final_numerico.csv"

CASAS = ["sup_esq", "sup_meio", "sup_dir", "meio_esq", "centro",
         "meio_dir", "inf_esq", "inf_meio", "inf_dir"]
CLASSES = ["tem_jogo", "x_venceu", "o_venceu", "empate"]
MAPA = {"x": 1, "o": -1, "b": 0}
LINHAS_TABULEIRO = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
SEMENTE = 42


def total_linhas_dataset():
    return sum(1 for _ in open(DATASET_NUMERICO, encoding="utf-8")) - 1


def traduzir_tabuleiro(tab, abordagem="A"):
    """tab: lista de 9 chars em {x,o,b} -> lista de features numéricas."""
    if abordagem == "A":
        return _abordagem_a(tab)
    return _abordagem_b(tab)


def _abordagem_a(tab):
    """9 features: cada casa convertida em 1 (x), -1 (o) ou 0 (b)."""
    return [MAPA[c] for c in tab]


def _abordagem_b(tab):
    """7 features derivadas do estado do tabuleiro."""
    qtd_x = tab.count("x")
    qtd_o = tab.count("o")
    posicoes_ocupadas = qtd_x + qtd_o
    casas_vazias = tab.count("b")
    jogador_da_vez = 1 if qtd_x == qtd_o else -1
    linhas_2x = sum(
        1 for l in LINHAS_TABULEIRO
        if sum(tab[i] == "x" for i in l) == 2 and sum(tab[i] == "b" for i in l) == 1
    )
    linhas_2o = sum(
        1 for l in LINHAS_TABULEIRO
        if sum(tab[i] == "o" for i in l) == 2 and sum(tab[i] == "b" for i in l) == 1
    )
    return [qtd_x, qtd_o, posicoes_ocupadas, linhas_2x, linhas_2o, casas_vazias, jogador_da_vez]


def _carregar(n_linhas, abordagem="A"):
    if abordagem == "A":
        df = pd.read_csv(DATASET_NUMERICO)
    else:
        df = pd.read_csv(DATASET)

    if n_linhas and n_linhas < len(df):
        partes = []
        for _, g in df.groupby("classe"):
            k = max(1, round(len(g) * n_linhas / len(df)))
            partes.append(g.sample(min(k, len(g)), random_state=SEMENTE))
        df = pd.concat(partes).reset_index(drop=True)

    if abordagem == "A":
        X = df[CASAS].values.astype(float)
    else:
        tabs = df[CASAS].astype(str).values.tolist()
        X = np.array([_abordagem_b(t) for t in tabs])

    y = np.asarray(df["classe"].tolist(), dtype=object)
    return X, y, len(df)


def _grade(parametros):
    chaves = list(parametros)
    if not parametros:
        return [{}]
    return [dict(zip(chaves, v)) for v in itertools.product(*parametros.values())]


def _f1m(y, p):
    return f1_score(y, p, average="macro", labels=CLASSES, zero_division=0)


def treinar(modelo, grade, n_linhas, abordagem="A"):
    """Executa o protocolo e devolve (modelo_final, relatorio_dict)."""
    X, y, usadas = _carregar(n_linhas, abordagem)

    def _split(Xa, ya, frac, strat):
        return train_test_split(Xa, ya, test_size=frac, random_state=SEMENTE,
                                stratify=strat)

    estratifica = y if min(np.bincount(pd.factorize(y)[0])) >= 2 else None
    X_resto, X_te, y_resto, y_te = _split(X, y, 0.20, estratifica)
    estratifica2 = y_resto if min(np.bincount(pd.factorize(y_resto)[0])) >= 2 else None
    X_tr, X_va, y_tr, y_va = _split(X_resto, y_resto, 0.25, estratifica2)

    combos = _grade(grade)
    tabela = []
    for combo in combos:
        m = clone(modelo).set_params(**combo).fit(X_tr, y_tr)
        p = m.predict(X_va)
        tabela.append({**combo,
                       "f1_macro_validacao": _f1m(y_va, p),
                       "acuracia_validacao": accuracy_score(y_va, p)})
    tabela = pd.DataFrame(tabela)
    i_melhor = tabela["f1_macro_validacao"].idxmax()
    melhores = combos[i_melhor]
    tabela = tabela.sort_values(
        "f1_macro_validacao", ascending=False, kind="stable").reset_index(drop=True)

    X_fit = np.vstack([X_tr, X_va])
    y_fit = np.concatenate([y_tr, y_va])
    final = clone(modelo).set_params(**melhores).fit(X_fit, y_fit)
    p_te = final.predict(X_te)

    relatorio = {
        "linhas_usadas": usadas,
        "n_treino": len(y_tr), "n_validacao": len(y_va), "n_teste": len(y_te),
        "melhores_hiperparametros": melhores,
        "tabela_selecao": tabela,
        "acuracia_teste": accuracy_score(y_te, p_te),
        "f1_macro_teste": _f1m(y_te, p_te),
        "matriz_confusao": confusion_matrix(y_te, p_te, labels=CLASSES),
        "relatorio_classes": classification_report(
            y_te, p_te, labels=CLASSES, zero_division=0, output_dict=True),
        "classes": list(final.classes_),
        "abordagem": abordagem,
    }
    modelo_jogo = clone(modelo).set_params(**melhores).fit(X, y)
    return modelo_jogo, relatorio


def classificar(modelo, tab, abordagem="A"):
    """Classifica o tabuleiro atual. Retorna (classe, {classe: prob}|None)."""
    x = np.array([traduzir_tabuleiro(tab, abordagem)])
    classe = modelo.predict(x)[0]
    probs = None
    if hasattr(modelo, "predict_proba"):
        probs = dict(zip(modelo.classes_, modelo.predict_proba(x)[0]))
    return classe, probs
