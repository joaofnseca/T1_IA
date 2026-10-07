"""
Treino e validação de um classificador escolhido, reutilizando a lógica do
pacote_padrao (Abordagem A + seleção por F1-macro).

O usuário escolhe:
  - qual algoritmo (dos classificadores/)
  - quantas linhas do dataset de referência usar

Fluxo:
  1. Carrega dataset_final.csv do pacote_padrao e corta para N linhas.
  2. Converte para Abordagem A: x=1, o=-1, b=0 (9 features).
  3. Divide em treino (60%) / validação (20%) / teste (20%), estratificado.
  4. Seleciona hiperparâmetros pela F1-macro na validação (hold-out).
  5. Retreina em treino+validação e avalia UMA vez no teste.

Também expõe traduzir_tabuleiro(), usado a cada jogada para montar a entrada
do classificador no mesmo formato da Abordagem A.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# dataset de referência vem do pacote_padrao do projeto
RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ / "pacote_padrao" / "dataset_final.csv"

CASAS = ["sup_esq", "sup_meio", "sup_dir", "meio_esq", "centro",
         "meio_dir", "inf_esq", "inf_meio", "inf_dir"]
CLASSES = ["tem_jogo", "x_venceu", "o_venceu", "empate"]
MAPA = {
    "x": 1, "o": -1, "b": 0,
    "1": 1, "-1": -1, "0": 0,
}
SEMENTE = 42


def total_linhas_dataset():
    return sum(1 for _ in open(DATASET, encoding="utf-8")) - 1  # menos o cabeçalho


def traduzir_tabuleiro(tab):
    """tab: lista de 9 chars em {x,o,b} -> lista de 9 ints (Abordagem A)."""
    return [MAPA[c] for c in tab]


def _carregar(n_linhas):
    df = pd.read_csv(DATASET)
    if n_linhas and n_linhas < len(df):
        # amostra estratificada por classe para não perder nenhuma das 4 classes
        partes = []
        for _, g in df.groupby("classe"):
            k = max(1, round(len(g) * n_linhas / len(df)))
            partes.append(g.sample(min(k, len(g)), random_state=SEMENTE))
        df = pd.concat(partes).reset_index(drop=True)
    X = np.array([[MAPA[c] for c in linha] for linha in df[CASAS].astype(str).values])
    y = np.asarray(df["classe"].tolist(), dtype=object)
    return X, y, len(df)


def _grade(parametros):
    chaves = list(parametros)
    if not parametros:
        return [{}]
    return [dict(zip(chaves, v)) for v in itertools.product(*parametros.values())]


def _f1m(y, p):
    return f1_score(y, p, average="macro", labels=CLASSES, zero_division=0)


def treinar(modelo, grade, n_linhas):
    """Executa o protocolo e devolve (modelo_final, relatorio_dict)."""
    X, y, usadas = _carregar(n_linhas)

    # garante ao menos uma amostra por classe presente para estratificar
    def _split(Xa, ya, frac, strat):
        return train_test_split(Xa, ya, test_size=frac, random_state=SEMENTE,
                                stratify=strat)

    estratifica = y if min(np.bincount(pd.factorize(y)[0])) >= 2 else None
    X_resto, X_te, y_resto, y_te = _split(X, y, 0.20, estratifica)
    estratifica2 = y_resto if min(np.bincount(pd.factorize(y_resto)[0])) >= 2 else None
    X_tr, X_va, y_tr, y_va = _split(X_resto, y_resto, 0.25, estratifica2)

    # seleção de hiperparâmetros por F1-macro na validação (hold-out)
    combos = _grade(grade)
    tabela = []
    for combo in combos:
        m = clone(modelo).set_params(**combo).fit(X_tr, y_tr)
        p = m.predict(X_va)
        tabela.append({**combo,
                       "f1_macro_validacao": _f1m(y_va, p),
                       "acuracia_validacao": accuracy_score(y_va, p)})
    tabela = pd.DataFrame(tabela)
    # índice do melhor combo preservando o dict ORIGINAL (sem converter tipos)
    i_melhor = tabela["f1_macro_validacao"].idxmax()
    melhores = combos[i_melhor]
    tabela = tabela.sort_values(
        "f1_macro_validacao", ascending=False, kind="stable").reset_index(drop=True)

    # retreina em treino+validação e avalia uma vez no teste
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
    }
    # treina de novo em TUDO (treino+val+teste) para jogar com o máximo de dados
    modelo_jogo = clone(modelo).set_params(**melhores).fit(X, y)
    return modelo_jogo, relatorio


def classificar(modelo, tab):
    """Classifica o tabuleiro atual. Retorna (classe, {classe: prob}|None)."""
    x = np.array([traduzir_tabuleiro(tab)])
    classe = modelo.predict(x)[0]
    probs = None
    if hasattr(modelo, "predict_proba"):
        probs = dict(zip(modelo.classes_, modelo.predict_proba(x)[0]))
    return classe, probs
