"""Treina e avalia uma MLP para classificar o estado do jogo da velha."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path(__file__).resolve().parent / "dataset_final_numerico.csv"
SEMENTE = 42
FRACAO_TESTE = 0.20

COLUNAS_TABULEIRO = [
    "sup_esq", "sup_meio", "sup_dir",
    "meio_esq", "centro", "meio_dir",
    "inf_esq", "inf_meio", "inf_dir",
]
CLASSES = ["tem_jogo", "x_venceu", "o_venceu", "empate"]
MAPA = {"x": 1, "o": -1, "b": 0}


def carregar_dados(caminho: Path = DATASET):
    dados = pd.read_csv(caminho)
    X = dados[COLUNAS_TABULEIRO].apply(pd.to_numeric, errors="raise")
    y = dados["classe"]
    return X, y


def criar_modelo() -> Pipeline:
    return Pipeline(
        steps=[
            ("padronizador", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def traduzir_tabuleiro(tabuleiro: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[MAPA[c] for c in tabuleiro]], columns=COLUNAS_TABULEIRO
    )


def classificar(modelo: Pipeline, tabuleiro: list[str], abordagem="A") -> tuple[str, dict]:
    entrada = traduzir_tabuleiro(tabuleiro)
    classe = str(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    classes = modelo.named_steps["mlp"].classes_
    return classe, dict(zip(classes, probabilidades))


def treinar_e_avaliar() -> tuple[Pipeline, dict]:
    X, y = carregar_dados()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=FRACAO_TESTE,
        random_state=SEMENTE,
        stratify=y,
    )

    modelo_teste = criar_modelo()
    modelo_teste.fit(X_treino, y_treino)
    previsoes = modelo_teste.predict(X_teste)

    metricas = {
        "linhas_usadas": len(X),
        "n_treino": len(X_treino),
        "n_teste": len(X_teste),
        "acuracia_teste": accuracy_score(y_teste, previsoes),
        "f1_macro_teste": f1_score(
            y_teste, previsoes, labels=CLASSES, average="macro", zero_division=0
        ),
        "matriz_confusao": confusion_matrix(y_teste, previsoes, labels=CLASSES),
        "relatorio_classes": classification_report(
            y_teste, previsoes,
            labels=CLASSES,
            zero_division=0,
            output_dict=True,
        ),
        "classes": CLASSES,
    }

    modelo_jogo = criar_modelo()
    modelo_jogo.fit(X, y)
    metricas["loss_curve"] = modelo_jogo.named_steps["mlp"].loss_curve_
    return modelo_jogo, metricas
