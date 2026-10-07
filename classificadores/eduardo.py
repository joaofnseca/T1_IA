"""Classificador AdaBoost para os estados do jogo da velha."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import treino
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

NOME = "AdaBoost"


def criar():
    """Retorna o modelo e a grade usada pela rotina comum de treinamento."""
    modelo = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        random_state=42,
    )
    grade = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0],
        "estimator__max_depth": [1, 2],
    }
    return modelo, grade


def treinar_e_avaliar(n_linhas=None, abordagem="A"):
    modelo, grade = criar()
    return treino.treinar(modelo, grade, n_linhas, abordagem)


def classificar(modelo, tabuleiro, abordagem="A"):
    return treino.classificar(modelo, tabuleiro, abordagem)
