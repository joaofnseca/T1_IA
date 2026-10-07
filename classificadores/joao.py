"""Classificador Random Forest para os estados do jogo da velha."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import treino
from sklearn.ensemble import RandomForestClassifier

NOME = "Random Forest"


def criar():
    modelo = RandomForestClassifier(random_state=42)
    grade = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    }
    return modelo, grade


def treinar_e_avaliar(n_linhas=None):
    modelo, grade = criar()
    return treino.treinar(modelo, grade, n_linhas)


def classificar(modelo, tabuleiro):
    return treino.classificar(modelo, tabuleiro)
