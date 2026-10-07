"""Classificador AdaBoost para os estados do jogo da velha."""

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
