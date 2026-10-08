import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import treino
from sklearn.tree import DecisionTreeClassifier


NOME = "Árvore de Decisão"


def criar():
    modelo = DecisionTreeClassifier(
        random_state=42
    )

    grade = {
    "criterion": [
        "gini"
    ],

    "max_depth": [
        None,
        15,
        20
    ],

    "min_samples_split": [
        2,
        5,
        10,
        15,
        20,
        30
    ],

    "min_samples_leaf": [
        1,
        2,
        5
    ], 
    
    "class_weight": [
    None,
    "balanced"
    ]

}
    return modelo, grade


def treinar_e_avaliar(n_linhas=None, abordagem="A"):
    modelo, grade = criar()
    return treino.treinar(modelo, grade, n_linhas, abordagem)


def classificar(modelo, tabuleiro, abordagem="A"):
    return treino.classificar(modelo, tabuleiro,abordagem)