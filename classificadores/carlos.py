"""
ALGORITMO 2 — Rede Neural (MLP)

Contrato (conforme o esqueleto do repositório):

    NOME = "Nome do algoritmo exibido no front"

    def criar():
        # retorna (modelo_sklearn, grade_de_hiperparametros)

Entrada: Abordagem A (9 valores x=1/o=-1/b=0).
Classes: tem_jogo, x_venceu, o_venceu, empate.
"""

from sklearn.neural_network import MLPClassifier

NOME = "Rede Neural (MLP)"


def criar():
    modelo = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        random_state=42,
    )

    grade = {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
    }

    return modelo, grade
