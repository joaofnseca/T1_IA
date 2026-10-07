from sklearn.neighbors import KNeighborsClassifier
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import treino

NOME = "k-NN (k vizinhos mais próximos)"


def criar():
    modelo = KNeighborsClassifier(algorithm="brute")
    grade = {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["hamming", "manhattan", "euclidean"],
    }
    return modelo, grade


def treinar_e_avaliar(n_linhas=None, abordagem="A"):
    modelo, grade = criar()
    return treino.treinar(modelo, grade, n_linhas, abordagem)


def classificar(modelo, tabuleiro, abordagem="A"):
    return treino.classificar(modelo, tabuleiro, abordagem)
