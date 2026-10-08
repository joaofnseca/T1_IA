import sys
from pathlib import Path
 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import treino
 
NOME = "k-NN (k vizinhos mais próximos)"
 
 
def _para_inteiros(X):
    return np.asarray(X).astype(int)
 
 
def criar():
    modelo = Pipeline([
        ("inteiros", FunctionTransformer(_para_inteiros)),
        ("knn", KNeighborsClassifier(algorithm="brute")),
    ])
    grade = {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["hamming", "manhattan", "euclidean"],
    }
    return modelo, grade
 
 
def treinar_e_avaliar(n_linhas=None, abordagem="A"):
    modelo, grade = criar()
    return treino.treinar(modelo, grade, n_linhas, abordagem)
 
 
def classificar(modelo, tabuleiro, abordagem="A"):
    return treino.classificar(modelo, tabuleiro, abordagem)
