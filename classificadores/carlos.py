from sklearn.neighbors import KNeighborsClassifier
 
NOME = "k-NN (k vizinhos mais próximos)"
 
 
def criar():
    modelo = KNeighborsClassifier(algorithm="brute")
    grade = {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["hamming", "manhattan", "euclidean"],
    }
    return modelo, grade
