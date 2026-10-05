from sklearn.neighbors import KNeighborsClassifier

NOME = "KNN (exemplo — troque pela sua implementação)"


def criar():
    return KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7]}
