"""Treina e avalia uma MLP para identificar se o jogo da velha terminou."""

from pathlib import Path

import matplotlib.pyplot as plt
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

# O CSV usa "centro" para a posição correspondente a "meio_meio".
COLUNAS_TABULEIRO = [
    "sup_esq",
    "sup_meio",
    "sup_dir",
    "meio_esq",
    "meio_meio",
    "meio_dir",
    "inf_esq",
    "inf_meio",
    "inf_dir",
]
ALIAS_COLUNAS = {"meio_meio": "centro"}

# empate tambem acaba
MAPA_CLASSES = {
    "tem_jogo": "nao_terminou",
    "x_venceu": "terminou",
    "o_venceu": "terminou",
    "empate": "terminou",
}
ROTULOS = ["nao_terminou", "terminou"]
NOMES_ROTULOS = ["Não terminou", "Terminou"]


def carregar_dados(caminho: Path = DATASET) -> tuple[pd.DataFrame, pd.Series]:
    """Lê apenas as nove casas do tabuleiro e deriva o alvo binário de classe."""
    dados = pd.read_csv(caminho)

    colunas_ausentes = {"classe"} - set(dados.columns)
    if colunas_ausentes:
        raise ValueError(
            f"Colunas obrigatórias ausentes no dataset: {sorted(colunas_ausentes)}"
        )

    colunas_fonte = [
        ALIAS_COLUNAS.get(coluna, coluna) for coluna in COLUNAS_TABULEIRO
    ]
    ausentes_tabuleiro = set(colunas_fonte) - set(dados.columns)
    if ausentes_tabuleiro:
        raise ValueError(
            "Colunas do tabuleiro ausentes no dataset: "
            f"{sorted(ausentes_tabuleiro)}"
        )

    X = dados[colunas_fonte].copy()
    X.columns = COLUNAS_TABULEIRO
    X = X.apply(pd.to_numeric, errors="raise")

    classes_desconhecidas = set(dados["classe"].dropna().unique()) - set(
        MAPA_CLASSES
    )
    if classes_desconhecidas or dados["classe"].isna().any():
        raise ValueError(
            "Valores ausentes ou não mapeados na coluna 'classe': "
            f"{sorted(classes_desconhecidas)}"
        )

    y = dados["classe"].map(MAPA_CLASSES).rename("jogo_terminou")
    return X, y


def criar_modelo() -> Pipeline:
    """Cria uma MLP com padronização ajustada exclusivamente no treino."""
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
    """Converte x/o/b para os valores numéricos usados no dataset."""
    mapa = {"x": 1, "o": -1, "b": 0}
    return pd.DataFrame(
        [[mapa[casa] for casa in tabuleiro]], columns=COLUNAS_TABULEIRO
    )


def classificar(modelo: Pipeline, tabuleiro: list[str]) -> tuple[str, dict[str, float]]:
    """Prevê se o jogo terminou e retorna as probabilidades por classe."""
    entrada = traduzir_tabuleiro(tabuleiro)
    classe = str(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]
    classes = modelo.named_steps["mlp"].classes_
    return classe, dict(zip(classes, probabilidades))


def treinar_e_avaliar() -> tuple[Pipeline, dict]:
    """Avalia em teste separado e retorna outro modelo treinado em todo o CSV."""
    X, y = carregar_dados()
    contagem_classes = y.value_counts()
    if len(contagem_classes) != len(ROTULOS) or contagem_classes.min() < 2:
        raise ValueError(
            "São necessárias ao menos duas amostras em cada classe para "
            "realizar a divisão estratificada treino/teste."
        )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
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
            y_teste, previsoes, labels=ROTULOS, average="macro", zero_division=0
        ),
        "matriz_confusao": confusion_matrix(y_teste, previsoes, labels=ROTULOS),
        "relatorio_classes": classification_report(
            y_teste,
            previsoes,
            labels=ROTULOS,
            target_names=NOMES_ROTULOS,
            zero_division=0,
            digits=4,
            output_dict=True,
        ),
    }

    # O teste fica reservado à avaliação; o modelo de jogo usa todo o dataset.
    modelo_jogo = criar_modelo()
    modelo_jogo.fit(X, y)
    metricas["loss_curve"] = modelo_jogo.named_steps["mlp"].loss_curve_
    return modelo_jogo, metricas


def main() -> None:
    """Treina, avalia no teste separado e exibe a curva de perda."""
    _, metricas = treinar_e_avaliar()
    print(f"Linhas no dataset: {metricas['linhas_usadas']}")
    print(f"Amostras de treino: {metricas['n_treino']}")
    print(f"Amostras de teste: {metricas['n_teste']}")
    print(f"Acurácia: {metricas['acuracia_teste']:.4f}")
    print(f"F1-macro: {metricas['f1_macro_teste']:.4f}")

    print("\nMatriz de confusão (linhas = real; colunas = previsto):")
    print(
        pd.DataFrame(
            metricas["matriz_confusao"],
            index=NOMES_ROTULOS,
            columns=NOMES_ROTULOS,
        )
    )
    print("\nRelatório de classificação:")
    print(pd.DataFrame(metricas["relatorio_classes"]).T)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(metricas["loss_curve"]) + 1), metricas["loss_curve"])
    plt.xlabel("Época")
    plt.ylabel("Perda (loss)")
    plt.title("Evolução da perda durante o treinamento da MLP")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
