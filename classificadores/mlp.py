"""Adaptador da MLP ao contrato de classificadores do app."""

import chies_MLP

NOME = "MLP (Multilayer Perceptron)"


def treinar_e_avaliar():
    """Treina e devolve o modelo e as métricas padronizadas do teste."""
    return chies_MLP.treinar_e_avaliar()


def classificar(modelo, tabuleiro, abordagem="A"):
    """Devolve a classe binária e as probabilidades para um tabuleiro."""
    return chies_MLP.classificar(modelo, tabuleiro)
