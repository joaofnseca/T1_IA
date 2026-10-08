"""Adaptador da MLP ao contrato de classificadores do app."""
 
import chies_MLP
 
NOME = "MLP (Multilayer Perceptron)"
 
 
def treinar_e_avaliar(n_linhas=None, abordagem="A"):
    """Treina na abordagem escolhida (A ou B) e devolve o modelo e as métricas do teste."""
    return chies_MLP.treinar_e_avaliar(n_linhas, abordagem)
 
 
def classificar(modelo, tabuleiro, abordagem="A"):
    """Devolve a classe prevista (4 classes) e as probabilidades para um tabuleiro."""
    return chies_MLP.classificar(modelo, tabuleiro, abordagem)
