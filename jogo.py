"""
Regras do jogo da velha + bot adversário (fácil / médio / difícil).

Importante para o T1: o bot NÃO usa o classificador de IA. O bot é um
adversário clássico (aleatório ou minimax). O classificador de IA apenas
ANALISA o estado atual do tabuleiro e devolve um rótulo; ele nunca escolhe
jogadas. As duas coisas são separadas de propósito.

Representação do tabuleiro: lista de 9 caracteres em {"x", "o", "b"}, nas
posições (índice 0..8):

    0 | 1 | 2        sup_esq  | sup_meio | sup_dir
    ---------        -----------------------------
    3 | 4 | 5        meio_esq | centro   | meio_dir
    ---------        -----------------------------
    6 | 7 | 8        inf_esq  | inf_meio | inf_dir
"""

import random

LINHAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),   # horizontais
          (0, 3, 6), (1, 4, 7), (2, 5, 8),   # verticais
          (0, 4, 8), (2, 4, 6)]              # diagonais

VAZIO = "b"


def tabuleiro_vazio():
    return [VAZIO] * 9


def jogadas_livres(tab):
    return [i for i, c in enumerate(tab) if c == VAZIO]


def vencedor(tab):
    """Retorna 'x', 'o' ou None."""
    for a, b, c in LINHAS:
        if tab[a] != VAZIO and tab[a] == tab[b] == tab[c]:
            return tab[a]
    return None


def fim_de_jogo(tab):
    return vencedor(tab) is not None or not jogadas_livres(tab)


def oponente(jogador):
    return "o" if jogador == "x" else "x"


# ---------------------------------------------------------------------------
# Bot adversário — três dificuldades
# ---------------------------------------------------------------------------
def _minimax(tab, jogador_bot, vez, prof=0):
    """Minimax padrão. Retorna (pontuação, jogada) do ponto de vista do bot."""
    vz = vencedor(tab)
    if vz == jogador_bot:
        return 10 - prof, None
    if vz == oponente(jogador_bot):
        return prof - 10, None
    if not jogadas_livres(tab):
        return 0, None

    maximiza = (vez == jogador_bot)
    melhor_valor = -float("inf") if maximiza else float("inf")
    melhor_jogada = None
    for i in jogadas_livres(tab):
        tab[i] = vez
        valor, _ = _minimax(tab, jogador_bot, oponente(vez), prof + 1)
        tab[i] = VAZIO
        if maximiza and valor > melhor_valor:
            melhor_valor, melhor_jogada = valor, i
        elif not maximiza and valor < melhor_valor:
            melhor_valor, melhor_jogada = valor, i
    return melhor_valor, melhor_jogada


def jogada_bot(tab, jogador_bot, dificuldade):
    """Escolhe a jogada do bot. Nunca usa o classificador de IA.

    - facil:   totalmente aleatório.
    - medio:   ganha/bloqueia se possível, senão aleatório.
    - dificil: minimax (ótimo, impossível de vencer).
    """
    livres = jogadas_livres(tab)
    if not livres:
        return None

    if dificuldade == "facil":
        return random.choice(livres)

    if dificuldade == "medio":
        # 1) jogada que vence agora
        for i in livres:
            tab[i] = jogador_bot
            if vencedor(tab) == jogador_bot:
                tab[i] = VAZIO
                return i
            tab[i] = VAZIO
        # 2) bloquear vitória do oponente
        adv = oponente(jogador_bot)
        for i in livres:
            tab[i] = adv
            if vencedor(tab) == adv:
                tab[i] = VAZIO
                return i
            tab[i] = VAZIO
        # 3) senão aleatório (prefere o centro)
        if 4 in livres:
            return 4
        return random.choice(livres)

    # dificil
    _, jogada = _minimax(list(tab), jogador_bot, jogador_bot)
    return jogada if jogada is not None else random.choice(livres)


DIFICULDADES = {"facil": "Fácil", "medio": "Médio", "dificil": "Difícil"}
