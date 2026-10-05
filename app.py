"""
Jogo da Velha vs Bot, com classificador de IA que ANALISA o estado do jogo.

Rodar:  streamlit run app.py

Passos:
  1. Setup: dificuldade do bot, jogar com X ou O, algoritmo classificador e
     quantas linhas do dataset usar.
  2. Treino: treina e valida o algoritmo escolhido (F1-macro) e mostra o
     resultado da validação/teste.
  3. Jogo: a cada jogada o tabuleiro atual é traduzido (x=1, o=-1, b=0) e
     enviado ao classificador, que devolve o rótulo analisado (NÃO decide a
     jogada do bot — o bot é um adversário clássico à parte).
"""

import pandas as pd
import streamlit as st

import classificadores
import jogo
import treino

ROTULO_CLASSE = {
    "tem_jogo": "🎮 Jogo em andamento",
    "x_venceu": "❌ X venceu",
    "o_venceu": "⭕ O venceu",
    "empate": "🤝 Empate",
}
SIMBOLO = {"x": "❌", "o": "⭕", "b": " "}

st.set_page_config(page_title="Jogo da Velha + IA", page_icon="🎯", layout="centered")

# Esconde o toolbar do Streamlit (botão Deploy, menu de 3 pontos e opções).
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
      [data-testid="stDecoration"] {display: none;}
      [data-testid="stStatusWidget"] {display: none;}
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_estado():
    st.session_state.setdefault("fase", "setup")   # setup -> treinado -> jogando
    st.session_state.setdefault("modelo", None)
    st.session_state.setdefault("relatorio", None)
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("tab", jogo.tabuleiro_vazio())
    st.session_state.setdefault("analise", None)


def nova_partida():
    st.session_state.tab = jogo.tabuleiro_vazio()
    st.session_state.analise = None
    cfg = st.session_state.config
    # se o jogador é O, o bot (X) começa
    if cfg["jogador"] == "o":
        i = jogo.jogada_bot(st.session_state.tab, "x", cfg["dificuldade"])
        st.session_state.tab[i] = "x"
    analisar()


def analisar():
    """Classifica o tabuleiro atual e guarda o resultado para exibir."""
    modelo = st.session_state.modelo
    if modelo is None:
        return
    tab = st.session_state.tab
    classe, probs = treino.classificar(modelo, tab)
    st.session_state.analise = {
        "entrada": treino.traduzir_tabuleiro(tab),
        "classe": classe,
        "probs": probs,
    }


def jogada_humano(i):
    cfg = st.session_state.config
    tab = st.session_state.tab
    if tab[i] != "b" or jogo.fim_de_jogo(tab):
        return
    tab[i] = cfg["jogador"]
    analisar()
    # vez do bot, se o jogo não acabou
    if not jogo.fim_de_jogo(tab):
        bot = jogo.oponente(cfg["jogador"])
        j = jogo.jogada_bot(tab, bot, cfg["dificuldade"])
        if j is not None:
            tab[j] = bot
        analisar()


# ===========================================================================
# TELA 1 — SETUP
# ===========================================================================
def tela_setup():
    st.title("🎯 Jogo da Velha vs Bot")
    st.caption("Com classificador de IA que **analisa** o estado do jogo a cada jogada. "
               "O bot adversário é clássico (minimax); a IA não decide suas jogadas.")

    algos = classificadores.descobrir()

    with st.form("setup"):
        c1, c2 = st.columns(2)
        with c1:
            dificuldade = st.radio("Dificuldade do bot",
                                   list(jogo.DIFICULDADES),
                                   format_func=lambda k: jogo.DIFICULDADES[k])
        with c2:
            jogador = st.radio("Você joga como",
                               ["x", "o"],
                               format_func=lambda j: "X (começa)" if j == "x" else "O (joga em segundo)")

        st.divider()
        st.subheader("Classificador de IA")

        if not algos:
            st.warning("Nenhum algoritmo implementado ainda. Preencha os arquivos "
                       "em `classificadores/algoritmo_N.py` (contrato: `NOME` e `criar()`).")
            algo_escolhido = None
        else:
            nomes = {a["chave"]: a["nome"] for a in algos}
            algo_escolhido = st.selectbox(
                "Algoritmo classificador",
                [a["chave"] for a in algos],
                format_func=lambda k: nomes[k])

        total = treino.total_linhas_dataset()
        n_linhas = st.slider("Quantas linhas do dataset de referência usar",
                             min_value=min(40, total), max_value=total, value=total, step=10)
        st.caption(f"Dataset de referência: **{total}** linhas (de `pacote_padrao/dataset_final.csv`).")

        enviar = st.form_submit_button("Treinar e validar ▶", type="primary",
                                       disabled=not algos)

    if enviar and algo_escolhido:
        mod = next(a["modulo"] for a in algos if a["chave"] == algo_escolhido)
        modelo, grade = mod.criar()
        with st.spinner("Treinando e validando o classificador…"):
            try:
                modelo_treinado, relatorio = treino.treinar(modelo, grade, n_linhas)
            except Exception as e:
                st.error(f"Falha ao treinar `{mod.NOME}`: {e}")
                return
        st.session_state.modelo = modelo_treinado
        st.session_state.relatorio = relatorio
        st.session_state.config = {"dificuldade": dificuldade, "jogador": jogador,
                                   "algoritmo": mod.NOME}
        st.session_state.fase = "treinado"
        st.rerun()


# ===========================================================================
# TELA 2 — RESULTADO DO TREINO/VALIDAÇÃO
# ===========================================================================
def tela_treino():
    r = st.session_state.relatorio
    cfg = st.session_state.config
    st.title("✅ Classificador treinado e validado")
    st.caption(f"Algoritmo: **{cfg['algoritmo']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("F1-macro (teste)", f"{r['f1_macro_teste']:.3f}")
    c2.metric("Acurácia (teste)", f"{r['acuracia_teste']:.3f}")
    c3.metric("Linhas usadas", r["linhas_usadas"])

    st.caption(f"Split — treino: {r['n_treino']} · validação: {r['n_validacao']} · teste: {r['n_teste']}")

    if r["melhores_hiperparametros"]:
        st.write("**Melhores hiperparâmetros** (escolhidos por F1-macro na validação):")
        st.json(r["melhores_hiperparametros"])

    with st.expander("Seleção de hiperparâmetros (validação)"):
        st.dataframe(r["tabela_selecao"], use_container_width=True)

    with st.expander("Matriz de confusão e relatório por classe (teste)"):
        cm = pd.DataFrame(r["matriz_confusao"], index=treino.CLASSES, columns=treino.CLASSES)
        st.write("Linhas = classe real · Colunas = classe prevista")
        st.dataframe(cm, use_container_width=True)
        st.dataframe(pd.DataFrame(r["relatorio_classes"]).T, use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("🎮 Começar a jogar", type="primary"):
        st.session_state.fase = "jogando"
        nova_partida()
        st.rerun()
    if c2.button("↩ Voltar ao setup"):
        st.session_state.fase = "setup"
        st.rerun()


# ===========================================================================
# TELA 3 — JOGO
# ===========================================================================
def tela_jogo():
    cfg = st.session_state.config
    tab = st.session_state.tab
    venc = jogo.vencedor(tab)
    acabou = jogo.fim_de_jogo(tab)

    esq, dir_ = st.columns([3, 2])

    with esq:
        st.subheader("Tabuleiro")
        st.caption(f"Você: {SIMBOLO[cfg['jogador']]}  ·  Bot: {SIMBOLO[jogo.oponente(cfg['jogador'])]} "
                   f"·  Dificuldade: {jogo.DIFICULDADES[cfg['dificuldade']]}")
        for linha in range(3):
            cols = st.columns(3)
            for col in range(3):
                i = linha * 3 + col
                marcado = tab[i] != "b"
                cols[col].button(
                    SIMBOLO[tab[i]] if marcado else "·",
                    key=f"cell_{i}",
                    disabled=marcado or acabou,
                    use_container_width=True,
                    on_click=jogada_humano, args=(i,))

        if acabou:
            if venc == cfg["jogador"]:
                st.success("Você venceu! 🎉")
            elif venc:
                st.error("O bot venceu. 🤖")
            else:
                st.info("Empate. 🤝")

        c1, c2 = st.columns(2)
        if c1.button("🔄 Nova partida"):
            nova_partida()
            st.rerun()
        if c2.button("↩ Trocar configuração"):
            st.session_state.fase = "setup"
            st.rerun()

    with dir_:
        st.subheader("🔍 Análise da IA")
        st.caption(f"Classificador: {cfg['algoritmo']}")
        a = st.session_state.analise
        if not a:
            st.info("Faça uma jogada para ver a análise.")
        else:
            st.write("**Classe analisada:**")
            st.success(ROTULO_CLASSE.get(a["classe"], a["classe"]))

            if a["probs"]:
                st.write("**Confiança por classe:**")
                probs = (pd.Series({ROTULO_CLASSE.get(k, k): v for k, v in a["probs"].items()})
                         .sort_values(ascending=False))
                st.bar_chart(probs)


# ===========================================================================
init_estado()
fase = st.session_state.fase
if fase == "setup":
    tela_setup()
elif fase == "treinado":
    tela_treino()
else:
    tela_jogo()
