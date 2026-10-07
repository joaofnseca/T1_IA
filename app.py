"""
Jogo da Velha vs Bot com MLP que estima se a partida terminou.

Rodar:  streamlit run app.py

Passos:
  1. Ao abrir, treina o MLP e avalia seu desempenho em um teste separado.
  2. Setup: escolha a dificuldade do bot e jogar com X ou O.
  3. Jogo: após cada jogada, o MLP estima se a partida terminou. O resultado
     oficial continua sendo determinado pelas regras do jogo.
"""

import pandas as pd
import streamlit as st

import classificadores
import jogo

ROTULOS = ["nao_terminou", "terminou"]
NOMES_ROTULOS = ["Não terminou", "Terminou"]

ROTULO_CLASSE = {
    "nao_terminou": "🎮 O MLP prevê: jogo não terminou",
    "terminou": "🏁 O MLP prevê: jogo terminou",
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
    st.session_state.setdefault("classificador", None)
    st.session_state.setdefault("config", {})
    st.session_state.setdefault("tab", jogo.tabuleiro_vazio())
    st.session_state.setdefault("analise", None)


@st.cache_resource(show_spinner=False)
def obter_modelo_treinado(chave: str):
    """Treina uma vez por classificador e reutiliza o modelo nas partidas."""
    algoritmo = next(
        item for item in classificadores.descobrir() if item["chave"] == chave
    )
    modulo = algoritmo["modulo"]
    return modulo.treinar_e_avaliar()


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
    classificador = st.session_state.classificador
    if modelo is None or classificador is None:
        return
    tab = st.session_state.tab
    classe, probs = classificador["modulo"].classificar(modelo, tab)
    st.session_state.analise = {
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
    st.caption(
        "Escolha um classificador para avaliar cada estado do tabuleiro. "
        "O bot escolhe as jogadas separadamente."
    )

    algoritmos = classificadores.descobrir()
    opcoes = {"": "Selecione um classificador"}
    opcoes.update({str(item["chave"]): str(item["nome"]) for item in algoritmos})
    opcoes["__em_breve__"] = "Outro algoritmo (em breve)"

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

        algoritmo_escolhido = st.selectbox(
            "Classificador",
            options=list(opcoes),
            format_func=lambda chave: opcoes[chave],
        )
        enviar = st.form_submit_button("Treinar e avaliar ▶", type="primary")

    if enviar:
        if not algoritmo_escolhido:
            st.warning("Selecione um classificador para continuar.")
            return
        if algoritmo_escolhido == "__em_breve__":
            st.info(
                "Essa opção está reservada. Ela ficará disponível quando "
                "outro classificador for implementado."
            )
            return

        algoritmo = next(
            item for item in algoritmos if item["chave"] == algoritmo_escolhido
        )
        try:
            with st.spinner(f"Treinando {algoritmo['nome']}…"):
                modelo, relatorio = obter_modelo_treinado(algoritmo_escolhido)
        except Exception as erro:
            st.error(f"Não foi possível treinar {algoritmo['nome']}: {erro}")
            return

        st.session_state.modelo = modelo
        st.session_state.relatorio = relatorio
        st.session_state.classificador = algoritmo
        st.session_state.config = {
            "dificuldade": dificuldade,
            "jogador": jogador,
            "algoritmo": algoritmo["nome"],
        }
        st.session_state.fase = "treinado"
        st.rerun()


# ===========================================================================
# TELA 2 — RESULTADO DO TREINO/VALIDAÇÃO
# ===========================================================================
def tela_treino():
    r = st.session_state.relatorio
    st.title(f"✅ {st.session_state.config['algoritmo']} treinado e avaliado")
    st.caption(
        "A avaliação abaixo usa um "
        "conjunto de teste separado, que não participou do treinamento."
    )

    c1, c2, c3 = st.columns(3)
    if "f1_macro_teste" in r:
        c1.metric("F1-macro (teste)", f"{r['f1_macro_teste']:.3f}")
    if "acuracia_teste" in r:
        c2.metric("Acurácia (teste)", f"{r['acuracia_teste']:.3f}")
    if "n_teste" in r:
        c3.metric("Amostras de teste", r["n_teste"])

    if all(chave in r for chave in ("n_treino", "n_teste", "linhas_usadas")):
        st.caption(
            f"Split estratificado — treino: {r['n_treino']} · "
            f"teste: {r['n_teste']} · total: {r['linhas_usadas']}"
        )

    if "matriz_confusao" in r:
        cm = pd.DataFrame(
            r["matriz_confusao"],
            index=NOMES_ROTULOS,
            columns=NOMES_ROTULOS,
        )
        with st.expander("Matriz de confusão e relatório por classe (teste)"):
            st.write("Linhas = classe real · Colunas = classe prevista")
            st.dataframe(cm, use_container_width=True)
            if "relatorio_classes" in r:
                st.dataframe(
                    pd.DataFrame(r["relatorio_classes"]).T, use_container_width=True
                )

    if "loss_curve" in r:
        with st.expander("Evolução da perda durante o treinamento"):
            curva = pd.DataFrame(
                {"Época": range(1, len(r["loss_curve"]) + 1), "Loss": r["loss_curve"]}
            ).set_index("Época")
            st.line_chart(curva)

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
        st.caption(
            f"Você: {SIMBOLO[cfg['jogador']]}  ·  "
            f"Bot: {SIMBOLO[jogo.oponente(cfg['jogador'])]} · "
            f"Dificuldade: {jogo.DIFICULDADES[cfg['dificuldade']]}"
        )
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
                    on_click=jogada_humano,
                    args=(i,),
                )

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
            st.write("**Previsão do classificador:**")
            st.info(ROTULO_CLASSE.get(a["classe"], a["classe"]))
            st.caption(
                "A previsão do classificador é uma análise. O encerramento oficial da "
                "partida é determinado pelas regras do jogo."
            )
            if a["probs"]:
                st.write("**Probabilidade estimada por classe:**")
                probs = pd.Series(
                    {
                        ROTULO_CLASSE.get(k, k): v
                        for k, v in a["probs"].items()
                    }
                ).sort_values(ascending=False)
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
