# T1 IA — Jogo da Velha vs Bot com classificador de IA

Front em **Streamlit** para jogar Jogo da Velha contra um bot. A cada jogada, o
estado atual do tabuleiro é enviado a um **algoritmo classificador de IA** que
apenas **analisa** a situação do jogo — ele **não** decide as jogadas do bot.

> O bot adversário é clássico (aleatório / heurística / minimax). O classificador
> de IA e o bot são coisas separadas de propósito.

## Fluxo

1. **Setup** — escolha:
   - Dificuldade do bot: **Fácil** (aleatório), **Médio** (ganha/bloqueia) ou **Difícil** (minimax, imbatível).
   - Jogar com **X** (começa) ou **O** (joga em segundo).
   - Qual **algoritmo classificador** usar.
   - **Quantas linhas** do dataset de referência usar no treino.
2. **Treino/validação** — o algoritmo é treinado e validado (seleção de
   hiperparâmetros por **F1-macro** na validação; avaliação única no teste).
   O resultado (F1-macro, acurácia, matriz de confusão) é exibido.
3. **Jogo** — a cada jogada o tabuleiro atual é traduzido para a **Abordagem A**
   (`X=1`, `O=-1`, `B=0`) nas posições `sup_esq, sup_meio, sup_dir, meio_esq,
   centro, meio_dir, inf_esq, inf_meio, inf_dir` e enviado ao classificador. O
   rótulo analisado (`tem_jogo`, `x_venceu`, `o_venceu`, `empate`) e a confiança
   por classe aparecem no painel lateral.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Os 5 algoritmos classificadores

Ficam em `classificadores/algoritmo_1.py` … `algoritmo_5.py`, **vazios**. Cada um
deve expor:

```python
NOME = "Nome exibido no front"

def criar():
    # retorna (modelo_sklearn, grade_de_hiperparametros)
    return MeuModelo(), {"param": [v1, v2]}
```

O app descobre automaticamente os arquivos que seguem esse contrato; os ainda
vazios são ignorados. Veja o cabeçalho de `algoritmo_1.py` para o exemplo.

## Estrutura

| Arquivo | Papel |
|---------|-------|
| `app.py` | Front Streamlit (setup → treino → jogo) |
| `jogo.py` | Regras do jogo da velha + bot (3 dificuldades) |
| `treino.py` | Treino/validação (Abordagem A, F1-macro) + tradução do tabuleiro |
| `classificadores/` | Os 5 algoritmos + descoberta automática |

O dataset de referência vem de `../pacote_padrao/dataset_final.csv` (616 linhas).
