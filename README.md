# T1 IA — Jogo da Velha vs Bot com classificador de IA

Front em **Streamlit** para jogar Jogo da Velha contra um bot. A cada jogada, o
estado atual do tabuleiro é enviado a uma rede MLP que estima se o jogo terminou;
ela não decide as jogadas do bot.

> O bot adversário é clássico (aleatório / heurística / minimax). O classificador
> de IA e o bot são coisas separadas de propósito.

## Fluxo

1. **Setup** — escolha:
   - Dificuldade do bot: **Fácil** (aleatório), **Médio** (ganha/bloqueia) ou **Difícil** (minimax, imbatível).
   - Jogar com **X** (começa) ou **O** (joga em segundo).
   - Classificador. O MLP está disponível e há uma opção reservada para
     algoritmos que ainda serão implementados.
2. **Treinamento e avaliação** — ao selecionar o MLP e continuar, ele é
   treinado com o dataset numérico e avaliado em um conjunto de teste separado.
   Acurácia, F1-macro, matriz de confusão, relatório e curva de perda são
   apresentados antes da partida.
3. **Jogo** — depois de cada jogada humana e do bot, as nove posições são
   enviadas ao MLP, que estima se o jogo terminou. A previsão é informativa; as
   regras do jogo determinam o resultado oficial e bloqueiam novas jogadas.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Classificador MLP: jogo terminado ou não

O script `mlp_classificador.py` contém o treinamento e a avaliação da rede
usados pelo app. Ele usa exclusivamente as nove posições do tabuleiro como
entradas (`id` é ignorado) e divide os dados em treino e teste com
estratificação. Na coluna `classe`, `tem_jogo` é convertido para `nao_terminou`,
enquanto `x_venceu`, `o_venceu` e `empate` são convertidos para `terminou`. No
CSV atual, a posição central está identificada como `centro`, equivalente a
`meio_meio`. O modelo é treinado uma vez quando você seleciona o MLP e envia o
formulário; ele é reutilizado nas partidas.

Para executar somente o treinamento e exibir as métricas no terminal e a curva
de perda:

```bash
python mlp_classificador.py
```

## Estrutura

| Arquivo | Papel |
|---------|-------|
| `app.py` | Front Streamlit (treino automático → configuração → jogo) |
| `jogo.py` | Regras do jogo da velha + bot (3 dificuldades) |
| `mlp_classificador.py` | Treinamento, avaliação e previsões do MLP |
| `classificadores/` | Adaptadores dos classificadores disponíveis no app |
| `dataset_final_numerico.csv` | Dataset usado pelo classificador |

### Adicionar outro classificador

Crie um módulo `.py` dentro de `classificadores/` com o contrato abaixo. O app
detecta módulos compatíveis automaticamente; os arquivos existentes que ainda
usam o contrato antigo não aparecem como opções até serem adaptados. O resultado
de `classificar()` deve usar os rótulos `nao_terminou` e `terminou`. As métricas
de `treinar_e_avaliar()` podem incluir `acuracia_teste`, `f1_macro_teste`,
`n_treino`, `n_teste`, `linhas_usadas`, `matriz_confusao`,
`relatorio_classes` e `loss_curve`; chaves indisponíveis são opcionais.

```python
NOME = "Nome exibido no seletor"

def treinar_e_avaliar():
    # Retorna (modelo, relatorio) após avaliar num teste separado.
    ...

def classificar(modelo, tabuleiro):
    # tabuleiro contém 9 valores: "x", "o" ou "b".
    # Retorna ("terminou" ou "nao_terminou", probabilidades_por_classe_ou_None).
    ...
```
