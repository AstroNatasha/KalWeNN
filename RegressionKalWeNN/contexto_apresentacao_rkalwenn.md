# Contexto para montar a apresentação do R-KalWeNN

Este documento reúne o que foi estabelecido ao longo do trabalho do artigo, para
que a apresentação seja montada sem precisar redescobrir nada. O PDF do artigo
será anexado junto; quando houver divergência entre este documento e o PDF, vale
o PDF, e a divergência deve ser apontada à usuária.

A apresentação reaproveita a do KalWeNN, que está no Google Slides dela, e é para
amanhã, 15/09/2026. Duração e idioma dos slides ainda não foram confirmados: a
apresentação do KalWeNN tinha 15 minutos e estava em português, mas isso precisa
ser perguntado.

---

## 1. Como trabalhar com a usuária

Estas regras foram construídas ao longo de várias sessões, a partir de erros que
aconteceram. Valem mais que qualquer conveniência.

**Papel.** Atuar como orientador de mestrado: conduzir o raciocínio junto com
ela, discordar quando for o caso e antecipar objeções de banca. Não substituir
as ideias dela por outras.

**Objetividade.** Resposta curta e direta. Modo didático só quando ela pedir com
todas as letras. Nada de resumo do que mudou, aviso preventivo ou "uma coisa que
vale notar". Fez o que foi pedido, diz que fez, para.

**Um passo por vez.** Propor uma coisa e parar. Ela decide. Quando ela pedir
passo a passo, entregar um passo, esperar ela avisar que cumpriu, e só então dar
o próximo. No máximo três perguntas por turno.

**Não afirmar sem verificar.** Se não deu para conferir, escrever "não
verifiquei" no corpo do texto. Número, fórmula e citação só a partir do PDF ou
deste documento.

**Imagens só sob pedido.** Não gerar nem alterar figura sem pedido explícito na
mensagem.

**Vocabulário é dela.** Usar os termos que ela já definiu e, quando faltar um,
perguntar em vez de inventar. Em particular, dizer "endereço" ou "endereço da
RAM", nunca "célula".

**Estilo de escrita.**
- Não usar travessão. Construir a coesão com conectivos, com parcimônia: "no
  entanto", "além disso", "por sua vez", "dessa forma", "uma vez que", "de modo
  que".
- Sem adjetivo desnecessário, metáfora, analogia de efeito, frase feita ou
  jargão de efeito. Proibido "o pulo do gato", "o gargalo" e similares.
- Evitar afirmação forte. Ela rejeitou frases como "é aqui que o contraste fica
  explícito". Preferir formulações medidas.
- Títulos simples e objetivos, nunca títulos de efeito.
- Não elogiar a pergunta nem repetir o que ela disse antes de responder.
- Fórmulas para slides devem vir em texto copiável para o Google Slides, com
  caracteres Unicode, por exemplo `x̂`, `P̂`, `ĉ`, `ŷ`, `K = P̂ / (P̂ + r)`.

---

## 2. Identidade visual

Paleta usada em todas as figuras, sempre com fundo branco.

| Uso | Cor |
|---|---|
| Linhas dos gráficos, ordem de uso | `#ffd23f` amarelo, `#16c1d4` ciano, `#db3763` rosa, `#5c7f44` verde |
| Eixos x e y, rótulos secundários | `#9c7b8e` |
| Título e subtítulo | `#2b2a5e` |
| Fundo | branco |

Convenção nas figuras do artigo: R-KalWeNN em ciano `#16c1d4`, Regression WiSARD
em rosa `#db3763`, terceira série em verde `#5c7f44`.

Ela prefere visual **minimalista**. A figura esquemática do artigo usa só
contorno `#9c7b8e`, texto `#2b2a5e` e um único destaque em `#16c1d4`, com
preenchimento cinza muito claro `#eef1f4` nos bits iguais a 1. Evitar blocos
coloridos de fundo e muitas cores ao mesmo tempo. Títulos de seção em negrito e
um pouco maiores que o resto.

---

## 3. O projeto

### 3.1 Linhagem

- **WiSARD** (Aleksander et al., 1984): rede neural sem peso. Aprende escrevendo
  em tabelas de consulta construídas sobre memórias RAM, em uma única passada
  pelos dados, sem gradiente. A entrada é binarizada por termômetro, os bits são
  embaralhados e cortados em tuplas, e cada tupla aponta um endereço numa RAM.
- **Regression WiSARD, ReW** (Lusquino Filho et al., Neurocomputing 416,
  280–291, 2020): extensão para regressão. Cada endereço guarda um contador `n_i`
  e a soma dos alvos `s_i`; o valor local é `μ_i = s_i / n_i`, e a predição
  combina os valores locais por uma função de agregação.
- **KalWeNN** (trabalho anterior dela, classificação, anais estendidos do SBBD
  2026): troca o contador de cada endereço por um filtro de Kalman escalar que
  guarda uma estimativa e sua incerteza. Em doze conjuntos de dados igualou ou
  melhorou a acurácia da WiSARD e reduziu a variação entre execuções.
- **R-KalWeNN** (este trabalho, regressão, submetido à LAFusion 2026, Springer
  CCIS, revisão duplo-cega): leva o KalWeNN para regressão.

### 3.2 O mecanismo do R-KalWeNN

Cada endereço da RAM guarda **dois filtros de Kalman escalares independentes**:

- **confiança** `ĉ_i`, com incerteza `p^c_i`, que observa `z = 1` a cada visita;
- **estimativa** `ŷ_i`, com incerteza `p^y_i`, que observa `z = y`, o alvo do
  exemplo que chegou ao endereço.

Cada visita roda a recursão, com `F = H = 1`:

```
P̂_pred = P̂ + q
K = P̂_pred / (P̂_pred + r)
x̂ ← x̂ + K (z − x̂)
P̂ ← (1 − K) P̂_pred
```

Ambos os estados começam em zero. Seis hiperparâmetros, três por campo: a
incerteza inicial `p0`, o ruído de processo `q` e o ruído de observação `r`.
Leitura didática que ela aprovou: `p0` diz o quanto o endereço confia no palpite
inicial; `r` diz o quanto confia em cada visita isolada; `q` controla a memória,
e com `q` perto de zero o endereço leva em conta todas as visitas.

Ponto central: **a recursão só age da segunda visita em diante**. Por isso o
tamanho da tupla, que decide quantas vezes cada endereço é revisitado, decide se
o filtro tem trabalho a fazer.

### 3.3 Funções de agregação, família K

Na ReW, só a média simples pondera pelos contadores; as outras dividem pelo
número `m` de RAM nodes com evidência. Na família K do R-KalWeNN, a confiança
multiplica a estimativa dentro da transformação e a normalização é feita pela
soma das confianças `C = Σ c_i`. As médias de potência usam `p = 2`, e as
harmônicas descartam endereços sem valor positivo.

| Média | Regression WiSARD | R-KalWeNN |
|---|---|---|
| Simple | `Σ s_i / Σ n_i` | `(1/C) Σ c_i y_i` |
| Power | `((1/m) Σ μ_i^p)^(1/p)` | `((1/C) Σ (c_i y_i)^p)^(1/p)` |
| Harmonic | `m / Σ μ_i^(−1)` | `C / Σ (c_i y_i)^(−1)` |
| Harmonic power | `(m / Σ μ_i^(−p))^(1/p)` | `(C / Σ (c_i y_i)^(−p))^(1/p)` |
| Exponential | `log((1/m) Σ e^μ_i)` | `log((1/C) Σ e^(c_i y_i))` |
| Median | `median_i μ_i` | `median_i c_i y_i` |

---

## 4. Protocolo experimental

- Conjunto: **House Prices**, apenas ele. YearPredictionMSD ficou fora desta
  submissão.
- Pré-processamento idêntico ao código publicado do trabalho de base: imputação,
  Box-Cox, atributos derivados, codificação disjuntiva e remoção de colunas quase
  constantes. Sobram 1.453 casas e 332 atributos, com 973 para treino e 480 para
  teste.
- Alvo em **`log(1+x)`**, que é a escala do código do trabalho de base. O 0,278
  publicado está nessa escala. Não converter para dólar: não foi medido.
- Métrica principal: **MAE**, porque é a que o trabalho de base reporta. RMSE e
  R² são reportados junto.
- Cinco execuções por configuração; o sorteio do mapeamento de bits para RAM
  nodes é a fonte de variação.
- Máquina: MacBook Pro 13", início de 2015, Core i5 dual-core de 2,7 GHz, 8 GB
  de RAM, sem GPU.
- Código anônimo: `https://anonymous.4open.science/r/rkalwenn`.

---

## 5. Resultados

### 5.1 Erro por função de agregação

Termômetro de 27 bits, tupla de 6 bits, hiperparâmetros vencedores da grade,
cinco execuções. Nessa codificação a memória tem 1.494 RAM nodes, 23.171
endereços ocupados e mediana de seis visitas por endereço.

| Média | ReW MAE | ReW RMSE | ReW R² | R-KalWeNN MAE | R-KalWeNN RMSE | R-KalWeNN R² |
|---|---|---|---|---|---|---|
| Simple | 0,2864 ± 0,0002 | 0,3764 | 0,1415 | 0,2471 ± 0,0002 | 0,3313 | 0,3352 |
| Power | 0,2481 ± 0,0001 | 0,3311 | 0,3359 | 0,2469 ± 0,0002 | 0,3314 | 0,3345 |
| Harmonic | 0,2481 ± 0,0001 | 0,3313 | 0,3350 | 0,2464 ± 0,0004 | 0,3323 | 0,3311 |
| Harmonic power | 0,2481 ± 0,0001 | 0,3314 | 0,3347 | **0,2459 ± 0,0003** | 0,3315 | 0,3342 |
| Exponential | **0,2480 ± 0,0001** | 0,3296 | 0,3418 | 0,2484 ± 0,0002 | 0,3322 | 0,3315 |
| Median | 0,2736 ± 0,0007 | 0,3592 | 0,2181 | 0,2744 ± 0,0014 | 0,3611 | 0,2101 |

Referências: o trabalho de base publica 0,278 com a mediana; chutar a média do
treino erra 0,3151. A média simples é onde a adaptação aparece mais: 0,2864 na
ReW e 0,2471 na família K. Em RMSE e R² a ordem entre os modelos se inverte, e o
texto diz que a comparação é lida no MAE.

### 5.2 Termômetro contra tupla

Termômetro de 6 a 30 bits, de 3 em 3; tuplas 2, 4, 6, 9, 12, 27, 36 e 54.

| Tupla | Mediana de visitas por endereço | % de endereços vistos uma vez |
|---|---|---|
| 2 | 57 | 8 |
| 4 | 14 | 16 |
| 6 | 6 | 23 |
| 9 | 3 | 32 |
| 12 | 2 | 39 |
| 27 | 1 | 63 |
| 36 | 1 | 73 |
| 54 | 1 | 85 |

Três leituras: o erro cai conforme a tupla cresce; as visitas por endereço caem
junto; e o termômetro quase não muda erro nem visitas, alterando só o número de
RAM nodes. Com tupla grande cada endereço guarda praticamente um exemplo e a
predição vira consulta, sem observação repetida para o filtro incorporar. Nas
tuplas grandes o R-KalWeNN colapsa: na tupla 27, 0,403 contra 0,161 da ReW.

**Ponto de operação**, escolhido pelo padrão de acesso e não pelo erro de teste:
mediana de pelo menos seis visitas e no máximo 25% dos endereços vistos uma vez;
entre as configurações que passam, a de menor MAE. Resultado: **termômetro de 9
bits, tupla de 6 bits, 498 RAM nodes**, mediana de seis visitas, 23% vistos uma
vez.

### 5.3 Hiperparâmetros do filtro

Produto cartesiano completo, 972 combinações, cinco execuções cada, termômetro 9,
tupla 6, agregação Harmonic power.

| Campo | Hiperparâmetro | Valores varridos | Vencedor |
|---|---|---|---|
| Confiança | `p0_c` | 10¹, 10², 10³ | 10¹ |
| | `q_c` | 10⁻³, 10⁻², 10⁻¹, 10⁰ | 10⁻² |
| | `r_c` | 10⁻¹, 10⁰, 10¹ | 10⁰ |
| Estimativa | `p0_y` | 10¹, 10², 10³ | 10² |
| | `q_y` | 10⁻⁴, 10⁻³, 10⁻² | 10⁻⁴ |
| | `r_y` | 10⁻¹, 10⁰, 10¹ | 10¹ |

Vencedor: **MAE 0,2451 ± 0,0003, RMSE 0,3305 ± 0,0004, R² 0,3381**. Rodando a
ReW no mesmo ponto e com o mesmo mapeamento: Exponential, MAE 0,2472, RMSE
0,3286, R² 0,3458.

Faixa de MAE ao longo da grade: de 0,2451 a 0,6416. 42 das 972 combinações ficam
acima de 0,3151, o erro de chutar a média.

Quanto cada eixo move o MAE mediano: `q_y` 0,0049, `p0_y` 0,0045, `r_y` 0,0025,
`p0_c` 0,0020, `r_c` 0,0018, `q_c` 0,0002. O campo da estimativa move mais.

Dois comportamentos explicam os painéis:
- **esquecimento**, controlado por `q`: como os exemplos de treino não têm
  ordem, esquecer só remove evidência; o MAE mediano cai de 0,2558, onde cada
  endereço guarda o equivalente às três últimas visitas, para 0,2472, onde guarda
  todas;
- **peso do estado inicial** depois das primeiras visitas: quando equivale a uma
  observação inteira, o MAE mediano sobe para 0,2926 na estimativa e 0,3073 na
  confiança; quando equivale a um décimo de observação, cai para 0,2462 e 0,2468.

Ela pediu que essa análise **não** seja explicada por razão entre
hiperparâmetros, que achou antididático.

---

## 6. Conclusões, limitações e trabalhos futuros

**Conclusões.** O R-KalWeNN leva para regressão a memória com estimativa e
incerteza. No House Prices alcança MAE de 0,2451 contra 0,2472 da ReW no mesmo
ponto de operação, ambos abaixo do 0,278 publicado. A condição sob a qual o filtro
tem efeito é a revisita, e ela pode ser medida antes de treinar, contando acessos.
Em classificação o endereço observa sempre o próprio acesso; em regressão observa
o alvo, que muda a cada visita.

**Limitações**, nas palavras dela:
1. o ganho sobre a ReW existe, mas é marginal;
2. só foi testado no House Prices;
3. como o número do trabalho de base não foi reproduzido exatamente, não dá para
   explicar por que a mediana venceu lá e perdeu aqui.

**Trabalhos futuros**, nas palavras dela:
1. mais conjuntos de dados, e maiores, com mais exemplos e portanto mais visitas;
2. novas funções de agregação que combinem melhor estimativa e confiança;
3. outras réguas de binarização;
4. séries temporais, onde o ruído de processo passa a ter papel.

---

## 7. Material pronto

- **Figura esquemática** do R-KalWeNN, em três faixas: *mapping* (entrada
  binária do termômetro, tuplas), *filtered memory* (três RAM nodes, cada um com o
  endereço acessado mostrando `ĉ`, `ŷ`, `p^c` e `p^y`, depois de 5, 3 e 2 visitas)
  e *prediction* (média simples ponderada pela confiança, resultado 11,50). Os
  valores saem da recursão com os hiperparâmetros vencedores. Arquivo:
  `artigo_rkalwenn/figures/rkalwenn_esquema.png`.
- **Figura termômetro contra tupla**, três painéis: MAE por tupla nos dois
  modelos; MAE contra termômetro com uma curva por tupla; mediana de visitas por
  tupla, com a linha do limiar de seis visitas.
- **Figura dos hiperparâmetros**, seis painéis, um por hiperparâmetro, com a
  mediana, o melhor valor e a faixa do melhor ao terceiro quartil em cada nível.

Figuras novas devem seguir a paleta da seção 2 e só ser feitas quando ela pedir.

## 8. Da apresentação do KalWeNN, o que já foi combinado

- Slide da WiSARD com uma frase de definição, três pontos positivos e três
  negativos, e os negativos já conduzindo aos benefícios que o Kalman traz dentro
  do endereço.
- Hipótese em frase curta, sem entregar a resposta.
- No slide do filtro, mostrar só o que fica guardado no endereço, `x̂` e `P̂`.
- Fórmulas em texto copiável para o Google Slides.
- Passo a passo da inferência como figura, no mesmo estilo das figuras de
  explicação.
