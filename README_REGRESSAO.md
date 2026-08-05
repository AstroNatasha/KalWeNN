# K-WiSARD — Estudo de Regressão: ReW vs KalWeNN

Extensão do estudo K-WiSARD para **problemas de regressão** (alvo contínuo). O
estudo de classificação está no [README.md](README.md); este documento é
autocontido e descreve o que foi implementado, como, com que dados, sob que
protocolo, e o que os experimentos mostraram.

**Resumo em uma frase:** em 8 datasets de regressão, a busca de hiperparâmetros
do KalWeNN converge sistematicamente para o caso limite em que ele *é* a ReW, e a
ReW vence nos 8; os ganhos reais que encontramos foram em outro lugar — em **como
agregar os neurônios**, onde a média canônica da ReW fica 15% atrás das
alternativas, e em manter a incerteza **por célula** e não compartilhada (até 12%).

Toda a nossa ReW foi **validada contra a implementação de referência** (wisardpkg
v2.0.0a7, do grupo da UFRJ): com a mesma entrada e o mesmo mapeamento, as
predições batem na precisão do `float64` para os 7 agregadores comparáveis (§3.2).

---

## 1. As cinco perguntas do estudo e onde cada uma foi respondida

| # | Pedido | Como foi operacionalizado | Resultado (§9) |
|---|---|---|---|
| 1 | KalWeNN com Regression WiSARD, acessando neurônios e contando os números | `RegressionWiSARD` guarda (contador, soma) por célula; 9 formas de agregar os números contados | §9.2 — contar **piora**; a média canônica fica 15% atrás |
| 2 | Testar o `p` em lugares diferentes | 3 escopos: por célula, por RAM, global | §9.3 — por célula sempre vence; RAM ≡ global (provado) |
| 3 | Kalman na inferência | fusão bayesiana `Σ(x/p)/Σ(1/p)` e versão sequencial com prior | §9.4 — perde para a média simples |
| 4 | Métricas MAE, RMSE, MSE (+ acurácia) | as três + R² + duas acurácias por tolerância + CV entre seeds | §8 e §9.1 |
| 5 | 8 datasets de regressão, 70/30 | 8 datasets, split 70/30, 30 seeds | §4 e §5 |

Além do pedido, o estudo fecha as lacunas de protocolo que ficaram abertas na
primeira rodada: validação contra a implementação de referência (§3.2), teste de
significância (§9.6), baselines da própria biblioteca incluindo a ClusReW (§9.7)
e varreduras de épocas, `p₀` e estratificação do split (§9.8).

---

## 2. Como reproduzir

O ambiente do projeto não tinha as bibliotecas científicas instaladas. Foi criado
um virtualenv local (`.venv/`, ignorado pelo git):

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn matplotlib scipy psutil
```

Pipeline completo (o tempo total medido nesta máquina foi de **4 min 22 s**):

```bash
# 1. modelos: verifica que a ReW é o caso limite do KalWeNN (§7)
.venv/bin/python -m wisard.models_regression

# 2. os 8 datasets: baixa, limpa e resume
.venv/bin/python -m datasets.loaders_regression

# 3. experimentos (todas as sete etapas)
.venv/bin/python analysis/05_regression.py all
#    ou por etapa/dataset:
.venv/bin/python analysis/05_regression.py grid final concrete yacht
```

As etapas, na ordem de dependência:

| Etapa | O que faz | Saída |
|---|---|---|
| `arch` | escolhe tupla × resolução na validação interna | `{ds}_arch.csv` |
| `grid` | busca (q, r, p₀) por escopo de p, na validação interna | `{ds}_grid.csv` |
| `final` | comparação no teste, 30 seeds, 24 variantes | `regression_report.csv` |
| `baselines` | modelos de referência do scikit-learn | `baselines.csv` |
| `signif` | Wilcoxon pareado, melhor ReW vs melhor KalWeNN | `significance.csv` |
| `epochs` | o KalWeNN melhora repassando o treino? | `epochs.csv` |
| `strat` | estratificar o split por faixas reduz o CV? | `stratification.csv` |

```bash
# 4. figuras e tabelas
.venv/bin/python analysis/06_regression_analysis.py
```

As etapas 5 e 6 — validação e baselines da biblioteca de referência — rodam em
outro ambiente, porque o `wisardpkg` não compila no Python 3.14. A receita
completa, sem sudo, está em §3.2:

```bash
/tmp/wpkg-env/bin/python analysis/07_wisardpkg_check.py     # equivalência numérica
/tmp/wpkg-env/bin/python analysis/08_wisardpkg_models.py    # ReW e ClusReW da referência
```

Ordens de grandeza nesta máquina: `arch` ~10 s · `grid` ~40 min (126 combos × 3
escopos × 8 datasets) · `final` ~15 min · `baselines` ~30 min (o random forest
domina) · `signif` ~10 min · `epochs`/`strat` ~5 min cada · `07` e `08` ~1 min
cada.

Variáveis de ambiente: `REG_FINAL_SEEDS` (padrão 30), `REG_SEL_SEEDS` (padrão 3)
e `REG_MAX_SAMPLES` (padrão 5000; use 0 para tirar o teto).

---

## 3. Implementação — o que é próprio e o que veio de fora

Esta seção existe porque a pergunta "o que foi feito à mão?" é a primeira que
aparece na banca.

### 3.1 Todas as WiSARDs deste estudo foram escritas à mão

Nenhum modelo do estudo vem de biblioteca de terceiros — o `wisardpkg` entra
só como **oráculo de validação** (§3.2), nunca para produzir resultados. Os dois
modelos comparados estão em
[wisard/models_regression.py](wisard/models_regression.py), escritos do zero em
NumPy puro:

- `RegressionWiSARD` — a ReW, com memória `{endereço: [contador, soma]}` por RAM;
- `KalmanRegressionWiSARD` — o KalWeNN de regressão, com memória
  `{endereço: [x, p, n]}` por RAM.

Ambos reaproveitam duas peças já existentes no repositório, para que os modelos
de regressão sejam comparáveis aos de classificação célula a célula:

- a codificação termômetro de [wisard/encoder.py](wisard/encoder.py);
- a lógica de mapeamento bit→RAM da `BaseWiSARD` de
  [wisard/models.py](wisard/models.py), replicada em `BaseRegressionWiSARD` sem a
  parte de classes (regressão não tem discriminador por classe). Com a mesma
  seed, o mapeamento sorteado é o mesmo.

### 3.2 Validação contra o pacote da literatura (`wisardpkg`)

O pacote de referência do grupo de WiSARD da UFRJ é o **`wisardpkg`**. Colocá-lo
para rodar deu trabalho e revelou duas armadilhas que valem registro:

**Armadilha 1 — a versão do PyPI não tem regressão.** A última versão publicada
no PyPI é a **1.6.3, de fevereiro de 2019**, e ela contém apenas
`Wisard`, `ClusWisard`, `KernelCanvas` e `Synthesizer`. Não há
`RegressionWisard` nenhuma. A ReW só existe na branch `develop` do repositório
oficial, sob as tags `v2.0.0a1`…`v2.0.0a7` (abril de 2020), que nunca foram
publicadas no PyPI. Ou seja: `pip install wisardpkg` **nunca** daria acesso à
ReW, com ou sem permissão de administrador.

**Armadilha 2 — não compila no Python do sistema.** É uma extensão C++ via
pybind11, sem wheel pré-compilado; no Python 3.14.4 desta máquina a compilação
falha por falta dos cabeçalhos (`Python.h`). Instalar `python3.14-dev` exigiria
sudo — mas **não é preciso**: um Python 3.11 autocontido instalado no espaço do
usuário já traz os próprios cabeçalhos, e o pacote (de 2019/2020) compila
naturalmente numa versão da época.

Receita completa, sem nenhum privilégio administrativo:

```bash
.venv/bin/pip install uv
.venv/bin/uv python install 3.11
.venv/bin/uv venv --python 3.11 /tmp/wpkg-env

git clone https://github.com/IAZero/wisardpkg.git /tmp/wisardpkg-src
git -C /tmp/wisardpkg-src fetch origin develop && git -C /tmp/wisardpkg-src checkout FETCH_HEAD

.venv/bin/uv pip install --python /tmp/wpkg-env/bin/python "pybind11<3" setuptools wheel
.venv/bin/uv pip install --python /tmp/wpkg-env/bin/python --no-build-isolation /tmp/wisardpkg-src
.venv/bin/uv pip install --python /tmp/wpkg-env/bin/python numpy pandas scikit-learn psutil

/tmp/wpkg-env/bin/python analysis/07_wisardpkg_check.py     # a validação
```

**A validação.** [analysis/07_wisardpkg_check.py](analysis/07_wisardpkg_check.py)
força os dois modelos a usarem a mesma entrada binarizada (o nosso termômetro), o
mesmo mapeamento bit→RAM (o wrapper aceita o kwarg `mapping`, um vetor achatado
em que a RAM *i* fica com `mapping[i·tupla : (i+1)·tupla]` — exatamente o layout
do nosso `_mapping`) e o mesmo alvo normalizado. Resultado nos 8 datasets × 2
seeds, como maior diferença absoluta dividida pela amplitude do alvo:

| Agregador | pior caso | |
|---|---|---|
| `simple` (SimpleMean) | 8,40e-16 | IGUAL |
| `unweighted` (PowerMean(1)) | 1,99e-16 | IGUAL |
| `power2` (PowerMean(2)) | 2,04e-16 | IGUAL |
| `harmonic` (HarmonicMean) | 2,85e-16 | IGUAL |
| `harmonic_power2` (HarmonicPowerMean(2)) | 2,04e-16 | IGUAL |
| `geometric` (GeometricMean) | 0,00e+00 | IGUAL |
| `exponential` (ExponentialMean) | 8,40e-16 | IGUAL |

Tudo no nível do epsilon do `float64`: a nossa ReW **é** a ReW da referência.

**O que a validação corrigiu.** A primeira versão deste estudo definia os
agregadores por conta própria, ponderando todos pelos contadores. Ao ler o fonte
(`src/models/regressionwisard/meanfunctions.cc`) ficou claro que **só a
`SimpleMean` pondera**: todas as outras dividem por N, o número de células
acessadas, ignorando os contadores. As fórmulas foram corrigidas e os
experimentos refeitos — o que mudou conclusões (§9.2), porque a família não
ponderada é justamente a que funciona melhor.

**Dois defeitos encontrados na referência.** Reproduzimos um e corrigimos o outro,
sempre declarando qual é qual:

1. `Median` (linhas 87–94) indexa o vetor ordenado com deslocamento de +1: para
   tamanho ímpar *s* usa `(s−1)/2 + 1` em vez de `(s−1)/2`, e para tamanho par
   usa `means[s/2]` e `means[s/2+1]`, que lê **fora dos limites** quando há
   exatamente 2 células acessadas. Implementamos a mediana correta, e por isso a
   `median` fica de fora da tabela de equivalência acima.
2. `ExponentialMean` (linha 233) inicializa o acumulador em `1` em vez de `0`,
   somando um `e⁰` fantasma. Com isso o resultado deixa de ser uma média (não fica
   mais entre o mínimo e o máximo das células). Reproduzimos esse comportamento em
   `exponential`, para a equivalência valer, **e** implementamos a versão sã em
   `exponential_fixed`. O defeito custa de 1,1% a 62,2% de RMSE (§9.2) — vale
   reportar aos mantenedores.
3. **Fallback para amostras sem cobertura.** Quando nenhuma célula endereçada foi
   vista no treino, a referência devolve **0** — que, na escala normalizada,
   equivale a prever o *mínimo* do alvo. Nós devolvemos a **média do treino**, que
   é o fallback padrão de qualquer regressor. Esta é uma divergência de projeto,
   não de fórmula, e por isso a tabela de equivalência acima compara apenas as
   amostras com ao menos uma célula conhecida. A diferença não é acadêmica: nos 5
   datasets em que os dois lados usam o mesmo agregador, ela sozinha vale de
   **1,2% a 4,8% de RMSE** a nosso favor (§9.7). Com tuplas pequenas ela nunca
   aparece — só surge nas tuplas grandes (48–96) que este estudo usa, onde parte
   das amostras de teste não encontra nenhum endereço conhecido.

### 3.3 O que é de biblioteca de terceiros

| Uso | Biblioteca |
|---|---|
| Álgebra, RNG, tudo numérico | NumPy |
| Tabelas e CSV | pandas |
| Split treino/teste, carga de datasets, modelos de referência | scikit-learn |
| Figuras | matplotlib |

Os **modelos de referência** de §9.6 (linear, kNN, árvore, random forest, dummy)
são do scikit-learn e servem só para dar escala aos números — não fazem parte da
comparação central.

---

## 4. Os dados — 8 datasets de regressão

Carregados por [datasets/loaders_regression.py](datasets/loaders_regression.py),
que segue o mesmo contrato do loader de classificação: cada função devolve
`(X, y)` em float.

| Dataset | n | atributos | alvo | faixa do alvo | média | σ | Origem |
|---|---|---|---|---|---|---|---|
| diabetes | 442 | 10 | progressão da doença em 1 ano | 25 – 346 | 152,13 | 77,01 | scikit-learn |
| california | 5.000\* | 8 | valor mediano das casas | 0,175 – 5,000 | 2,07 | 1,15 | scikit-learn |
| airfoil | 1.503 | 5 | pressão sonora (dB) | 103,38 – 140,99 | 124,84 | 6,90 | OpenML 44957 |
| energy | 768 | 8 | carga de aquecimento | 6,01 – 43,10 | 22,31 | 10,08 | OpenML 44960 |
| concrete | 1.030 | 8 | resistência à compressão (MPa) | 2,33 – 82,60 | 35,82 | 16,70 | OpenML 44959 |
| abalone | 4.177 | 8 | idade (nº de anéis) | 1 – 29 | 9,93 | 3,22 | OpenML 44956 |
| auto_mpg | 392 | 7 | consumo (milhas por galão) | 9,00 – 46,60 | 23,45 | 7,80 | OpenML `autoMpg` v1 |
| yacht | 308 | 6 | resistência residual do casco | 0,01 – 62,42 | 10,50 | 15,14 | OpenML `yacht_hydrodynamics` v1 |

\* subamostrado de 20.640 com semente fixa (`rng=0`), teto de 5.000 por dataset,
para manter o custo do estudo viável. É o único dataset afetado pelo teto.

Três decisões de carga que valem registro:

1. **Os `data_id` da suíte OpenML-CTR23** (44956–44960) foram preferidos aos
   nomes antigos. As versões acessadas por nome (`energy-efficiency`,
   `abalone`) trazem o alvo **já discretizado em categorias** — usá-las
   transformaria o problema em classificação disfarçada. Isso foi detectado
   durante a preparação: a `energy-efficiency` v1 devolvia 37 valores distintos
   no lugar dos 587 valores reais da carga de aquecimento.
2. **Colunas categóricas** (ex.: `Sex` no abalone) viram códigos inteiros por
   `factorize`, mesma convenção do loader de classificação.
3. **Linhas com NaN/inf** em X ou y são descartadas (só afeta o auto_mpg:
   398 → 392 linhas).

---

## 5. Protocolo experimental

```
dados
 └─ split externo 70/30  (30 seeds na comparação final)
     ├─ TREINO 70%
     │   └─ split interno 70/30  (3 seeds)   ← toda a seleção acontece aqui
     │       ├─ treino interno → ajusta o modelo
     │       └─ VALIDAÇÃO      → escolhe tupla, resolução, q e r
     └─ TESTE 30%  ← tocado uma única vez, na comparação final
```

- **70/30**, como pedido, **sem estratificação** — não existe estratificação para
  alvo contínuo. (Estratificar por *bins* de y é possível e está na lista de
  pendências.)
- **Seleção de hiperparâmetros em validação interna**, nunca no teste. Isso é
  mais rigoroso do que o estudo de classificação deste repositório, em que a
  arquitetura é escolhida com a métrica do próprio teste
  (`select_architecture` em [analysis/01_common.py](analysis/01_common.py)).
- **30 seeds** na comparação final; cada seed é um split 70/30 diferente **e** um
  mapeamento bit→RAM diferente. Toda métrica reportada é média sobre as 30, com
  desvio e CV%.
- **Mesma arquitetura para os dois modelos** dentro de cada dataset: a tupla e a
  resolução são escolhidas uma vez (com a ReW) e reaproveitadas pelo KalWeNN.
  Isso isola a regra de atualização, que é o objeto do estudo.

### Normalização do alvo

Os dois modelos escalam `y` para [0,1] com o mínimo/máximo **do treino** e
devolvem a predição na escala original. Duas razões:

- é obrigatório no Kalman — `p0 = 1` pressupõe alvo de ordem de grandeza 1;
- é aplicado igualmente à ReW para que a comparação seja justa: as médias não
  lineares (harmônica, geométrica, potência) **não** são invariantes a mudança de
  escala, então normalizar só um dos lados enviesaria o resultado.

### Grades de busca

| O que | Valores | Combinações |
|---|---|---|
| tupla (`address_size`) | 16, 32, 48, 64 | 20 combinações de arquitetura |
| resolução do termômetro | 16, 32, 64, 128, 256 | (avaliadas com a ReW) |
| `q` (ruído de processo) | 0, 1e-4, 1e-3, 1e-2, 1e-1, 1 | 126 triplas (q, r, p₀) |
| `r` (ruído de observação) | 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2 | × 3 escopos de p |
| `p₀` (incerteza inicial) | 0,1 / 1 / 10 | |

O grid inclui **q = 0** de propósito: é o caso limite que reproduz a ReW (§7).
Cada modelo treinado no grid é avaliado nos 5 modos de inferência de uma vez — a
inferência não altera o treino —, então o melhor `(q, r)` é escolhido
separadamente para cada par escopo × inferência.

**Arquiteturas escolhidas** (validação interna, ReW de média simples):

| Dataset | tupla | resolução | R² na validação |
|---|---|---|---|
| diabetes | 32 | 256 | 0,444 |
| california | 64 | 128 | 0,557 |
| airfoil | 64 | 128 | 0,617 |
| energy | 48 | 256 | 0,972 |
| concrete | 48 | 256 | 0,551 |
| abalone | 64 | 128 | 0,443 |
| auto_mpg | 48 | 256 | 0,843 |
| yacht | 48 | 256 | 0,650 |

Vale notar que a busca foi consistentemente para **tuplas grandes** (32–64). Com
tuplas pequenas (8, como no regime denso do estudo de classificação) o modelo
subajusta feio: cada célula agrega amostras demais e a predição colapsa na média
global — o R² na california fica em 0,09 com tupla 8 contra 0,56 com tupla 64.
Faz sentido: em classificação basta ordenar as classes corretamente, enquanto em
regressão a célula precisa estimar um valor, e para isso ela precisa ser
específica.

---

## 6. Os modelos

Notação: a amostra é codificada em bits pelo termômetro e o mapeamento sorteado
distribui esses bits entre `n_rams` RAMs; cada RAM `r` é endereçada por uma
tupla de bits, produzindo um endereço `a`. Em regressão **não há discriminador
por classe** — existe um único conjunto de RAMs, e cada célula estima o alvo.

Toda a memória é **esparsa** (dicionários): só os endereços vistos são
guardados, o que dispensa alocar `2^tupla` células e é o que viabiliza tuplas de
48 a 96 bits.

### 6.1 ReW — Regression WiSARD

Cada célula guarda `(n, s)` = número de acessos e soma dos alvos. O treino é uma
passada: para cada amostra, cada RAM incrementa a célula endereçada.

Na predição, acessamos os neurônios e agregamos os números contados. Sendo
`m_i = s_i/n_i` a média local da célula `i`, `n_i` o contador e `N` o número de
células acessadas com `n_i > 0`:

| Variante | Fórmula | Nome na referência |
|---|---|---|
| `simple` | `Σ s_i / Σ n_i` | `SimpleMean` — **a ReW canônica**, a única que pondera pelos contadores |
| `unweighted` | `(1/N) Σ m_i` | `PowerMean(1)` — média aritmética das células |
| `power2` | `√( Σ m_i² / N )` | `PowerMean(2)` — média quadrática |
| `median` | mediana dos `m_i` | `Median` (com a indexação corrigida) |
| `harmonic` | `N / Σ (1/m_i)` | `HarmonicMean` |
| `harmonic_power2` | `√( N / Σ (1/m_i²) )` | `HarmonicPowerMean(2)` |
| `geometric` | `(Π m_i)^(1/N)` | `GeometricMean` |
| `exponential` | `log( (1 + Σ e^{m_i}) / N )` | `ExponentialMean` — inclui o "+1" defeituoso da referência |
| `exponential_fixed` | `log( Σ e^{m_i} / N )` | a mesma, sem o "+1" (nossa) |

O ponto que mais importa aqui: **só a `simple` pondera pelos contadores**. Todas
as demais são médias simples sobre as células. Isso não é escolha nossa — é o que
a implementação de referência faz, e foi o que a leitura do fonte revelou (§3.2).

Fica de fora a `LogisticMean` da referência, que não é um estimador puro: aplica
uma sigmoide sobre uma combinação linear com três parâmetros (β₁, β₂, γ) que
precisariam ser ajustados, o que a tornaria um modelo diferente e não uma forma
de agregar.

Células nunca vistas no treino (`n = 0`) não entram na conta. Se nenhuma célula
endereçada for conhecida, a predição é a média do treino. Como o treino não
depende do agregador, um único modelo treinado serve para as 9 variantes
(`set_estimator`).

### 6.2 KalWeNN de regressão

Cada célula guarda `(x, p, n)`. A diferença conceitual para a versão de
classificação é a **observação**: lá `z = 1,0` significa "este endereço foi
visto"; aqui `z = y` é o próprio alvo normalizado da amostra, e a célula passa a
estimar o valor do alvo naquele endereço.

```
p_pred = p + q
k      = p_pred / (p_pred + r)      # ganho de Kalman
x      = x + k · (z − x)            # z = alvo normalizado
p      = (1 − k) · p_pred
```

Como em classificação, a ordem de apresentação das amostras é sorteada por seed,
então a aleatoriedade do KalWeNN vem de duas fontes (mapeamento e ordem) contra
uma só da ReW (mapeamento) — a ReW é indiferente à ordem, porque soma.

### 6.3 Onde vive o `p` — três escopos

| Escopo | O que significa |
|---|---|
| `cell` | um `p` por célula (RAM, endereço). Cada endereço tem seu próprio ritmo: endereços raros seguem com `p` alto e continuam aprendendo rápido. |
| `ram` | um `p` por RAM, compartilhado por todos os endereços dela. |
| `global` | um único `p` para o modelo inteiro; o ganho vira função só do tempo. |

### 6.4 Como as RAMs viram uma predição — cinco modos de inferência

| Modo | Fórmula | O que é |
|---|---|---|
| `mean` | `(1/N) Σ x_i` | média simples das evidências |
| `count` | `Σ n_i x_i / Σ n_i` | ponderada pelos contadores — o análogo direto da ReW |
| `conf` | `Σ (1−p_i) x_i / Σ (1−p_i)` | heurística de confiança (a mesma do `04_experiments.py fusion`) |
| `fusion` | `Σ (x_i/p_i) / Σ (1/p_i)` | **fusão bayesiana**: peso = precisão. É o Kalman aplicado na inferência |
| `kalman_seq` | `(μ₀/P₀ + Σ x_i/p_i) / (1/P₀ + Σ 1/p_i)` | a mesma fusão partindo do prior do treino (μ₀, P₀ = média e variância do alvo) |

`fusion` é a versão principiada do que o estudo de classificação faz de forma
heurística em `conf`: lá o peso é `(1−p)`, aqui é `1/p`, que é o peso correto
para fundir observações independentes com variâncias diferentes.

---

## 7. Resultado teórico: a ReW é um caso limite do KalWeNN

Com `q = 0`, a recursão da incerteza vira `1/pₙ = 1/p₀ + n/r`, e portanto

```
kₙ = 1 / (n + r/p₀)
```

Quando `r/p₀ → 0`, isso é exatamente `kₙ = 1/n`, o ganho da **média aritmética
incremental**: cada célula passa a guardar a média exata dos alvos vistos nela.
Agregando com os contadores (`count`), obtém-se `Σ nᵢmᵢ / Σ nᵢ = Σ sᵢ / Σ nᵢ`, que
é a ReW de média simples.

Ou seja: **a ReW não é um modelo concorrente do KalWeNN — é o KalWeNN num canto
específico do seu espaço de parâmetros.** O que o filtro de Kalman acrescenta é a
possibilidade de sair desse canto: com `q > 0` o ganho não decai até zero e a
célula passa a esquecer o passado (média com desconto exponencial).

Verificação numérica (`python3 -m wisard.models_regression`) — maior diferença
absoluta entre as predições dos dois modelos, em % da amplitude do alvo,
conforme `p₀` cresce:

| dataset | seed | p₀=10 | p₀=10² | p₀=10⁴ | p₀=10⁶ | p₀=10⁸ |
|---|---|---|---|---|---|---|
| yacht | 0 | 2,82e-01% | 2,86e-02% | 2,86e-04% | 2,86e-06% | 2,68e-08% |
| auto_mpg | 0 | 3,39e-01% | 3,46e-02% | 3,46e-04% | 3,46e-06% | 3,42e-08% |
| energy | 0 | 4,41e-01% | 4,50e-02% | 4,51e-04% | 4,51e-06% | 4,51e-08% |
| concrete | 0 | 3,05e-01% | 3,09e-02% | 3,09e-04% | 3,09e-06% | 3,18e-08% |

A diferença cai proporcionalmente a `1/p₀` em todos os casos, exatamente como a
álgebra prevê, até chegar ao piso de precisão do `float64`. (Acima de `p₀ ≈ 10¹⁰`
o cálculo de `1 − k` sofre cancelamento catastrófico e a diferença volta a subir;
por isso a demonstração é feita como convergência e não com um `p₀` "infinito".)

Este resultado é a chave para ler todo o restante: nos experimentos, a busca de
`(q, r, p₀)` **escolheu esse canto — ou a sua vizinhança imediata — nos 8
datasets**: `q = 0` em cinco deles e o menor `q` positivo da grade nos outros
três, com ganho abaixo de 0,1%.

---

## 8. Métricas

Reportadas por dataset e por modelo, sempre como média sobre as 30 seeds, com
desvio-padrão e CV%:

| Métrica | Definição |
|---|---|
| **MAE** | média de \|ŷ − y\| |
| **MSE** | média de (ŷ − y)² |
| **RMSE** | √MSE — mesma informação do MSE, na unidade do alvo |
| **R²** | 1 − SQres/SQtot; 0 = tão bom quanto prever a média |
| **acc ±10%** | fração das predições com erro ≤ 10% do valor verdadeiro |
| **acc ±0,25σ** | fração com erro ≤ 0,25 desvio-padrão do alvo |
| **CV%** | desvio/média do RMSE **entre as 30 seeds** — a métrica de estabilidade, que é a tese central do projeto |

Sobre a **acurácia**, que foi pedida explicitamente: regressão não tem
acerto/erro, então "acurácia" só existe com uma tolerância declarada. Foram
usadas duas, porque nenhuma serve sozinha. A `±10%` é relativa e intuitiva, mas
fica indefinida perto de `y = 0` (no yacht, cujo alvo começa em 0,01, ela é
severa: 13% mesmo com R² = 0,74). A `±0,25σ` é absoluta e funciona para qualquer
alvo. As duas estão em todas as tabelas.

Nos gráficos comparativos o RMSE aparece **dividido pelo σ do alvo**, o que o
torna adimensional e permite pôr os 8 datasets no mesmo eixo (1,0 = tão ruim
quanto prever sempre a média).

---

## 9. Resultados

Fonte: `results/regression/regression_report.csv` (192 linhas = 8 datasets × 24
variantes), `synthesis_regression.csv` e as figuras em
`results/figures/regression/`.

### 9.1 Melhor ReW vs melhor KalWeNN, por dataset

Cada lado é a melhor variante da sua família naquele dataset (30 seeds, teste).

| Dataset | ReW (variante) | RMSE | MAE | R² | acc±10% | acc±0,25σ | CV% | KalWeNN (variante) | RMSE | MAE | R² | acc±10% | acc±0,25σ | CV% | Δ RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diabetes | exponential_fixed | **58,16** | 47,09 | 0,412 | 0,190 | 0,244 | 4,49 | cell/conf | 58,41 | 47,16 | 0,407 | 0,181 | 0,242 | 4,67 | −0,43% |
| california | exponential_fixed | **0,673** | 0,498 | 0,655 | 0,248 | 0,365 | 2,03 | cell/conf | 0,675 | 0,495 | 0,653 | 0,250 | 0,373 | 2,03 | −0,30% |
| airfoil | harmonic_power2 | **3,271** | 2,456 | 0,774 | 0,994 | 0,449 | 4,70 | cell/mean | 3,335 | 2,554 | 0,765 | 0,994 | 0,416 | 4,92 | −1,96% |
| energy | median | **1,047** | 0,594 | 0,988 | 0,941 | 0,971 | 25,90 | cell/mean | 1,476 | 0,998 | 0,978 | 0,897 | 0,910 | **15,11** | −41,02% |
| concrete | power2 | **9,192** | 6,735 | 0,693 | 0,382 | 0,435 | 5,97 | cell/mean | 9,291 | 6,858 | 0,686 | 0,370 | 0,427 | 5,83 | −1,08% |
| abalone | power2 | **2,290** | 1,618 | 0,501 | 0,397 | 0,358 | 3,08 | cell/mean | 2,295 | 1,608 | 0,498 | 0,403 | 0,364 | 3,18 | −0,22% |
| auto_mpg | unweighted | **2,838** | 1,944 | 0,869 | 0,696 | 0,649 | 12,35 | cell/mean | 2,838 | 1,944 | 0,869 | 0,696 | 0,649 | 12,35 | −0,0015% |
| yacht | geometric | **7,226** | 2,950 | 0,753 | 0,353 | 0,837 | 15,92 | cell/mean | 7,394 | 3,556 | 0,742 | 0,135 | 0,771 | 14,76 | −2,32% |

Δ RMSE negativo = ReW melhor. A tabela completa, com MSE, desvios, tempos e
memória, está em `results/regression/synthesis_regression.csv`.

**Como ler isso.** A ReW vence nos **8 de 8**, mas é preciso entender de onde vem
a vitória, porque ela não é sobre a regra de atualização:

- O `(q, r)` escolhido pela validação foi `q = 0, r = 0,001` em 5 dos 8 datasets,
  que é precisamente o canto do §7 onde o KalWeNN **é** a ReW. Nos outros três o
  `q` escolhido é o menor positivo da grade (1e-4 ou 1e-3), praticamente o mesmo
  canto. O melhor KalWeNN é, em todos os casos, a ReW de média não ponderada
  usando outro nome.
- Logo, a diferença que aparece na tabela é quase toda **diferença de agregador**,
  não de modelo: o KalWeNN só dispõe dos 5 modos de inferência do §6.4, enquanto a
  ReW dispõe das 9 médias do §6.1. Nos datasets em que a melhor média é uma das
  exóticas (mediana no energy, geométrica no yacht, harmônica de potência no
  airfoil), a ReW abre vantagem simplesmente porque tem essa opção no cardápio.
- Onde os dois usam agregação equivalente (auto_mpg, abalone, california), a
  diferença cai para 0,02–0,22%: ruído de ordem de apresentação.

O caso extremo é o **energy**: a mediana chega a RMSE 1,047 contra 1,476 do
KalWeNN (−41%), mas com CV de 25,9% contra 15,1%. É o único caso em que aparece o
trade-off clássico do projeto — bem mais preciso na média, bem menos estável
entre execuções.

Na média dos 8 datasets, o CV% do RMSE é **9,31% para a ReW e 7,86% para o
KalWeNN** — vantagem de estabilidade do KalWeNN, mas quase inteiramente produzida
pelo energy.

→ figuras `1_rmse_normalizado`, `2_estabilidade_cv`

### 9.2 Contar os números **piora** a predição

Este é o resultado mais forte do estudo, e é sobre a ReW, não sobre o Kalman.
Média, nos 8 datasets, de quanto cada agregador fica acima do melhor daquele
dataset:

| Agregador | % acima do melhor | Vitórias |
|---|---|---|
| `median` | **+3,90%** | 1 (energy) |
| `unweighted` | +5,72% | 1 (auto_mpg) |
| `exponential_fixed` | +5,89% | 2 (diabetes, california) |
| `geometric` | +6,29% | 1 (yacht) |
| `power2` | +7,24% | 2 (abalone, concrete) |
| `harmonic` | +9,44% | 0 |
| `harmonic_power2` | +14,33% | 1 (airfoil) |
| **`simple` (a ReW canônica)** | **+15,68%** | **0** |
| `exponential` (com o defeito) | +33,36% | 0 |

**A média canônica da ReW não vence em nenhum dataset e fica em penúltimo lugar**
— só não é a pior porque a `ExponentialMean` da referência tem um defeito (§3.2).
Quanto a `simple` fica acima do melhor agregador de cada dataset: energy +52,5%,
airfoil +21,8%, california +17,4%, yacht +12,7%, concrete +10,5%, abalone +5,6%,
auto_mpg +3,2%, diabetes +1,8%.

Comparando diretamente com a média aritmética não ponderada (que é a mesma conta,
só que sem os contadores), `simple` perde em todos os 8: airfoil +19,3%,
california +15,5%, yacht +10,1%, energy +9,5%, concrete +9,4%, abalone +4,7%,
auto_mpg +3,2%, diabetes +1,4%.

**Interpretação.** Ponderar pelo contador dá mais voz às células mais acessadas,
que são justamente as **mais genéricas** — endereços que muitas amostras
diferentes compartilham, e cuja média local é próxima da média global. As células
raras são as específicas, as que carregam informação sobre aquela região do
espaço de entrada. Ponderar por `n` sistematicamente afoga o sinal específico no
genérico. O mesmo mecanismo explica por que `count` é o pior modo de inferência do
KalWeNN (§9.4). Vale notar que a própria biblioteca de referência já reflete isso
sem dizer: de seus oito agregadores, só o `SimpleMean` pondera.

**O custo do defeito na `ExponentialMean`.** Comparando `exponential` (com o "+1"
da referência) contra `exponential_fixed` (sem ele), o RMSE piora em: airfoil
+62,2%, auto_mpg +48,9%, energy +47,5%, abalone +20,6%, california +9,7%,
concrete +5,1%, yacht +3,8%, diabetes +1,1%. Corrigido, o estimador passa de
último para terceiro lugar e vence em 2 dos 8 datasets. É um bug de uma linha com
efeito de até 62%.

→ figura `5_estimadores_rew`

### 9.3 Onde manter o `p`: por célula, sempre

Quanto cada escopo fica acima do melhor RMSE do dataset:

| Dataset | `cell` | `ram` | `global` |
|---|---|---|---|
| airfoil | 0% | +12,40% | +12,40% |
| concrete | 0% | +8,74% | +8,74% |
| california | 0% | +5,92% | +5,92% |
| abalone | 0% | +5,86% | +5,86% |
| yacht | 0% | +3,86% | +3,86% |
| auto_mpg | 0% | +2,44% | +2,44% |
| diabetes | 0% | +1,59% | +1,59% |
| energy | 0% | +1,51% | +1,51% |

`cell` vence nos 8, com margem de 1,5% a 12,4%.

**E há um resultado estrutural aqui:** `ram` e `global` produzem predições
**idênticas bit a bit** — verificado numericamente (diferença máxima 0,00e+00 em
todos os 5 modos de inferência). A razão é que numa WiSARD toda amostra endereça
**todas** as RAMs exatamente uma vez, então todos os `p` por RAM seguem a mesma
recursão, com o mesmo número de atualizações, partindo do mesmo `p₀`: eles
permanecem iguais entre si e iguais ao `p` global para sempre. Manter um `p` por
RAM é, portanto, redundante por construção — não é uma descoberta empírica, é uma
consequência da topologia.

Isso reduz o experimento pedido a duas condições realmente distintas —
**incerteza por célula** vs **incerteza compartilhada** — e a conclusão é que só
a primeira serve para alguma coisa. Faz sentido: se o `p` é compartilhado, o
ganho `k` depende apenas de quantas amostras já passaram, e não de quanta
evidência aquela célula específica acumulou. Toda a adaptatividade do filtro,
que é o que o distingue de uma média móvel comum, se perde.

→ figura `3_escopo_p`

### 9.4 Kalman na inferência: perde para a média simples

Com `p` por célula (o único escopo em que `conf` e `fusion` se distinguem da
média — nos demais os pesos são todos iguais e as três fórmulas colapsam),
quanto cada modo fica acima do melhor:

| Modo de inferência | % acima do melhor (média dos 8) |
|---|---|
| `mean` — média simples | **+0,01%** |
| `conf` — peso (1−p) | +0,04% |
| `fusion` — peso 1/p (**Kalman na inferência**) | +3,62% |
| `kalman_seq` — fusão com prior | +4,11% |
| `count` — peso pelos contadores | +9,31% |

Por dataset, a fusão bayesiana custa de +1,2% (auto_mpg) a +8,0% (airfoil).

**Interpretação.** A fusão `Σ(x/p)/Σ(1/p)` é a combinação ótima *se* as
observações forem independentes e `p` for a variância real do erro de cada uma.
Nenhuma das duas hipóteses vale aqui: as RAMs veem a mesma amostra e são
fortemente correlacionadas, e `p` não mede o erro da estimativa — mede quantas
vezes aquela célula foi atualizada. Como `p` cai rapidamente com o número de
acessos, o peso `1/p` explode justamente para as células mais acessadas, e a
fusão reproduz o mesmo viés do `count`, que é o pior modo de todos (§9.2). O peso
`(1−p)`, por ser limitado, sofre muito menos com isso — o que explica ele ficar
praticamente empatado com a média simples.

→ figura `4_inferencia`

### 9.5 O ruído de processo `q` não ajuda

Menor RMSE de validação para cada valor de `q` (varrendo todos os `r`), em % do
valor obtido com `q = 0`:

- o **mínimo da curva está em `q = 0` em 5 dos 8 datasets** (abalone, concrete,
  diabetes, energy, yacht);
- nos outros três o mínimo está no menor `q` positivo da grade e a melhora é
  desprezível: airfoil e auto_mpg em `q = 0,001` (99,91%) e california em
  `q = 0,0001` (99,98%). São ganhos da terceira casa decimal;
- em `q = 1` a piora chega a +3,57% (california), +2,71% (abalone) e +2,45%
  (airfoil).

Ou seja: em nenhum dataset o ruído de processo compra mais de **0,1%** de RMSE, e
em todos ele degrada assim que sai da vizinhança de zero.

Ou seja: deixar a célula esquecer o passado degrada a estimativa. Isso é
coerente com o que o alvo é — em regressão, `x` estima a **média condicional**
do alvo naquele endereço, e a melhor estimativa de uma média estacionária é a
média de todas as observações, com peso `1/n`. Esquecer só ajudaria se a relação
entre entrada e alvo mudasse ao longo do fluxo de treino, o que não acontece com
dados i.i.d. embaralhados.

→ figuras `7_efeito_q`, `6_heatmap_qr`

### 9.6 A diferença é estatisticamente significativa — e quase sempre irrelevante

As 30 seeds são pareadas (mesmo split para os dois modelos), então o teste de
postos sinalizados de **Wilcoxon** é o adequado: não pressupõe normalidade e
respeita o pareamento.

| Dataset | ReW | KalWeNN | Δ RMSE | seeds em que a ReW vence | p |
|---|---|---|---|---|---|
| energy | 1,047 | 1,476 | −41,0% | 30/30 | 1,9e-09 |
| yacht | 7,226 | 7,394 | −2,3% | 21/30 | 1,2e-03 |
| airfoil | 3,271 | 3,335 | −2,0% | 29/30 | 5,6e-09 |
| concrete | 9,192 | 9,291 | −1,1% | 27/30 | 1,3e-07 |
| diabetes | 58,158 | 58,409 | −0,4% | 23/30 | 1,0e-03 |
| california | 0,673 | 0,675 | −0,3% | 29/30 | 2,3e-08 |
| abalone | 2,290 | 2,295 | −0,22% | 27/30 | 4,7e-06 |
| auto_mpg | 2,8379 | 2,8379 | −0,0015% | 22/30 | 2,9e-02 |

A ReW vence com **p < 0,05 nos 8 datasets**. Mas o auto_mpg mostra exatamente por
que significância não é o mesmo que relevância: a diferença é de **0,0015%** — a
quarta casa decimal — e ainda assim é detectada, porque com 30 pares uma
vantagem minúscula e consistente basta para rejeitar a hipótese nula. Lido junto
com o §9.1: o efeito é real, e é pequeno em 7 dos 8 casos.

→ `results/regression/significance.csv`

### 9.7 A ReW da referência, a ClusReW e o efeito do fallback

Rodando os modelos da própria biblioteca nos mesmos splits e com a mesma
binarização (10 seeds, `analysis/08_wisardpkg_models.py`):

| Dataset | nossa ReW | wisardpkg ReW | wisardpkg ClusReW | agregador |
|---|---|---|---|---|
| diabetes | 57,155 | 57,200 | 58,657 | exponential_fixed / unweighted\* |
| california | 0,668 | 0,672 | 0,769 | exponential_fixed / unweighted\* |
| airfoil | **3,200** | 3,279 | 3,967 | harmonic_power2 |
| energy | 1,046 | 1,441 | 1,840 | median / unweighted\* |
| concrete | **9,335** | 9,755 | 10,648 | power2 |
| abalone | **2,285** | 2,331 | 2,430 | power2 |
| auto_mpg | **2,817** | 2,959 | 3,111 | unweighted |
| yacht | **7,135** | 7,223 | 8,596 | geometric |

\* nesses três, o nosso melhor agregador não existe na referência (é o
`exponential_fixed`, ou a `Median`, que devolve NaN por causa do defeito de
indexação), então a referência roda com o melhor agregador que ela sabe calcular.

Duas leituras:

- **A ClusReW não ajudou.** Perde para a ReW pura nos 8 datasets, de +5% a +26%
  de RMSE (de +2,5% no diabetes a +21% no yacht). Cabe a ressalva de que varremos apenas 4 combinações de
  (`minScore`, `threshold`, `discriminatorLimit`); com uma busca mais larga o
  ensemble provavelmente melhora. Como baseline mínimo, porém, ele não muda
  nenhuma conclusão do estudo.
- **O fallback vale RMSE.** Nas 5 linhas em negrito os dois lados usam o mesmo
  agregador e o mesmo mapeamento, então tudo que resta é a escolha do fallback
  (§3.2, divergência 3). Só isso vale +2,4% (airfoil), +4,3% (concrete), +2,0%
  (abalone), +4,8% (auto_mpg) e +1,2% (yacht) a nosso favor. Prever a média do
  treino em vez do mínimo é uma correção de uma linha com efeito prático.

### 9.8 Épocas, `p₀` e estratificação do split

Três varreduras que fechavam lacunas do protocolo:

**Número de épocas — não muda nada.** Com `n_epochs` ∈ {1, 2, 3, 5}, a variação
máxima de RMSE em qualquer dataset foi de **1,27%** (energy), e em cinco dos oito
a diferença aparece só na quarta casa decimal. É consequência direta do §7: com
`q = 0` o ganho é `1/n`, a célula guarda a média aritmética das observações, e
repetir o mesmo conjunto de treino não muda uma média. Só o energy reage, e é o
único dataset cujo `(q, r)` ótimo não está no canto degenerado.
→ `results/regression/epochs.csv`

**`p₀` entrou na busca** (0,1 / 1 / 10), junto com `q` e `r`. O ótimo migra entre
os três valores conforme o dataset (10 no california, concrete, diabetes, abalone,
auto_mpg e yacht; 1 no airfoil; 0,1 no energy) — o que é esperado, já que com
`q = 0` o comportamento depende só da razão `r/p₀`. Ampliar a busca não mudou
nenhuma conclusão: o `q` escolhido continua sendo 0 ou o menor positivo da grade.

**Estratificar o split por faixas de quantil do alvo reduz o CV nos datasets
pequenos**, que era a suspeita:

| Dataset | CV sem | CV com | |
|---|---|---|---|
| diabetes | 4,67% | **3,57%** | −23,5% |
| yacht | 14,76% | **12,48%** | −15,5% |
| auto_mpg | 12,35% | **10,95%** | −11,4% |
| abalone | 3,18% | **2,93%** | −7,8% |
| energy | 15,55% | **14,61%** | −6,0% |
| concrete | 5,83% | **5,64%** | −3,2% |
| california | 2,03% | 2,04% | +0,7% |
| airfoil | 4,86% | 5,52% | +13,6% |

O efeito é o previsto e vale a pena onde importa: os dois datasets com maior CV
(yacht e auto_mpg, ~300 amostras) melhoram 15,5% e 11,4%. Nos datasets grandes o
CV já é baixo e a estratificação chega a piorar um pouco. Os resultados
principais deste estudo **não** usam estratificação — o pedido era 70/30 simples,
e mudá-la depois invalidaria a comparação —, mas para um próximo estudo com
datasets pequenos a recomendação é estratificar por faixas.
→ `results/regression/stratification.csv`

### 9.9 Custo e contexto

Média sobre os 8 datasets, na melhor configuração de cada família:

| | Memória | Treino | Inferência |
|---|---|---|---|
| ReW | 233 KB | 0,021 s | 0,009 s |
| KalWeNN (`p` por célula) | 310 KB | 0,028 s | 0,010 s |
| KalWeNN descartando `p` após o treino | **233 KB** | — | — |

O KalWeNN custa **33% mais memória** (24 → 32 bytes por célula) e ~33% mais tempo
de treino. Mas como `p` não é usado por `mean` nem por `count` — os dois melhores
modos de inferência — ele pode ser descartado depois do treino, e o footprint
fica **idêntico ao da ReW**. É a mesma conclusão do estudo de classificação
(`04_experiments.py memory`), agora confirmada em regressão.

Modelos de referência do scikit-learn (RMSE, 10 seeds, mesmos splits), só para
dar escala:

| Dataset | melhor WiSARD | dummy (média) | linear | kNN k=5 | árvore | random forest |
|---|---|---|---|---|---|---|
| diabetes | 58,16 | 74,89 | **53,94** | 59,81 | 81,20 | 57,93 |
| california | 0,673 | 1,143 | 1,670 | 1,153 | 0,800 | **0,553** |
| airfoil | 3,271 | 6,954 | 4,796 | 6,156 | 2,730 | **1,871** |
| energy | 1,047 | 10,077 | 3,018 | 2,376 | 0,572 | **0,503** |
| concrete | 9,192 | 16,595 | 10,655 | 9,582 | 6,958 | **5,233** |
| abalone | 2,290 | 3,223 | 2,236 | 2,277 | 3,039 | **2,207** |
| auto_mpg | **2,838** | 7,968 | 3,296 | 4,265 | 3,840 | 2,865 |
| yacht | 7,226 | 14,691 | 8,943 | 11,960 | 1,306 | **1,063** |

Contando vitórias da melhor WiSARD (ReW ou KalWeNN) contra cada referência: bate
o dummy em **8/8**, o kNN em **7/8**, a regressão linear em **6/8**, a árvore em
**4/8** e o random forest em **1/8** (só no auto_mpg, 2,838 contra 2,865). A
derrota é em geral por margem moderada, com a exceção do yacht (7,39 vs 1,06),
cuja relação entrada-alvo é quase determinística e sofre muito com a quantização
do termômetro.

Ressalva: os baselines rodaram com 10 seeds contra as 30 das WiSARDs (são
contexto, não parte da comparação central), então essas contagens são
indicativas, não um confronto pareado.

---

## 10. Conclusões

1. **A ReW é um caso limite do KalWeNN** (`q = 0`, `r/p₀ → 0`, agregação por
   contagem) — demonstrado algebricamente e verificado numericamente com
   convergência em `1/p₀` (§7). As duas famílias não são alternativas
   independentes.

2. **Em regressão, o filtro de Kalman não melhora o erro.** A busca de
   hiperparâmetros escolheu `q = 0` em 5 dos 8 datasets, e nos outros três o menor
   `q` positivo da grade, com ganho abaixo de 0,1%. A ReW vence em 8 de 8 — mas a
   vantagem vem de ter mais formas de agregar no cardápio, não da regra de
   atualização: onde os dois usam agregação equivalente, a diferença é de
   0,02–0,22%.

3. **A vantagem de estabilidade é pequena e concentrada.** CV médio 7,86% contra
   9,31%, praticamente todo vindo do energy (15,1% vs 25,9%). Não se reproduz aqui
   o ganho consistente de estabilidade encontrado em classificação.

4. **Contar os números atrapalha.** A média canônica da ReW (`SimpleMean`, a
   única que pondera pelos contadores) não vence em nenhum dataset e fica em
   penúltimo lugar entre nove agregadores, 15,7% acima do melhor em média e até
   52,5% no energy. Ponderar por `n` privilegia as células genéricas em detrimento
   das específicas.

5. **A incerteza precisa ser por célula.** `p` compartilhado custa de 1,5% a 12,4%
   de RMSE; e `p` por RAM é matematicamente idêntico a `p` global numa WiSARD, por
   construção.

6. **Kalman na inferência não compensa.** A fusão por precisão `1/p` perde 3,6%
   para a média simples, pelo mesmo motivo do item 4: `1/p` é um proxy do
   contador. O peso limitado `(1−p)` empata com a média.

7. **O custo extra do KalWeNN é descartável.** 33% mais memória durante o treino,
   footprint idêntico ao da ReW depois, porque `p` não é usado na inferência
   pelos modos que funcionam.

8. **A nossa ReW é fiel à da literatura.** Equivalência numérica com o wisardpkg
   v2.0.0a7 na precisão do `float64` para os 7 agregadores comparáveis, com
   entrada e mapeamento forçados a serem os mesmos. No caminho, três divergências
   na referência: a indexação da `Median` (que devolve NaN), o "+1" da
   `ExponentialMean` (até 62% de RMSE) e o fallback que devolve 0 em vez da média
   do treino (de 1,2% a 4,8%).

9. **A vantagem da ReW é significativa, mas quase sempre minúscula.** Wilcoxon
   pareado dá p < 0,05 nos 8 datasets; ainda assim, em 7 deles a diferença fica
   abaixo de 2,4%, e no auto_mpg é de 0,0015%. Significância com 30 pares não é
   relevância prática.

10. **Épocas não mudam nada** (≤ 1,3%), porque com `q = 0` a célula guarda uma
    média e repetir o treino não muda uma média. **Estratificar o split por
    faixas do alvo** reduz o CV em até 23% nos datasets pequenos, e é a
    recomendação para trabalhos futuros com poucas amostras.

Lendo em conjunto: o mecanismo que fez o KalWeNN valer a pena em classificação —
saturar a evidência e estabilizar a memória — não tem função equivalente em
regressão, onde a célula precisa estimar uma média estacionária e a melhor forma
de fazer isso já é a média incremental, que é o que a ReW faz. O resultado é
negativo para a hipótese central, mas produziu três achados úteis e reaproveitáveis
sobre a própria ReW (itens 4, 5 e 8).

---

## 11. Limitações e o que ficaria melhor conferir

1. **A busca do ClusReW foi mínima** — 4 combinações de
   (`minScore`, `threshold`, `discriminatorLimit`), contra as 126 do KalWeNN.
   Ele perde nos 8 datasets (§9.7), mas com busca comparável provavelmente
   melhora, e é o ensemble que a literatura recomenda. Refazer com uma grade
   larga é o próximo passo natural — o ambiente já está pronto (§3.2).
2. **california subamostrada** para 5.000 de 20.640, por custo. O teto é
   removível com `REG_MAX_SAMPLES=0`, e os resultados desse recorte não foram
   refeitos com a base inteira.
3. **A `LogisticMean` da referência ficou de fora** (§6.1): tem três parâmetros
   a ajustar, o que a torna um modelo e não um agregador.
4. **Um único regime de arquitetura por dataset.** A tupla/resolução é escolhida
   com a ReW e reaproveitada pelo KalWeNN, para isolar a regra de atualização.
   É a escolha certa para a pergunta do estudo, mas não descarta que o KalWeNN
   preferisse outra arquitetura.
5. **Datasets pequenos seguem com CV alto** (yacht 15,9%, auto_mpg 12,4%). O
   §9.8 mostra que estratificar por faixas resolve boa parte disso, mas os
   resultados principais mantêm o 70/30 simples que foi pedido.

---

## 12. Arquivos

**Código novo**

| Arquivo | O que é |
|---|---|
| [datasets/loaders_regression.py](datasets/loaders_regression.py) | os 8 datasets de regressão |
| [wisard/models_regression.py](wisard/models_regression.py) | ReW e KalWeNN de regressão + validação do §7 |
| [analysis/05_regression.py](analysis/05_regression.py) | experimentos: `arch`, `grid`, `final`, `baselines` |
| [analysis/06_regression_analysis.py](analysis/06_regression_analysis.py) | 7 figuras + tabela síntese |
| [analysis/07_wisardpkg_check.py](analysis/07_wisardpkg_check.py) | equivalência numérica contra o wisardpkg (§3.2) |
| [analysis/08_wisardpkg_models.py](analysis/08_wisardpkg_models.py) | ReW e ClusReW da referência como baseline (§9.7) |

**Saídas** (em `results/regression/` e `results/figures/regression/`)

| Arquivo | Conteúdo |
|---|---|
| `{ds}_arch.csv` | as 25 arquiteturas avaliadas na validação interna |
| `{ds}_grid.csv` | os 42 pares (q,r) × 3 escopos × 5 inferências, RMSE de validação |
| `{ds}_config.json` | arquitetura e melhores (q,r) escolhidos |
| `regression_report.csv` | comparação final: 8 datasets × 24 variantes, todas as métricas |
| `baselines.csv` | modelos de referência do scikit-learn |
| `significance.csv` | Wilcoxon pareado por dataset (§9.6) |
| `wisardpkg_models.csv` | ReW e ClusReW da biblioteca de referência (§9.7) |
| `epochs.csv` | efeito do número de épocas (§9.8) |
| `stratification.csv` | efeito da estratificação do split (§9.8) |
| `synthesis_regression.csv` / `.md` | tabela síntese do §9.1 |
| `1_rmse_normalizado` … `7_efeito_q` | figuras (SVG transparente + PNG) |
