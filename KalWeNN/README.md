# K-WiSARD

Estudo comparativo entre a **WiSARD clássica** (rede neural sem peso, baseada em
contagem) e a **KalWeNN**, uma variação em que cada célula de memória deixa de ser
uma contagem e passa a ser um estado estimado por uma atualização escalar do
filtro de Kalman, acompanhado de uma incerteza.

A pergunta central: tratar a memória como *estimativa* (e não como contagem)
melhora a acurácia e, principalmente, a estabilidade entre execuções? O estudo
roda a comparação em doze datasets, em dois regimes de arquitetura, com 30 seeds.

## Os quatro modelos

Comparamos quatro modelos, em dois regimes de (tupla, termômetro):

| Modelo | Memória | Tupla / termômetro | O que é |
|---|---|---|---|
| **WiSARD** | densa | 8 / 16 | clássica por contagem + bleaching (`bleach=0` → clássica pura) |
| **KalWeNN** | densa | 8 / 16 | clássica + filtro de Kalman por célula |
| **WiSARD (dict)** | esparsa | 20 / 20 | clássica em dicionários (viabiliza tuplas grandes) |
| **KalWeNN (dict)** | esparsa | 20 / 20 | Kalman em dicionários |

A versão esparsa guarda só os endereços vistos, o que permite tuplas grandes sem
alocar `2^address_size` células. Ela é matematicamente equivalente à densa
(mesma seed → mesmos números); a validação está no `__main__` de
`wisard/models.py`.

## Estrutura do repositório

```
wisard/
  models.py        todos os modelos (Base, WiSARD, KalWeNN e variantes esparsas)
  encoder.py       codificação thermometer
  metrics.py       utilidades da API antiga (run_experiment)

datasets/
  loaders.py       carregamento dos datasets (get_wine, get_mnist, ...)

analysis/          pipeline numerado, na ordem de execução
  01_common.py     base compartilhada: datasets, split, avaliação, paleta, caminhos
  02_search.py     grid search dos 4 modelos → relatório CSV
  03_analysis.py   todas as figuras e tabelas
  04_experiments.py  experimentos extras (fusão na inferência, MNIST oficial, memória)
  legacy/          versões antigas, mantidas só como histórico

results/           saídas geradas (CSV da busca, configs e figuras)
Kalman-WiSARD.ipynb  notebook original do projeto (referência histórica)
```

## Como rodar

Os arquivos de `analysis/` seguem a ordem do número. O `01_common` não roda
sozinho — é importado pelos demais.

```bash
# 1. busca de hiperparâmetros + comparação dos 4 modelos → results/v2/tuple_comparison_report.csv
python3 analysis/02_search.py                  # lista padrão de datasets
python3 analysis/02_search.py wine iris mnist  # datasets específicos

# 2. figuras e tabelas (lê o CSV do passo anterior) → results/figures/presentation/
python3 analysis/03_analysis.py                # todas
python3 analysis/03_analysis.py 8 11 table     # apenas algumas
python3 analysis/03_analysis.py 12 vowel       # figura 12 com outro dataset

# 3. (opcional) experimentos extras
python3 analysis/04_experiments.py fusion all
python3 analysis/04_experiments.py mnist-official
python3 analysis/04_experiments.py memory
```

Algumas variáveis de ambiente ajustam runs pesados, sem editar código:

```bash
WISARD_BACKEND=sparse python3 analysis/02_search.py mnist   # memória esparsa
WISARD_SEEDS=5 WISARD_GRID_SEEDS=2 python3 analysis/02_search.py wine
```

## Metodologia

- **Datasets (12):** wine, iris, breast cancer, digits, vehicle, segment, vowel,
  letter, pendigits, satellite, texture e MNIST. Outros loaders estão disponíveis
  em `datasets/loaders.py`.
- **Split:** 80/20 estratificado por seed. MNIST usa o split **oficial** 60k/10k.
- **Seeds:** 30 na comparação final. A aleatoriedade vem do mapeamento bit→RAM e,
  no Kalman, da ordem de apresentação dos exemplos — por isso toda métrica é média
  sobre as seeds.
- **Codificação:** thermometer (`wisard/encoder.py`) — cada atributo numérico vira
  bits ordenados (valores maiores acendem mais bits).
- **Busca de (q, r) do Kalman:** grid sobre `q ∈ {1e-4, 1e-3, 1e-2}` e
  `r ∈ {1e-2, 1e-1, 5e-1, 1, 2}`, com tupla e termômetro fixos do regime.

### A atualização do Kalman

Para cada célula ativada (classe, RAM, endereço), o modelo guarda um estado `x`
(evidência) e uma incerteza `p`, atualizados por:

```
p_pred = p + q
k      = p_pred / (p_pred + r)      # ganho de Kalman
x      = x + k * (1 - x)            # observação z = 1.0 (endereço visto)
p      = (1 - k) * p_pred
```

`q` é o ruído de processo (quanto a incerteza cresce), `r` o ruído de observação
(quanto o modelo confia em cada ativação). Com `p` alto o ganho é grande e a
célula se ajusta rápido; conforme `p` cai, cada nova ativação mexe menos — daí a
memória ficar mais estável. Na predição, o score de uma classe é a soma dos `x`
nos endereços ativados.

## Principais conclusões

- A **KalWeNN iguala ou supera** a WiSARD clássica em acurácia, com os maiores
  ganhos nos problemas mais difíceis (onde a clássica vai pior).
- O ganho mais consistente é em **estabilidade**: a KalWeNN tem coeficiente de
  variação bem menor entre execuções (memória mais previsível).
- O custo de memória e tempo é modesto. Como o `p` não é usado na inferência, ele
  pode ser descartado após o treino, igualando o footprint da clássica sem perda
  de acurácia (ver `04_experiments.py memory`).

As tabelas e figuras que sustentam isso são geradas pelo `03_analysis.py`.

## Dependências

```bash
pip install numpy pandas scikit-learn matplotlib scipy psutil
```

`scipy` é usado nos heatmaps; `psutil` só na API antiga (`wisard/metrics.py`).

## Notebook original

`Kalman-WiSARD.ipynb` é a primeira versão do projeto (apenas Wine, poucas seeds,
a WiSARD por contagem crua). Foi mantido como referência histórica — o estudo
atual está todo no pipeline de `analysis/`.
