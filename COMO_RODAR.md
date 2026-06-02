# Como rodar os experimentos K-WiSARD

## Estrutura do projeto

```
K-WiSARD/
├── wisard/          ← modelos e utilitários (não editar para rodar)
├── datasets/        ← loaders de todos os datasets
├── experiments/     ← um arquivo por dataset
└── results/         ← CSVs gerados automaticamente após cada experimento
```

---

## 1. Pré-requisitos

Instale as dependências antes de rodar qualquer experimento:

```bash
pip install numpy pandas scikit-learn psutil
```

> Os datasets do OpenML (glass, vehicle, segment, vowel, satellite, MNIST) são
> baixados automaticamente na primeira execução e ficam em cache local
> (`~/scikit_learn_data/`). Requer conexão com a internet apenas na primeira vez.

---

## 2. Rodar um dataset específico

Todos os scripts são executados a partir da **raiz do projeto** (`K-WiSARD/`).
Cada script roda **SimpleWiSARD e Kalman-WiSARD** sobre o mesmo dataset e salva
os resultados em `results/`.

### Wine
```bash
python experiments/run_wine.py
```
- 178 amostras · 13 features · 3 classes
- Parâmetros: `address_size=4`, `resolution=16`

### Iris
```bash
python experiments/run_iris.py
```
- 150 amostras · 4 features · 3 classes
- Parâmetros: `address_size=4`, `resolution=16`

### Breast Cancer
```bash
python experiments/run_breast_cancer.py
```
- 569 amostras · 30 features · 2 classes
- Parâmetros: `address_size=4`, `resolution=16`

### Digits (8×8)
```bash
python experiments/run_digits.py
```
- 1797 amostras · 64 features · 10 classes
- Parâmetros: `address_size=8`, `resolution=8`

### Glass
```bash
python experiments/run_glass.py
```
- 214 amostras · 9 features · 6 classes  *(download ~100 KB na primeira vez)*
- Parâmetros: `address_size=4`, `resolution=16`

### Vehicle
```bash
python experiments/run_vehicle.py
```
- 846 amostras · 18 features · 4 classes  *(download ~50 KB na primeira vez)*
- Parâmetros: `address_size=4`, `resolution=16`

### Segment
```bash
python experiments/run_segment.py
```
- 2310 amostras · 19 features · 7 classes  *(download ~500 KB na primeira vez)*
- Parâmetros: `address_size=4`, `resolution=16`

### Vowel
```bash
python experiments/run_vowel.py
```
- 990 amostras · 10 features · 11 classes  *(download ~100 KB na primeira vez)*
- Parâmetros: `address_size=4`, `resolution=16`

### Satellite
```bash
python experiments/run_satellite.py
```
- 6435 amostras · 36 features · 6 classes  *(download ~1 MB na primeira vez)*
- Parâmetros: `address_size=4`, `resolution=16`

### MNIST
```bash
python experiments/run_mnist.py
```
- 70 000 amostras (usa 10 000 por padrão) · 784 features · 10 classes
- *(download ~55 MB na primeira vez)*
- Parâmetros: `address_size=8`, `resolution=8`

> Para usar mais amostras do MNIST, edite `MAX_SAMPLES` em
> `experiments/run_mnist.py`. Quanto maior, mais lento o treino.

---

## 3. Rodar todos os datasets de uma vez

```bash
python experiments/run_all.py
```

Isso executa todos os 10 datasets em sequência, imprime o resumo de cada um e
salva dois arquivos consolidados:

| Arquivo | Conteúdo |
|---|---|
| `results/all_raw.csv` | uma linha por (dataset, modelo, seed) |
| `results/all_summary.csv` | médias e desvios por (dataset, modelo) |

---

## 4. Saída gerada por experimento

Cada script individual gera dois arquivos em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `results/<dataset>_raw.csv` | acurácia e métricas por seed |
| `results/<dataset>_summary.csv` | médias agregadas das 5 seeds |

### Colunas dos CSVs

| Coluna | Descrição |
|---|---|
| `model` | `SimpleWiSARD` ou `KalmanWiSARD` |
| `seed` | seed usada na divisão treino/teste |
| `accuracy` | acurácia no conjunto de teste |
| `train_wall_s` | tempo real de treino (segundos) |
| `test_wall_s` | tempo real de inferência (segundos) |
| `train_cpu_s` | tempo de CPU no treino (segundos) |
| `test_cpu_s` | tempo de CPU na inferência (segundos) |
| `train_peak_mb` | pico de memória RSS durante o treino (MB) |
| `test_peak_mb` | pico de memória RSS durante a inferência (MB) |
| `train_energy_j` | energia consumida no treino em joules (somente Linux com Intel RAPL) |
| `test_energy_j` | energia consumida na inferência em joules (somente Linux com Intel RAPL) |
| `rapl_available` | `True` se a leitura de energia foi possível |

> Em máquinas sem suporte a Intel RAPL, as colunas de energia ficam como `NaN`.
> O restante das métricas funciona normalmente em qualquer sistema.

---

## 5. Parâmetros dos modelos

### SimpleWiSARD

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `address_size` | `4` | número de bits por endereço de RAM |
| `resolution` | `16` | bits por feature no thermometer encoding |
| `seed` | `42` | seed do embaralhamento do mapeamento |
| `use_negative_update` | `False` | se `True`, decrementa a memória das classes erradas |

### KalmanWiSARD

Herda todos os parâmetros do SimpleWiSARD, mais:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `q` | `1e-3` | ruído de processo — controla a taxa de esquecimento |
| `r` | `5e-2` | ruído de observação — controla a confiança em novas amostras |
| `p0` | `1.0` | incerteza inicial de cada endereço |
| `n_epochs` | `1` | quantas vezes o conjunto de treino é percorrido |

> Valores maiores de `q` fazem o modelo esquecer mais rápido (mais plástico).
> Valores maiores de `r` fazem o modelo confiar menos em cada nova amostra
> (mais conservador).

---

## 6. Alterar parâmetros sem modificar o código

Edite os dicionários no topo do arquivo do experimento desejado. Exemplo para
testar a Kalman-WiSARD com mais épocas no Wine:

```python
# experiments/run_wine.py
KALMAN_PARAMS = dict(address_size=4, resolution=16, q=1e-4, r=1e-2, p0=1.0, n_epochs=3)
```

Depois rode normalmente:

```bash
python experiments/run_wine.py
```
