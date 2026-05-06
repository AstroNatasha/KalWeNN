# K-WiSARD

Este repositorio contem um notebook experimental, `Kalman-WiSARD.ipynb`, que compara duas formas de classificacao inspiradas na arquitetura WiSARD:

1. **WiSARD simples**, usada como baseline, em que cada endereco ativado nas memorias RAM recebe uma contagem.
2. **Kalman-WiSARD**, uma variacao em que cada celula de memoria deixa de ser apenas uma contagem e passa a ser um estado estimado por uma atualizacao escalar do filtro de Kalman.

O objetivo do programa e verificar se uma memoria WiSARD atualizada com uma regra probabilistica suave, baseada em incerteza, consegue melhorar a acuracia e a estabilidade em relacao a uma WiSARD por contagem no dataset Wine do `scikit-learn`.

## Arquivo principal

- `Kalman-WiSARD.ipynb`: notebook com instalacao de dependencias, implementacao dos modelos, execucao dos experimentos, tabelas de resultados e graficos.

## Dependencias

O notebook usa:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `psutil`
- `notebook`
- `pyarrow`

As primeiras celulas instalam parte dessas dependencias:

```python
!pip install numpy pandas scikit-learn notebook
!pip install pyarrow
!pip install matplotlib
```

A ultima versao do experimento tambem importa `psutil` para medir CPU e memoria. Se o ambiente nao tiver `psutil`, instale com:

```bash
pip install psutil
```

## Ideia conceitual

WiSARD e uma arquitetura de classificacao baseada em memorias RAM. Em vez de treinar pesos por gradiente, o metodo:

1. transforma atributos numericos em bits;
2. divide esses bits em grupos;
3. interpreta cada grupo como um endereco de memoria;
4. registra quais enderecos foram ativados por exemplos de cada classe;
5. classifica uma nova amostra pela classe cujas memorias reconhecem mais enderecos ativados.

No notebook, cada classe possui seu proprio conjunto de RAMs. Durante a predicao, uma amostra ativa um endereco em cada RAM. O modelo soma as evidencias armazenadas nesses enderecos para cada classe e retorna a classe com maior score.

A diferenca cientifica entre os dois modelos esta na forma como a evidencia e armazenada:

- na WiSARD simples, a evidencia e uma contagem direta;
- na Kalman-WiSARD, a evidencia e uma estimativa continua atualizada por filtro de Kalman, acompanhada de uma incerteza.

Isso faz com que a Kalman-WiSARD trate cada ativacao como uma observacao ruidosa sobre a relevancia de uma celula de memoria, em vez de simplesmente somar ocorrencias.

## Dataset usado

O experimento usa o dataset Wine carregado por:

```python
from sklearn.datasets import load_wine

data = load_wine()
X = data.data
y = data.target
```

Esse dataset contem amostras de vinhos descritas por atributos fisico-quimicos e rotuladas em tres classes. O notebook usa todos os atributos numericos disponiveis em `X` e os rotulos inteiros em `y`.

Para cada seed, os dados sao divididos com:

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=seed,
    stratify=y
)
```

Assim:

- 70% dos dados sao usados para treino;
- 30% sao usados para teste;
- `stratify=y` preserva a proporcao das classes em treino e teste;
- `random_state=seed` torna cada divisao reprodutivel.

As seeds usadas no experimento final sao:

```python
seeds = (0, 1, 2, 3, 4)
```

## Codificacao thermometer

Como WiSARD trabalha com bits, cada atributo numerico real precisa ser convertido para uma representacao binaria. O notebook implementa isso na classe `ThermometerEncoder`.

Para cada atributo, o encoder guarda:

- minimo observado no conjunto de treino;
- maximo observado no conjunto de treino;
- resolucao, isto e, quantidade de bits usada para representar aquele atributo.

O valor e limitado ao intervalo conhecido:

```python
value = np.clip(value, self.minimum, self.maximum)
```

Depois e normalizado:

```python
normalized = (value - minimum) / ((maximum - minimum) + 1e-9)
```

Em seguida, o numero de bits ligados e calculado por:

```python
k = int(normalized * resolution)
```

O vetor final tem `resolution` posicoes. Os `k` primeiros bits recebem 1 e os demais recebem 0.

Exemplo conceitual com `resolution = 8`:

- valor baixo: `[1, 0, 0, 0, 0, 0, 0, 0]`
- valor medio: `[1, 1, 1, 1, 0, 0, 0, 0]`
- valor alto: `[1, 1, 1, 1, 1, 1, 1, 0]`

Esse tipo de codificacao preserva ordem: valores maiores tendem a ativar mais bits.

## Estrutura WiSARD comum aos dois modelos

A classe `BaseWiSARD` implementa a parte compartilhada entre WiSARD simples e Kalman-WiSARD.

### Parametros estruturais

- `address_size`: quantidade de bits usada para formar o endereco de cada RAM.
- `resolution`: quantidade de bits usada para codificar cada atributo numerico.
- `seed`: semente aleatoria usada para embaralhar o mapeamento dos bits.

No experimento principal:

```python
address_size = 4
resolution = 16
```

Como o dataset Wine tem 13 atributos, a representacao binaria de cada amostra tem:

```text
13 atributos * 16 bits por atributo = 208 bits
```

Com `address_size = 4`, cada RAM recebe 4 bits, entao o modelo usa:

```text
ceil(208 / 4) = 52 RAMs
```

Cada RAM pode acessar:

```text
2^4 = 16 enderecos possiveis
```

### Mapeamento de bits para RAMs

O metodo `_build_mapping` cria uma permutacao aleatoria dos indices dos bits e os reorganiza em grupos de tamanho `address_size`.

Cada grupo define quais bits da amostra codificada serao usados por uma RAM. Se o numero total de bits nao for multiplo de `address_size`, o codigo adiciona padding por repeticao aleatoria de indices existentes.

Esse mapeamento e fixado por seed, por isso os resultados podem mudar entre seeds, mas sao reprodutiveis para uma mesma seed.

### Conversao de bits em endereco

O metodo `_bits_to_address` interpreta um grupo de bits como um numero binario. Por exemplo:

```text
bits = [1, 0, 1, 1]
endereco = 11
```

Na predicao, cada amostra ativa exatamente um endereco por RAM.

## WiSARD simples

A classe `SimpleWiSARD` herda de `BaseWiSARD`.

Durante o treino:

1. converte `X` para `float` e `y` para array;
2. identifica as classes com `np.unique(y)`;
3. ajusta os encoders thermometer usando os dados de treino;
4. codifica todas as amostras em bits;
5. constroi o mapeamento aleatorio bit-RAM;
6. cria uma memoria separada para cada classe;
7. para cada amostra, calcula os enderecos ativados;
8. incrementa a memoria da classe correta nos enderecos ativados.

A memoria e uma estrutura esparsa:

```python
{
    classe: [defaultdict(float), defaultdict(float), ...]
}
```

Ou seja, para cada classe existe uma lista de RAMs, e cada RAM e um dicionario que associa:

```text
endereco -> contagem
```

A atualizacao positiva e:

```python
self.memory_[y_true][ram_idx][address] += 1.0
```

Tambem existe a opcao `use_negative_update`. Quando ativada, a mesma amostra reduz a memoria das outras classes:

```python
self.memory_[other_class][ram_idx][address] -= 1.0
```

No experimento principal, essa opcao fica desligada:

```python
use_negative_update = False
```

Durante a predicao:

1. a amostra e codificada;
2. os enderecos ativados sao obtidos;
3. para cada classe, o modelo soma as contagens nesses enderecos;
4. a classe com maior soma e retornada.

Formalmente, para uma classe `c`, o score e:

```text
score(c) = soma_i memoria[c][RAM_i][endereco_i]
```

## Kalman-WiSARD

A classe `KalmanWiSARD` tambem herda de `BaseWiSARD`, mas substitui a contagem simples por uma atualizacao escalar inspirada no filtro de Kalman.

Para cada classe, RAM e endereco, o modelo armazena dois valores:

- `x`: estado estimado, interpretado como evidencia daquele endereco para aquela classe;
- `p`: incerteza associada ao estado.

As memorias sao:

```python
self.memory_x_  # estados estimados
self.memory_p_  # incertezas
```

### Parametros Kalman

No experimento principal, os parametros sao:

```python
q = 1e-3
r = 5e-2
p0 = 1.0
n_epochs = 1
```

Interpretacao:

- `q`: ruido de processo. Controla quanto a incerteza cresce antes de cada nova observacao.
- `r`: ruido de observacao. Controla o quanto o modelo confia na observacao recebida.
- `p0`: incerteza inicial de cada celula de memoria.
- `n_epochs`: numero de passagens pelo conjunto de treino.

### Atualizacao escalar

Para cada celula ativada, o codigo executa:

```python
x_pred = current_x
p_pred = current_p + q
k_gain = p_pred / (p_pred + r)
updated_x = x_pred + k_gain * (z - x_pred)
updated_p = (1.0 - k_gain) * p_pred
```

Onde:

- `current_x` e a evidencia atual;
- `current_p` e a incerteza atual;
- `z` e a observacao;
- `k_gain` e o ganho de Kalman.

No treino positivo, a observacao e:

```python
z = 1.0
```

Isso significa: "este endereco foi observado em uma amostra desta classe".

Se `use_negative_update=True`, as outras classes receberiam:

```python
z = 0.0
```

No experimento principal, a atualizacao negativa tambem fica desligada na Kalman-WiSARD.

### Interpretacao cientifica

A WiSARD simples trata toda nova ativacao como evidencia inteira e permanente. Ja a Kalman-WiSARD trata cada ativacao como uma observacao ruidosa sobre um estado latente.

O ganho de Kalman controla o tamanho da correcao:

- se a incerteza `p` esta alta, o ganho tende a ser maior, e a observacao muda mais o estado;
- se a incerteza `p` esta baixa, o ganho tende a ser menor, e a memoria fica mais estavel;
- se `r` aumenta, o modelo confia menos na observacao;
- se `q` aumenta, a incerteza cresce mais entre atualizacoes, deixando o modelo mais adaptavel.

Assim, a atualizacao evita que a memoria seja apenas uma soma bruta de frequencias. Ela produz uma evidencia continua, suavizada e dependente da confianca acumulada.

Durante a predicao, o score da classe e:

```text
score(c) = soma_i memory_x[c][RAM_i][endereco_i]
```

A classe escolhida e aquela com maior soma dos estados `x`.

## Avaliacao experimental

O notebook avalia os dois modelos pela funcao `evaluate_model_full`.

Para cada seed:

1. divide o dataset em treino e teste com estratificacao;
2. instancia o modelo com a seed atual;
3. mede o treino;
4. mede a inferencia;
5. calcula a acuracia no conjunto de teste.

As metricas coletadas sao:

- acuracia;
- media e desvio-padrao da acuracia;
- tempo real de treino, em segundos;
- tempo real de inferencia, em segundos;
- tempo de CPU de treino, em segundos;
- tempo de CPU de inferencia, em segundos;
- pico de memoria RSS durante treino;
- pico de memoria RSS durante inferencia;
- energia via Intel RAPL, quando disponivel no Linux.

### Tempo real e tempo de CPU

O tempo real e medido com:

```python
time.time()
```

Ele representa o tempo de relogio entre o inicio e o fim de uma operacao.

O tempo de CPU e medido com:

```python
process.cpu_times()
```

Ele soma tempo de usuario e tempo de sistema consumidos pelo processo:

```python
cpu_time = (user_final - user_inicial) + (system_final - system_inicial)
```

### Memoria

O notebook define a classe `PeakMemoryMonitor`, que cria uma thread para amostrar periodicamente o RSS do processo:

```python
psutil.Process(os.getpid()).memory_info().rss
```

O maior valor observado durante treino ou inferencia e convertido para MB.

### Energia

O notebook tenta medir energia por Intel RAPL procurando arquivos em:

```text
/sys/class/powercap/intel-rapl*/energy_uj
/sys/class/powercap/intel-rapl:*/*/energy_uj
/sys/class/powercap/intel-rapl:*/energy_uj
```

Quando disponivel, a energia e calculada pela diferenca entre leituras acumuladas em microjoules e convertida para joules:

```text
energia_j = delta_uj / 1e6
```

O codigo tambem tenta tratar wrap-around do contador usando `max_energy_range_uj`.

No ambiente em que o notebook foi executado, RAPL nao estava disponivel. Por isso, os campos de energia ficaram como `NaN`.

## Parametros do experimento principal

Baseline:

```python
baseline_params = {
    "address_size": 4,
    "resolution": 16,
    "use_negative_update": False
}
```

Kalman-WiSARD:

```python
best_kalman_params = {
    "address_size": 4,
    "resolution": 16,
    "q": 1e-3,
    "r": 5e-2,
    "p0": 1.0,
    "n_epochs": 1,
    "use_negative_update": False
}
```

Seeds:

```python
seeds = (0, 1, 2, 3, 4)
```

## Resultados obtidos

Na execucao registrada no notebook, a comparacao final foi:

| Modelo | Acuracia media | Desvio-padrao | Treino real medio (s) | Inferencia real media (s) | CPU treino medio (s) | CPU inferencia medio (s) | Pico RSS treino (MB) | Pico RSS inferencia (MB) | RAPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kalman-WiSARD | 0.974074 | 0.009072 | 0.018393 | 0.009062 | 0.020 | 0.008 | 275.925781 | 275.925781 | indisponivel |
| WiSARD simples | 0.948148 | 0.041243 | 0.017203 | 0.008704 | 0.014 | 0.008 | 275.925781 | 275.925781 | indisponivel |

O ganho medio de acuracia da Kalman-WiSARD sobre a WiSARD simples foi:

```text
0.974074 - 0.948148 = 0.025926
```

Ou seja, aproximadamente **2,59 pontos percentuais**.

As acuracias por seed foram:

| Seed | WiSARD simples | Kalman-WiSARD |
|---:|---:|---:|
| 0 | 0.981481 | 0.981481 |
| 1 | 0.907407 | 0.962963 |
| 2 | 0.981481 | 0.981481 |
| 3 | 0.981481 | 0.962963 |
| 4 | 0.888889 | 0.981481 |

Observacao importante: esses tempos sao muito pequenos e podem variar bastante conforme maquina, carga do sistema, versao das bibliotecas e ambiente de execucao. A parte mais robusta da comparacao e a repeticao por seeds com a mesma divisao estratificada e os mesmos parametros.

## Graficos gerados

O notebook gera graficos para inspecionar visualmente os resultados:

- barras com media e desvio-padrao da acuracia;
- curva de acuracia por seed;
- boxplot das acuracias;
- barra do ganho medio de acuracia;
- graficos de custo com acuracia, tempo, CPU e memoria;
- heatmap de sensibilidade para `address_size` e `resolution`;
- heatmap de sensibilidade para os parametros Kalman `q` e `r`.

Esses graficos ajudam a validar nao apenas qual modelo teve maior media, mas tambem se o resultado foi estavel entre seeds e qual foi o custo computacional associado.

## Experimentos de sensibilidade

O notebook inclui dois estudos adicionais.

### `address_size` e `resolution`

O primeiro heatmap recomputa a Kalman-WiSARD para:

```python
address_sizes = [3, 4, 5]
resolutions = [8, 16]
q = 1e-3
r = 5e-2
seeds = [0, 1, 2]
```

Esse teste verifica como a arquitetura muda quando:

- `resolution` aumenta ou reduz a granularidade da codificacao dos atributos;
- `address_size` aumenta ou reduz o tamanho dos enderecos e, portanto, o numero de enderecos possiveis por RAM.

### `q` e `r`

O segundo heatmap fixa:

```python
address_size = 4
resolution = 16
```

E testa:

```python
q_values = [1e-4, 1e-3, 1e-2]
r_values = [1e-1, 5e-2, 1e-2]
seeds = [0, 1, 2]
```

Esse teste avalia a sensibilidade da atualizacao de Kalman:

- `q` maior torna o modelo mais adaptavel, pois aumenta a incerteza prevista;
- `r` maior torna o modelo mais conservador diante da observacao;
- combinacoes diferentes podem alterar acuracia e estabilidade.

## Como validar o que foi feito

Para validar o experimento, siga estes passos.

1. Abra `Kalman-WiSARD.ipynb`.
2. Execute as celulas de instalacao de dependencias.
3. Se necessario, instale `psutil`.
4. Execute a celula final marcada como experimento completo com metricas de custo.
5. Confirme que os parametros principais sao:

```python
address_size = 4
resolution = 16
q = 1e-3
r = 5e-2
p0 = 1.0
n_epochs = 1
use_negative_update = False
seeds = (0, 1, 2, 3, 4)
```

6. Confirme que `train_test_split` usa `test_size=0.30`, `stratify=y` e `random_state=seed`.
7. Confirme que a WiSARD simples incrementa `+1.0` nos enderecos ativados da classe correta.
8. Confirme que a Kalman-WiSARD usa a atualizacao:

```python
p_pred = current_p + q
k_gain = p_pred / (p_pred + r)
updated_x = current_x + k_gain * (z - current_x)
updated_p = (1.0 - k_gain) * p_pred
```

9. Confirme que a predicao dos dois modelos escolhe a classe com maior soma de evidencias nos enderecos ativados.
10. Compare as listas `all_accuracies` dos dois modelos com a tabela de resultados.

Uma validacao manual minima pode ser feita recalculando as medias:

```python
baseline_accs = [
    0.9814814814814815,
    0.9074074074074074,
    0.9814814814814815,
    0.9814814814814815,
    0.8888888888888888,
]

kalman_accs = [
    0.9814814814814815,
    0.9629629629629629,
    0.9814814814814815,
    0.9629629629629629,
    0.9814814814814815,
]

print(np.mean(baseline_accs), np.std(baseline_accs))
print(np.mean(kalman_accs), np.std(kalman_accs))
```

Os valores esperados sao:

```text
WiSARD simples: media 0.948148, desvio-padrao 0.041243
Kalman-WiSARD: media 0.974074, desvio-padrao 0.009072
```

## Limitacoes e cuidados cientificos

Este projeto e um experimento controlado, nao uma prova definitiva de superioridade geral da Kalman-WiSARD.

Principais limitacoes:

- o teste usa apenas o dataset Wine;
- o conjunto e pequeno;
- os parametros da Kalman-WiSARD foram escolhidos experimentalmente;
- nao ha validacao cruzada aninhada;
- a diferenca de acuracia deve ser reavaliada em outros datasets;
- os tempos medidos sao muito curtos e sujeitos a ruido;
- a medicao de energia depende de suporte Intel RAPL no sistema operacional;
- a memoria RSS medida inclui o processo Python inteiro, nao apenas as estruturas internas do modelo.

Mesmo assim, o experimento e validavel porque fixa seeds, registra parametros, imprime resultados por seed e compara os dois modelos sob a mesma divisao treino/teste.

## Conclusao

No experimento registrado, a Kalman-WiSARD obteve maior acuracia media e menor desvio-padrao do que a WiSARD simples:

```text
Kalman-WiSARD: 0.974074 +- 0.009072
WiSARD simples: 0.948148 +- 0.041243
```

A interpretacao principal e que a atualizacao tipo Kalman produziu memorias mais estaveis para este problema, suavizando a evidencia armazenada em cada endereco. O custo computacional aumentou pouco neste experimento, mas essa conclusao deve ser reavaliada em bases maiores.
