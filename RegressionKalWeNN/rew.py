"""RegressionKalWeNN: Regression WiSARD e a variante com filtro de Kalman.

Arquivo único. Antes eram seis arquivos numa pasta `rew/`; foram unidos aqui
para o projeto ficar mais fácil de ler e de mover, sem quebrar os imports que já
existem nos notebooks (ver a seção de compatibilidade no fim do arquivo).

ORGANIZAÇÃO, na ordem em que as coisas acontecem num treino:

    1. CODIFICAÇÃO   ThermometerEncoder, build_encoders, apply_encoders
                     Converte atributos numéricos em bits monotônicos.

    2. AGREGAÇÃO     Mean e as nove subclasses
                     Como as N estimativas locais viram uma predição só.
                     Port fiel de meanfunctions.cc do wisardpkg (branch develop),
                     exceto a Median, que corrige um defeito da referência.

    3. MODELO BASE   RegressionWiSARD
                     Célula = [contador, soma(y)]. Predição = Σsoma / Σcontador.
                     Validado contra a implementação oficial: onze caminhos com
                     diferença exatamente zero (ver test_vs_wisardpkg.py).

    4. KALMAN        KalmanReW e KalmanReW_apenasY
                     A primeira é o modelo; a segunda existe só para demonstrar
                     que filtrar y sozinho não muda nada.

    5. MÉTRICAS      mae, rmse, r2, within_tolerance, all_metrics

    6. DADOS         load_house_prices, impute_train_median, split

REFERÊNCIA OFICIAL: a Regression WiSARD não está no branch `master` do wisardpkg,
que é o que o PyPI distribui. Só existe no `develop`. Ver requirements.txt.
"""

import math
import sys as _sys

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# 1. CODIFICAÇÃO DA ENTRADA
#
# Cópia idêntica do encoder do projeto de classificação, de propósito: assim
# qualquer diferença de resultado entre os dois projetos não pode ser atribuída
# ao encoder. `build_encoders` deve receber SOMENTE o split de treino.
# ════════════════════════════════════════════════════════════════════════════

class ThermometerEncoder:
    def __init__(self, minimum: float, maximum: float, resolution: int):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution
        self._range = (maximum - minimum) + 1e-9

    def encode(self, column: np.ndarray) -> np.ndarray:
        col = np.clip(column, self.minimum, self.maximum)
        k = np.clip(
            ((col - self.minimum) / self._range * self.resolution).astype(int),
            0,
            self.resolution,
        )
        return (np.arange(self.resolution) < k[:, np.newaxis]).astype(np.uint8)


def build_encoders(X: np.ndarray, resolution: int) -> list:
    mins, maxs = X.min(axis=0), X.max(axis=0)
    return [ThermometerEncoder(mins[j], maxs[j], resolution) for j in range(X.shape[1])]


def apply_encoders(X: np.ndarray, encoders: list) -> np.ndarray:
    return np.concatenate([enc.encode(X[:, j]) for j, enc in enumerate(encoders)], axis=1)

# ════════════════════════════════════════════════════════════════════════════
# 2. FUNÇÕES DE AGREGAÇÃO
#
# Cada classe recebe `output_rams`, uma lista de pares [contador, soma_y], um
# por RAM node, e devolve um número. Só a SimpleMean pondera pela evidência
# acumulada; as outras tratam cada RAM node como voto igual.
# ════════════════════════════════════════════════════════════════════════════

class Mean:
    def calculate(self, output_rams):
        raise NotImplementedError


class SimpleMean(Mean):
    """Σy / Σc. A única que aproxima o estimador de Nadaraya-Watson.

    Note que ela soma TODAS as RAM nodes, inclusive as que voltaram (0, 0):
    elas simplesmente não contribuem para nenhuma das duas somas.
    """

    def calculate(self, output_rams):
        sum_counters = sum(c for c, _ in output_rams)
        sum_y = sum(y for _, y in output_rams)
        if sum_counters > 0:
            return sum_y / sum_counters
        return 0.0


class PowerMean(Mean):
    def __init__(self, power: int):
        self.power = power

    def calculate(self, output_rams):
        sum_y, counter = 0.0, 0
        for c, y in output_rams:
            if c != 0:
                sum_y += (y / c) ** self.power
                counter += 1
        if counter > 0:
            return (sum_y / counter) ** (1.0 / self.power)
        return 0.0


class Median(Mean):
    """Valor central das razões `sum_y / counter`.

    DIVERGE da referência de propósito. O `Median::calculate` do wisardpkg tem
    dois defeitos: divide `outputRams[i][1]/outputRams[i][0]` sem checar counter
    zero (o que produz inf/nan para address nunca visto, enquanto todas as
    outras médias protegem com `if(outputRams[i][0]!=0)`), e usa índices
    deslocados em um na hora de pegar o valor central, o que no caso par ainda
    lê uma posição além do fim do vetor. Aqui está a mediana correta, sobre as
    RAM nodes que de fato têm evidência.
    """

    def calculate(self, output_rams):
        means = sorted(y / c for c, y in output_rams if c != 0)
        n = len(means)
        if n == 0:
            return 0.0
        if n % 2 == 0:
            return (means[n // 2 - 1] + means[n // 2]) / 2.0
        return means[(n - 1) // 2]


class HarmonicMean(Mean):
    def calculate(self, output_rams):
        hmean, counter = 0.0, 0
        for c, y in output_rams:
            if c != 0:
                amean = y / c
                if amean != 0:
                    hmean += 1.0 / amean
                    counter += 1
        if counter > 0:
            return counter / hmean
        return 0.0


class HarmonicPowerMean(Mean):
    def __init__(self, power: int):
        self.power = power

    def calculate(self, output_rams):
        hmean, counter = 0.0, 0
        for c, y in output_rams:
            if c != 0:
                amean = y / c
                if amean != 0:
                    hmean += 1.0 / (amean ** self.power)
                    counter += 1
        if counter > 0:
            return (counter / hmean) ** (1.0 / self.power)
        return 0.0


class GeometricMean(Mean):
    def calculate(self, output_rams):
        gmean, counter = 1.0, 0
        for c, y in output_rams:
            if c != 0:
                amean = y / c
                if amean != 0:
                    gmean *= amean
                    counter += 1
        if counter > 0:
            return gmean ** (1.0 / counter)
        return 0.0


class ExponentialMean(Mean):
    """Port fiel, inclusive do `emean` inicializado em 1 e não em 0.

    Isso soma um termo constante que não vem de RAM node nenhuma. Está assim na
    referência; mantido para não divergir.
    """

    def calculate(self, output_rams):
        emean, counter = 1.0, 0
        for c, y in output_rams:
            if c != 0:
                emean += math.exp(y / c)
                counter += 1
        if counter > 0:
            return math.log(emean / counter)
        return 0.0


class LogisticMean(Mean):
    """Regressão logística reaproveitando a arquitetura da ReW (Seção 5.3.6).

    A função logística é aplicada sobre estatísticas já agregadas pela memória,
    não sobre as features originais.
    """

    def __init__(self, beta1: float, beta2: float, gamma: float):
        self.beta1, self.beta2, self.gamma = beta1, beta2, gamma

    def calculate(self, output_rams):
        sum_y, counter = 0.0, 0
        for c, y in output_rams:
            if c != 0:
                sum_y += y / c
                counter += 1
        if counter > 0:
            z = (sum_y / counter) * self.beta1 + (counter / len(output_rams)) * self.beta2 + self.gamma
            return 1.0 / (1.0 + math.exp(-z))
        return 0.0


class UnweightedMean(Mean):
    """Média simples das estimativas locais, sem ponderar por counter.

    É o limite da SaturatedMean quando a saturação é total. Serve como um dos
    extremos da família, para saber se a saturação parcial ganha de ambos os
    lados ou se é melhor ir direto ao peso uniforme.
    """

    def calculate(self, output_rams):
        vals = [y / c for c, y in output_rams if c != 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

# ════════════════════════════════════════════════════════════════════════════
# 3. O MODELO BASE
#
# Port fiel de src/models/regressionwisard/ do wisardpkg, branch develop.
# ════════════════════════════════════════════════════════════════════════════

class RegressionWiSARD:
    def __init__(
        self,
        address_size: int = 3,
        resolution: int = 16,
        seed: int = 42,
        mean=None,
        min_zero: int = 0,
        min_one: int = 0,
        steps: int = 0,
        complete_addressing: bool = True,
        ordered_mapping: bool = False,
        mapping=None,
    ):
        """Padrões iguais aos do construtor da referência.

        address_size        : bits por RAM node (`n`). Padrão 3, como no wisardpkg.
                              É também o smoothing parameter do kernel implícito.
        resolution          : thermometer size (não existe no wisardpkg, que
                              recebe a entrada já binarizada).
        mean                : instância de `means.Mean`. Padrão SimpleMean.
        min_zero / min_one  : mínimo de zeros / uns que o address precisa ter
                              para a RAM node contribuir na predição.
        steps               : passadas de correção de resíduo após o treino.
                              Padrão 0, ou seja, desligado.
        complete_addressing : se o retina size não divide por `n`, completa com
                              índices sorteados em vez de descartar bits.
        ordered_mapping     : se True, não embaralha o mapping.
        mapping             : mapping explícito (lista de índices de bit).
        """
        if address_size < 2:
            raise ValueError("address size não pode ser menor que 2")

        self.address_size = address_size
        self.resolution = resolution
        self.seed = seed
        self.mean = mean if mean is not None else SimpleMean()
        self.min_zero = min_zero
        self.min_one = min_one
        self.steps = steps
        self.complete_addressing = complete_addressing
        self.ordered_mapping = ordered_mapping
        self._given_mapping = None if mapping is None else list(mapping)

        self._rng = np.random.default_rng(seed)
        self._encoders = None
        self._mapping = None
        self.n_rams_ = None
        self.retina_size_ = None
        self.n_trainings_ = 0

    # ── etapas 1 a 6: hiperparâmetros, mapping e memória vazia ────────────────

    def _set_ram_shuffle(self, entry_size: int):
        """Port de `setRAMShuffle`."""
        self.retina_size_ = entry_size

        if self._given_mapping is None:
            n_rams = entry_size // self.address_size
            remain = entry_size % self.address_size
            indexes_size = entry_size
            if self.complete_addressing and remain > 0:
                n_rams += 1
                indexes_size += self.address_size - remain

            mapping = list(range(entry_size))
            # completa com índices sorteados COM reposição, como na referência
            mapping += [
                int(self._rng.integers(0, entry_size))
                for _ in range(indexes_size - entry_size)
            ]
            if not self.ordered_mapping:
                self._rng.shuffle(mapping)
        else:
            mapping = list(self._given_mapping)
            n_rams = len(mapping) // self.address_size

        self._mapping = np.array(mapping, dtype=int).reshape(n_rams, self.address_size)
        self.n_rams_ = n_rams
        # memória esparsa: {address: [counter, sum_y, fit]}, uma por RAM node.
        # A referência usa std::map, ou seja, só aloca address efetivamente visto.
        # Denso seria inviável: com address_size=24 seriam 2^24 posições por RAM.
        self._memory = [dict() for _ in range(n_rams)]

    def _addresses(self, Xb: np.ndarray) -> np.ndarray:
        """Etapas 8 e 9. Port de `getIndex`: peso 1 no primeiro bit da tupla,
        dobrando a cada bit seguinte (LSB-first), como no wisardpkg."""
        powers = 1 << np.arange(self.address_size)          # [1, 2, 4, ...]
        bits = Xb[:, self._mapping]                         # (n, n_rams, addr)
        return (bits * powers).sum(axis=2).astype(int)

    def _ones_count(self, Xb: np.ndarray) -> np.ndarray:
        """Quantidade de bits 1 em cada address, para o filtro min_zero/min_one."""
        return Xb[:, self._mapping].sum(axis=2).astype(int)

    # ── etapas 7 a 11: treino ────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionWiSARD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._encoders = build_encoders(X, self.resolution)   # etapa 1
        Xb = apply_encoders(X, self._encoders)                # etapa 7
        self._set_ram_shuffle(Xb.shape[1])                    # etapas 2 a 6

        addresses = self._addresses(Xb)                       # etapas 8 e 9
        for i in range(len(Xb)):                              # etapa 10
            self.n_trainings_ += 1
            for r in range(self.n_rams_):
                a = int(addresses[i, r])
                cell = self._memory[r].get(a)
                if cell is None:
                    self._memory[r][a] = [1.0, y[i], 0.0]
                else:
                    cell[0] += 1.0
                    cell[1] += y[i]
        # a ordem dos exemplos não importa: somar é comutativo (etapa 11)

        for _ in range(self.steps):
            self._fit_step(Xb, addresses, y)
        return self

    def _fit_step(self, Xb: np.ndarray, addresses: np.ndarray, y: np.ndarray):
        """Port de `calculateFit` + `applyFit`.

        Acumula o resíduo `y - ŷ` num terceiro campo da célula e, ao fim da
        passada, soma o resíduo MÉDIO ao `sum(y)`. É a única correção iterativa
        que existe na ReW, e vem desligada por padrão (`steps = 0`).
        """
        y_pred = self._predict_from(Xb)
        for i in range(len(Xb)):
            residuo = y[i] - y_pred[i]
            for r in range(self.n_rams_):
                cell = self._memory[r].get(int(addresses[i, r]))
                if cell is not None:
                    cell[2] += residuo
        for r in range(self.n_rams_):
            for cell in self._memory[r].values():
                cell[1] += cell[2] / cell[0]
                cell[2] = 0.0

    # ── etapas 12 a 15: inferência ───────────────────────────────────────────

    def _read(self, Xb: np.ndarray):
        """Etapas 12 e 13. Port de `getVote`, incluindo min_zero/min_one.

        Devolve (counters, sums), cada um (n_amostras, n_rams). Uma RAM node
        devolve (0, 0) quando o address nunca foi visto OU quando o address não
        satisfaz o mínimo de zeros/uns exigido.
        """
        addresses = self._addresses(Xb)
        ones = self._ones_count(Xb)
        zeros = self.address_size - ones

        n = len(Xb)
        counters = np.zeros((n, self.n_rams_))
        sums = np.zeros((n, self.n_rams_))
        filtra = (self.min_one > 0) or (self.min_zero > 0)

        for i in range(n):
            for r in range(self.n_rams_):
                if filtra and (ones[i, r] < self.min_one or zeros[i, r] < self.min_zero):
                    continue
                cell = self._memory[r].get(int(addresses[i, r]))
                if cell is not None:
                    counters[i, r] = cell[0]
                    sums[i, r] = cell[1]
        return counters, sums

    def _predict_from(self, Xb: np.ndarray) -> np.ndarray:
        counters, sums = self._read(Xb)
        return np.array([
            self.mean.calculate(list(zip(counters[i], sums[i])))   # etapas 14 e 15
            for i in range(len(Xb))
        ])

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        return self._predict_from(Xb)

    # ── diagnóstico ──────────────────────────────────────────────────────────

    def dont_know_fraction(self, X: np.ndarray) -> np.ndarray:
        """Fração das `N` RAM nodes que não contribuíram, por amostra.

        Não é métrica de erro, é a medida direta do fenômeno que o ESANN 2019 e
        a tese mencionam sem formalizar: quando `n` cresce a rede fica esparsa e
        "a confiança da rede diminui". É a linha de base contra a qual a
        incerteza do Kalman será comparada.
        """
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        counters, _ = self._read(Xb)
        return (counters == 0).mean(axis=1)

    def addresses_used(self) -> np.ndarray:
        """Quantos addresses distintos cada RAM node chegou a usar."""
        return np.array([len(m) for m in self._memory])

    def memory_bytes(self) -> int:
        # chave (8) + counter, sum_y, fit (8 cada)
        return int(sum(len(m) for m in self._memory) * 32)

# ════════════════════════════════════════════════════════════════════════════
# 4. AS VARIANTES COM KALMAN
#
# KalmanReW filtra os DOIS estados da célula e é o modelo de fato.
# KalmanReW_apenasY filtra só a estimativa e serve de contraprova.
# ════════════════════════════════════════════════════════════════════════════

class KalmanReW(RegressionWiSARD):
    """Filtro de Kalman nos dois estados da célula.

    célula = [c, y, p_c, p_y]

        c : estado de ocupação. Observação z = 1.0, como no KalWeNN de
            classificação. Satura em [0,1) e serve de PESO na agregação.
        y : estado da estimativa. Observação z = y_i. É a ESTIMATIVA local.

    Predição:  ŷ = Σ_k c_k · y_k / Σ_k c_k
    """

    def __init__(self, address_size=17, resolution=16, seed=42,
                 q_c=1e-3, r_c=2.0, q_y=0.0, r_y=1.0, shuffle=True, **kw):
        """
        q_c, r_c : ruídos do filtro do PESO. `r_c` controla a velocidade de
                   saturação: grande satura devagar e se aproxima do contador
                   cru; pequeno satura rápido e se aproxima do peso uniforme.
        q_y, r_y : ruídos do filtro da ESTIMATIVA. `q_y = 0` significa sem
                   esquecimento. `r_y` precisa estar na escala da variância de
                   `y` dentro dos grupos, senão o filtro fica mal calibrado.
        shuffle  : embaralha a ordem de treino. Com `q_y > 0` a ordem passa a
                   importar, então use `False` para preservar ordem temporal.
        """
        super().__init__(address_size=address_size, resolution=resolution,
                         seed=seed, **kw)
        self.q_c, self.r_c, self.q_y, self.r_y = q_c, r_c, q_y, r_y
        self.shuffle = shuffle

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._encoders = build_encoders(X, self.resolution)
        Xb = apply_encoders(X, self._encoders)
        self._set_ram_shuffle(Xb.shape[1])
        addr = self._addresses(Xb)
        ordem = self._rng.permutation(len(Xb)) if self.shuffle else np.arange(len(Xb))

        self._memory = [dict() for _ in range(self.n_rams_)]
        for i in ordem:
            self.n_trainings_ += 1
            for r_ in range(self.n_rams_):
                a = int(addr[i, r_])
                cell = self._memory[r_].get(a)
                if cell is None:
                    # peso: primeira observação de z=1 partindo de x=0, p0=1
                    kc = (1.0 + self.q_c) / (1.0 + self.q_c + self.r_c)
                    # estimativa: prior não informativo, aceita o valor de face
                    self._memory[r_][a] = [kc, y[i], (1.0 - kc) * (1.0 + self.q_c), self.r_y]
                    continue

                pc = cell[2] + self.q_c                  # atualiza o PESO
                kc = pc / (pc + self.r_c)
                cell[0] = cell[0] + kc * (1.0 - cell[0])
                cell[2] = (1.0 - kc) * pc

                py = cell[3] + self.q_y                  # atualiza a ESTIMATIVA
                ky = py / (py + self.r_y)
                cell[1] = cell[1] + ky * (y[i] - cell[1])
                cell[3] = (1.0 - ky) * py
        return self

    def predict(self, X):
        Xb = apply_encoders(np.asarray(X, float), self._encoders)
        addr = self._addresses(Xb)
        out = np.empty(len(Xb))
        for i in range(len(Xb)):
            num = den = 0.0
            for r_ in range(self.n_rams_):
                cell = self._memory[r_].get(int(addr[i, r_]))
                if cell is not None:
                    num += cell[0] * cell[1]             # c_k · y_k
                    den += cell[0]                       # c_k
            out[i] = num / den if den > 0 else 0.0
        return out

    def dont_know_fraction(self, X):
        Xb = apply_encoders(np.asarray(X, float), self._encoders)
        addr = self._addresses(Xb)
        return np.array([
            np.mean([self._memory[r_].get(int(addr[i, r_])) is None
                     for r_ in range(self.n_rams_)])
            for i in range(len(Xb))
        ])

    def uncertainty(self, X):
        """Média do `p_y` das células endereçadas, por amostra."""
        Xb = apply_encoders(np.asarray(X, float), self._encoders)
        addr = self._addresses(Xb)
        return np.array([
            float(np.mean([self._memory[r_].get(int(addr[i, r_]), [0, 0, 0, self.r_y])[3]
                           for r_ in range(self.n_rams_)]))
            for i in range(len(Xb))
        ])

    def memory_bytes(self):
        # chave (8) + c, y, p_c, p_y (8 cada)
        return int(sum(len(m) for m in self._memory) * 40)


class KalmanReW_apenasY(RegressionWiSARD):
    """Filtra só a estimativa de `y`; o contador continua cru e continua sendo o peso.

    NÃO É UM MODELO PROPOSTO. Existe para demonstrar, numericamente, que filtrar
    a estimativa sozinha não muda nada: com `q = 0` as predições são idênticas às
    da ReW (diferença medida: 0.000e+00). É o contraste que isola o mecanismo
    real, que é o peso saturado da `KalmanReW`.
    """

    def __init__(self, address_size=17, resolution=16, seed=42, mean=None,
                 q=0.0, r=1.0, shuffle=True, **kw):
        super().__init__(address_size=address_size, resolution=resolution,
                         seed=seed, mean=mean if mean is not None else SimpleMean(), **kw)
        self.q, self.r = q, r
        self.shuffle = shuffle

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._encoders = build_encoders(X, self.resolution)
        Xb = apply_encoders(X, self._encoders)
        self._set_ram_shuffle(Xb.shape[1])
        addr = self._addresses(Xb)
        ordem = self._rng.permutation(len(Xb)) if self.shuffle else np.arange(len(Xb))

        # célula: [counter, x, p]
        self._memory = [dict() for _ in range(self.n_rams_)]
        for i in ordem:
            self.n_trainings_ += 1
            for r_ in range(self.n_rams_):
                a = int(addr[i, r_])
                cell = self._memory[r_].get(a)
                if cell is None:
                    self._memory[r_][a] = [1.0, y[i], self.r]
                    continue
                p_pred = cell[2] + self.q
                k = p_pred / (p_pred + self.r)
                cell[0] += 1.0
                cell[1] = cell[1] + k * (y[i] - cell[1])
                cell[2] = (1.0 - k) * p_pred
        return self

    def _read(self, Xb):
        """Devolve (counters, sums) com `sums = x * counter`, para que a divisão
        que toda função de média faz internamente recupere o estado `x`."""
        addr = self._addresses(Xb)
        ones = self._ones_count(Xb)
        zeros = self.address_size - ones
        n = len(Xb)
        counters = np.zeros((n, self.n_rams_))
        sums = np.zeros((n, self.n_rams_))
        filtra = (self.min_one > 0) or (self.min_zero > 0)
        for i in range(n):
            for r_ in range(self.n_rams_):
                if filtra and (ones[i, r_] < self.min_one or zeros[i, r_] < self.min_zero):
                    continue
                cell = self._memory[r_].get(int(addr[i, r_]))
                if cell is not None:
                    counters[i, r_] = cell[0]
                    sums[i, r_] = cell[1] * cell[0]
        return counters, sums

# ════════════════════════════════════════════════════════════════════════════
# 5. MÉTRICAS
#
# MAE é a principal, por comparabilidade com a tese. RMSE porque MAE sozinho
# esconde erro concentrado. R² porque responde se o modelo bate o chute da
# média, e nenhum artigo da ReW reporta isso.
# ════════════════════════════════════════════════════════════════════════════

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro absoluto médio, na unidade do alvo."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raiz do erro quadrático médio. Pune erros grandes desproporcionalmente."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinação.

    1 = perfeito, 0 = empata com chutar a média, negativo = pior que a média.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.10) -> float:
    """Fração de predições a menos de `tol` de erro relativo.

    Métrica auxiliar de leitura, não substitui MAE nem R² no texto. A tolerância
    tem que ser escolhida antes de olhar os resultados, nunca depois.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true) <= tol))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.10) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        f"dentro_de_{int(tol * 100)}%": within_tolerance(y_true, y_pred, tol),
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. CARREGAMENTO DOS DADOS
#
# House Prices via OpenML. As decisões de descarte de coluna e de imputação
# estão comentadas junto de cada constante.
# ════════════════════════════════════════════════════════════════════════════

# categórica disfarçada de número: 20 = "1 andar, pós-1946", 190 = "conversão de
# duas famílias". Thermometer afirmaria que 190 é "mais" que 20.
DROP_CODIGO_CATEGORICO = ["MSSubClass"]

# NaN aqui significa ausência de garagem, não dado faltante. Imputar pela mediana
# diria que uma casa sem garagem teve a garagem construída em 1980.
DROP_NAN_SEMANTICO = ["GarageYrBlt"]

# quase constantes (≥96% de zeros). Cada coluna consome `resolution` bits da
# retina independentemente de discriminar algo, e esses bits entram no sorteio do
# mapping como ruído quase constante dentro dos addresses.
DROP_QUASE_CONSTANTE = ["PoolArea", "3SsnPorch", "LowQualFinSF", "MiscVal"]

DROP = DROP_CODIGO_CATEGORICO + DROP_NAN_SEMANTICO + DROP_QUASE_CONSTANTE

# os NaN restantes, imputados pela mediana DO TREINO
IMPUTAR = ["LotFrontage", "MasVnrArea"]


def load_house_prices():
    """Devolve (X, y) com as 30 features numéricas escolhidas e o alvo em dólares.

    Ainda contém NaN em LotFrontage e MasVnrArea: a imputação depende do split e
    é feita por `impute_train_median`.
    """
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    df = d.frame

    y = df["SalePrice"].astype(float)
    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Id", "SalePrice"])
    X = X.drop(columns=DROP)
    return X, y


def impute_train_median(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Preenche NaN com a mediana calculada SOMENTE no treino.

    Calcular a mediana no dataset completo vazaria informação do teste, pela
    mesma razão que `build_encoders` só pode ver o treino.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in IMPUTAR:
        if col in X_train.columns:
            m = X_train[col].median()
            X_train[col] = X_train[col].fillna(m)
            X_test[col] = X_test[col].fillna(m)
    return X_train, X_test


def split(X, y, seed: int, test_size: float = 0.30):
    """70/30 sem estratificação (não há classes em regressão).

    Fica próximo do split da tese, que usou 973 de treino e 480 de teste, ou
    seja, cerca de 67/33 das 1460 linhas do arquivo rotulado.

    O custo de treinar com menos dados é que sobram mais addresses nunca vistos,
    que é o modo de falha da ReW, então a fração de don't know sobe em relação a
    um 80/20.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=seed)

# ════════════════════════════════════════════════════════════════════════════
# COMPATIBILIDADE COM OS IMPORTS ANTIGOS
#
# Os notebooks e o test_vs_wisardpkg.py foram escritos quando isto era um pacote
# com seis módulos, e usam caminhos como `from rew.kalman import KalmanReW` ou
# `import rew.models as M`. Registrar os nomes antigos em sys.modules faz esses
# imports continuarem funcionando, todos apontando para este mesmo arquivo.
#
# Isso importa também para o teste diferencial, que troca `rew.models.build_encoders`
# por um encoder identidade. Como aqui tudo divide o mesmo espaço de nomes, a
# troca alcança a `RegressionWiSARD` normalmente.
# ════════════════════════════════════════════════════════════════════════════

for _apelido in ("encoder", "means", "models", "kalman", "metrics", "data"):
    _sys.modules[__name__ + "." + _apelido] = _sys.modules[__name__]
    setattr(_sys.modules[__name__], _apelido, _sys.modules[__name__])

__all__ = [
    "ThermometerEncoder", "build_encoders", "apply_encoders",
    "Mean", "SimpleMean", "PowerMean", "Median", "HarmonicMean",
    "HarmonicPowerMean", "GeometricMean", "ExponentialMean", "LogisticMean",
    "UnweightedMean",
    "RegressionWiSARD", "KalmanReW", "KalmanReW_apenasY",
    "mae", "rmse", "r2", "within_tolerance", "all_metrics",
    "load_house_prices", "impute_train_median", "split",
]
