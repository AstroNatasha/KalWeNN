"""
WiSARD para REGRESSÃO — implementação própria (nenhum pacote externo).

  BaseRegressionWiSARD  : codificação termômetro + mapeamento bit→RAM, sem classes.
  RegressionWiSARD      : a ReW da literatura — cada célula guarda (contador n,
                          soma dos alvos s); a predição acessa os neurônios
                          endereçados e agrega os números contados.
  KalmanRegressionWiSARD: KalWeNN de regressão — cada célula guarda (x, p, n) e é
                          atualizada por um passo escalar de Kalman com z = y.

Diferença estrutural para a versão de classificação: não há discriminador por
classe. Existe UM único conjunto de RAMs, e cada célula estima o valor do alvo
naquele endereço. Toda a memória é esparsa (dicionários), o que dispensa alocar
2^address_size células e permite tuplas grandes.

Normalização do alvo
  Ambos os modelos escalam y para [0, 1] com o mínimo/máximo do TREINO e
  devolvem a predição na escala original. Isso é obrigatório no Kalman (p0=1
  pressupõe alvo de ordem 1) e é aplicado igualmente à ReW para que a
  comparação seja justa — médias não lineares (harmônica, geométrica, potência)
  não são invariantes a mudanças de escala.

Resultado que liga os dois modelos
  Com q=0 e p0→∞, o ganho de Kalman vira k_n = 1/n e cada célula passa a
  guardar a média aritmética exata dos alvos vistos nela. Agregando por
  contagem, o KalWeNN reproduz numericamente a ReW de média simples — ou seja,
  a ReW é um caso limite do KalWeNN. Verificar:  python3 -m wisard.models_regression
"""

import numpy as np

from .encoder import build_encoders, apply_encoders

# modos de agregação da ReW (como os neurônios acessados viram uma predição).
# As fórmulas seguem a implementação de referência do wisardpkg v2.0.0a7
# (src/models/regressionwisard/meanfunctions.cc) — ver nota em `_aggregate`.
REW_ESTIMATORS = ["simple", "unweighted", "power2", "median", "harmonic",
                  "harmonic_power2", "geometric", "exponential", "exponential_fixed"]

# onde vive a incerteza no KalWeNN
P_SCOPES = ["cell", "ram", "global"]

# como as evidências das RAMs viram uma predição
KAL_INFERENCES = ["mean", "count", "conf", "fusion", "kalman_seq"]

_EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# base
# ══════════════════════════════════════════════════════════════════════════════

class BaseRegressionWiSARD:
    """Codificação e mapeamento bit→RAM, idênticos aos da versão de classificação.

    Mesma seed → mesmo mapeamento, então os modelos de regressão são comparáveis
    entre si célula a célula.
    """

    # O endereço de uma RAM é o valor binário da tupla de bits, guardado num
    # inteiro de 64 bits. Acima disso os bits mais significativos recebem peso
    # zero e são silenciosamente descartados — a RAM passa a enxergar só parte
    # da tupla. É um limite da representação, não uma escolha de projeto, e vale
    # igualmente para a implementação de referência em C++.
    MAX_ADDRESS_SIZE = 64

    def __init__(self, address_size: int = 8, resolution: int = 16, seed: int = 42):
        if address_size > self.MAX_ADDRESS_SIZE:
            raise ValueError(
                f"address_size={address_size} excede {self.MAX_ADDRESS_SIZE}: o "
                f"endereço não cabe em 64 bits e {address_size - 64} bits da "
                f"tupla seriam ignorados sem aviso.")
        self.address_size = address_size
        self.resolution = resolution
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._encoders = None
        self._mapping = None
        self._powers = None
        self.n_rams_ = None
        self.y_min_ = None
        self.y_max_ = None
        self.y_mean_ = None      # média do treino (fallback e prior)
        self.y_var_ = None       # variância do treino normalizado (prior)

    # ── preparação ────────────────────────────────────────────────────────────
    def _setup(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._encoders = build_encoders(X, self.resolution)
        Xb = apply_encoders(X, self._encoders)
        n_bits = Xb.shape[1]
        self.n_rams_ = int(np.ceil(n_bits / self.address_size))
        total = self.n_rams_ * self.address_size
        indices = np.arange(n_bits)
        if total > n_bits:
            indices = np.concatenate(
                [indices, self._rng.choice(indices, total - n_bits, replace=True)]
            )
        self._rng.shuffle(indices)
        self._mapping = indices.reshape(self.n_rams_, self.address_size)
        self._powers = 1 << np.arange(self.address_size - 1, -1, -1)

        # normalização do alvo pelo treino
        self.y_min_, self.y_max_ = float(y.min()), float(y.max())
        self._y_range = (self.y_max_ - self.y_min_) or 1.0
        yn = self._normalize(y)
        self.y_mean_ = float(yn.mean())
        self.y_var_ = float(yn.var()) or 1.0
        return X, yn, Xb

    def _normalize(self, y):
        return (np.asarray(y, dtype=float) - self.y_min_) / self._y_range

    def _denormalize(self, yn):
        return np.asarray(yn, dtype=float) * self._y_range + self.y_min_

    def _all_addresses(self, Xb: np.ndarray) -> np.ndarray:
        bits = Xb[:, self._mapping]                       # (n, n_rams, address_size)
        return (bits * self._powers).sum(axis=2).astype(int)   # (n, n_rams)

    def _addresses_of(self, X: np.ndarray) -> np.ndarray:
        Xb = apply_encoders(np.asarray(X, dtype=float), self._encoders)
        return self._all_addresses(Xb)


# ══════════════════════════════════════════════════════════════════════════════
# ReW — Regression WiSARD (contador + soma por célula)
# ══════════════════════════════════════════════════════════════════════════════

class RegressionWiSARD(BaseRegressionWiSARD):
    """Regression WiSARD: cada célula guarda (n, s) = (nº de acessos, soma dos alvos).

    `estimator` escolhe como os neurônios acessados são agregados. Sendo
    m_i = s_i/n_i a média local da célula i, n_i o seu contador e N o número de
    células acessadas com n_i > 0:

      simple           Σs_i / Σn_i               média aritmética PONDERADA pelos
                                                 contadores (a ReW canônica)
      unweighted       (1/N) Σ m_i               média aritmética simples das
                                                 células, ignorando os contadores
      power2           √( Σ m_i² / N )           média quadrática
      median           mediana dos m_i           robusta a células ruidosas
      harmonic         N / Σ(1/m_i)              média harmônica
      harmonic_power2  √( N / Σ(1/m_i²) )        harmônica de potência 2
      geometric        (Π m_i)^(1/N)             média geométrica
      exponential      log( (1 + Σ e^{m_i}) / N) média exponencial (como na
                                                 referência, com o "+1" indevido)
      exponential_fixed log( Σ e^{m_i} / N )      a mesma, sem o "+1"

    IMPORTANTE — só a `simple` pondera pelos contadores. Todas as demais são
    médias simples sobre as células, exatamente como na implementação de
    referência (wisardpkg v2.0.0a7, `meanfunctions.cc`): lá, `SimpleMean` soma
    contadores e somas, enquanto `PowerMean`, `Median`, `HarmonicMean`,
    `HarmonicPowerMean`, `GeometricMean` e `ExponentialMean` dividem por N.
    `unweighted` é o `PowerMean(1)` da referência.

    Células nunca vistas no treino (n_i = 0) não entram na conta. Se nenhuma
    célula endereçada for conhecida, devolve a média do treino.
    """

    def __init__(self, address_size: int = 8, resolution: int = 16,
                 seed: int = 42, estimator: str = "simple"):
        super().__init__(address_size, resolution, seed)
        if estimator not in REW_ESTIMATORS:
            raise ValueError(f"estimator inválido: {estimator!r}. Opções: {REW_ESTIMATORS}")
        self.estimator = estimator

    def fit(self, X, y):
        X, yn, Xb = self._setup(X, y)
        addresses = self._all_addresses(Xb)
        # _mem[r] = {endereco: [n, soma]}
        self._mem = [dict() for _ in range(self.n_rams_)]
        for i in range(len(addresses)):
            row, z = addresses[i], yn[i]
            for r in range(self.n_rams_):
                cell = self._mem[r].get(int(row[r]))
                if cell is None:
                    self._mem[r][int(row[r])] = [1.0, z]
                else:
                    cell[0] += 1.0
                    cell[1] += z
        return self

    def set_estimator(self, estimator: str):
        """Troca o agregador sem retreinar (o treino da ReW não depende dele)."""
        if estimator not in REW_ESTIMATORS:
            raise ValueError(f"estimator inválido: {estimator!r}")
        self.estimator = estimator
        return self

    def _aggregate(self, n, m):
        """n = contadores, m = médias locais das células acessadas (todas com n > 0).

        Fórmulas conferidas linha a linha contra o wisardpkg v2.0.0a7,
        `src/models/regressionwisard/meanfunctions.cc`. Duas divergências
        deliberadas em relação ao código de referência, ambas por serem defeitos
        evidentes dele (ver README_REGRESSAO.md §3.2):

        - `Median`: a referência indexa o vetor ordenado com deslocamento de +1
          (e lê fora dos limites quando há exatamente 2 células). Aqui usamos a
          mediana correta.
        - `ExponentialMean`: a referência inicializa o acumulador em 1 em vez de
          0, o que soma um e^0 fantasma. Reproduzimos esse "+1" para manter a
          equivalência numérica — é o que a definição da biblioteca faz.
        """
        est = self.estimator
        if est == "simple":                       # SimpleMean
            return float((n * m).sum() / n.sum())
        if est == "unweighted":                   # PowerMean(1)
            return float(m.mean())
        if est == "power2":                       # PowerMean(2)
            return float(np.sqrt((m ** 2).sum() / len(m)))
        if est == "median":                       # Median (versão corrigida)
            return float(np.median(m))
        if est == "harmonic":                     # HarmonicMean
            nz = m[m != 0]
            if len(nz) == 0:
                return 0.0
            return float(len(nz) / (1.0 / nz).sum())
        if est == "harmonic_power2":              # HarmonicPowerMean(2)
            nz = m[m != 0]
            if len(nz) == 0:
                return 0.0
            return float(np.sqrt(len(nz) / (1.0 / nz ** 2).sum()))
        if est == "geometric":                    # GeometricMean
            nz = m[m != 0]
            if len(nz) == 0:
                return 0.0
            return float(np.prod(nz) ** (1.0 / len(nz)))
        if est == "exponential":                  # ExponentialMean (com o "+1")
            return float(np.log((1.0 + np.exp(m).sum()) / len(m)))
        if est == "exponential_fixed":            # ExponentialMean sem o "+1"
            # Sem o acumulador iniciado em 1, log(média de e^m) é uma média de
            # verdade: fica sempre entre min(m) e max(m). Com o "+1" da
            # referência, não fica — e o erro explode. Serve para medir o custo
            # daquele defeito.
            return float(np.log(np.exp(m).mean()))
        raise ValueError(self.estimator)

    def predict(self, X):
        addresses = self._addresses_of(X)
        out = np.empty(len(addresses))
        for i in range(len(addresses)):
            row = addresses[i]
            ns, ms = [], []
            for r in range(self.n_rams_):
                cell = self._mem[r].get(int(row[r]))
                if cell is not None:
                    ns.append(cell[0])
                    ms.append(cell[1] / cell[0])
            out[i] = (self._aggregate(np.asarray(ns), np.asarray(ms))
                      if ns else self.y_mean_)
        return self._denormalize(out)

    def memory_bytes(self):
        entries = sum(len(d) for d in self._mem)
        return int(entries * 24)   # chave (8) + contador (8) + soma (8)

    def n_cells(self):
        return int(sum(len(d) for d in self._mem))


# ══════════════════════════════════════════════════════════════════════════════
# KalWeNN de regressão — Kalman por célula, com z = y
# ══════════════════════════════════════════════════════════════════════════════

class KalmanRegressionWiSARD(BaseRegressionWiSARD):
    """Cada célula estima o alvo por um passo escalar de Kalman.

    Atualização (z = alvo normalizado da amostra):
        p_pred = p + q ; k = p_pred/(p_pred+r) ; x += k(z-x) ; p = (1-k)p_pred

    `p_scope` — ONDE a incerteza é mantida:
        cell    um p por célula (RAM, endereço). Cada endereço tem seu próprio
                ritmo de convergência: endereços raros seguem com p alto e
                continuam aprendendo rápido.
        ram     um p por RAM, compartilhado por todos os endereços dela. Como
                toda amostra acessa toda RAM, o ganho passa a depender só de
                quantas amostras já foram vistas.
        global  um único p para o modelo inteiro. Ablação mais extrema: o ganho
                vira uma função só do tempo, igual para todas as células.

    `inference` — COMO os neurônios acessados viram uma predição:
        mean        média simples dos x
        count       média dos x ponderada pelos contadores n (o análogo direto
                    da ReW: "contando os números")
        conf        média ponderada por (1-p) — heurística de confiança
        fusion      Σ(x/p) / Σ(1/p) — fusão bayesiana, peso = precisão 1/p.
                    É o filtro de Kalman aplicado na INFERÊNCIA.
        kalman_seq  fusion com prior do treino: (μ0/P0 + Σx/p) / (1/P0 + Σ1/p),
                    com μ0, P0 = média e variância do alvo no treino. Equivale a
                    fundir sequencialmente as RAMs partindo do prior.

    Em p_scope='ram'/'global' todos os p acessados são iguais, então conf e
    fusion degeneram na média simples — o efeito está documentado no README.
    """

    def __init__(self, address_size: int = 8, resolution: int = 16, seed: int = 42,
                 q: float = 1e-3, r: float = 1e-1, p0: float = 1.0,
                 n_epochs: int = 1, p_scope: str = "cell", inference: str = "count"):
        super().__init__(address_size, resolution, seed)
        if p_scope not in P_SCOPES:
            raise ValueError(f"p_scope inválido: {p_scope!r}. Opções: {P_SCOPES}")
        if inference not in KAL_INFERENCES:
            raise ValueError(f"inference inválido: {inference!r}. Opções: {KAL_INFERENCES}")
        self.q, self.r, self.p0 = q, r, p0
        self.n_epochs = n_epochs
        self.p_scope = p_scope
        self.inference = inference

    # ── treino ────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        X, yn, Xb = self._setup(X, y)
        addresses = self._all_addresses(Xb)
        n_rams = self.n_rams_

        # _mem[r] = {endereco: [x, p, n]}   (p só é usado quando p_scope='cell')
        self._mem = [dict() for _ in range(n_rams)]
        self._p_ram = np.full(n_rams, self.p0)   # p_scope='ram'
        self._p_global = self.p0                 # p_scope='global'

        q, r, scope = self.q, self.r, self.p_scope
        for _ in range(self.n_epochs):
            for i in self._rng.permutation(len(addresses)):
                row, z = addresses[i], yn[i]

                if scope == "global":
                    p_pred = self._p_global + q
                    k = p_pred / (p_pred + r)
                    self._p_global = (1.0 - k) * p_pred

                for ram in range(n_rams):
                    a = int(row[ram])
                    cell = self._mem[ram].get(a)
                    if cell is None:
                        cell = [0.0, self.p0, 0.0]
                        self._mem[ram][a] = cell

                    if scope == "cell":
                        p_pred = cell[1] + q
                        k = p_pred / (p_pred + r)
                        cell[1] = (1.0 - k) * p_pred
                    elif scope == "ram":
                        p_pred = self._p_ram[ram] + q
                        k = p_pred / (p_pred + r)
                        self._p_ram[ram] = (1.0 - k) * p_pred
                    # 'global': k já calculado uma vez por amostra

                    cell[0] += k * (z - cell[0])
                    cell[2] += 1.0
        return self

    def set_inference(self, inference: str):
        """Troca o agregador sem retreinar (o treino não depende dele)."""
        if inference not in KAL_INFERENCES:
            raise ValueError(f"inference inválido: {inference!r}")
        self.inference = inference
        return self

    # ── inferência ────────────────────────────────────────────────────────────
    def _cell_p(self, ram: int, cell) -> float:
        if self.p_scope == "cell":
            return cell[1]
        if self.p_scope == "ram":
            return float(self._p_ram[ram])
        return float(self._p_global)

    def _aggregate(self, x, p, n):
        mode = self.inference
        if mode == "mean":
            return float(x.mean())
        if mode == "count":
            return float((n * x).sum() / n.sum())
        if mode == "conf":
            w = np.maximum(1.0 - p, _EPS)
            return float((w * x).sum() / w.sum())
        if mode == "fusion":
            w = 1.0 / np.maximum(p, _EPS)
            return float((w * x).sum() / w.sum())
        if mode == "kalman_seq":
            w = 1.0 / np.maximum(p, _EPS)
            prec0 = 1.0 / max(self.y_var_, _EPS)
            return float((prec0 * self.y_mean_ + (w * x).sum()) / (prec0 + w.sum()))
        raise ValueError(mode)

    def predict(self, X):
        addresses = self._addresses_of(X)
        out = np.empty(len(addresses))
        for i in range(len(addresses)):
            row = addresses[i]
            xs, ps, ns = [], [], []
            for ram in range(self.n_rams_):
                cell = self._mem[ram].get(int(row[ram]))
                if cell is not None:
                    xs.append(cell[0])
                    ps.append(self._cell_p(ram, cell))
                    ns.append(cell[2])
            out[i] = (self._aggregate(np.asarray(xs), np.asarray(ps), np.asarray(ns))
                      if xs else self.y_mean_)
        return self._denormalize(out)

    def predicted_uncertainty(self, X):
        """(predições, incerteza) — incerteza = média dos p das células acessadas."""
        addresses = self._addresses_of(X)
        unc = np.empty(len(addresses))
        for i in range(len(addresses)):
            row = addresses[i]
            ps = [self._cell_p(ram, self._mem[ram][int(row[ram])])
                  for ram in range(self.n_rams_) if int(row[ram]) in self._mem[ram]]
            unc[i] = float(np.mean(ps)) if ps else self.p0
        return self.predict(X), unc

    def memory_bytes(self):
        """Bytes da memória. Em p_scope='cell' o p é por célula; nos demais
        escopos o p custa um punhado de floats, praticamente nada."""
        entries = sum(len(d) for d in self._mem)
        if self.p_scope == "cell":
            return int(entries * 32)          # chave + x + p + n
        extra = self.n_rams_ * 8 if self.p_scope == "ram" else 8
        return int(entries * 24 + extra)      # chave + x + n

    def memory_bytes_no_p(self):
        """Footprint descartando p após o treino (p não é usado em mean/count)."""
        entries = sum(len(d) for d in self._mem)
        return int(entries * 24)

    def n_cells(self):
        return int(sum(len(d) for d in self._mem))


# ══════════════════════════════════════════════════════════════════════════════
# validação: a ReW é um caso limite do KalWeNN (q=0, p0→∞, agregação por contagem)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from datasets.loaders_regression import get_regression

    print("A ReW é o caso limite do KalWeNN com q=0 e p0→∞ ?\n")
    print("Com q=0, a recursão dá k_n = 1/(n + r/p0). Quando r/p0 → 0 o ganho vira")
    print("1/n, cada célula guarda a média aritmética dos alvos vistos nela, e a")
    print("agregação por contagem reproduz a ReW de média simples. Abaixo, a maior")
    print("diferença entre as duas predições (em % da amplitude do alvo) conforme")
    print("p0 cresce — deve cair proporcionalmente a 1/p0 até o piso do float64.\n")

    P0S = [1e1, 1e2, 1e4, 1e6, 1e8]
    print(f"{'dataset':10s} {'seed':>4s} " + " ".join(f"{'p0=1e%d' % int(np.log10(p)):>10s}" for p in P0S))
    for name in ["yacht", "auto_mpg", "energy", "concrete"]:
        X, y = get_regression(name)
        rng_y = y.max() - y.min()
        for seed in range(2):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
            rew = RegressionWiSARD(8, 16, seed=seed, estimator="simple").fit(Xtr, ytr)
            base = rew.predict(Xte)
            diffs = []
            for p0 in P0S:
                kal = KalmanRegressionWiSARD(8, 16, seed=seed, q=0.0, r=1.0, p0=p0,
                                             p_scope="cell", inference="count").fit(Xtr, ytr)
                diffs.append(float(np.abs(base - kal.predict(Xte)).max()) / rng_y * 100)
            print(f"{name:10s} {seed:4d} " + " ".join(f"{d:9.2e}%" for d in diffs))
