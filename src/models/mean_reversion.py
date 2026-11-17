"""Knowledge-guided mean-reversion forecaster.

For product p at origin t and horizon h,

    y_hat = y + a * (mu_short - y) + b * (mu_long - y)

where y is the last observed price, mu_short the trailing 750-business-day mean,
mu_long the expanding mean of the whole history up to t, and (a, b) with
a, b >= 0 and a + b <= 1 the shares of the two gaps closed within h days.
With gap_state, (a, b) takes one value when the relative gap |mu_short - y| / y
is at or below the cell's median and another when it is above, so large
deviations can revert at a different speed from small ones. Parameters are
fitted per pooling cell (commodity group or family) and horizon by grid
least-absolute-deviation.
"""
import numpy as np

FIT_STRIDE = 5         # fitting origins are subsampled; consecutive origins overlap almost completely
MIN_HISTORY = 250      # origins need a year of history so both anchors are defined
GRID1 = np.round(np.linspace(0, 1, 101), 2)
GRID2 = np.round(np.arange(0, 1 + 1e-9, 0.02), 2)


def anchors(P, window=750):
    """Trailing window-day and expanding means of a (products x days) price matrix, known at each day."""
    T = P.shape[1]
    C = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
    idx = np.arange(T)
    lo = np.maximum(idx + 1 - window, 0)
    return (C[:, idx + 1] - C[:, lo]) / (idx + 1 - lo), C[:, 1:] / (idx + 1)


def _fit_cell(r, g1, g2, w, two):
    if not two:
        e = [np.sum(w * np.abs(r - a * g1)) for a in GRID1]
        return float(GRID1[int(np.argmin(e))]), 0.0
    best = (np.inf, 0.0, 0.0)
    for a in GRID2:
        bs = GRID2[GRID2 <= 1 - a + 1e-9]
        e = np.abs(r[None, :] - a * g1[None, :] - bs[:, None] * g2[None, :]) @ w
        k = int(np.argmin(e))
        if e[k] < best[0]:
            best = (float(e[k]), float(a), float(bs[k]))
    return best[1], best[2]


class MeanReversion:
    """structure in {single, two, state, two_state}; objective in {thb, relative}.

    window (trailing-mean length) and cut_q (deviation-state quantile) default to
    the reported 750 days and the median, and exist for the sensitivity checks.
    """

    def __init__(self, P, level, cells, structure="two_state", objective="thb", window=750, cut_q=0.5):
        self.P, self.level, self.cells = P, level, cells
        self.structure, self.objective = structure, objective
        self.two = structure in ("two", "two_state")
        self.state = structure in ("state", "two_state")
        self.cut_q = cut_q
        self.m_short, self.m_long = anchors(P, window)
        self.params = {}

    def _gap(self, g, o):
        return np.abs(self.m_short[g, o] - self.P[g, o]) / self.P[g, o]

    def fit(self, h, label_end):
        """Fit horizon h on every fifth origin whose label falls before index label_end."""
        P, T, n = self.P, self.P.shape[1], self.P.shape[0]
        o = np.arange(MIN_HISTORY, T - h, FIT_STRIDE)
        o = o[o + h < label_end]
        g = np.repeat(np.arange(n), len(o)); o = np.tile(o, n)
        r = P[g, o + h] - P[g, o]
        g1 = self.m_short[g, o] - P[g, o]; g2 = self.m_long[g, o] - P[g, o]
        w = np.ones(len(g)) if self.objective == "thb" else 1.0 / self.level[g]
        prm = {}
        for c in np.unique(self.cells):
            k = np.flatnonzero(self.cells[g] == c)
            if self.state:
                gs = self._gap(g[k], o[k]); cut = float(np.quantile(gs, self.cut_q))
                for s, ks in (("small", gs <= cut), ("large", gs > cut)):
                    i = k[ks]
                    prm[(c, s)] = _fit_cell(r[i], g1[i], g2[i], w[i], self.two)
                prm[(c, "cut")] = cut
            else:
                prm[(c, "all")] = _fit_cell(r[k], g1[k], g2[k], w[k], self.two)
        self.params[h] = prm
        return prm

    def predict(self, h, g, o):
        P, prm = self.P, self.params[h]
        p = P[g, o]; g1 = self.m_short[g, o] - p; g2 = self.m_long[g, o] - p
        a = np.zeros(len(g)); b = np.zeros(len(g)); cg = self.cells[g]
        gs = self._gap(g, o) if self.state else None
        for c in np.unique(cg):
            k = cg == c
            if self.state:
                lg = k & (gs > prm[(c, "cut")]); sm = k & ~lg
                a[sm], b[sm] = prm[(c, "small")]; a[lg], b[lg] = prm[(c, "large")]
            else:
                a[k], b[k] = prm[(c, "all")]
        return p + a * g1 + b * g2
