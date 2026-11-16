"""Localized Semi / Conf estimators of zeta_X(s0), zeta_W(s0) and Delta(s0) for the real data.

Root solving, the spacing estimate of g and the variance formulas come from
conditional_mc.estimators. Only the nuisances are specific to the real data:
m is a gradient boosted tree regression on the covariates, and the residual density is a
location-scale model eps = sigma(V) U, with sigma from boosting |eps| (inner two-fold
residuals) and a Gaussian KDE of U with Silverman's bandwidth.
"""
import math, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import KFold, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), 'conditional')]
from conditional_mc.estimators import kernel_h, bandwidths, spacing_c, _fold_quantiles, K2_INT  # noqa: E402
from config import CS  # noqa: E402

CN_FACTOR = 0.75
METHODS = ('semi', 'conf', 'conf_hsemi')


class Setup:
    """Dataset configuration plus the prediction level alpha."""
    def __init__(self, cfg, alpha):
        self.X, self.CAT, self.ZS, self.GRIDS, self.folds = cfg['X'], cfg['CAT'], cfg['Z'], cfg['GRIDS'], cfg['folds']
        self.TARGETS = {'x': self.X, **{k: self.X + v for k, v in self.ZS.items()}}
        self.ALPHA, self.P = alpha, 1 - alpha

    def gbm(self, cols, seed):
        return HGB(max_iter=500, learning_rate=0.03, min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=True, validation_fraction=0.15, random_state=seed,
                   categorical_features=[c in self.CAT for c in cols])

    def make_folds(self, df, seed):
        if self.folds == 'random':
            return [ev for _, ev in KFold(2, shuffle=True, random_state=seed).split(df)]
        if self.folds == 'by_study':      # every study appears in both folds
            return [ev for _, ev in StratifiedKFold(2, shuffle=True, random_state=seed).split(df, df.study_c)]
        if self.folds == 'spatial_blocks':  # randomly assigned 100 km blocks
            ub = df.block.unique(); first = set(np.random.default_rng(seed).permutation(ub)[:len(ub) // 2])
            m = df.block.isin(first).values
            return [np.where(m)[0], np.where(~m)[0]]
        raise ValueError(self.folds)


class LSPrep:
    def __init__(self, sigma, grid, dens):
        self.sigma, self.grid, self.dens = sigma, grid, dens

    def A(self, q):
        u = q / self.sigma
        f = lambda t: np.interp(t, self.grid, self.dens, left=0.0, right=0.0)
        return (f(u) - f(-u)) / self.sigma


def nuisance(st, Vtr, ytr, Vev, yev, cols, seed):
    m = st.gbm(cols, seed).fit(Vtr, ytr)
    e = np.zeros(len(ytr))  # inner out-of-fold residuals for the density model
    for a, b in KFold(2, shuffle=True, random_state=seed + 7).split(Vtr):
        e[b] = ytr[b] - st.gbm(cols, seed).fit(Vtr.iloc[a], ytr[a]).predict(Vtr.iloc[b])
    sm = st.gbm(cols, seed).fit(Vtr, np.abs(e))
    s_tr = np.clip(sm.predict(Vtr), 0.02, None)
    s_ev = np.clip(sm.predict(Vev), 0.02, None)
    u = e / s_tr
    bw = 0.9 * min(u.std(), np.subtract(*np.quantile(u, [.75, .25])) / 1.34) * len(u) ** -0.2
    grid = np.linspace(-15, 15, 6001)
    dens = np.zeros_like(grid)
    for i in range(0, len(u), 500):
        z = (grid[:, None] - u[None, i:i + 500]) / bw
        dens += np.exp(-0.5 * z * z).sum(1)
    dens /= len(u) * bw * math.sqrt(2 * math.pi)
    return yev - m.predict(Vev), LSPrep(s_ev, grid, dens), m


def estimate(st, Sv, folds, resid, prep, tx, tw, s0, method, n, scale, c=1.5):
    """One (s0, method): mirrors conditional_mc.estimators.estimate_at_s0."""
    P, ALPHA = st.P, st.ALPHA
    marginal = s0 is None
    hs = bandwidths(n, c, c)
    h = 1.0 if marginal else hs[method] * scale
    c_n = spacing_c(n, 2.0 if marginal else h / scale, CN_FACTOR)
    kind = 'semi' if method == 'semi' else 'conf'
    fq, fig, phi, ind, fw, ff, neff = {tx: [], tw: []}, {tx: [], tw: []}, {tx: [], tw: []}, {tx: [], tw: []}, [], [], []
    for j, ev in enumerate(folds):
        w = np.ones(len(ev)) if marginal else kernel_h(Sv[ev], s0, h)
        fw.append(w); ff.append(float(np.mean(w))); neff.append(float(np.sum(w) * (1 if marginal else h)))
        for t in (tx, tw):
            r = resid[t][j]
            fq_ = _fold_quantiles(r, w, prep[t][j].A, P, c_n)
            q = fq_[kind]
            fq[t].append(q); fig[t].append(fq_['inv_g'])
            a = np.asarray(prep[t][j].A(q)) if method == 'semi' else 0.0
            phi[t].append(P - (np.abs(r) <= q) - (r * a if method == 'semi' else 0.0))
            ind[t].append((np.abs(r) <= q).astype(float))
    f_s = float(np.mean(ff))
    g = {t: 1.0 / np.nanmean(fig[t]) for t in (tx, tw)}
    zq = {t: float(np.mean(fq[t])) for t in (tx, tw)}
    cross = lambda a, b: float(np.mean([np.mean(a[j] * b[j] * fw[j]) for j in range(2)]))
    K = 1.0 if marginal else K2_INT
    gx, gw = g[tx], g[tw]
    if method == 'semi':
        sxx, sww, sxw = cross(phi[tx], phi[tx]), cross(phi[tw], phi[tw]), cross(phi[tx], phi[tw])
        vx, vw = sxx / (f_s * gx * gx), sww / (f_s * gw * gw)
        vd = (K / f_s) * (vx + vw - 2 * sxw / (f_s * gx * gw))
        vx, vw = K / f_s * vx, K / f_s * vw
    else:
        rho = cross(ind[tx], ind[tw]) / f_s - P * P
        vx, vw = K / f_s * ALPHA * P / gx ** 2, K / f_s * ALPHA * P / gw ** 2
        vd = vx + vw - 2 * K / f_s * rho / (gx * gw)
    nh = n * h if not marginal else n
    return dict(zeta_x=zq[tx], zeta_w=zq[tw], delta=zq[tx] - zq[tw],
                se_x=math.sqrt(vx / nh), se_w=math.sqrt(vw / nh),
                se=math.sqrt(vd / nh) if vd > 0 else math.nan, h=h, n_eff=float(np.mean(neff)) * 2)


def fit_split(st, df, seed):
    y = df.Y.values
    folds = st.make_folds(df, seed)
    resid, prep = {}, {}
    for t, cols in st.TARGETS.items():
        resid[t], prep[t] = [], []
        for j, ev in enumerate(folds):
            tr = folds[1 - j]
            r, pr, _ = nuisance(st, df.iloc[tr][cols], y[tr], df.iloc[ev][cols], y[ev], cols, seed + j)
            resid[t].append(r); prep[t].append(pr)
    return folds, resid, prep


def run_split(st, df, split):
    n = len(df)
    folds, resid, prep = fit_split(st, df, 20261019 + split)
    rows = []
    for z in st.ZS:
        for method in METHODS:
            rows.append(dict(split=split, S='marginal', s0=np.nan, Z=z, method=method, c=np.nan,
                             **estimate(st, None, folds, resid, prep, 'x', z, None, method, n, 1.0)))
        for S, grid in st.GRIDS.items():
            Sv = df[S].values
            scale = Sv.std() * math.sqrt(3)  # rescale the Unif(-1,1) bandwidth rule of the simulation
            for s0 in grid:
                for c in CS:
                    for method in METHODS:
                        rows.append(dict(split=split, S=S, s0=s0, Z=z, method=method, c=c,
                                         **estimate(st, Sv, folds, resid, prep, 'x', z, s0, method, n, scale, c)))
    return rows


def holdout(st, df):
    """Bandwidth-constant check: estimate on the part with cycle == 0, compute the kernel-weighted
    coverage of |Y - m(V)| <= zeta(s0) on the part with cycle == 1."""
    tr_df, te_df = df[df.cycle == 0].reset_index(drop=True), df[df.cycle == 1].reset_index(drop=True)
    n = len(tr_df)
    folds, resid, prep = fit_split(st, tr_df, 777)
    z0 = list(st.ZS)[0]
    out = []
    for t in ['x'] + list(st.ZS):
        cols = st.TARGETS[t]
        m = st.gbm(cols, 1).fit(tr_df[cols], tr_df.Y.values)
        ar = np.abs(te_df.Y.values - m.predict(te_df[cols]))
        for S, grid in st.GRIDS.items():
            scale = tr_df[S].std() * math.sqrt(3)
            for s0 in grid:
                for c in CS:
                    for method in METHODS:
                        e = estimate(st, tr_df[S].values, folds, resid, prep, 'x', t if t != 'x' else z0, s0, method, n, scale, c)
                        q = e['zeta_x'] if t == 'x' else e['zeta_w']
                        w = kernel_h(te_df[S].values, s0, e['h'])
                        out.append(dict(target=t, S=S, s0=s0, c=c, method=method, zeta=q,
                                        cover_test=float(np.sum(w * (ar <= q)) / np.sum(w)), n_test_eff=float(np.sum(w > 0))))
    return out
