"""Run the Section 7 analysis for one data set.

    python run_estimation.py nhanes --alpha 0.1 --jobs 10

Writes to results/<dataset>_a<alpha>_{raw,summary,final,holdout}.csv:
  raw      one row per split, (S, s0), Z, method and bandwidth constant c
  summary  median over the splits; se^2 = median(se_b^2 + (Delta_b - Delta)^2); p0 tests Delta <= 0
  final    the rows used in the paper (Semi c = 2, Conf c = 3, Conf with the Semi bandwidth c = 2, marginal)
  holdout  out-of-sample local coverage used to check the bandwidth constants
"""
import argparse, math, os, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
import estimation as E
from config import DATASETS, CHOSEN, SPLITS

HERE = os.path.dirname(os.path.abspath(__file__))


def aggregate(g):
    d = g.delta.median()
    se = math.sqrt(np.nanmedian(g.se ** 2 + (g.delta - d) ** 2))
    return pd.Series(dict(zeta_x=g.zeta_x.median(), zeta_w=g.zeta_w.median(), delta=d, se=se,
                          split_sd=g.delta.std(), h=g.h.iloc[0], n_eff=g.n_eff.median(), p0=1 - norm.cdf(d / se)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(DATASETS)); ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--splits', type=int, default=SPLITS); ap.add_argument('--jobs', type=int, default=10)
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]; st = E.Setup(cfg, a.alpha)
    df = pd.read_csv(os.path.join(HERE, cfg['file'])).reset_index(drop=True)
    tasks = [delayed(E.run_split)(st, df, s) for s in range(a.splits)] + [delayed(E.holdout)(st, df)]
    res = Parallel(n_jobs=a.jobs, batch_size=1)(tasks)
    out = os.path.join(HERE, 'results', f'{a.dataset}_a{a.alpha:g}')
    raw = pd.DataFrame([r for rr in res[:-1] for r in rr]); raw.to_csv(out + '_raw.csv', index=False)
    pd.DataFrame(res[-1]).to_csv(out + '_holdout.csv', index=False)
    summ = raw.groupby(['S', 's0', 'c', 'Z', 'method'], dropna=False).apply(aggregate).reset_index()
    summ.to_csv(out + '_summary.csv', index=False)
    keep = (summ.S == 'marginal') | (summ.c == summ.method.map(CHOSEN))
    summ[keep].to_csv(out + '_final.csv', index=False)
    print(f'wrote {out}_*.csv')


if __name__ == '__main__':
    main()
