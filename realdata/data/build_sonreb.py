"""SonReb database: concrete analytic sample (Section 7.2).
Usage: python build_sonreb.py <dir containing sonreb.csv from Zenodo> <output dir>
"""
import sys, numpy as np, pandas as pd
R, W = sys.argv[1], sys.argv[2]
d = pd.read_csv(f'{R}/sonreb.csv').rename(columns={'Literature': 'study', 'Median Rebound Number, RN': 'RN', 'Average Velocity, Vp (m/s)': 'Vp',
    'fc,cyl (MPa)': 'Y', 'Specimen Age (days)': 'age', 'NDT Specimen Type': 'ndt_type', 'Compression Specimen': 'comp', 'Orientation': 'orient',
    'Rebar Present': 'rebar', 'Max Aggregate Size (mm)': 'agg', 'W/C Ratio': 'wc', '28-day Design Strength (MPa)': 'fd', 'Test Type': 'upv_test'})
log = [('records', len(d))]
d = d[d[['RN', 'Vp', 'Y']].notna().all(axis=1)]; log.append(('RN, Vp, strength present', len(d)))
d = d.drop_duplicates(['study', 'RN', 'Vp', 'Y']).reset_index(drop=True); log.append(('exact duplicates within study removed', len(d)))
for c in ['study', 'ndt_type', 'comp', 'orient']:
    d[c + '_c'] = d[c].astype('category').cat.codes.astype(float).where(d[c].notna())
d['rebar'] = d.rebar.astype(float); d['log_age'] = np.log(d.age)
rng = np.random.default_rng(20261023)
d['cycle'] = 0   # holdout half, stratified by study
for _, idx in d.groupby('study').groups.items():
    idx = rng.permutation(np.array(idx)); d.loc[idx[: len(idx) // 2], 'cycle'] = 1
d.to_csv(f'{W}/sonreb.csv', index=False)
for k, v in log: print(f'{k:40s}{v}')
print('studies', d.study.nunique(), '; per study: median', int(d.study.value_counts().median()), 'min', d.study.value_counts().min(), 'max', d.study.value_counts().max())
print('Y (MPa):', d.Y.describe().round(1).to_dict(), 'skew %.2f' % d.Y.skew())
print('RN quantiles:', d.RN.quantile([.02, .05, .1, .25, .5, .75, .9, .95, .98]).round(1).to_dict())
print('missing:', d[['log_age', 'ndt_type_c', 'comp_c', 'orient_c', 'agg', 'wc', 'fd']].isna().mean().round(2).to_dict())
