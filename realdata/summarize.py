"""Print the numbers quoted in Section 7 from results/*_a0.1_final.csv.

    python summarize.py

For each data set: the grid points at which each test rejects H0: Delta(s0) <= 0
(one-sided, level 0.05), and Delta-hat, SE and the lower bound L = Delta-hat - 1.645 SE.
For NHANES also the counts over the 38 fasting glucose and HbA1c grid points.
"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); Z95 = 1.6448536269514722
pd.set_option('display.width', 200)
for name in ['nhanes', 'sonreb', 'lucas', 'warfarin']:
    T = pd.read_csv(os.path.join(HERE, 'results', f'{name}_a0.1_final.csv'))
    T = T[(T.S != 'marginal') & T.method.isin(['semi', 'conf'])].copy(); T['L'] = T.delta - Z95 * T.se
    print(f'\n===== {name} =====')
    for (S, z), g in T.groupby(['S', 'Z']):
        p = g.pivot_table(index='s0', columns='method', values=['delta', 'se', 'L'])
        rej = {m: [f'{s:g}' for s in p.index if p.loc[s, ('L', m)] > 0] for m in ['semi', 'conf']}
        print(f'\n{z} by {S}: Semi rejects at {rej["semi"] or "none"}; Conf rejects at {rej["conf"] or "none"}')
        print(p.round(2).to_string())
    if name == 'nhanes':
        k = T[T.Z.isin(['fpg', 'hba1c'])]
        s = k[k.method == 'semi'].set_index(['S', 's0', 'Z']); c = k[k.method == 'conf'].set_index(['S', 's0', 'Z']).reindex(s.index)
        rs, rc = s.p0 < .05, c.p0 < .05
        print(f'\nFasting glucose and HbA1c, {len(s)} grid points: Semi rejects {rs.sum()}, Conf {rc.sum()}, disagree {(rs != rc).sum()} '
              f'(Semi only {(rs & ~rc).sum()}); mean |Delta_Semi - Delta_Conf| = {(s.delta - c.delta).abs().mean():.2f}; '
              f'median SE ratio Semi/Conf = {np.median(s.se / c.se):.2f}')
