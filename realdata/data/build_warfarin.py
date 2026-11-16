"""IWPC warfarin (simsl 0.2.1 subset): analytic sample (Section 7.2).
Usage: python build_warfarin.py <csv written by simsl_to_csv.R> <output dir>
Y is the stable weekly dose; `cycle` is a random half split used only for the
holdout check of the bandwidth constants.
"""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1])
d.columns = [c.replace('.', '_') for c in d.columns]
d['cycle'] = np.random.default_rng(20261022).permutation(np.r_[np.zeros(len(d) // 2), np.ones(len(d) - len(d) // 2)]).astype(int)
d.to_csv(f'{sys.argv[2]}/warfarin.csv', index=False)
print(d.shape)
