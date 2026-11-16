"""LUCAS 2015 topsoil: soil-carbon analytic sample (Section 7.2).
Usage: python build_lucas.py <ESDAC download dir> <spectra10nm.npz from lucas_spectra.py> <output dir>
"""
import sys, numpy as np, pandas as pd
from sklearn.decomposition import PCA
R, SPEC, W = sys.argv[1], sys.argv[2], sys.argv[3]
t = pd.read_csv(f'{R}/topsoil/LUCAS_Topsoil_2015_20200323.csv', low_memory=False)
a = pd.read_csv(f'{R}/LUCAS2015_AncillaryData_20201007.csv', low_memory=False).rename(columns={'POI': 'Point_ID', 'Elevation': 'elev_dem'})
sp = np.load(SPEC); A = pd.DataFrame(sp['A'], index=sp['pid'])
log = [('topsoil points', len(t))]
d = t[['Point_ID', 'OC', 'NUTS_0']].merge(a, on='Point_ID', how='inner'); log.append(('with ancillary', len(d)))
d = d[d.Point_ID.isin(A.index)]; log.append(('with spectra', len(d)))
d = d[d.OC.notna()]; log.append(('OC present', len(d)))
d = d[d.OC <= 120].reset_index(drop=True); log.append(('mineral soils (OC <= 120 g/kg)', len(d)))
d = d[d.BIO1.notna()].reset_index(drop=True); log.append(('climate covariates present', len(d)))
d['xkm'] = d.Point_ID // 10000; d['ykm'] = d.Point_ID % 10000
d['asp_sin'] = np.sin(np.deg2rad(d.Aspect)); d['asp_cos'] = np.cos(np.deg2rad(d.Aspect))
for c in ['CLIMA_Cod', 'BioGeo', 'Soil_Group', 'NUTS_0']:
    d[c + '_c'] = d[c].astype('category').cat.codes.astype(float).where(d[c].notna())
pca = PCA(n_components=30).fit(A.loc[d.Point_ID].values)
Z = pca.transform(A.loc[d.Point_ID].values); ev = np.cumsum(pca.explained_variance_ratio_)
for k in range(30): d[f'PC{k+1}'] = Z[:, k]
d['Y'] = d.OC
d['block'] = (d.xkm // 100).astype(int) * 1000 + (d.ykm // 100).astype(int)
blocks = d.block.unique(); rng = np.random.default_rng(20261022); half = set(rng.permutation(blocks)[:len(blocks) // 2])
d['cycle'] = d.block.isin(half).astype(int)          # spatial-block half split, used only for the holdout check
d.to_csv(f'{W}/lucas.csv', index=False)
for k, v in log: print(f'{k:34s}{v}')
print('PCA cumulative variance: PC1 %.3f, PC3 %.3f, PC5 %.3f, PC10 %.4f, PC30 %.5f' % (ev[0], ev[2], ev[4], ev[9], ev[29]))
print('Y (OC g/kg):', d.Y.describe().round(1).to_dict(), 'skew %.2f' % d.Y.skew())
print('BIO1 quantiles:', d.BIO1.quantile([.02, .05, .1, .25, .5, .75, .9, .95, .98]).round(1).to_dict())
print('blocks:', len(blocks), 'points per block median', int(d.block.value_counts().median()), '; holdout halves', d.cycle.value_counts().to_dict())
print('missing in X:', d[['elev_dem', 'Slope', 'Aspect', 'BIO1', 'BIO12', 'Soil_Group']].isna().mean().round(3).to_dict())
print('corr Y with BIO1 %.2f, BIO12 %.2f, elev %.2f, PC1 %.2f, PC2 %.2f' % tuple(d.Y.corr(d[c]) for c in ['BIO1', 'BIO12', 'elev_dem', 'PC1', 'PC2']))
