"""Stream LUCAS 2015 spectra from the zip: average replicate scans per PointID, 10 nm bin means."""
import sys, zipfile, numpy as np, pandas as pd
ZIP, OUT = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(ZIP); ids, rows = [], []
for name in sorted(z.namelist()):
    if not name.endswith('.csv'): continue
    for ch in pd.read_csv(z.open(name), chunksize=4000):
        wl = np.array([float(c) for c in ch.columns[5:]]); A = ch.iloc[:, 5:].to_numpy(np.float32)
        g = pd.DataFrame(A).groupby(ch.PointID.values).mean()
        b = ((wl - 400) // 10).astype(int)          # 210 bins of 10 nm
        B = np.stack([g.values[:, b == k].mean(1) for k in range(b.max() + 1)], 1)
        ids.append(g.index.values); rows.append(B)
    print(name, flush=True)
P = np.concatenate(ids); X = np.concatenate(rows)
df = pd.DataFrame(X).groupby(P).mean()   # a point split across chunks is averaged again
np.savez_compressed(OUT, pid=df.index.values, A=df.values.astype(np.float32), wl=400 + 10 * np.arange(df.shape[1]) + 4.75)
print('points', len(df), 'bands', df.shape[1])
