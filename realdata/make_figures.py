"""Figures 6-9 of Section 7 from results/*_a0.1_final.csv.

    python make_figures.py [output dir]      (default: figures/ next to this file)

ogtt_delta_s0.pdf        Figure 6, NHANES estimates with one-sided 95% lower bounds
ogtt_decision_map.pdf    Figure 7, NHANES decision map over (s0, tau)
further_delta_s0.pdf     Figure 8, concrete, soil carbon and warfarin estimates
further_decision_map.pdf Figure 9, decision maps for the three further data sets
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures')
Z95 = 1.6448536269514722
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'hatch.linewidth': 0.8})
# Semi / Conf styles follow the simulation figures (Figures 1 and 3)
COL = {'semi': '#1f77b4', 'conf': '#ff7f0e'}; MK = {'semi': 'o', 'conf': 's'}; LS = {'semi': '-', 'conf': '--'}; LAB = {'semi': 'Semi', 'conf': 'Conf'}
# decision-map categories, coded 2*[Semi rejects] + [Conf rejects]; Okabe-Ito colours plus hatching
CAT_COL = ['#f2f2f2', '#E69F00', '#0072B2', '#3d3d3d']
CAT_HATCH = ['', '\\\\\\', '///', '']
CAT_LAB = ['Neither', 'Conf only', 'Semi only', 'Both']


def load(name):
    T = pd.read_csv(os.path.join(HERE, 'results', f'{name}_a0.1_final.csv'))
    T = T[(T.S != 'marginal') & T.method.isin(['semi', 'conf'])].copy()
    T['L'] = T.delta - Z95 * T.se          # one-sided 95% lower bound; H0: Delta <= tau rejected iff L > tau
    return T


def panel(x, T):
    for m in ['conf', 'semi']:
        t = T[T.method == m].sort_values('s0')
        x.fill_between(t.s0, t.L, t.delta + Z95 * t.se, color=COL[m], alpha=.14, lw=0)
        x.plot(t.s0, t.delta, LS[m], color=COL[m], marker=MK[m], ms=4.0, lw=1.6, label=f'{LAB[m]} estimate')
        x.plot(t.s0, t.L, LS[m], color=COL[m], lw=0.9, alpha=.9, label=f'{LAB[m]} lower bound')
    x.axhline(0, color='0.45', lw=.8)


def dmap(x, T, top, xlabel, show_y=True):
    g = sorted(T.s0.unique())
    Ls = T[T.method == 'semi'].set_index('s0').L.reindex(g).values; Lc = T[T.method == 'conf'].set_index('s0').L.reindex(g).values
    for k in range(len(g)):
        cuts = sorted({0.0, top, *[min(max(v, 0.0), top) for v in (Ls[k], Lc[k])]})   # category changes only at the bounds
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi <= lo: continue
            mid = (lo + hi) / 2; c = int(Ls[k] > mid) * 2 + int(Lc[k] > mid)
            x.add_patch(Rectangle((k - .5, lo), 1, hi - lo, facecolor=CAT_COL[c], hatch=CAT_HATCH[c],
                                  edgecolor='white' if c in (1, 2) else CAT_COL[c], lw=0))
    for v in np.arange(len(g) + 1) - .5: x.axvline(v, color='white', lw=1.0)
    x.set_xlim(-.5, len(g) - .5); x.set_ylim(0, top)
    x.set_xticks(range(len(g))); x.set_xticklabels([f'{v:g}' for v in g], fontsize=8.5); x.set_xlabel(xlabel)
    if show_y: x.set_ylabel(r'$\tau$ (reduction in half-width)')
    for sp in x.spines.values(): sp.set_visible(False)


def map_legend(fig):
    fig.legend(handles=[Patch(facecolor=CAT_COL[i], hatch=CAT_HATCH[i], edgecolor='white' if i in (1, 2) else '0.6', label=CAT_LAB[i])
                        for i in [3, 2, 1, 0]], loc='lower center', ncol=4, frameon=False, fontsize=9, handlelength=2.6, handleheight=1.4)


def nhanes():
    T = load('nhanes')
    ZL = {'fpg': 'Fasting glucose', 'hba1c': 'HbA1c', 'nonglyc': 'Insulin and lipids'}; SL = {'age': 'Age (years)', 'bmi': 'BMI (kg/m$^2$)'}
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.8), sharey='row')
    for i, S in enumerate(['age', 'bmi']):
        for j, z in enumerate(ZL):
            panel(ax[i, j], T[(T.S == S) & (T.Z == z)])
            if i == 0: ax[i, j].set_title(ZL[z], fontsize=10.5)
            ax[i, j].set_xlabel(f'$s_0$: {SL[S]}')
            if j == 0: ax[i, j].set_ylabel(r'$\widehat\Delta(s_0)$ (mg/dL)')
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig(f'{FIG}/ogtt_delta_s0.pdf'); plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for i, S in enumerate(['age', 'bmi']):
        for j, z in enumerate(['fpg', 'hba1c']):
            dmap(ax[i, j], T[(T.S == S) & (T.Z == z)], 10.0, f'$s_0$: {SL[S]}', show_y=(j == 0))
            if i == 0: ax[i, j].set_title(ZL[z], fontsize=10.5)
    map_legend(fig); plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig(f'{FIG}/ogtt_decision_map.pdf'); plt.close()


def further():
    D = {'sonreb': ('RN', 'rebound number', 'MPa', 0.25), 'lucas': ('BIO1', r'mean annual temperature ($^\circ$C)', 'g/kg', 18.0),
         'warfarin': ('Weight', 'weight (SD units)', 'mg/week', 3.0)}
    T = {k: load(k) for k in D}; bold = dict(fontsize=11, fontweight='bold')
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for x, (k, z, tt) in zip(ax.ravel(), [('sonreb', 'upv', '(a) Concrete strength'), ('lucas', 'spectra', '(b) Soil organic carbon'),
                                          ('warfarin', 'vkorc1', '(c) Warfarin, VKORC1'), ('warfarin', 'cyp2c9', '(d) Warfarin, CYP2C9')]):
        panel(x, T[k][T[k].Z == z]); x.set_title(tt, **bold); x.set_xlabel(f'$s_0$: {D[k][1]}'); x.set_ylabel(rf'$\widehat\Delta(s_0)$ ({D[k][2]})')
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig(f'{FIG}/further_delta_s0.pdf'); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for x, (k, z, tt) in zip(ax, [('sonreb', 'upv', '(a) Concrete strength'), ('lucas', 'spectra', '(b) Soil organic carbon'), ('warfarin', 'vkorc1', '(c) Warfarin, VKORC1')]):
        dmap(x, T[k][T[k].Z == z], D[k][3], f'$s_0$: {D[k][1]}'); x.set_title(tt, **bold)
        x.set_ylabel(rf'$\tau$ ({D[k][2]}, reduction in half-width)')
    map_legend(fig); plt.tight_layout(rect=(0, .08, 1, 1)); plt.savefig(f'{FIG}/further_decision_map.pdf'); plt.close()


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True); nhanes(); further(); print('figures written to', FIG)
