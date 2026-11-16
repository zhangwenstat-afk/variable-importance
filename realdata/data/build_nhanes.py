"""NHANES 2011-2016: OGTT analytic sample (Section 7.1).
Usage: python build_nhanes.py <dir with the downloaded .xpt files> <output dir>
"""
import sys, pandas as pd, numpy as np
R, W = sys.argv[1], sys.argv[2]
rd = lambda n: pd.read_sas(f'{R}/{n}.xpt')
parts = []
for s in 'GHI':
    d = rd(f'DEMO_{s}')[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR', 'DMDEDUC2']]
    b = rd(f'BMX_{s}')[['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXHT']]
    bp = rd(f'BPX_{s}'); bp['SBP'] = bp[['BPXSY1', 'BPXSY2', 'BPXSY3']].mean(axis=1); bp['DBP'] = bp[['BPXDI1', 'BPXDI2', 'BPXDI3']].mean(axis=1)
    bp = bp.rename(columns={'BPXPLS': 'PULSE'})[['SEQN', 'SBP', 'DBP', 'PULSE']]
    mq = rd(f'MCQ_{s}')[['SEQN', 'MCQ300C']]
    gl = rd(f'GLU_{s}'); gl = gl[['SEQN', 'LBXGLU'] + (['LBXIN'] if 'LBXIN' in gl else [])]
    if s != 'G': gl = gl.merge(rd(f'INS_{s}')[['SEQN', 'LBXIN']], on='SEQN', how='left')
    gh = rd(f'GHB_{s}')[['SEQN', 'LBXGH']]
    tr = rd(f'TRIGLY_{s}')[['SEQN', 'LBXTR', 'LBDLDL']]; hd = rd(f'HDL_{s}')[['SEQN', 'LBDHDD']]
    dq = rd(f'DIQ_{s}')[['SEQN', 'DIQ010', 'DIQ050', 'DIQ070']]
    og = rd(f'OGTT_{s}')[['SEQN', 'LBXGLT']]
    m = d
    for t in [b, bp, mq, gl, gh, tr, hd, dq, og]: m = m.merge(t, on='SEQN', how='left')
    m['cycle'] = 1 if s == 'I' else 0; m['wave'] = s; parts.append(m)
df = pd.concat(parts, ignore_index=True); log = [('all persons', len(df))]
df = df[df.RIDAGEYR >= 20]; log.append(('age>=20', len(df)))
df = df[df.LBXGLT.notna()]; log.append(('OGTT 2h glucose present', len(df)))
df = df[(df.DIQ010 != 1) & (df.DIQ050 != 1) & (df.DIQ070 != 1)]; log.append(('no diagnosed diabetes / meds', len(df)))
df = df[df.LBXGLU.notna()]; log.append(('fasting glucose present', len(df)))
df = df[df.BMXBMI.notna()]; log.append(('BMI present', len(df)))
df = df.rename(columns={'RIDAGEYR': 'age', 'RIAGENDR': 'sex', 'RIDRETH3': 'race', 'INDFMPIR': 'pir', 'DMDEDUC2': 'educ',
                        'BMXBMI': 'bmi', 'BMXWAIST': 'waist', 'BMXHT': 'height', 'LBXGLT': 'Y', 'LBXGLU': 'glu', 'LBXIN': 'ins',
                        'LBXGH': 'HbA1c', 'LBXTR': 'tg', 'LBDLDL': 'ldl', 'LBDHDD': 'hdl'})
df['educ'] = df.educ.where(df.educ <= 5)
df['famhist'] = df.MCQ300C.map({1.0: 1.0, 2.0: 0.0})
df = df.reset_index(drop=True)
for k, v in log: print(f'{k:32s}{v}')
print('by wave:', df.wave.value_counts().sort_index().to_dict())
print('Y (2h glucose, mg/dL):', df.Y.describe().round(1).to_dict())
print('Y>=140 (IGT or worse): %.3f   Y>=200 (DM by OGTT): %.3f' % ((df.Y >= 140).mean(), (df.Y >= 200).mean()))
print('missing rates:', df[['pir', 'educ', 'waist', 'SBP', 'famhist', 'ins', 'HbA1c', 'tg', 'ldl', 'hdl']].isna().mean().round(3).to_dict())
print('corr with Y:', df[['glu', 'HbA1c', 'ins', 'tg', 'hdl', 'age', 'bmi', 'waist']].corrwith(df.Y).round(2).to_dict())
df.to_csv(f'{W}/nhanes_ogtt.csv', index=False)
