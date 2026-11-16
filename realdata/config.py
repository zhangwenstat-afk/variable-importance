"""Data sets of Section 7. Paths are relative to this folder.

X: cheap covariates; CAT: columns treated as categorical by the trees;
Z: the additional covariates compared with X alone; GRIDS: S and its s0 grid;
folds: how the two cross-fitting folds are formed.
"""
DATASETS = {
    'nhanes': dict(  # Section 7.1, Y = OGTT two-hour glucose (mg/dL)
        file='data/nhanes_ogtt.csv',
        X=['age', 'sex', 'race', 'pir', 'educ', 'bmi', 'waist', 'height', 'SBP', 'DBP', 'PULSE', 'famhist', 'cycle'],
        CAT={'sex', 'race', 'cycle'},
        Z={'fpg': ['glu'], 'hba1c': ['HbA1c'], 'nonglyc': ['ins', 'tg', 'ldl', 'hdl']},
        GRIDS={'age': list(range(25, 80, 5)), 'bmi': [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5]},
        folds='random'),
    'sonreb': dict(  # Section 7.2, Y = compressive strength (MPa)
        file='data/sonreb.csv',
        X=['RN', 'study_c', 'log_age', 'ndt_type_c', 'comp_c', 'orient_c', 'rebar', 'agg', 'wc', 'fd'],
        CAT={'study_c', 'ndt_type_c', 'comp_c', 'orient_c'},
        Z={'upv': ['Vp']},
        GRIDS={'RN': [20, 24, 28, 32, 36, 40, 44]},
        folds='by_study'),
    'lucas': dict(  # Section 7.2, Y = soil organic carbon (g/kg); restricted data
        file='data_restricted/lucas/lucas.csv',
        X=['xkm', 'ykm', 'elev_dem', 'Slope', 'asp_sin', 'asp_cos'] + [f'BIO{k}' for k in range(1, 20)]
          + ['CLIMA_Cod_c', 'BioGeo_c', 'Soil_Group_c', 'NUTS_0_c'],
        CAT={'CLIMA_Cod_c', 'BioGeo_c', 'Soil_Group_c', 'NUTS_0_c'},
        Z={'spectra': [f'PC{k}' for k in range(1, 31)]},
        GRIDS={'BIO1': [2, 4, 6, 8, 10, 12, 14, 16]},
        folds='spatial_blocks'),
    'warfarin': dict(  # Section 7.2, Y = stable weekly warfarin dose (mg/week)
        file='data/warfarin.csv',
        X=['Weight', 'Height', 'Age', 'Enzyme', 'Amiodarone', 'Gender', 'Black', 'Asian'],
        CAT=set(),
        Z={'vkorc1': ['VKORC1_AG', 'VKORC1_AA'], 'cyp2c9': ['CYP2C9_12', 'CYP2C9_13', 'CYP2C9_other']},
        GRIDS={'Weight': [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]},
        folds='random'),
}

# Bandwidth h = c * sqrt(3) * sd(S) * n^{-1/4} (Semi) or n^{-2/5} (Conf). The constants in CHOSEN are the
# ones used in the paper; CS is the grid checked by the holdout coverage (*_holdout.csv).
CS = [0.75, 1.0, 1.5, 2.0, 3.0]
CHOSEN = {'semi': 2.0, 'conf': 3.0, 'conf_hsemi': 2.0}
SPLITS = 20          # random two-fold cross-fitting splits, aggregated by the median
