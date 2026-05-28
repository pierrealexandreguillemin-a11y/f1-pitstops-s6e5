# ps-s6-e5-realmlp-pytabkit

## [cell 0 MD]
## Setup Libraries

## [cell 1 CODE]
```python
%%time
!pip install -q pytabkit
```

## [cell 2 CODE]
```python
import random
import warnings
import numpy as np, pandas as pd
from colorama import Fore, Style
from importlib.metadata import version
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
import pytabkit
from pytabkit import RealMLP_TD_Classifier

warnings.filterwarnings('ignore')
print("PyTorch  version:", torch.__version__)
print("PyTabKit version:", version("pytabkit"))

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(42)
```

## [cell 3 MD]
## Load Data

## [cell 4 CODE]
```python
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/train.csv")
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e5/test.csv")
orig = pd.read_csv("/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv")
print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Orig shape :", orig.shape)
```

## [cell 5 MD]
## Preprocess Features

## [cell 6 CODE]
```python
%%time
ID = 'id'
TARGET = 'PitNextLap'
orig = orig.drop(['Normalized_TyreLife'], axis=1)
y_orig = orig[TARGET]; orig = orig.drop([TARGET], axis=1)
X = train.drop([ID, TARGET], axis=1); train_id = train[ID]
y = train[TARGET]
X_test = test.drop([ID], axis=1); test_id = test[ID]
del train, test
print("X      init shape:", X.shape)
print("X_test init shape:", X_test.shape)
print("orig   init shape:", orig.shape, "\n")

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols), "\n")

category_map = {}
important_combos = [
    ('Race', 'Compound'),
    ('Race', 'Year'),
]
def feature_engineering(df, fit=False):
    # Arithmetic interaction
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
    df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')

    # Categorize numericals
    for col in num_cols + ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype(str)

    # Count encoding
    for col in cat_cols + ['Year_cat_', 'PitStop_cat_']:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # Discretize numericals
    bin_config = {'RaceProgress': [200], 'LapTime (s)': [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

    # Create interaction categories
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype(str)

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols, combo_names

X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, new_cat_cols, new_num_cols, combo_names = feature_engineering(X_test, fit=False)
orig, new_cat_cols, new_num_cols, combo_names = feature_engineering(orig, fit=False)
cat_cols += new_cat_cols; num_cols += new_num_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols), "\n")

print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols), "\n")
print("X      prep shape:", X.shape)
print("X_test prep shape:", X_test.shape)
print("orig   prep shape:", orig.shape, "\n")
```

## [cell 7 MD]
## Config

## [cell 8 CODE]
```python
class CFG:
    FOLDS = 5
    SEED = 42
    TE = True

params = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',

    'n_ens': 20,
    'n_epochs': 5,
    'batch_size': 256,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,

    'lr': 0.019,
    'wd': 0.01,
    'sq_mom': 0.99,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,

    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'invsqrtp1e-3',

    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,

    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',

    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}
```

## [cell 9 MD]
## Train K-Fold

## [cell 10 CODE]
```python
%%time
skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
          zip(skf.split(X, y), skf.split(orig, y_orig)), 1):
    X_tr = X.iloc[tr_idx].copy()
    orig_tr = orig.iloc[or_tr_idx].copy()
    X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
    y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
    X_val = X.iloc[val_idx].copy()
    y_val = y.iloc[val_idx]
    X_tst = X_test.copy()

    # Target encoding
    if CFG.TE:
        te_cols = combo_names
        TE = TargetEncoder(cv=CFG.FOLDS, smooth='auto', shuffle=True, random_state=CFG.SEED)
        tr_enc = TE.fit_transform(X_tr[te_cols], y_tr)
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])

        te_names = [f"_{col}TE" for col in te_cols]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

    if fold == 1: print("len(FEATURES):", len(X_tr.columns.tolist()), "\n")
    print("#"*16)
    print(f"### Fold {fold}/{CFG.FOLDS} ...")
    print("#"*16)

    model = RealMLP_TD_Classifier(**params)
    model.fit(X_tr, y_tr, X_val, y_val)

    val_preds = model.predict_proba(X_val)[:, 1]
    fold_test_preds = model.predict_proba(X_tst)[:, 1]

    oof_preds[val_idx] = val_preds
    test_preds += fold_test_preds / CFG.FOLDS

    fold_score = roc_auc_score(y_val, val_preds)
    print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | AUC Score: {fold_score:.5f}{Style.RESET_ALL}\n")
    torch.cuda.empty_cache()
```

## [cell 11 MD]
## Evaluation and Submission

## [cell 12 CODE]
```python
oof_score = roc_auc_score(y, oof_preds)
print("\n" + "="*24)
print(f"Overall OOF AUC: {Fore.BLACK}{Style.BRIGHT}{oof_score:.5f}{Style.RESET_ALL}")
print("="*24)

oof_df = pd.DataFrame({ID: train_id, TARGET: oof_preds})
oof_df.to_csv('oof_preds.csv', index=False)

sub = pd.DataFrame({ID: test_id, TARGET: test_preds})
sub.to_csv('submission.csv', index=False)
sub.head()
```

## [cell 13 CODE]
```python

```
