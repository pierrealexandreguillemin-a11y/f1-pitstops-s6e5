# Top kernels synthesis — what makes 0.95+ on S6E5

> Date: 2026-05-28 | Triggered by: notre LB 0.94457 (top 68.4%) vs leaders 0.95493 (top 0.1%) | Sources pulled local : `docs/research/competitor_kernels/`

## TL;DR — 6 tricks universels pour passer 0.94 → 0.95+

1. **External dataset MANDATORY** : `aadigupta1601/f1-strategy-dataset-pit-stop-prediction` (`f1_strategy_dataset_v4.csv`). Présent dans 100% des top kernels analysés. Concaténé au train avec flag `IsOriginalData`.
2. **NOT GroupKFold** : top kernels utilisent `StratifiedKFold` ou `StratifiedGroupKFold` (jamais vanilla GroupKFold). Notre fold variance ±0.0084 est artefact de pessimisme — Stratification gérée par target = CV-LB gap réduit.
3. **FE row-level massive** : ~50+ features dérivées (domain, crosses, frequency, target encoding, group stats, digit/signature). Le 0.95 plateau est atteint via FE, pas via tuning.
4. **CatBoost params upgrades** : `depth=8, lr=0.018, l2_leaf_reg=8.5, auto_class_weights="Balanced", early_stopping_rounds=500, bootstrap_type="Bayesian"`.
5. **Seed ensembles** : 2-N seeds, average predictions.
6. **Synthetic-data exploits** (avancé) : digit features (chaque digit de chaque numeric en cat), signature features (scale*100 round), string-precision encoding. Exploite les fingerprints du générateur synthétique.

## Sources analysées (4 kernels, raw .ipynb dans `competitor_kernels/`)

| Kernel | Author | Votes | LB | Stack |
|---|---|---|---|---|
| `eda-predicting-f1-pit-stops` | Chris Deotte (GM) | 82 | n/a (EDA only) | matplotlib viz |
| `f1-best-catboost-solution-0-95259` | flexonafft | 48 | **0.95259** | CatBoost + ensemble seeds + stacking |
| `ps-s6-e5-realmlp-pytabkit` | Vladimir Demidov | **126** | n/a (training nb) | RealMLP (NN tabulaire) via PyTabKit |
| `f1-pit-stops-blender-0-95454` | Raunak Dey | 85 | **0.95454** | Pure post-hoc CSV blending (hors scope péda) |

## Détails par trick

### Trick #1 — External Aadi Gupta dataset

URL : <https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction>
File: `f1_strategy_dataset_v4.csv`
Status: déjà autorisé par le règlement competition (white-listed external data).

**Pourquoi ça aide** : c'est la version "clean" du dataset, chronologie intra-driver préservée (cf EDA #02). Concaténer au train apporte signal time-series **valide** (que le shuffled competition data détruit). Le flag `IsOriginalData` permet à CatBoost de partitionner — la portion shuffled vs original est traitée différemment.

**Implémentation** (flexonafft) :
```python
train["IsOriginalData"] = 0
original["IsOriginalData"] = 1
original = original.drop(columns=["Normalized_TyreLife"], errors="ignore")  # col extra à drop
# Concat en TRAIN time (pas en val) pour augmenter le signal
X_tr = pd.concat([X_tr_comp, X_orig], axis=0)
```

### Trick #2 — Stratified split au lieu de GroupKFold

Justifications :
- Notre EDA #01 montrait que GroupKFold est pessimiste vs LB (généralisation à course inconnue vs laps inconnus dans même course).
- Top kernels utilisent `StratifiedKFold(5, shuffle=True, seed=42)` random row split OR `StratifiedGroupKFold(5)` (groups=Race+Year, stratify=target).
- flexonafft mode prod = `single_split` (80/20 stratifié), `run_validation: False`, iter=8140 fixe — ils sont si confiants dans leur CV qu'ils skip validation en final training.

**Impact sur notre cas** : on est passé de OOF 0.929 → LB 0.945 (+1.5pp). Stratified random split nous ferait OOF ~0.945 (matche LB). Le gap CV-LB serait ~0, on pourrait shipper directement après CV.

### Trick #3 — Row-level FE catalog (flexonafft, ~50 features)

**Domain (10)** :
- `EstimatedTotalLaps = LapNumber / RaceProgress` (clip 1-120)
- `LapsRemaining = EstimatedTotalLaps - LapNumber`
- `RemainingRaceProgress = 1 - RaceProgress`
- `LapProgress_x_LapNumber = LapNumber * RaceProgress`
- `Early_Race`, `Mid_Race`, `Late_Race` (binaires sur RaceProgress thresholds)
- `RacePhase` (5-bin cat sur RaceProgress)
- `LapBin` (6-bin cat sur LapNumber)
- `PitWindowPressure = TyreLife * RaceProgress`

**Tyre (8)** :
- `TyreAgeRatio = TyreLife / LapNumber`
- `LapPerTyreLife = LapNumber / (TyreLife + 1)`
- `TyreLifeMinusLap`, `LapMinusTyreLife`
- `TyreAgeVsRace = TyreLife / EstimatedTotalLaps`
- `TyreLife_x_RaceProgress` (redondant avec PitWindowPressure)
- `TyreLifeBin` (6-bin cat)
- `TyreLife_to_LapsRemaining` ratio

**Stint (5)** :
- `StintPressure = Stint * TyreLife`
- `TyreLife_x_Stint`
- `Is_First_Stint` (Stint==1)
- `Is_Late_Stint` (Stint>=3)
- `Stint_x_LapNumber`

**Degradation/Delta (8)** :
- `DegPerRaceLap`, `DegPerTyreLap`, `AbsDegPerTyreLap`
- `Abs_Cumulative_Degradation`, `Positive_Degradation`
- `DeltaAbs`, `LapTimeDeltaPositive`, `LapTimeDeltaNegative`
- `DeltaPerTyreLap`

**Position (4)** :
- `Abs_Position_Change`
- `Gained_Position`, `Lost_Position`
- `PositionPressure = Position * RaceProgress`
- `PositionBin` (4-bin cat)

**Cross-categorical (9)** — concat strings :
- `Race_Year`
- `Compound_Stint`
- `Driver_Race`
- `Driver_Compound`
- `Race_Compound`
- `Race_Compound_Stint`
- `Compound_RacePhase`
- `Compound_TyreLifeBin`
- `RacePhase_TyreLifeBin`

### Trick #4 — Frequency encoding (sur train+test+original combiné)

Pour chaque cat (Driver, Race, Compound, plus les 9 crosses) :
```python
counts = pd.concat([train[col], test[col], original[col]]).value_counts()
total = sum(len(x) for x in [train, test, original])
df[f"{col}_count"] = df[col].map(counts).astype(np.int32)
df[f"{col}_freq"] = df[f"{col}_count"] / total
```

**Pourquoi ça aide** : transforme la cardinalité 887 de `Driver` en 2 features numériques (count + freq) que CatBoost peut splitter, sans subir le full one-hot ou target encoding.

### Trick #5 — Group statistics (mean/std/diff par cat)

Pour chaque `group_col ∈ {Race_Year, Race_Compound_Stint, Driver_Race, Compound_Stint}` × `value_col ∈ {LapTime_Delta, Position_Change, RaceProgress, TyreLife}` :
```python
stats = combined.groupby(group_col)[value_col].agg(["mean", "std"])
df[f"{value_col}_mean_by_{group_col}"] = df[group_col].map(stats["mean"])
df[f"{value_col}_std_by_{group_col}"]  = df[group_col].map(stats["std"])
df[f"{value_col}_diff_mean_by_{group_col}"] = df[value_col] - df[f"{value_col}_mean_by_{group_col}"]
```

4 group_cols × 4 value_cols × 3 stats = **48 group features**. CatBoost adore.

### Trick #6 — Synthetic-data exploits (advanced)

**Digit features** : décompose chaque feature numérique en ses digits individuels :
```python
out[f"{col}_int_digit_1"] = (values // 1) % 10   # digit des unités
out[f"{col}_int_digit_2"] = (values // 10) % 10  # digit des dizaines
out[f"{col}_int_digit_3"] = (values // 100) % 10 # digit des centaines
out[f"{col}_dec_digit_1"] = ((values * 10).round() % 10)
out[f"{col}_dec_digit_2"] = ((values * 100).round() % 10)
```

Le générateur synthétique laisse des fingerprints au niveau du digit (modes de génération, biases d'arrondi). CatBoost peut splitter sur "digit des unités de LapTime est 0" et capter le pattern.

**Signature features** : `(value * 100).round() % 10^i` traité comme cat string.

**String precision encoding** : `df["RaceProgress_str"] = df["RaceProgress"].round(4).astype(str)`. Idem `EstimatedTotalLaps` (1 decimal), `TyreAgeRatio` (3 decimals).

→ Ces tricks captent des artefacts du générateur synthétique. C'est probablement ce qui explique le plateau dur à 0.954-0.955.

### CatBoost params optimisés (flexonafft, LB 0.95259)

```python
{
    "iterations": 11000,                # vs 1000 baseline
    "learning_rate": 0.018,             # vs auto
    "depth": 8,                         # vs 6 default
    "l2_leaf_reg": 8.5,                 # vs 3 default
    "random_strength": 0.65,            # vs 1 default
    "bootstrap_type": "Bayesian",       # vs Bernoulli default
    "bagging_temperature": 0.45,        # Bayesian-specific
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",   # vs None baseline (test impact!)
    "task_type": "GPU",                 # vs CPU
    "early_stopping_rounds": 500,       # vs 50 baseline (10x more patience)
}
```

**Final training** : iterations fixées via `max(1800, mean(best_iters) * 1.12)` — entraîne 12% plus longtemps que la mean OOF best_iter, sur full train+original (pas de val set).

### RealMLP variant (yekenot)

Path alternative : RealMLP_TD_Classifier (PyTabKit), modèle neural net spécialisé tabulaire.
- `n_ens=20` (ensemble interne de 20 modèles)
- `hidden_sizes=[512, 256, 128]`, `act=silu`
- `lr=0.019`, `wd=0.01`
- 5 epochs, no early stopping
- Target Encoding sur les combos Race_Compound + Race_Year
- KBinsDiscretizer 200 bins sur RaceProgress (!) — confirme l'exploitation fingerprint
- `floor()` + `factorize()` pour discrétiser tous les numerics → cat string

Pas notre stack mais utile à savoir pour submit-03 alternative.

## Implication pour notre roadmap

| Submit | Action | Cible OOF | Cible LB | Effort |
|---|---|---|---|---|
| **01 ✓ shipped** | CatBoost defaults + GroupKFold | 0.9294 | 0.94457 | 1.5h |
| **02 strategy A — incremental FE** | + Aadi Gupta dataset + 30 features FE (domain+crosses+freq) + StratifiedKFold + params tunés | ~0.95 OOF | 0.950-0.953 | 4-6h |
| **02 strategy B — full flexonafft replica** | + Aadi Gupta + 80 features (incl. digit/signature) + StratifiedKFold + params tunés + seed ensemble (42, 777) | ~0.952 OOF | 0.952-0.953 | 6-10h |
| **03 (optionnel J-1)** | + meta CatBoost stacking sur OOF | ~0.953 OOF | 0.953-0.954 | 4-6h |

**Recommandation submit-02 = strategy A.** Incremental, beats baseline of 0.94 nettement, garde de la marge pour 02b/03 si on a le temps. Strategy B = couvre tout d'un coup mais explore beaucoup d'inconnues en parallèle (debug nightmare si OOF ≠ LB).

Pour atteindre **top 5% (0.954+)** il faudrait submit-03 stacking + ensemble seeds + tuning Optuna sur tout ça. Probablement hors-scope en J-3 mais on verra après 02.

---

**Décision attendue user** : confirmer strategy A submit-02 (incremental), ou pivot vers B (full replica), ou autre.
