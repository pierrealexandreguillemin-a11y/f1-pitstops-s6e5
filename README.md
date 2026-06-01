# f1-pitstops-s6e5

Kaggle Playground Series S6E5 — **Predicting F1 Pit Stops** (binary AUC).
Sponsor : Google LLC. Tier : Swag (1 t-shirt). Deadline : **2026-05-31 23:59 UTC**.

## Goal

Submit a CSV of probabilistic predictions (`id,PitNextLap ∈ [0,1]`) for the 188 165 test rows. AUC scoring. 10 submissions/day allowed, 2 final selectable for private leaderboard.

Not a leaderboard hunt — pedagogical journey through tabular ML techniques on a real synthetic dataset, with the journey documented in commits + this README.

## Stack

| Layer | Tool |
|---|---|
| Python | 3.13.13 venv-scoped (`.venv/`) |
| Tabular | pandas 2.3.3, numpy, scikit-learn, scipy |
| Boosting | catboost 1.2.10, lightgbm 4.6.0 |
| Plots | matplotlib, seaborn |
| Kaggle | `kaggle` CLI 2.1.2 (submission via `.venv/Scripts/kaggle.exe`) |
| Dev | ruff, mypy, pre-commit, pytest, detect-secrets |

Pas de jupyter, pas de notebooks, pas de torch/lifelines/pycox. Tout en `.py` purs.

## Setup (one-shot)

```bash
cd C:/Dev/f1-pitstops-s6e5
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pre_commit install
.venv/Scripts/python.exe -m pre_commit install --hook-type pre-push
.venv/Scripts/kaggle.exe competitions download -c playground-series-s6e5 -p data/raw/
# unzip data/raw/playground-series-s6e5.zip in place, then remove the zip
```

## Plan v2 (3 submissions max, J-4)

| # | Approach | Script | What it teaches |
|---|---|---|---|
| 01 | **CatBoost defaults + GroupKFold(Race+Year)** | `scripts/run_01_baseline.py` | Baseline-first principle, group-aware CV, AUC anchor |
| 02 | **+ Row-level FE** (Compound×Stint/Year, TyreLife/Stint, LapTime vs Race-median, target-enc Driver) | `scripts/run_02_row_fe.py` | FE vs hyperparam tuning ROI on tabular |
| 03 | **+ Ensemble LightGBM** or **external F1 dataset augmentation** (time-permitting) | `scripts/run_03_*.py` | Model diversity vs more data |

> **Data constraint** (found 2026-05-28, see `docs/eda/2026-05-28-data-integrity.md`) : the competition dataset is row-shuffled at the `(Race, Year, Driver)` level — **62% of train groups have non-monotonic `Stint`**, 12.75% of consecutive-lap transitions show Stint decreasing. Intra-stint `TyreLife` is preserved (100%). All time-series FE (`laps_since_pit`, rolling lap-time, undercut, stint progression) is **broken** and excluded from submit-02. Only row-level FE.

Ship a submission only if `AUC_CV ≥ baseline + 0.5pp` AND `paired fold majority positive`. CV-LB gap = data point, not a target. Expected fold variance ~±0.010 under GroupKFold (data noise, not bug — confirmed by other competitors).

## Tracking table

| # | Date | Approach | OOF AUC | Public LB | Δ vs baseline | Key insight |
|---|------|----------|---------|-----------|---------------|-------------|
| 01 | 2026-05-28 | CatBoost defaults + GroupKFold(Race+Year) | 0.9294 | **0.94457** | — (anchor) | CV-LB gap +1.52pp (GroupKFold pessimiste); LB rank #1849/2703 (top 68%); median 0.948, top 0.9549 → baseline insuffisant, need FE |
| 02 | 2026-05-29 | **Plan-2 "killer FE"** : (Y,R,D) aggregates + Aadi + 9 crosses + freq enc + StratifiedKFold + CatBoost 16k iter | 0.9794 | **0.94296** | **−0.16pp (RÉGRESSION)** | **CV MIRAGE** : CatBoost categorical-CTR overfit la CV random-row (mesuré : numeric-only OOF 0.9433≈LB, all-cats 0.9752). OOF non fiable. Post-mortem : [`docs/research/2026-05-29-submit-02-cv-lb-gap-postmortem.md`](docs/research/2026-05-29-submit-02-cv-lb-gap-postmortem.md) |
| 03 | 2026-05-29 | **Plan-2c** : Plan-2 MINUS les 5 crosses high-card (cause mesurée du mirage) + class weights, GPU 8k iter. | 0.95676 | **0.95083** | **+0.63pp (MEILLEUR)** | **DIAGNOSTIC CONFIRMÉ PAR LE LB** : droguer les 5 crosses high-card bat submit-01 de +0.63pp ET resserre le gap OOF-LB de 3.6pp→0.6pp (0.957 vs 0.951 → OOF redevient fiable). Les crosses high-card *étaient* la cause du mirage CatBoost-CTR. Plan-2b (all-cats, OOF 0.9754) = mirage confirmé, non soumis. |
| 04 | 2026-05-30 | **LightGBM diversity blend** : LightGBM 2-seed sur features Plan-2c (mêmes folds, Aadi inclus) blendé avec CatBoost Plan-2c. `scripts/run_04_lgbm_blend.py`. | LGBM 0.94799 / blend 0.95676 | non soumis | **+0.000pp (DEAD END)** | **Confirme 699182 sur nos données** : ρ(CatBoost,LGBM)=0.958 → blend optimal w=1.00 (= CatBoost pur), LightGBM intégralement jeté. Deux GBDT ne se blendent pas. Le seul levier de gain = diversité d'architecture (RealMLP). |
| 05 | 2026-05-30 | **Diagnostic ablation Aadi** : A/B WITH vs NO-AADI, LightGBM 1-seed mêmes folds. `kernels/run_05_aadi_ablation` (Kaggle CPU). | WITH 0.94776 / NO 0.94698 | diagnostic | **−0.078pp si drop (GARDER Aadi)** | 696380 a mesuré 25% de labels Aadi altérés → testé si on injecte du bruit. Verdict : Aadi **aide** (+0.078pp, toutes années) malgré le bruit. Cohérent 699182. Branche drop-Aadi morte. |
| 06 | 2026-05-30 | **RealMLP (PyTabKit) SEUL** : recette yekenot, StratifiedKFold seed=42 + Aadi, GPU T4. `kernels/run_06_realmlp`. | 0.95409 | **0.95407** | **+0.95pp (MEILLEUR)** | **RENVERSEMENT LB** : RealMLP seul bat CatBoost Plan-2c de **+0.324pp** (0.95407 vs 0.95083) ALORS QUE l'OOF disait l'inverse (0.954 < 0.957). **CatBoost overfit la CV random-row bien plus que RealMLP** → sur le vrai test, le NN est le moteur. Gap OOF-LB RealMLP ≈ 0 (fiable). |
| 07 | 2026-05-30 | **Blend isotonique CatBoost + RealMLP** (w=0.73, puis 0.50). `scripts/run_07_isotonic_blend.py`. | 0.95726 | 0.95269 (w.73) / 0.95365 (w.50) | sous RealMLP seul | **Monotone : moins de CatBoost = meilleur LB** (w1.0→0.95083, w.73→0.95269, w.50→0.95365, w0→0.95407). Les blends sont *tirés vers le bas* par CatBoost. Le poids OOF-optimal (0.73) était trop CatBoost-lourd. Vrai blend gagnant = RealMLP + TabPFN (non-GBDT), CatBoost ~0. |
| 06v2 | 2026-05-30 | **RealMLP-v2** : config research-backed (`ls_eps=0` + `n_epochs=16`, PyTabKit docs). `kernels/run_06_realmlp`. | 0.95396 | 0.95398 | ≈ v1 (−0.009pp) | **Tuner RealMLP ne le décorrèle PAS** : corr(v1,v2)=0.999. Le changement de config est neutre sur LB. v1 reste le best single. Conséquence : pour un gain en blend il faut une *autre architecture*, pas une autre config. |
| 08 | 2026-05-30 | **TabPFN-v2** (foundation model in-context) : 10k contexte + bagging×4, RAW features. `kernels/run_08_tabpfn`. | 0.93381 | **0.93318** | −1.14pp vs RealMLP | **OOF honnête (≈LB, ≠ mirage CatBoost) mais FAIBLE** : subsample 10k/350k + 14 features raw l'affament. Décorrélé (corr 0.919 vs RealMLP) mais 2.1pp trop faible. ⚠️ Coûteux : 1ʳᵉ config à 100k contexte = hors-specs (limite design 10k, arxiv 2502.17361) → 2h24 T4 gâchées avant fix. Leçon : `--accelerator NvidiaTeslaT4` obligatoire (pool peut donner P100/sm_60 incompatible torch cu128). |
| 09 | 2026-05-30 | **Blend isotonique RealMLP-v1 + TabPFN** w_tp=0.10. `scripts/` inline. | 0.95433 | 0.95399 | sous RealMLP seul (−0.008pp) | **Gain OOF non transféré** : OOF +0.024pp mais LB −0.008pp. Un modèle faible (même décorrélé) prend un poids ~0 utile. Confirme : la diversité ne paie QUE si le partenaire est fort. TabPFN écarté. |
| 10 | 2026-05-30 | **TabM** (PyTabKit, ICLR 2025) : 2ᵉ NN fort, archi différente. Même pipeline que submit-06. `kernels/run_10_tabm`. | 0.95211 | non soumis seul | — | **NN FORT confirmé** (0.952, à 0.002 de RealMLP, ≠ TabPFN faible) MAIS **trop corrélé** : corr(RealMLP,TabM)=**0.989** (deux MLP/même FE). Run lent (~110min, pas d'early-stop = mon erreur de config). |
| 11 | 2026-05-30 | **Blend isotonique RealMLP-v1 + TabM** w_tm=0.25. | 0.95429 | 0.95398 | sous RealMLP seul (−0.009pp) | **3ᵉ échec de diversité** : OOF +0.011pp non transféré, comme TabPFN. À corr 0.989 le blend = clone de RealMLP. **Les 3 familles (GBDT/foundation/MLP) épuisées → RealMLP-v1 seul (0.95407) = plafond LB public.** |
| — | conclu | **Plan-Ensemble** (ex-"submit-03") : diversité + blend isotonic. [`docs/plans/2026-05-28-plan-ensemble-FROZEN.md`](docs/plans/2026-05-28-plan-ensemble-FROZEN.md) | — | — | — | **CONCLU** : isotonic blend opérationnel mais AUCUN partenaire (CatBoost ρ.958 / TabPFN ρ.919-faible / TabM ρ.989) ne bat RealMLP-seul. Le gain OOF des blends ne transfère jamais. Plateau LB ~0.954. |

| 12 | 2026-05-31 | **RealMLP + agrégats target (Y,R,D,Compound)** in-data, leak-free `TargetEncoder(cv=5)`. `kernels/run_12_realmlp_agg` (D10 lever 1). | 0.95399 | 0.95401 | tie (−0.00006 = 0.08·SE) | **Agrégats REDONDANTS** : OOF plat, LB = égalité statistique avec RealMLP-v1. Confirme le plafond. **SE(AUC) public ≈ 0.00078** → l'écart à #1 (0.0009) = 1.15·SE (bruit). Données synthétiques+anonymisées (Driver 887 hashé) → pas de levier externe non plus. |
| 13 | 2026-05-31 | **Probe TabR fold0 partial** (full-430k, retrieval) + RealMLP-v1 rank-blend w=0.20. `kernels/run_13_tabr`. | OOF 0.95025 (TabR) | **0.95411** | **+0.04pp** | **1ʳᵉ diversité non-dégradante** : ρ=0.945 (full) le plus décorrélé du projet. Premier blend > RealMLP-v1. Privé 0.95428. |
| 14-15 | 2026-05-31 | **TabR full-430k / TabPFN-rich** (scope coupé deadline : TabR subsample 100k, TabPFN 2 bags). `kernels/run_14_15`. | — | diagnostic | — | Scope réduit pour tenir la deadline (TabR O(N) ~270s/epoch full vs 36s 100k). A2 TabPFN-rich = négatif (voir row 16-bis). |
| 16 | 2026-05-31 | **TabR 3-fold 100k + RealMLP-v1** rank-blend w=0.10. `kernels/run_16_tabr3fold`. **→ FINAL A.** | TabR ρ=0.901 | **0.95414** | **+0.07pp** | **Cohérence 4/4 blends TabR ≥ v1.** Privé **0.95423**. Sélectionné comme final A. |
| 17 | 2026-05-31 | **TabR 5-fold 100k + RealMLP-v1** rank-blend w=0.10 (repro confirm). `kernels/run_17_tabr5fold`. | TabR ρ=0.914 | **0.95415** | **+0.08pp** | **Meilleur public du projet.** Privé 0.95426 (> final A retenu — mais 0.06·SE = bruit, 0 impact rang). Repro confirme submit-16. |
| A2 | 2026-05-31 | **TabPFN-rich FE** (14→41 features) seul. Test : la richesse FE dé-affame-t-elle TabPFN ? | — | **0.93485** | −1.9pp | **NÉGATIF** : FE riche n'a rien dé-affamé, TabPFN reste faible ici. Hypothèse D7 réfutée. |
| B2 | 2026-05-31 | **Structural aggregates** (reconstruction d'ordre lap). | OOF 0.95394 | non soumis | sous v1 | **NÉGATIF + voie MORTE** : ρ.99 redondant ; `StintStartLap` incohérent → reconstruction d'ordre impossible (confirme EDA-02 : 62% Stint non-monotone). |

**Finals SÉLECTIONNÉS (deadline passée, réouverts puis confirmés)** : **A = blend RealMLP-v1 + TabR 3-fold `16` (public 0.95414 / privé 0.95423)** · **B = RealMLP-v1 seul (public 0.95407 / privé 0.95412)** = blend gagnant + moteur pur décorrélé en hedge LB-privé. TabR = 4ᵉ famille (retrieval), 1er partenaire non-dégradant du projet.

## Résultat final — 523ème / shakeup −40 places

Compétition **fermée 2026-05-31 23:59 UTC**. Classement final **523ème** (privé), −40 places vs public — shakeup modéré, dans le bruit du plateau (top-464 tenait en ~4.4·SE, voir D10).

| | Public | Privé |
|---|---|---|
| **Final A** — submit-16 blend RealMLP+TabR | 0.95414 | **0.95423** |
| **Final B** — RealMLP-v1 seul | 0.95407 | 0.95412 |
| Meilleur privé non sélectionné (submit-13) | 0.95411 | 0.95428 |
| 🥇 1ère place (vainqueur) | 0.95488 | 0.95503 |

**Lecture honnête** : (1) le pari TabR a payé en privé aussi — nos 3 meilleurs scores privés sont les 3 blends TabR ; (2) A > B en privé (le blend bat le moteur, **cohérent avec le public** → décision robuste, pas de retournement) ; (3) nos deux meilleures submissions privées (submit-13 0.95428, submit-17 0.95426) **n'étaient PAS sélectionnées** — écart à A retenu = 0.00005 ≈ 0.06·SE = bruit pur, 0 impact rang (thèse du « nuage de bruit » D10 confirmée : impossible de trapper le meilleur final dans ce régime). La discipline LB-only + plafond reconnu + no-grinding nous a protégés d'une vraie dégringolade de shakeup.

Filled after each submit. Empty cells = not yet measured (never "expected").

## Run a submission

```bash
# 1. Train + predict + write CSV
.venv/Scripts/python.exe scripts/run_NN_<approach>.py
# (prints OOF AUC per fold + mean ± std + Brier ; writes submissions/NN_<approach>.csv)

# 2. Submit
.venv/Scripts/kaggle.exe competitions submit playground-series-s6e5 \
    -f submissions/NN_<approach>.csv \
    -m "submit-NN: <approach> CV=X.XXX"

# 3. Wait for scoring + fill tracking row
.venv/Scripts/kaggle.exe competitions submissions playground-series-s6e5 | head -5
```

## Repo layout

```
.
├── README.md                    # this file (plan vivant + tracking)
├── CLAUDE.md                    # project rules (Claude only)
├── pyproject.toml               # deps exact-pinned, Python 3.13
├── requirements.lock            # frozen resolved deps (144 pkgs)
├── .pre-commit-config.yaml      # ruff + detect-secrets + lock-sync (pre-push) + pytest (pre-push)
├── data/
│   └── raw/                     # train/test/sample_submission CSVs (gitignored)
├── src/
│   ├── data.py                  # load_train, load_test, make_group_kfold(Race+Year)
│   ├── evaluate.py              # compute_metrics (AUC/logloss/Brier), plot_calibration
│   └── submit.py                # write_submission, submit_to_kaggle
├── tests/                       # 5 tests (3 data + 2 evaluate)
├── scripts/
│   ├── check_lock_sync.py       # pre-push hook
│   └── run_NN_*.py              # one script per submission (created at run time)
├── submissions/                 # NN_<approach>.csv (gitignored)
└── docs/
    ├── research/2026-05-26-deps-versions.md   # deps decisions
    └── superpowers/             # original spec + plan (SUPERSEDED, archive only)
```

## Honesty / retrospective

This project was over-scoped initially (6 notebooks, ISO/SOTA cadre, MoE bonus, survival analysis, session-end skill projection — for a 1-t-shirt prize). Phase 0 bootstrap (Tasks 1-13 of the original plan) was useful infrastructure. Re-scope on 2026-05-27 dropped notebooks + heavy ML deps. The journey from "tuto-fun-as-architecture-fest" to "ship a CSV in 4 days" is itself the lesson.

Tracking is honest : if a submission doesn't beat baseline, the row says so. If we run out of time before submit-03, the row stays at "—".

## License

CC BY 4.0 (matches Kaggle competition data license).
