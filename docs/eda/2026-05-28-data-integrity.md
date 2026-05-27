# EDA #02 — Data integrity (follow-up on Kaggle discussion alert)

> Date: 2026-05-28 | Git ref at investigation: `cac8438` | Triggered by Kaggle discussion (Huzaifa Mankda, Fernando Arizmendi, Alibay Alili) flagging non-monotonic Stint per (Race, Year, Driver).

## TL;DR

The Kaggle discussion is **right**. The competition dataset is **row-shuffled at the (Race, Year, Driver) level**:

- **62% of (Race, Year, Driver) groups in train have non-monotonic `Stint`** (in test: 31%).
- **12.75% of consecutive-lap transitions per driver have `Stint` going DOWN** (physically impossible).
- **`LapTime_Delta` is NOT `LapTime[L] − LapTime[L−1]`** — only 0.27% match within 0.01s.
- **`Cumulative_Degradation` is NOT cumulative** — only 1.00% match.
- **`TyreLife` IS monotone within `(R, Y, D, Stint)`** (100% of stints) — intra-stint integrity preserved.

→ The original `f1-strategy-dataset-pit-stop-prediction` preserves race chronology ; the synthetic competition version **destroys inter-stint chronology** but preserves intra-stint structure. Quote Huzaifa: *"The whole data for a race, year, driver is inaccurate. […] The actual dataset preserves the race integrity but the competition dataset does not."*

**Modeling implication** : treat each row as **independent**. Any feature requiring chronological ordering per driver (`laps_since_pit`, rolling-N lap-time, "is undercut") is **broken** and will inject noise. Confirms Alibay Alili's intuition: "treat each row as an independent sample". Confirms Fernando Arizmendi's observation: GroupKFold fold variance ±0.010 was a symptom of this corruption.

## Verification — Huzaifa's specific case in our train.csv

D109, Dutch Grand Prix, 2022 (11 train rows, sorted by `LapNumber`) :

| LapN | Cmp | Stint | TL | PS | LT(s) | Delta | CumDeg | NextPit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4  | MED | 1 |  4 | 0 |  77.1 | -37.2 | -162.9 | 0 |
| 14 | MED | 1 | 14 | 1 |  76.6 |  -7.9 |   13.6 | 0 |
| 17 | MED | **2** |  8 | 0 |  77.9 | -38.1 |  -16.7 | **1** |
| 18 | MED | **1** | 18 | 1 |  75.8 |  -4.0 |  -59.3 | 0 |
| 19 | MED | 1 | 19 | 0 |  76.6 |  -8.4 |  102.1 | 0 |
| 23 | MED | 2 | 11 | 1 |  93.9 |   7.2 |   -4.1 | 1 |
| 27 | SFT | 3 | 15 | 0 |  77.1 |  -0.1 |  -23.1 | 0 |
| 40 | MED | 2 | 24 | 0 |  76.9 |  -9.6 | -141.6 | 0 |
| 47 | SFT | 4 |  4 | 1 | 108.9 |  -2.1 |  -18.7 | 0 |
| 56 | SFT | 5 |  7 | 1 |  84.0 |   5.1 |  -52.8 | 0 |
| 69 | SFT | 4 | 20 | 0 |  75.8 |  -0.1 |  -43.8 | 0 |

Stint sequence : `[1, 1, 2, 1, 1, 2, 3, 2, 4, 5, 4]` → impossible chronologically. Lap 17 is Stint 2 then Lap 18 is Stint 1. Lap 40 (Stint 2, TyreLife 24) then Lap 47 (Stint 4, TyreLife 4) skips Stint 3 = OK but Lap 56 (Stint 5) then Lap 69 (Stint **4**) is impossible.

## Global corruption quantification

### A. Stint monotonicity per (Race, Year, Driver)

| Set | Groups | Monotonic | Non-monotonic |
|---|---:|---:|---:|
| **train** | 40 869 | 15 493 (37.91%) | **25 376 (62.09%)** |
| **test**  | 37 038 | (68.72%) | **(31.28%)** |

Test corruption is lower because test groups have fewer rows on average (less opportunity to observe the chaos), not because test is cleaner.

Per year (train) :

| Year | Non-monotonic % |
|---|---:|
| 2022 | 58.81% |
| 2023 | 63.32% |
| 2024 | 64.84% |
| 2025 | 59.79% |

Uniform corruption across years. Not a year-specific bug.

### B. TyreLife monotonicity within (Race, Year, Driver, **Stint**)

| Set | Stint-groups | Monotonic | Non-monotonic |
|---|---:|---:|---:|
| train | 113 567 | **113 567 (100.00%)** | 0 |

**Intra-stint integrity is preserved.** `TyreLife` is a valid row-level feature, just don't expect "next row by lap" to be "next lap actually run by driver".

### C. `LapTime_Delta` reconstruction (consecutive laps, lap_diff=1)

Hypothesis tested : `LapTime_Delta[L] == LapTime(s)[L] − LapTime(s)[L−1]`.

- Rows with consecutive `LapNumber` per (R, Y, D) : **127 250**
- Match within ±0.01s : **348 (0.27%)**
- Median absolute error : **6.6 s**
- Mean absolute error : **10.0 s**

→ `LapTime_Delta` is NOT a within-driver lap-over-lap delta. It is computed against some other reference (fastest-lap of race? median? quali time?). **Treat as a black-box numeric feature.**

### D. `Cumulative_Degradation` reconstruction

Hypothesis tested : `CumDeg[L] == CumDeg[L−1] + LapTime_Delta[L]`.

- Match within ±0.01 : **1 270 (1.00%)**
- Median absolute error : **16.3** (units unclear, presumably seconds-equivalent)

→ Also NOT a cumulative quantity in the naive sense. Black-box feature.

### E. Stint transitions on consecutive laps

`Stint[L] − Stint[L−1]` over 127 250 consecutive-lap pairs in train :

| Diff | Count | Notes |
|---:|---:|---|
| -5  | 3 | |
| -4  | 23 | |
| -3  | 216 | |
| -2  | 1 276 | |
| -1  | **14 701** | impossible |
|  0  | 91 087 | same stint |
|  +1 | 18 083 | normal pit transition |
|  +2 | 1 578 | "skipped" 1 stint |
|  +3 | 255 | |
|  +4 | 25 | |
|  +5 | 3 | |

**12.75% of consecutive-lap pairs have Stint decreasing.** 1.46% skip stints upward.

## Year 2023 anomaly — revisited

Now that we know the data is shuffled, the 2023 anomaly looks **structurally identical to other years, scaled ~30× down on P(pit)** :

P(`PitNextLap=1` | Year × Stint) :

| Year \ Stint | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.087 | **0.515** | 0.423 | 0.366 | 0.071 |
| **2023** | **0.010** | **0.011** | **0.007** | **0.008** | **0.007** |
| 2024 | 0.076 | **0.565** | 0.518 | 0.333 | 0.153 |
| 2025 | 0.074 | **0.569** | 0.506 | 0.125 | 0.066 |

For 2022/24/25, target probability **peaks at Stint 2-3 around 50-57%**. For 2023, it's **flat at ~1%** across all stints.

Hypothesis: 2023 labels were generated under a different probabilistic rule — perhaps the synthetic generator output ~30× fewer positive labels for that year (sub-sampling, threshold, or seed quirk). The Stint **structure** of the data is the same, only the label rate differs.

**Implication for modeling** : `Year` is critical as feature. Tree boosters will partition on Year early. Do NOT down-weight or drop 2023 — let the model output ~1% probabilities for 2023 test rows (35 348 of 188 165 = 18.8% of test). Dropping 2023 would force the model to extrapolate from 28% positives → 18.8% of test rows where truth is 1%.

## Revised submit-01 baseline plan

What stays from `2026-05-27-overview.md`:
- CatBoost default + GroupKFold(Race+Year), seed=42
- Native categoricals: `Driver`, `Compound`, `Race`, `Year`
- No clipping (trees robust)
- All columns except `id`
- AUC + Brier + logloss + frac-2023-in-val per fold

What changes:
- **No surprise on high fold variance** — Fernando Arizmendi already saw ±0.010 with GroupKFold. Expect the same. **Not a bug, just the data.**
- **No naive time-series FE in submit-02** — drop "laps_since_pit", "rolling lap time", "is undercut", "stint progression" from the plan. They will inject noise.
- **README plan v2 row 02** ("+ Domain FE: pit window, undercut, year×compound, degradation rate, stint progression") needs revision before submit-02 is written. Most of those features are nonsense on this corrupted data.

## Revised submit-02 FE candidates (replacing time-series FE)

Row-level, no chronological assumption :

| Feature | Type | Why |
|---|---|---|
| `Compound × Stint` interaction | crossed cat | Strong signal in target rate table |
| `Compound × Year` interaction | crossed cat | 2023 anomaly may interact with compound |
| `TyreLife / Stint` ratio | numeric | Proxy for "tires worn relative to stint number" — row-level, valid |
| `LapTime(s) − median(LapTime by Race×Year)` | numeric | Normalized pace, row-level, valid |
| Target encoding on `Driver` | regularized cat | Replace 887-card cat with target mean (CV-aware) |
| Frequency encoding on `Driver`, `Race` | numeric | Cheap alternative to TE |
| `RaceProgress` × `Stint` | crossed | Captures "early stint near end of race" vs "late stint at start" |
| `Position` × `Compound` | crossed | Slow tires for back-of-grid? |

**Excluded** : everything that requires `groupby(Race, Year, Driver).shift()` or `.rolling()`. Confirmed broken.

## Could we *unshuffle* the data?

Probably not. The intra-stint TyreLife monotonicity is the only invariant. We could try to reconstruct stint order using:
- `LapNumber` within Stint (yes, monotonic within stint by construction)
- Compound stickiness (a stint = single compound, so compound transitions ≈ stint transitions in raw F1)
- `Cumulative_Degradation` direction (probably broken, see D)

But the official inter-stint ordering signal is destroyed. We'd be inferring with significant noise. Not worth the time on a 1-T-shirt comp with 3 days left. **Skip.**

## Sources / acknowledgments

- Huzaifa Mankda (503rd) — flagged Stint non-monotonicity with the D109 Dutch GP 2022 sample, identified the `LapTime_Delta` and `Cumulative_Degradation` relation violations
- Fernando Arizmendi (1194th) — observed ±0.010 fold variance under GroupKFold, attributed to `Stint`-based grouping
- Alibay Alili (725th) — proposed "treat each row as independent"
- Original Kaggle discussion : `https://www.kaggle.com/competitions/playground-series-s6e5/discussion` (page behind CloudFlare bot protection from local fetch ; thread content received via user paste)
- Original source dataset (clean) : `https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction`

---

**Next step** (en attente signal user) : update README plan v2 row 02 (drop time-series FE, add row-level FE candidates), then create `scripts/run_01_baseline.py`.
