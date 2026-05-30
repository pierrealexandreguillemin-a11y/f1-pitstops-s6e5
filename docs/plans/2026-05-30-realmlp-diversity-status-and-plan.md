# Status & forward plan — 2026-05-30 — RealMLP diversity push

> Durable record of the live plan (was only in conversation). Deadline **2026-05-31
> 23:59 UTC** (J-1). Mandate from user: **push score to max** (TabPFN + other levers),
> act autonomously on each verdict, probe-submit principled candidates (no LB grinding).

## Current standings (LB = the only trustworthy metric here)

| Submit | Model | OOF AUC | **Public LB** | Note |
|---|---|---|---|---|
| 01 | CatBoost defaults | 0.9294 | 0.94457 | anchor |
| 02 | killer-FE all-cats | 0.9794 | 0.94296 | CV mirage (regression) |
| 03 | Plan-2c CatBoost (drop 5 high-card crosses) | 0.95676 | 0.95083 | honest CatBoost |
| 07 | blend CatBoost+RealMLP w=0.73 | 0.95726 | 0.95269 | dragged down by CatBoost |
| 07b | blend w=0.50 | 0.95695 | 0.95365 | less CatBoost = better |
| 06v2 | RealMLP-v2 (ls_eps=0, n_epochs=16) | 0.95396 | 0.95398 | ≈ v1, corr 0.999 — config doesn't decorrelate |
| 08 | TabPFN-v2 (10k ctx + bagging×4) ALONE | 0.93381 | 0.93318 | honest OOF but WEAK (−2.1pp), corr 0.919 |
| 09 | blend RealMLP-v1 + TabPFN w_tp=0.10 | 0.95433 | 0.95399 | OOF gain did NOT transfer (−0.008pp vs RealMLP) |
| 10 | **TabM** (ICLR 2025, 2nd strong NN) | IN FLIGHT | IN FLIGHT | the real diversity lever (strong + different arch) |
| **06** | **RealMLP-v1 ALONE** | 0.95409 | **0.95407** | **BEST so far** |

### TabPFN verdict (submit-08/09, measured) — line CLOSED
TabPFN's OOF is HONEST here (OOF 0.93381 ≈ LB 0.93318 — unlike CatBoost's CTR mirage),
but it is genuinely WEAK: subsampling 10k of 350k rows + 14 raw features starves it.
At corr 0.919 it IS more decorrelated than CatBoost (0.958), but 2.1pp too weak -> in a
blend it earns ~0 useful weight (submit-09: OOF +0.024pp, LB −0.008pp, did not transfer).
**Diversity only pays when the partner is STRONG.** Hence TabM. Also cost a lot of T4:
the first config used 100k context (10x TabPFN-v2's documented 10k design limit, arxiv
2502.17361) -> "degraded performance" + ~10x slower, killed at 8650s. Fixed to 10k +
bagging. Infra lesson: ALWAYS push GPU kernels with `--accelerator NvidiaTeslaT4` (the
Kaggle pool may assign a P100/sm_60, incompatible with the preinstalled torch cu128 ->
`no kernel image is available` crash; submit-08 v3 died this way in 20s).

### Selected finals (deadline 2026-05-31 23:59 UTC) — revise if TabM wins
- **A = RealMLP-v1 ALONE** (LB 0.95407) — best public, safe pick.
- **B = blend RealMLP-v1 + TabPFN `09`** (LB 0.95399) — top-2 public + architectural
  diversity (NN + transformer) as private-LB insurance.
- Kaggle has no CLI for final selection; pick on the UI (Submissions -> "Use for Final
  Score"). Kaggle's default top-2-public already = these two. **REVISE if submit-10 TabM
  or a RealMLP+TabM blend beats 0.95407 on LB.**

## THE key finding (LB-reversal, measured 2026-05-30)

**RealMLP alone (LB 0.95407) beats CatBoost Plan-2c (LB 0.95083) by +0.324pp — even
though the OOF said the opposite (RealMLP 0.954 < CatBoost 0.957).** CatBoost overfits
the random-row StratifiedKFold CV far more than RealMLP; on the true test the NN wins.
Blends are MONOTONE in CatBoost weight: w(CatBoost)=1.0→0.95083, 0.73→0.95269,
0.50→0.95365, 0.0→0.95407. **CatBoost should get ~0 weight.** RealMLP is the engine.

**Consequence:** the OOF is NOT a reliable guide for model/blend selection here. Use LB
probes (principled, ≤ a few) to decide. The winning blend is RealMLP + TabPFN (the two
strong non-GBDT families), CatBoost weight ~0.

## In-flight kernels (Kaggle T4, monitor armed)

- **TabM (submit-10)** — slug `pguillemin/f1-s6e5-tabm`. Second STRONG NN of a different
  architecture (parameter-efficient MLP-ensemble, ICLR 2025). Same pipeline as submit-06
  (yekenot FE, folds, Aadi), only model swapped to `TabM_D_Classifier` (pytabkit defaults).
  `kernels/run_10_tabm`. Pushed with `--accelerator NvidiaTeslaT4`.

(DONE this session: RealMLP-v2 → 0.95398, TabPFN-v2 → 0.93318, blend 09 → 0.95399. See
standings + TabPFN verdict above.)

## Forward plan (execute on TabM completion)

1. **TabM alone → probe LB.** Beats 0.95407? If yes, new best + final candidate.
2. **corr(RealMLP-v1, TabM)** on aligned OOF/test. Lower corr + comparable strength =
   blend likely to lift (the case TabPFN could not provide).
3. **Blend RealMLP-v1 + TabM** (isotonic-calibrate each, LB-informed weight — NOT OOF
   weight, proven misleading). Probe 1 (maybe 2) principled variant(s). Does it beat 0.95407?
4. **Finalize the 2 finals.** Currently A=RealMLP-v1 (0.95407), B=blend-09 (0.95399).
   If TabM or the RealMLP+TabM blend beats 0.95407 → it becomes slot A, and the best
   decorrelated runner-up becomes slot B. Never two near-identical.

## Tooling / artifacts

- Blend script: `scripts/run_07_isotonic_blend.py` (vectors-only, safe locally, no OOM).
  Ad-hoc weight blends done via inline python (see git log).
- Pulled outputs (local temp): `plan2c_out/`, `realmlp_out/` (= v1). v2/tabpfn to pull.
- **No local heavy training** (laptop OOM risk, only 3.7 GB free observed) — all model
  training on Kaggle. Local = only vector blends + submissions.

## Guardrails

seed=42 everywhere. No competition submit was blind — each probe answers a question.
No LB grinding (no fine weight sweeps on LB). LB is the only trustworthy metric (OOF
overfits via CatBoost CTR / random-row CV). Aadi external data HELPS (ablation measured
+0.078pp), keep it. GBDT+GBDT blend is a dead end (submit-04, rho=0.958).
