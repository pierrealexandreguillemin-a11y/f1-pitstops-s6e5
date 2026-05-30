# DECISION LOG — f1-pitstops-s6e5

> Append-only chain of strategic decisions, their **completion conditions**, and what
> each **triggered next**. This is the navigation index over the scattered plan docs /
> README tracking / post-mortems / session memos. For fast reorientation without
> re-reading everything. Newest frontier at the bottom. **LB is the only trustworthy
> metric here** (OOF overfits via CatBoost CTR on random-row CV).

Legend: ✅ done/proven · ⏳ in-flight · ❌ dead end (kept as negative knowledge).

---

### D0 — Honest baseline first ✅
- **Decision:** ship CatBoost defaults + GroupKFold(Race+Year), zero FE (submit-01).
- **Why:** establish an honest LB anchor + measure the CV↔LB gap before optimizing.
- **Completion condition:** submitted, LB measured. → **LB 0.94457**, CV-LB gap +1.52pp
  (GroupKFold pessimistic).
- **Triggered:** baseline insufficient → need feature engineering → **D1**.
- Artifacts: `kernels/run_01_baseline`, README row 01.

### D1 — "Killer FE" (Plan-2) ❌ MIRAGE
- **Decision:** heavy FE — (Y,R,D) aggregates + Aadi external + 9 crosses + freq enc +
  StratifiedKFold + CatBoost 16k iter (submit-02).
- **Why:** competitor 699182 lifted LightGBM 0.91→0.947 with stint aggregates.
- **Completion condition:** submitted. OOF 0.9794 looked great but **LB 0.94296 = REGRESSION**
  vs baseline. → CV mirage detected.
- **Triggered:** must find WHY OOF lied before trusting it again → **D2**.
- Artifacts: `kernels/run_02_killer_fe`, README row 02, post-mortem
  `docs/research/2026-05-29-submit-02-cv-lb-gap-postmortem.md`.
- **Lesson banked:** OOF is untrustworthy here; only LB counts.

### D2 — Ablation: locate the mirage cause ✅
- **Decision:** measure (not assert) the cause — CatBoost ablation numeric-only vs
  +all-cats vs −5-high-card-crosses.
- **Why:** 2 prior wrong diagnoses asserted without measuring (OOF≈LB, cross-fold leak).
- **Completion condition:** measured — numeric-only OOF 0.9433 (≈LB), all-cats 0.9752
  (mirage), minus-5-high-card-crosses 0.9524. **→ the 5 high-card crosses' categorical
  CTR overfit the random-row CV.**
- **Triggered:** drop those 5 crosses → **D3** (Plan-2c).
- **Lesson banked:** measure the cause before asserting it.

### D3 — Plan-2c: drop the 5 high-card crosses ✅ (first real win)
- **Decision:** Plan-2 MINUS {Driver_Race, Race_Compound_Stint, Race_Year, Driver_Compound,
  Race_Compound}, GPU 8k iter, 2-seed (submit-03).
- **Why:** D2 measured these as the mirage source.
- **Completion condition:** submitted. OOF 0.95676 → **LB 0.95083**, +0.63pp vs baseline,
  AND OOF-LB gap tightened 3.6pp→0.6pp (diagnosis confirmed by LB).
- **Triggered:** (a) a base now beats 0.944 → diversity-ensemble gate met → **D4**;
  (b) Plan-2b all-cats kept the mirage (OOF 0.9754) → not submitted (dead).
- Artifacts: README row 03.

### D4 — GBDT diversity blend (LightGBM) ❌ dead end
- **Decision:** LightGBM on Plan-2c features, blend with Plan-2c CatBoost (submit-04).
- **Why:** test if a second GBDT adds blend diversity.
- **Completion condition:** measured — ρ(CatBoost,LGBM)=0.958, best blend w=1.00 (CatBoost
  pure), **+0.000pp**. Confirmed discussion 699182 on our own data.
- **Triggered:** GBDT+GBDT doesn't blend → need a NON-GBDT architecture (RealMLP) → **D6**.
  Also raised the Aadi-noise question → **D5**.

### D5 — Aadi external-data ablation ✅ (keep Aadi)
- **Decision:** A/B WITH vs NO-AADI (LightGBM, same folds) (submit-05 diagnostic).
- **Why:** discussion 696380 measured 25% of Aadi targets are altered → maybe noise.
- **Completion condition:** measured — WITH 0.94776 vs NO 0.94698 → **Aadi helps +0.078pp**
  (all years) despite the noise.
- **Triggered:** keep Aadi in all subsequent models. Drop-Aadi branch closed.

### D6 — RealMLP (architectural diversity) ✅ → reframed the whole problem
- **Decision:** RealMLP (PyTabKit, yekenot recipe), then isotonic blend with Plan-2c
  (submit-06 + submit-07).
- **Why:** 699182 — architectural (not implementation) diversity is the lever.
- **Completion condition (multi-probe):**
  - blend w=0.73 → LB 0.95269; w=0.50 → 0.95365; **RealMLP ALONE → LB 0.95407 (new best)**.
  - **LB-REVERSAL:** RealMLP alone beats CatBoost (0.95083) by +0.324pp though OOF said the
    opposite. Blends monotone in CatBoost weight → CatBoost should be ~0.
- **Triggered:** RealMLP is the engine, not CatBoost → push the non-GBDT direction →
  **D7**. User mandate: push score to max.
- Artifacts: `kernels/run_06_realmlp`, `scripts/run_07_isotonic_blend.py`, README rows 06/07,
  commit `44bc568`.
- **Lesson banked:** don't anchor on the OOF-strong model; LB-probe the candidates.

### D7 — Push to max: improve RealMLP + add TabPFN ✅ RESOLVED (negative)
- **Decision:** (a) RealMLP-v2 research-backed config (`ls_eps=0`, `n_epochs=16`);
  (b) TabPFN-v2 3rd family pinned `tabpfn==2.0.9`; (c) explore 2-way blend weight.
- **Result (measured):** RealMLP-v2 LB **0.95398** (≈ v1, corr 0.999 — config doesn't
  decorrelate). TabPFN-v2 LB **0.93318** (honest OOF but 2.1pp too weak). Blend-09
  (RealMLP+TabPFN w=0.10) LB **0.95399** < RealMLP-alone 0.95407. **No new best.**
- **Lesson banked:** diversity pays ONLY when the partner is strong; tuning the same
  architecture won't decorrelate it. → D9.

### D8 — Cut TabPFN, keep quota for a strong partner ✅ DONE
- **Decision:** stop the TabPFN line. First config (100k context) was out of spec
  (TabPFN-v2 design limit = 10k, arxiv 2502.17361) → killed at 8650s; refit to 10k +
  bagging×4 confirmed the model is simply weak here. No further TabPFN runs.
- **Infra lessons banked:** (1) ALWAYS push GPU kernels with `--accelerator NvidiaTeslaT4`
  — the pool may give a P100/sm_60 that the preinstalled torch cu128 can't run
  (`no kernel image`, submit-08 v3 died in 20s). (2) Pull kernel logs with `PYTHONUTF8=1`
  (emoji in title truncates the download via charmap). (3) Per-bag checkpoint write so a
  container restart/timeout still yields coherent outputs.

### D9 — TabM: the second STRONG NN ✅ RESOLVED (negative)
- **Decision:** add TabM (PyTabKit `TabM_D_Classifier`, ICLR 2025). Same submit-06 pipeline.
- **Result (measured):** TabM OOF **0.95211** — genuinely STRONG (0.002 below RealMLP, the
  opposite of weak TabPFN). BUT corr(RealMLP,TabM) = **0.989** (both MLPs on the same FE →
  near-identical). Blend-11 (w_tm=0.25) OOF 0.95429, LB **0.95398** < RealMLP-alone 0.95407.
  OOF +0.011pp did NOT transfer — same failure mode as TabPFN.
- **Config miss (mine):** no early-stopping / bounded n_epochs on TabM (left `_D` defaults)
  → trained ~110min past the per-fold optimum (best epoch ~2). Quality OK (best-epoch
  checkpoint kept) but wasteful GPU. Next time: set `n_epochs` + early stopping like RealMLP.
- **Lesson banked:** a diversity partner must be BOTH strong AND decorrelated. We have now
  exhausted all three families — CatBoost/GBDT (ρ.958), TabPFN/foundation (ρ.919 but weak),
  TabM/MLP (ρ.989). **None beats RealMLP-v1 alone; every blend's OOF gain fails to transfer.**

### CONCLUSION — RealMLP-v1 alone is the ceiling ✅
The public LB plateaus at **~0.954**. RealMLP-v1 alone (**LB 0.95407**) is the best
single AND beats every blend we built. No principled lever left within the time/compute.
**Finals LOCKED:** A = RealMLP-v1 (0.95407), B = blend-09 RealMLP+TabPFN (0.95399, the most
decorrelated near-tied candidate = private-LB insurance). Select on Kaggle UI (no CLI);
Kaggle's top-2-public default already = these two.

### D10 — Improve RealMLP-v1 ITSELF (blends exhausted) ⏳ IN FLIGHT (option 1)
Context: public LB is a density wall — top 0.95497, us 0.95407 (#464/2904), 229 teams in
[0.9515, 0.95407]. Blends all failed; the only path up is a stronger RealMLP standalone.
Three levers evaluated (RealMLP-v1 already has: `n_ens=20` internal ensemble, yekenot FE
[arithmetic interactions, count-enc, KBins, TargetEncoder on combos Race_Compound/Race_Year],
`n_epochs` tested v1=5/v2=16=tie, calibration N/A since AUC is rank-based):

- **Lever 1 — add (Y,R,D) single-column target-mean features (CHOSEN, building).** RealMLP
  lacks explicit out-of-fold target means by Driver/Race/Year/Compound (it only freq-encodes
  them + target-encodes 2 combos). Plan-2c's measured ablation: single-column aggregates
  GENERALIZE (the mirage was the 5 high-card CROSSES, not these). = NEW information for the
  NN, not just variance reduction. Leak-free via sklearn `TargetEncoder(cv=5)` (same tool the
  kernel already uses). Highest upside. Risk: OOF won't guide → LB probe decides; may not
  transfer like everything else at this plateau. Kernel `run_12_realmlp_agg`, ~40min T4.
- **Lever 2 — seed-ensemble the full run (3-5 seeds, average test preds).** Adds fold-split +
  init diversity beyond the internal n_ens=20. Reliable but small (+0.0001-0.0003), zero
  CV-overfit risk. Not chosen first (smaller upside than lever 1).
- **Lever 3 — RealMLP_HPO (per-dataset tuning) — REJECTED.** Optimises on the UNRELIABLE OOF
  → re-creates the submit-02 mirage trap (CV-overfit config that regresses on LB). Worst
  choice at J-0.

- **Completion condition (PENDING):** run_12 COMPLETE → probe alone on LB → beats 0.95407?
  If yes → new best, becomes final slot A. If no → lever 1 dead, fall back to lever 2 or stop.
- Ship gate unchanged: submit only if hypothesis-backed to beat 0.95407 (LB the verdict).

---

## Frontier pointer
**Now:** D10 lever-1 IN FLIGHT — RealMLP-v1 + (Y,R,D) target-mean FE (`run_12_realmlp_agg`,
`pguillemin/f1-s6e5-realmlp-agg`). All blends exhausted (D6/D7/D8/D9). Best = RealMLP-v1 alone
**LB 0.95407** (#464/2904, density wall). On completion: probe → if > 0.95407 it becomes final
slot A, else fall back. Finals currently LOCKED A=RealMLP-v1 / B=blend-09. Deadline 2026-05-31
23:59 UTC.

## Superseded / frozen plan docs (for reference, not active)
- `docs/plans/2026-05-28-submit-02-plan.md` — Plan-2 (D1), shipped, regressed.
- `docs/plans/2026-05-29-submit-02-checkpoint-resume-plan.md` — the timeout-proof net (done).
- `docs/plans/2026-05-28-plan-ensemble-FROZEN.md` — ex-ensemble plan; gate met at D3,
  realised via D6/D7.
- `docs/superpowers/specs|plans/2026-05-26-*` — original tuto spec/plan, SUPERSEDED (archive).
