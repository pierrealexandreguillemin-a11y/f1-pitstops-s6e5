# Levers to improve candidates 06 & 09 — SOTA ML + audacious angles

> Date: 2026-05-31 (J-0, deadline 23:59 UTC) | Triggered by user "reprise" — reopen the
> LOCKED finals to look for genuine improvements. Web research authorized.
> **Honest frame first (CLAUDE.md rigour rule):** the plateau is real. SE(AUC) public ≈
> 0.00078 (Hanley-McNeil, DECISION-LOG D10). Our best RealMLP-v1 = **0.95407** (#464/2904).
> The **top-5 public notebook ("Tail53 Top 05") = 0.95453 = +0.59·SE above us** — i.e. the
> public ceiling itself is within noise of us. Every blend we built saw its OOF gain **fail
> to transfer** to LB. So the realistic payoff of everything below is NOT "smash the public
> plateau" — it is **(a) a better, genuinely-decorrelated final slot B for private-LB
> insurance, and (b) cheap shots with asymmetric upside.** Ranked by realistic EV, not hype.

## What 06 & 09 are (recap)

- **06 = RealMLP-v1 alone** (PyTabKit, yekenot FE, StratifiedKFold5+Aadi, n_ens=20). LB **0.95407**. Best single. OOF≈LB reliable for this model.
- **09 = isotonic blend RealMLP-v1 + TabPFN-v2 (w_tp=0.10)**. LB **0.95399**. Current final slot B = "most decorrelated insurance" — but its partner (TabPFN) is **weak (0.93318)** because it was **double-starved**: capped at 10k context AND fed only 14 RAW features.

## Diversity coverage so far (the gap)

| Family | Model | Strong? | Decorrelated vs RealMLP? | Verdict |
|---|---|---|---|---|
| GBDT | CatBoost/LGBM | weak-ish (0.951) | ρ.958 | drags blend down |
| Foundation (ICL) | TabPFN-v2 | **weak (0.933)** | ρ.919 (best) | starved → too weak |
| MLP | TabM | strong (0.952) | **ρ.989 (clone)** | too correlated |
| MLP | RealMLP (self) | 0.95407 | — | the engine |

**Untested family = retrieval / neighbour-based** (TabR, ModernNCA). This is the one
architectural inductive bias we never tried, and it is exactly the "strong AND decorrelated"
gap. See SOTA-1.

---

## ANGLE A — SOTA ML (latest tabular research, early 2026)

### A1 — ModernNCA / TabR: the untested retrieval architecture ⭐ best SOTA merit
- **What:** ModernNCA (revived deep Neighbourhood Component Analysis) and TabR
  (retrieval-augmented). On **TabArena** (living benchmark, arXiv 2506.16791) ModernNCA is
  **top-ranked among deep methods**, often above RealMLP/TabM. Inductive bias = soft nearest
  neighbours in a learned embedding space — **fundamentally different from feedforward MLP**
  → the best theoretical shot at a partner that is BOTH strong (≈0.95) AND decorrelated
  (ρ well below TabM's 0.989).
- **Why it attacks our exact gap:** TabM failed only on correlation (ρ.989), not strength.
  A retrieval model keeps the strength but breaks the correlation by construction.
- **Actionability:** not in PyTabKit (RealMLP/TabM only). Available in **TALENT toolkit**
  (LAMDA-Tabular/TALENT, 35+ deep methods incl. ModernNCA & TabR) — research repo, less
  plug-and-play than PyTabKit's sklearn API but runnable on Kaggle T4. ~1 GPU kernel, reuse
  the submit-06 fold/FE scaffold.
- **EV:** the highest-merit *new* model. If ρ(RealMLP, ModernNCA) lands ~0.90–0.95 with
  strength ~0.95, the blend could finally transfer — and even if it doesn't beat 0.95407 on
  public, a strong decorrelated model is a **strictly better final slot B than blend-09**
  (whose partner is weak). Risk: integration friction (TALENT API), and the persistent
  "gain doesn't transfer" pattern could bite again.

### A2 — TabPFN-2.5 / TabICLv2: un-starve the foundation model ⭐ fixes 09's known weakness
- **What changed since our submit-08:** we pinned `tabpfn==2.0.9` and capped context at 10k
  (the v2 design limit, arXiv 2502.17361). **TabPFN-2.5** (Prior Labs, model report Nov 2025;
  `pip install tabpfn` now defaults to it; weights on HF `Prior-Labs/tabpfn_2_5`) handles
  **~100k rows × ~2000 features**. **TabICLv2** (arXiv 2602.11139, Feb 2026; column-then-row
  attention) scales to **million-scale** under 50 GB GPU, "without tuning surpasses current
  SOTA", faster than RealTabPFN-2.5.
- **Why:** our TabPFN was crippled on BOTH axes. Feed TabPFN-2.5 (i) ~100k stratified
  context (10× ours) and (ii) the **rich yekenot FE** (not 14 raw cols) → plausibly lifts it
  from 0.933 toward ~0.95, while staying a foundation-model architecture (decorrelated from
  gradient-MLP). That would make blend-09 actually transfer, OR give a much stronger slot B.
- **Caveats (honest):** (1) **License-gate infra risk** — our memo recorded TabPFN-3 weights
  fail in non-interactive batch (`TabPFNLicenseError`); verify TabPFN-2.5's `tabpfn-2.5-license-v1.1`
  doesn't block headless Kaggle before burning a kernel. The "no commercial/production" clause
  is fine for a competition. (2) Strengthening it via FE risks **raising** correlation toward
  RealMLP — the decorrelation that made it good insurance could shrink. (3) TabICLv2 is very
  fresh (Feb 2026) — packaging maturity unverified.
- **EV:** medium. Directly repairs the documented flaw of our current slot B. Cheaper concept
  than A1 (reuse submit-08 scaffold), but A1's architecture is the more principled diversity.

### A3 — Seed/fold ensemble of RealMLP-v1 (D10 lever-2, previously rejected)
- Average 3–5 RealMLP runs over different fold seeds + inits (beyond internal n_ens=20).
  Reliable, zero CV-overfit risk, **but expected < 0.5·SE ≈ < 0.0004** — noise. Listed for
  completeness; **not worth J-0 GPU**.

---

## ANGLE B — Audacious (unconventional, asymmetric upside)

### B1 — Rank-average our RealMLP with strong PUBLIC submissions ⭐ cheapest, most likely to move
- **What:** the public top ("Tail53 Top 05", 0.95453) is almost certainly itself a **blend of
  public notebooks** (GBDT-heavy: the "tail/rank-average" pattern is the dominant public meta).
  Our RealMLP-v1 is the **rare strong standalone NN** — plausibly decorrelated from the
  GBDT-heavy public consensus. **Rank-average** (not value-average; AUC is rank-based) our
  06 preds with 1–3 of the best public submission CSVs.
- **Why it can actually transfer (unlike our internal blends):** our prior blend partners were
  all trained on the *same* FE/folds → correlated failure modes. Public GBDT blends are a
  genuinely different lineage → real decorrelation is plausible.
- **Cost:** ZERO GPU. Download public CSVs + `scipy.stats.rankdata` + weighted average.
- **Caveats:** (1) ethically it is "borrowing" public work — fine for a swag-tier Playground,
  but flag it; the project's pedagogical honesty norm matters. (2) public CSVs may already
  contain RealMLP-like models → less decorrelation than hoped. (3) only public-LB measurable;
  private reshuffle still applies.
- **EV:** the single most likely lever to produce a *new public best*, and it's free. Strong
  candidate to at least *test* with remaining submission quota.

### B2 — Reconstruct lap chronology → unlock time-series FE (reopen the "Skip" decision)
- **What:** EDA-02 dismissed un-shuffling ("not worth it, 3 days left"). At J-0 it's still the
  one source of **information nobody else has** (everyone treats rows as independent — confirmed
  in the discussion). Invariants we CAN use: intra-stint `TyreLife` is **100% monotone**; a
  stint = a single compound. Attempt per-(Race,Year,Driver) order via (compound-run grouping →
  TyreLife-within-stint → LapNumber) to recover stint sequence, then derive `laps_since_pit`,
  `tyre_age_vs_typical_window`, undercut proxies.
- **EV:** high variance. Our own analysis says inter-stint order is destroyed (12.75% of
  consecutive pairs have Stint decreasing) → reconstruction is noisy and may inject more noise
  than signal (the exact failure submit-02 taught us). But it's **pure pandas, no GPU**, and if
  even partially valid it's genuinely orthogonal information. Worth a *quick* OOF probe, not a
  big investment.

### B3 — Adversarial validation / generator artifact hunt
- Train a classifier train-vs-test. If AUC ≫ 0.5 → exploitable distribution shift (test-like
  sample weighting) or a generator artifact (id-ordering signal, near-duplicate train/test rows
  giving near-free labels). The 2023 ~30× label-rate anomaly is already a known generator
  artifact (exploited via `Year`). Cheap diagnostic; low odds on a well-built synthetic set but
  high payoff if it hits. No GPU.

### B4 — Pseudo-labeling / transductive learning
- Add high-confidence test rows (p<0.02 or p>0.8) as pseudo-labels, refit RealMLP. Classic
  semi-supervised lift; TabPFN/ICL models are transductive by nature. Modest upside, real
  overfit/confirmation-bias risk on a noisy synthetic target. 1 GPU kernel. Lower priority.

### B5 — Per-group rank normalization (post-processing)
- 2023 base rate ~1% vs ~28% elsewhere. AUC is **global** rank-based, so per-Year rank
  re-scaling *can* help or hurt depending on cross-group ordering. Cheap to OOF-probe, no
  training. Small expected effect; quick to rule in/out.

---

## Recommendation (ranked for J-0, given GPU quota + SE reality)

1. **B1 (rank-avg with public CSVs)** — free, fastest, most likely *new public best*. Test first.
2. **A1 (ModernNCA via TALENT)** — best *principled* lever; its real value is a **superior
   final slot B** (strong + truly decorrelated) even if public score ties. 1 GPU kernel.
3. **A2 (TabPFN-2.5 un-starved)** — repairs 09's exact flaw; cheaper to wire than A1 but
   check the license-gate first.
4. **B3 + B5** — cheap diagnostics, run opportunistically (no GPU).
5. **B2** — high-variance, pandas-only; quick OOF probe only.
6. Skip **A3, B4** at J-0 (noise / overfit risk vs cost).

**Strategic reframing:** beating 0.95407 on the *public* LB is chasing 0.x·SE noise. The
durable win is **picking the best 2 finals for the PRIVATE LB**: keep A=RealMLP-v1, and
*upgrade slot B* from weak-partner blend-09 to a **strong + decorrelated** model (A1 or A2) if
one materializes. That is the honest, defensible improvement this research points to.

## Sources
- Tail53 top-5 public notebook (0.95453): https://www.kaggle.com/code/beicicc/f1-pit-stops-tail53-top-05-0-95453
- TabArena living benchmark (ModernNCA top deep method): https://arxiv.org/pdf/2506.16791
- TabR (ICLR 2024, retrieval): https://openreview.net/pdf?id=rhgIgTSSxW
- TabM (ICLR 2025): https://arxiv.org/html/2410.24210v1
- TALENT toolkit (ModernNCA/TabR impls): https://github.com/LAMDA-Tabular/TALENT
- PyTabKit (RealMLP/TabM): https://github.com/dholzmueller/pytabkit
- TabICL (ICML 2025): https://arxiv.org/pdf/2502.05564
- TabICLv2 (Feb 2026, million-scale): https://arxiv.org/pdf/2602.11139
- TabPFN-2.5 report (100k×2000): https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report
- TabPFN-2.5 weights: https://huggingface.co/Prior-Labs/tabpfn_2_5
- TabPFN-v2 large-data method (10k limit, our prior config): https://arxiv.org/pdf/2502.17361
