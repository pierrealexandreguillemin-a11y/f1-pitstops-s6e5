# topic 701714 — submission-generalization-LB

fetched: 2026-05-28 ; n_messages=3

## msg 0 — unknown (? votes)

During my submission with a SLSQP Ensemble blend, I noticed a trend followed by my previous submissions also, the external dataset has a significant effect. The external dataset has a smoother distribution, in my analysis I got -

![External vs Raw Covariate KDEs](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21414749%2F33172641aafa462e366c8ba360276f11%2Fexternal_vs_raw_distribution.png?generation=1779169345858209&alt=media)

The distributions are much smooth of external features.

![OOF vs LB Comparison Chart](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F21414749%2F82e9ea931c9919996180acdd9c209438%2Foof_vs_lb_comparison.png?generation=1779169490052070&alt=media)

So the external data actually hurt local CV while improving LB. It is close enough to transfer useful structure, but different enough to regularize the ensemble.

---

## msg 1 — unknown (? votes)

When you add external data, do you only add it to train folds and keep the validation folds the same in both your (1) w/ external and (2) w/out external experiments?

---

## msg 2 — unknown (? votes)

Are you using stratified Kfold or just normal Kfold?

---
