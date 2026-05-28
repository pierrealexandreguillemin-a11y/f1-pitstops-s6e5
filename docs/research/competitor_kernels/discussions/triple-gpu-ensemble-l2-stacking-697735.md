# topic 697735 — triple-gpu-ensemble-l2-stacking

fetched: 2026-05-28 ; n_messages=3

## msg 0 — unknown (? votes)

Does LGBM + Optuna approach gives any real benefits here?

---

## msg 1 — unknown (? votes)

Hello everyone,
Like many of you, I hit a plateau around the 0.946+ mark using standard tree-based models. I noticed that simple blending (averaging predictions) wasn't giving that extra push. I wanted to share a framework that helped me squeeze out more performance and stabilize my CV.
Here are three key takeaways from my recent experiments:
1. The Power of Diversity (CatBoost is a Must)
While LightGBM and XGBoost are incredibly fast and powerful, they often "think" similarly. Adding CatBoost (on GPU) to the mix provided the necessary model diversity. CatBoost's handling of the categorical features (like Driver and Race) seems to capture hidden patterns that the others miss.
2. Stop Simple Averaging (Use a Meta-Model)
Instead of doing (LGBM + XGB + CAT) / 3, I fed their out-of-fold (OOF) predictions as new features into a Logistic Regression Meta-Model (Level-2 Stacking).
Why? Because simple averaging assumes all models are equally correct in every scenario. The Logistic Regression "manager" learns when to trust CatBoost (e.g., specific track conditions) and when to trust XGBoost. This simple switch gave me a solid boost!
3. The OOM (Out Of Memory) Trap & The Fix
Training three heavy models across 5 folds on a GPU will inevitably crash your Kaggle session (OOM).
The fix: Force memory cleanup after every single fold. Do not rely on Python's default memory management.

import gc

# At the end of your fold loop:
del X_tr, y_tr, X_vl, y_vl
del model_lgb, model_xgb, model_cat
# ... delete your prediction arrays for the fold ...
gc.collect() # Force garbage collection



I hope this helps anyone struggling to push past their current limit. Let me know if you’ve found other algorithms (like Neural Networks / MLP) that add good diversity to tree-based stacks!
Happy modeling! 🚀


Abdulrahman Ragheb

---

## msg 2 — unknown (? votes)

but pytoch and scikit can be helped to help

---
