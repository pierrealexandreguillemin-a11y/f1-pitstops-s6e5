# Analyse de la solution gagnante — PS S6E5 (Optimistix, 1ʳᵉ place)

> Source : writeup « 1st place by the skin of my teeth » + thread de commentaires
> (Masaya Kawamata 8ème, Ravi Ramakrishnan 17ème, Ayush Khaire).
> Date : 2026-06-01. Notre résultat : **523ème** (privé 0.95423, final A = blend RealMLP+TabR).

---

## 1. Le résultat brut

- **1ʳᵉ place privé 0.95503**, devant @cdeotte de **0.00001** — gagné avec une submission de
  *dernière minute* (la compète s'est fermée pendant qu'il soumettait le blend suivant).
- Public 0.95488 → privé 0.95503 : **monte au shakeup** (sélection robuste, pas d'overfit public).
- Notre gap au vainqueur : **0.0008 ≈ 1·SE** (SE public ≈ 0.00078, mesuré D10). Pas un gouffre
  de méthode — un gouffre de **volume de diversité** + un levier qu'on n'a pas vu (Driver-drop).

## 2. L'architecture gagnante (stacking 2 niveaux)

```
Niveau 0 (base models)    ~30 familles × variantes FE × variantes HP  → 186 OOFs
Niveau 1 (ensemblers)     Autogluon (best) ≈ LR-Logits (close 2nd)
Niveau final              blend 50/50 (Autogluon, LR-Logits) = 0.95503
```

- **186 OOFs** au total (objectif 200, pas atteint). Familles citées : XGB, LGBM, CatBoost,
  RealMLP, TabM, HGB, RF, YDF, FT-Transformer, MLP-PLR, + AutoML (Autogluon, LightAutoML,
  FLAML, PyTabKit). Métriques variées mais AUC dans la vaste majorité.
- **Ensembleur = Autogluon**, talonné par **LR sur logits** (logistic regression sur les
  log-odds des OOFs, pas sur les probas brutes — linéarise l'espace AUC).
- **Le blend de deux ensembleurs forts bat chacun pris seul** : 50/50 Autogluon+LR-Logits.
- **186 OOFs = 0.95487/0.95503** ; **182 OOFs L1-pure = 0.95480/0.95495**. Les **4 OOFs L2**
  (stacking récursif) ajoutent +0.00008 privé — mais avec un risque de leak → il a gardé une
  submission L1-pure en **hedge** (exactement notre logique de final-B décorrélé).

## 3. Le VRAI breakthrough — distribution shift sur `Driver`

C'est le point central, et c'est **ce qu'on a manqué**.

1. Après 3 semaines, plateau public ~0.95470 — « spinning my wheels ».
2. Retour à la planche : relecture de la **relation dataset original ↔ dataset compétition**.
   Plusieurs (siukeitin, ravi20076) avaient montré que les deux datasets diffèrent.
3. **Adversarial validation par variable** (ravi20076) : entraîner un classifieur orig-vs-comp,
   regarder l'AUC par feature. **`Driver` ressort comme la variable la plus différente.**
4. Conséquence appliquée : entraîner des modèles **`Driver`-dropped** + certains **sans la
   donnée originale du tout** → boost après le plateau.
5. **Sample weights sur la donnée externe** : poids ∈ [0.5, 1.0] ≈ même score ; **0.25 dégrade**
   CV et LB.

> **Lien avec nos mesures** : on a mesuré que garder Aadi (externe) aide +0.078pp (D5, ablation),
> et on l'a gardé — cohérent avec « weights 0.5-1.0 OK, 0.25 dégrade ». MAIS on n'a JAMAIS testé
> l'adversarial validation par feature, donc on n'a pas vu que `Driver` était le poste de shift.
> On avait noté « Driver 887 hashé, pas de levier externe » (D10) → c'était vrai pour l'*ajout*
> d'info, FAUX pour le *retrait* : drop Driver = un levier de robustesse au shift, pas d'info.

## 4. HPO & génération d'OOFs (du thread de commentaires)

Question de Masaya (8ème) : comment fixe-t-il les HPs avant d'ajouter un OOF au pool ?

Réponse d'Optimistix — **diversité et vitesse > tuning parfait** :
- **Pas de HPO systématique.** Part des HPs d'anciens modèles ou de bons notebooks publics,
  tweake quelques HPs importants.
- Optuna / FLAML seulement *parfois* (FLAML « great for quick HPO »), souvent au début.
- Tuning fait sur **notebook rapide** : sinon restreint à un **subsample** ou **1-2 folds**.
- **Garde souvent plusieurs jeux de HPs du même notebook** → la variance HP *elle-même* devient
  une source de diversité.
- Invoque @cdeotte : les Kagglers très expérimentés ne passent pas beaucoup de temps sur le HPO
  non plus.

Réponse de Masaya (sa propre pratique, complémentaire) :
- Tuning manuel léger des params clés d'abord (XGB : `max_depth`, `colsample_bytree`, `max_bin`),
  **fixé** pendant l'exploration des feature sets, puis Optuna **sérieux en étape finale** une
  fois un bon FE trouvé.
- Hypothèse validée par l'échange : **garder plusieurs variantes HP (arbres shallow + deep) du
  même FE+modèle ajoute de la diversité utile** — vu chez cdeotte aussi.

> **Implication directe pour nous** : notre conclusion « tuner RealMLP ne le décorrèle pas
> (corr v1/v2 = 0.999) » (D7) est vraie *pour cette paire de configs*, mais on a généralisé trop
> vite à « le tuning n'apporte rien ». Le vainqueur n'utilise pas le tuning pour *améliorer* un
> modèle — il utilise les **variantes HP comme générateur de diversité d'OOFs**. C'est un usage
> orthogonal qu'on n'a pas exploité.

## 5. Ce que le vainqueur a fait et qu'on n'a PAS fait

| Levier | Vainqueur | Nous | Verdict |
|---|---|---|---|
| Nb de base models / OOFs | **186** | 2 (RealMLP, TabR) + diagnostics | **écart #1** |
| Familles d'archi | ~30 (GBDT, NN, RF, FT-Transformer, AutoML…) | 4 testées (GBDT/TabPFN/TabM/TabR) | volume |
| Méta-ensembleur | Autogluon + LR-Logits, blend 50/50 | rank-blend 2-vecteurs w=0.10 | sophistication |
| Adversarial validation par feature | **oui → drop Driver** | non | **levier manqué** |
| Variantes HP comme diversité | **oui** | non (jugé inutile à tort) | usage manqué |
| Sample-weight sur externe | testé (0.5-1.0 OK, 0.25 KO) | gardé Aadi w=1 (ablation +0.078pp) | aligné |
| L2 stacking + hedge L1-pure | oui | analogue (final-B décorrélé) | aligné |
| LB probing | évité (discipline) | évité (discipline) | **aligné** ✅ |

## 6. Ce qu'on a BIEN fait (à ne pas dévaloriser)

- **Discipline LB-only + no grinding** : identique au vainqueur (« I'd been careful not to do any
  LB probing »). On n'a pas overfit le public → shakeup −40 places seulement (modéré).
- **Hedge décorrélé en final-B** : même principe que sa submission L1-pure gardée contre le leak L2.
- **Diagnostic du mirage CatBoost-CTR** (D2) : compréhension fine qui nous a évité de ship le
  mirage OOF 0.979. Rigueur méthodologique réelle.
- **TabR = vrai gain de diversité** : 4ᵉ famille (retrieval), 1er partenaire non-dégradant,
  cohérent public ET privé. La thèse « diversité d'archi est le levier » était la BONNE thèse —
  on a juste manqué de **volume** pour la pousser à l'échelle du vainqueur.

## 7. Leçons transférables (prochaine compétition)

1. **Adversarial validation systématique orig-vs-comp ET train-vs-test, par feature.** C'est le
   diagnostic qui a débloqué le #1. Cheap, à faire en EDA. → variable la plus « shiftée » =
   candidate au drop, pas à l'ajout.
2. **Penser en POOL d'OOFs, pas en « meilleur modèle ».** L'objectif n'est pas un modèle parfait
   mais 100+ OOFs honnêtes et diversifiés. Industrialiser la génération (template kernel
   paramétré par {FE set, modèle, HP variant, seed}).
3. **Variantes HP = diversité gratuite.** Ne pas tuner pour la perf d'un modèle ; lancer 2-3
   configs (shallow/deep, plus/moins de régul) et tout verser au pool.
4. **Méta-ensembleur appris (Autogluon / LR-Logits sur logits) > rank-blend manuel** dès qu'on a
   > ~10 OOFs. Le LR-Logits est trivial à mettre en place et finit 2ème ensembleur.
5. **Tuning rapide = subsample / 1-2 folds**, jamais full-data — vitesse > perfection (notre
   contrainte laptop 15.4 GB rendait ça d'autant plus pertinent, on aurait dû systématiser).
6. **Le shakeup récompense la robustesse, pas le score public.** Notre SE-analysis (D10) était
   correcte ; sa conclusion « le top public est un nuage de bruit » s'est vérifiée (on a glissé
   de 40 places sans changer de régime). Garder un final décorrélé reste la bonne assurance.

## 8. Honnêteté finale

On a fini **523ème**, à ~1·SE du vainqueur. Le projet était pédagogique (1 t-shirt), pas un
leaderboard hunt — et sur ce plan il a réussi : mirage CV diagnostiqué et mesuré, LB-reversal
documenté, 4 familles d'archi testées proprement, TabR validé comme diversité réelle, discipline
anti-overfit tenue. Le **delta au podium est un delta d'échelle de diversité** (2 vs 186 OOFs) et
**un levier de diagnostic manqué** (adversarial-validation → Driver-drop), pas un delta de rigueur.
Les deux sont des compétences reproductibles, identifiées, et bankées pour la prochaine fois.
