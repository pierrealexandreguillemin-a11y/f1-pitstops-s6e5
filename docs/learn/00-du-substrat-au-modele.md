# Module 00 — Du substrat au modèle : poser les objets

> Série « du bare-metal à la prédiction d'arrêt au stand ». Niveau : tu connais train/test/
> k-fold et tu lis des cours intro-ML/DL ; ici on **pose proprement les objets mathématiques**
> qui sous-tendent tout le reste, avec chaque symbole défini. Peu d'analogies (une analogie
> forcée est un mensonge sur la réalité scientifique). Pré-requis : algèbre de base (vecteurs,
> somme), notion de fonction.

## Conventions de notation (utilisées dans toute la série)

| Symbole | Type | Signification |
|---|---|---|
| $n$ | $\mathbb{N}$ | nombre d'exemples (lignes du dataset). Ici $n = 439\,140$ en train. |
| $d$ | $\mathbb{N}$ | nombre de *features* (colonnes numériques en entrée). Ici $d \approx 41$ après préparation. |
| $x \in \mathbb{R}^d$ | vecteur | un exemple : les $d$ nombres décrivant **une** ligne (un tour de course). |
| $x_j \in \mathbb{R}$ | scalaire | la $j$-ième feature de $x$ (ex. $x_j = $ `TyreLife`). $j \in \{1,\dots,d\}$. |
| $X \in \mathbb{R}^{n\times d}$ | matrice | le dataset entier : $n$ lignes empilées, chaque ligne un $x$. |
| $y \in \{0,1\}$ | scalaire | la *cible* (label) d'un exemple : 1 = arrêt au stand au tour suivant, 0 sinon. |
| $\theta \in \mathbb{R}^p$ | vecteur | les **paramètres** du modèle ($p$ nombres). Pour RealMLP, $p \approx 683\,000$. |
| $f_\theta$ | fonction | le **modèle** : une fonction de $\mathbb{R}^d$ vers $[0,1]$, *paramétrée* par $\theta$. |
| $\hat{y} \in [0,1]$ | scalaire | la **prédiction** : $\hat{y} = f_\theta(x)$, interprétée comme une probabilité. |

## 1. Le substrat : ce que la machine sait calculer

Un processeur exécute, sur des nombres codés en binaire, un répertoire minuscule d'opérations
arithmétiques (addition, multiplication) et logiques. Les nombres décimaux (un temps au tour
$76.491$) sont stockés au format **virgule flottante** : un réel $r$ est approché par

$$r \approx (-1)^{s}\,(1 + m)\,2^{\,e}$$

où $s\in\{0,1\}$ est le signe, $m\in[0,1)$ la *mantisse* (les chiffres significatifs) et
$e\in\mathbb{Z}$ l'*exposant* (l'ordre de grandeur), le tout codé sur 32 bits (`float32`). La
conséquence pratique qui te suivra partout : **les calculs sont approchés**, pas exacts
(d'où, plus tard, les soucis de précision numérique). Mais l'essentiel pour ce module :

> La machine ne manipule que des **tableaux de nombres** via additions et multiplications.
> « Apprentissage automatique » n'ajoute aucune capacité primitive — il *organise* ces
> opérations.

## 2. L'opération atomique : le produit scalaire

Presque tout le calcul d'un réseau se ramène à une seule opération, le **produit scalaire**
entre un vecteur de paramètres $w \in \mathbb{R}^d$ et un exemple $x \in \mathbb{R}^d$ :

$$z = \langle w, x\rangle = w^\top x = \sum_{j=1}^{d} w_j\,x_j$$

- $w_j$ : le poids associé à la feature $j$ (combien cette feature « compte »),
- $x_j$ : la valeur de la feature $j$ pour cet exemple,
- $z \in \mathbb{R}$ : un score scalaire, combinaison linéaire pondérée des features.

Empiler beaucoup de tels produits scalaires (une matrice $W$ de poids) puis intercaler des
fonctions non linéaires : c'est l'intégralité de l'architecture d'un réseau. On déroule ça au
module 02.

## 3. Ce qu'est un « modèle », formellement

Un modèle est une **fonction paramétrée**

$$f_\theta : \mathbb{R}^d \longrightarrow [0,1], \qquad x \longmapsto \hat{y} = f_\theta(x).$$

- l'**entrée** $x$ : les $d$ features d'un tour de course,
- la **sortie** $\hat{y}\in[0,1]$ : la probabilité estimée d'un arrêt au stand au tour suivant,
- les **paramètres** $\theta$ : les $p$ nombres ($\approx 683\,000$) qui *définissent* la
  fonction. Deux $\theta$ différents = deux fonctions différentes.

Physiquement, dans la RAM, $\theta$ est exactement ce qui existe : un grand tableau de `float32`.
« Faire une prédiction » = évaluer $f_\theta(x)$, c.-à-d. enchaîner des produits scalaires et des
non-linéarités sur les nombres de $x$ et $\theta$.

## 4. Ce qu'est « apprendre », formellement

On dispose d'exemples étiquetés $\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$ (l'indice $(i)$ = numéro de
l'exemple). On choisit une **fonction de perte** (*loss*) $\ell(\hat{y}, y)$ qui mesure l'écart
entre une prédiction et la vérité. Le coût total sur le dataset est la perte moyenne :

$$L(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x^{(i)}),\, y^{(i)}\big).$$

**Apprendre = résoudre un problème d'optimisation** : trouver les paramètres qui minimisent ce
coût,

$$\theta^\star = \arg\min_{\theta \in \mathbb{R}^p} L(\theta).$$

- $L(\theta)$ : une surface dans un espace à $p \approx 683\,000$ dimensions ; un seul nombre
  réel (le coût) en fonction des $p$ paramètres,
- $\arg\min$ : « la valeur de $\theta$ qui rend $L$ minimal »,
- on n'a pas de formule fermée pour $\theta^\star$ : on l'**approche** itérativement (descente de
  gradient, module 03). C'est *là* que la dérivée intervient — exactement ce que tu appelais
  « les maths qui s'impriment sur les couches » : à chaque itération, on met à jour $\theta$.

Au départ $\theta$ est initialisé aléatoirement : $L(\theta)$ est élevé, les prédictions sont
mauvaises. L'optimisation fait décroître $L$.

## 5. Raccrochage à ce qu'on a mesuré

- `Total params: 683 K` (vu dans le log TabR) $= p$, la dimension de $\theta$.
- La saturation RAM de ton laptop : il faut loger en mémoire la matrice $X \in
  \mathbb{R}^{n\times d}$ ($439\,140 \times 41$ `float32` $\approx$ 72 Mo, gérable) **plus** les
  états intermédiaires du réseau pour les $n$ exemples pendant l'optimisation (gradients,
  activations), qui eux explosent — d'où le passage sur Kaggle (16 Go + GPU).
- Le GPU vs CPU : un GPU exécute des milliers de produits scalaires $\langle w,x\rangle$ en
  parallèle (algèbre matricielle massivement parallèle), d'où l'accélération d'un facteur 10–100
  sur l'évaluation de $f_\theta$ et de ses gradients.

## À retenir (formulation compacte)

1. Modèle = fonction paramétrée $f_\theta:\mathbb{R}^d\to[0,1]$, avec $\theta\in\mathbb{R}^p$ un
   tableau de $\approx 683\text{k}$ réels.
2. Le calcul élémentaire est le produit scalaire $z=\langle w,x\rangle=\sum_j w_j x_j$.
3. Apprendre = $\theta^\star=\arg\min_\theta\frac1n\sum_i \ell(f_\theta(x^{(i)}),y^{(i)})$,
   résolu *itérativement*.

## Suite

**Module 01 — Du tableur au dataset** : construire concrètement $X \in \mathbb{R}^{n\times d}$ à
partir de colonnes mixtes (`Driver` texte, `Compound` catégoriel, `LapTime` réel). On
formalisera l'encodage one-hot (vecteurs de base $e_k$), l'encodage par cible comme espérance
conditionnelle $\mathbb{E}[y \mid \text{catégorie}]$, et la standardisation
$z = (x-\mu)/\sigma$ — et on verra mathématiquement *où* notre « mirage » CV prend racine.
