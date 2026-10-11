# Note technique : transformée de Fourier d'une image, décomposition périodique, déquantification et indice de netteté

*eoqual-core. Code : [`eoqual.core.fourier`](../src/eoqual/core/fourier.py) (TFD, décomposition périodique, translation, déquantification) ; l'indice de netteté SI est un indicateur et vit dans eoqual-metrics (`eoqual.metrics.nr.sharpness.backends.phase.sharpness_index`). Tests : [`tests/test_fourier.py`](../tests/test_fourier.py) et, pour le SI, `tests/test_sharpness_index.py` d'eoqual-metrics. Illustrations : [`examples/TF_image_demo.ipynb`](../examples/TF_image_demo.ipynb).*

---

## 1. Le problème : la TFD suppose la périodicité

La transformée de Fourier discrète (TFD) d'une image $u$ de taille $M \times N$ est

$$
\hat u(\xi_y, \xi_x) = \sum_{y=0}^{M-1}\sum_{x=0}^{N-1} u(y,x)\, e^{-2i\pi\left(\frac{\xi_y y}{M} + \frac{\xi_x x}{N}\right)} .
$$

Elle est exacte (inversible), mais elle décrit un signal **périodique** de période $(M, N)$. Analyser $u$ par TFD revient donc à analyser le **pavage infini du plan** par $u$.

Sur une image réelle, les bords opposés ne se raccordent pas : éclairage non uniforme, objets coupés, etc. Le pavage présente alors des **coutures**, c'est-à-dire des discontinuités le long de lignes verticales et horizontales. Or :

* une marche verticale a une TF concentrée sur l'axe $\xi_y = 0$, avec une amplitude qui décroît lentement (en $1/|\xi_x|$) ;
* une marche horizontale donne de même l'axe $\xi_x = 0$.

D'où la **croix** caractéristique dans le spectre. Elle n'a rien à voir avec le contenu de l'image et fausse toute analyse fréquentielle : mesure de netteté, estimation de FTM ou de bruit, filtrage, détection de raies, énergie haute fréquence. Dans l'image test du notebook, le saut moyen entre bords opposés (0.54) est **20 fois** plus grand que la différence moyenne entre pixels voisins (0.025). Sur l'axe $\xi_y = 0$, l'amplitude spectrale est ×3.7 trop forte.

## 2. La solution : décomposition périodique + lisse (Moisan, 2011)

On écrit

$$
u = p + s ,
$$

où :

* $p$ est la **composante périodique**. Elle se raccorde sur le tore et contient tous les détails de $u$. C'est elle qu'on analyse par TFD.
* $s$ est la **composante lisse** (presque harmonique, de moyenne nulle). Elle ne porte que la variation basse fréquence qui absorbe les sauts de bord.

### 2.1 Définition

Notons $\Delta$ le laplacien discret **périodique** (les voisins sont pris modulo $M$ et $N$) et $\Delta_i$ le laplacien **intérieur** (seuls les voisins situés dans l'image sont comptés). Moisan définit $p$ comme l'unique image telle que

$$
\Delta p = \Delta_i u , \qquad \operatorname{moy}(p) = \operatorname{moy}(u).
$$

$p$ a donc exactement les mêmes variations locales que $u$ à l'intérieur de l'image, mais elle est périodique.

### 2.2 Calcul

Comme $s = u - p$, on a $\Delta s = \Delta u - \Delta_i u =: v$. L'image $v$ n'est non nulle que sur le bord, où elle vaut le saut « de l'autre côté du tore » :

| pixel | contribution à $v$ |
|---|---|
| première ligne $y=0$ | $+\big(u(M-1,x) - u(0,x)\big)$ |
| dernière ligne $y=M-1$ | $+\big(u(0,x) - u(M-1,x)\big)$ |
| première colonne $x=0$ | $+\big(u(y,N-1) - u(y,0)\big)$ |
| dernière colonne $x=N-1$ | $+\big(u(y,0) - u(y,N-1)\big)$ |

Les coins cumulent deux contributions. Le laplacien périodique est **diagonal en Fourier**, de valeurs propres $2\cos(2\pi\xi_y/M) + 2\cos(2\pi\xi_x/N) - 4$. D'où :

$$
\hat s(\xi) = \frac{\hat v(\xi)}{2\cos\frac{2\pi\xi_y}{M} + 2\cos\frac{2\pi\xi_x}{N} - 4} \quad (\xi \ne 0), \qquad \hat s(0) = 0, \qquad p = u - s .
$$

Le coût est celui de deux FFT, en $O(MN \log MN)$.

### 2.3 Propriétés utiles

* $p$ conserve les bords internes, la texture et les raies de $u$. Seule une composante lisse est retirée.
* $s = 0$ **si et seulement si** les bords opposés sont égaux ($v = 0$). Une image déjà périodique *au sens échantillonné* (par exemple une sinusoïde à nombre entier de périodes) n'a pas $u(0,\cdot) = u(M-1,\cdot)$ : $s$ y est petit mais pas nul. Ce n'est pas un défaut, la décomposition reste inoffensive.
* Pour un gradient purement horizontal (problème 1D), $s$ est **affine** : une fonction harmonique 1D est une droite. $p$ est alors **constante**, égale à la moyenne, car tout le gradient est absorbé par $s$. En 2D, $s$ est seulement presque harmonique.
* Le spectre de $p$ ne présente plus de croix. Celui de $s$ est concentré aux basses fréquences et décroît rapidement.

### 2.4 Comment l'utiliser : analyse ou traitement

Il n'y a pas d'« algorithme inverse » : revenir à $u$, c'est rajouter $s$.

* **Analyse du spectre** (netteté, bruit, FTM, raies, énergie haute fréquence) : on calcule la TFD de $p$ et on s'arrête là. Aucun retour n'est nécessaire.
* **Traitement dans Fourier puis retour** (filtrage, déconvolution de Wiener ou Tikhonov) : on traite $p$, puis on rajoute $s$, soit $u_f = \mathcal F^{-1}(H \cdot \hat p) + s$. Comme $s$ ne contient presque que des basses fréquences, un flou ou un filtre le modifie peu. Si le traitement agit sur les basses fréquences (passe-haut, par exemple), on traite $s$ à part dans le domaine spatial ou on l'écarte. Sans cette précaution, une déconvolution amplifie la croix de bord et produit des oscillations le long des bords.
* **Cas où ce n'est pas nécessaire** : image déjà raccordée (`boundary_jump` de l'ordre des différences entre pixels voisins), analyse par blocs fenêtrés (méthode de Welch), convolution par FFT avec des zéros ajoutés autour de l'image (*zero-padding*).

### 2.5 Comparaison avec le fenêtrage (Hann, Hamming…)

Multiplier par une fenêtre qui s'annule sur les bords supprime aussi la croix, mais :

* **l'image est modifiée partout** : les bords sont écrasés et l'information qui s'y trouve est perdue ;
* le spectre est **convolué** par la TF de la fenêtre, ce qui élargit les raies et dégrade la résolution fréquentielle ;
* **l'énergie est biaisée** (il faut la renormaliser), ce qui gêne les mesures quantitatives.

`perdecomp` évite ces trois défauts. Le fenêtrage reste pertinent pour une analyse par blocs, où l'on cherche justement à réduire la variance.

## 3. Déquantification : translation de $(1/2, 1/2)$ pixel

Les niveaux de gris d'une image numérique sont **quantifiés** (entiers). Pour les mesures calculées dans le domaine spatial, comme la variation totale de l'indice de netteté (§5), cette structure entière introduit un biais. Les plateaux de valeurs identiques donnent des différences exactement nulles. La parade consiste à **translater l'image d'un demi-pixel dans chaque direction par interpolation de Shannon**. Les nouvelles valeurs sont interpolées entre pixels voisins : elles ne sont plus entières, et l'histogramme « en peigne » disparaît.

La translation se fait dans le domaine de Fourier, par un déphasage :

$$
\widehat{u_{\tau}}(\xi) = \hat u(\xi)\, e^{-2i\pi\left(\frac{\xi_y \tau_y}{M} + \frac{\xi_x \tau_x}{N}\right)}, \qquad \tau = (1/2, 1/2),
$$

avec les fréquences prises dans $[-M/2, M/2[ \times [-N/2, N/2[$ (`np.fft.fftfreq`).

**Effet sur le spectre.** Une translation ne modifie que la **phase**. Sur le spectre d'**amplitude**, la déquantification est donc :

* **sans effet** pour une taille impaire ;
* limitée à la **ligne et à la colonne de Nyquist** pour une taille paire. Une translation d'un demi-pixel n'y est pas représentable par une image réelle, et garder la partie réelle atténue ces coefficients, voire les annule.

L'option `spectrum(..., dequant=True)` n'agit donc vraiment qu'avec `output="complex"`. Son vrai usage est l'indice de netteté.

**Ordre des opérations.** L'interpolation de Shannon est périodique : appliquée à $u$ non périodique, elle produit des **oscillations de Gibbs** le long des bords. On déquantifie donc $p$, **après** `perdecomp`. Dans le notebook, l'erreur maximale de translation passe de 0.56 (Shannon sur $u$) à 0.03 (Shannon sur $p$ + spline sur $s$).

## 4. Axes de fréquences et Nyquist

Avec un pas d'échantillonnage d'un pixel, la fréquence d'échantillonnage vaut 1 cycle/pixel. **La fréquence de Nyquist vaut 0.5 cycle/pixel sur chaque axe, quelle que soit la taille de l'image.** Le spectre centré (`fftshift`) couvre $[-0.5, 0.5[$ sur chaque axe. En cycles par *largeur d'image*, cela correspond à $N/2$ (et $M/2$ verticalement).

La TFD ne contient rien « au-delà de Nyquist » : un éventuel repliement (aliasing) apparaît **sous** Nyquist, mêlé au vrai contenu. Ce qu'on mesure, c'est l'énergie au-dessus d'une fréquence de coupure $f_c$. Dans le spectre centré, le rayon `min(M, N) // 4` (en indices) vaut $f_c = 0.25$ cycle/pixel, soit Nyquist/2, et non Nyquist. Cette mesure se fait sur $p$ : sans `perdecomp`, la croix de bord gonfle l'énergie haute fréquence (0.91 % au lieu de 0.36 % dans le notebook).

## 5. Indice de netteté SI (Blanchet & Moisan, 2012)

### 5.1 Principe : la netteté se lit dans la phase

Si l'on garde l'amplitude de Fourier d'une image et qu'on tire les phases au hasard, les contours disparaissent et l'on obtient une texture. Les contours nets n'existent que parce que les phases sont **cohérentes**. Le flou, le bruit et le ringing détruisent cette cohérence.

La cohérence se mesure par la **variation totale** :

$$
TV(u) = \sum_x |u(x+e_1) - u(x)| + |u(x+e_2) - u(x)| .
$$

Une image nette a une TV nettement **plus faible** que ses versions à phase aléatoire, car son énergie est concentrée sur peu de contours.

* **GPC** (*Global Phase Coherence*, Blanchet, Moisan, Rougé, 2008) : $GPC(u) = -\log_{10} P\big(TV(u_\text{phase aléatoire}) \le TV(u)\big)$. La probabilité ne s'estime que par Monte-Carlo, ce qui est coûteux, et elle est inaccessible pour une image nette.
* **SI** : on remplace la phase aléatoire par le champ gaussien $V = u * W$, avec $W$ un bruit blanc de variance $1/MN$. $V$ a en moyenne le même spectre d'amplitude que $u$. Moyenne et variance de $TV(V)$ ont alors une **formule fermée**, d'où

$$
SI(u) = -\log_{10}\Phi\!\left(\frac{\mu - TV(u)}{\sigma}\right), \qquad \Phi(t) = P(\mathcal N(0,1) > t).
$$

* **Indice S** (Leclaire & Moisan, 2015) : une variante simplifiée et dérivable, utilisée pour la déconvolution aveugle. Elle n'est pas implémentée ici.

### 5.2 Formules fermées

Soit $g_x, g_y$ les gradients (différences avant **périodiques**), $\Gamma_{ab}(z) = \sum_w g_a(w)\, g_b(w+z)$ leurs autocorrélations croisées circulaires, et $\rho_{ab} = \Gamma_{ab} / \sqrt{\Gamma_{aa}(0)\Gamma_{bb}(0)}$. Chaque $g_a(V)(x)$ est gaussien et $E|\mathcal N(0, \alpha^2)| = \alpha\sqrt{2/\pi}$. Pour deux gaussiennes de corrélation $\rho$, $\operatorname{Cov}(|X|, |Y|) = \frac{2}{\pi}\alpha_X\alpha_Y\, \omega(\rho)$, avec $\omega(t) = t \arcsin t + \sqrt{1-t^2} - 1$. On en déduit :

$$
\mu = \sqrt{\tfrac{2}{\pi}}\sqrt{MN}\,\big(\|g_x\|_2 + \|g_y\|_2\big), \qquad
\sigma^2 = \frac{2}{\pi}\sum_{a,b \in \{x,y\}} \sqrt{\Gamma_{aa}(0)\Gamma_{bb}(0)}\; \sum_z \omega\big(\rho_{ab}(z)\big).
$$

Le calcul se fait en quelques FFT. **Pré-traitement** : `perdecomp`, car le modèle est circulaire et les coutures fausseraient $\mu$, $\sigma$ et la TV. Puis `dequantize`, car les plateaux entiers abaissent la TV et gonflent le SI : 1559 au lieu de 638 sur la scène 8 bits du notebook, 2569 au lieu de 453 sur 16 niveaux.

**Validation** (notebook, §7.1) : sur 2000 tirages, μ et σ empiriques de $TV(u * W)$ coïncident avec les formules (écart < 0.01 % sur μ et ≈ 1 % sur σ, dans l'erreur statistique). Avec une **vraie** phase aléatoire, la moyenne est la même, mais l'écart-type est environ **8 fois plus petit** : seule la phase varie, pas l'amplitude. Le SI est donc **plus prudent** que la GPC. Il sous-estime la significativité, mais l'ordre entre images est préservé.

### 5.3 Peut-on dire qu'une image est plus nette qu'une autre ?

**Oui, pour deux versions de la même scène, à taille égale et avec le même pré-traitement.** Le SI est alors un bon indicateur sans référence : réglage de mise au point, choix entre plusieurs restaurations, suivi d'une acquisition. Sur la scène structurée du notebook :

| flou gaussien σ (px) | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| SI | 638 | 396 | 92 | 40 | 23 | 11 | 7 |

| bruit σ (niveaux) | 0 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| SI | 638 | 535 | 371 | 79 | 4.8 | 0.5 |

Le SI est très sensible (un demi-pixel de flou suffit) et décroît avec le flou comme avec le bruit. Sur une scène pauvre en contours, il atteint aux flous forts un **plancher** de quelques unités, où il ne discrimine plus.

**Non, pour des scènes différentes ou des tailles différentes :**

* **Le contenu domine.** Une texture nette mais à phase « aléatoire » (herbe, sable, eau) a un SI de 0.4, moins que la scène structurée *floue* (23). L'image test des §1 à 4, nette mais dominée par sa texture, n'atteint que 5.
* **La taille compte.** Le SI est un −log de probabilité et croît avec la quantité de structure cohérente. À densité d'objets constante, il vaut 64, 262 puis 638 pour des imagettes de 64, 128 et 256 pixels de côté. Il faut comparer des imagettes de même emprise.
* **Ce n'est pas une mesure physique.** Le SI ne donne ni FTM ni largeur de PSF. C'est un score ordinal, complémentaire des méthodes par bord ou par cible.
* **Point de vigilance.** Le SI récompense la cohérence de phase. Un renforcement de netteté trop fort ou certaines restaurations peuvent l'augmenter sans que l'image soit plus fidèle. C'est le principe même de la déconvolution aveugle par maximisation de S, avec le même risque de sur-netteté.

## 6. Pièges courants

| Piège | Conséquence | Bonne pratique |
|---|---|---|
| Terme source de `perdecomp` construit avec `u[0] - u[-1]` au lieu de `u[-1] - u[0]` | on calcule $-s$, et $p$ **double** les sauts de bord (0.54 → 1.07 dans le notebook), tout en restant plausible à l'œil | contrôler `boundary_jump(p)` |
| Croire que multiplier par $(-1)^{x+y}$ décale l'image d'un demi-pixel | $(-1)^{x+y} = e^{i\pi(x+y)}$ est une **modulation** : elle translate le *spectre* de $(M/2, N/2)$, ce qui équivaut à un `fftshift` (astuce classique pour centrer le spectre). L'image n'est pas déplacée et la quantification n'est pas touchée | `dequantize` (déphasage de Fourier) |
| Exprimer Nyquist en « $N/2$ cycles par unité » | confusion entre cycles/image et cycles/pixel | Nyquist = 0.5 cycle/pixel |
| Rayon `min(M, N)//4` pris pour Nyquist | on mesure en fait au-dessus de Nyquist/2 | raisonner en cycles/pixel (`frequencies`) |
| Déquantifier ou translater $u$ non périodique | oscillations de Gibbs sur les bords | `perdecomp` d'abord |
| Comparer le SI de scènes ou de tailles différentes | conclusions sans fondement | même scène, même emprise |

## 7. Utilisation

```python
from eoqual.core.fourier import perdecomp, dequantize, translate, spectrum, frequencies
from eoqual.metrics.nr.sharpness.backends.phase.sharpness_index import (
    sharpness_index, sharpness_stats)   # eoqual-metrics

p, s = perdecomp(u)                                 # u = p + s
A, fy, fx = spectrum(u)                             # log(1+|TFD(p)|), axes en cycles/pixel
A, fy, fx = spectrum(u, method="raw")               # TFD brute (croix d'artefacts)
A, fy, fx = spectrum(u, method="hann")              # fenêtre de Hann
A, fy, fx = spectrum(u, output="amplitude")         # |TFD(p)|
U, fy, fx = spectrum(u, output="complex", dequant=True)  # TFD complexe de p déquantifiée
w = dequantize(p)                                   # translation (1/2, 1/2) par Shannon
w = translate(p, dy, dx)                            # translation sous-pixellique quelconque
si = sharpness_index(u)                             # perdecomp + dequantize + SI (pmode=3)
st = sharpness_stats(u)                             # {"tv", "mu", "sigma", "z", "si"}
si = sharpness_index(u, pmode=1)                    # perdecomp sans dequantize
```

* Entrée : image 2D (pour une image couleur, traiter chaque canal séparément). Calcul en `float64`.
* Dépendances : NumPy, et SciPy (`scipy.special.log_ndtr`, pour un $-\log_{10}\Phi$ stable quand le SI vaut plusieurs centaines).
* Tests : `pytest tests/test_fourier.py` (eoqual-core) et `pytest tests/test_sharpness_index.py` (eoqual-metrics). Ils vérifient :
  * $u = p + s$ et la caractérisation $\Delta p = \Delta_i u$, y compris en taille impaire ;
  * $s = 0$ quand les bords coïncident ;
  * les translations (translation entière égale à `np.roll`, deux demi-translations égales à une translation d'un pixel) ;
  * l'effet de `dequant` sur l'amplitude et la phase ;
  * μ et σ du SI contre un Monte-Carlo ;
  * la décroissance du SI avec le flou et le bruit.

## Références

* L. Moisan, *Periodic plus smooth image decomposition*, J. Math. Imaging Vision 39(2), 2011, p. 161-179.
* G. Blanchet, L. Moisan, B. Rougé, *Measuring the global phase coherence of an image*, ICIP 2008.
* G. Blanchet, L. Moisan, *An explicit sharpness index related to global phase coherence*, ICASSP 2012.
* A. Leclaire, L. Moisan, *No-reference image quality assessment and blind deblurring with sharpness metrics exploiting Fourier phase information*, J. Math. Imaging Vision 52(1), 2015, p. 145-172.
