# Référence technique — eoqual-core

Principe, conventions et limites de chaque brique. Le code de chaque
fonction porte le détail (docstrings) ; ce document donne la vue
d'ensemble et les pièges.

## 1. Conventions communes

- Images 2D `(M, N)`, indices `(ligne, colonne)`.
- Fréquences en cycles par unité d'échantillonnage : cycles/pixel pour une
  image, 0.5 = Nyquist.
- MTF normalisées à 1 en fréquence nulle.
- Aucune fonction ne modifie ses entrées.

## 2. `fourier` — TFD d'image sans artefacts de bord

**Problème.** La TFD suppose l'image périodique. Les bords opposés d'une
image réelle ne se raccordent pas : le pavage du plan présente des
coutures, dont la TF forme une « croix » sur les axes du spectre,
étrangère au contenu de l'image. Elle fausse toute analyse fréquentielle
(netteté, MTF, bruit, détection de raies).

**`perdecomp` (Moisan, 2011).** Décompose `u = p + s` : `s` est lisse,
de moyenne nulle, solution d'une équation de Poisson périodique dont le
second membre ne vit que sur le bord ; `p` est périodique et garde tout
le contenu de l'image. Le laplacien périodique étant diagonal en Fourier,
le calcul coûte `O(MN log MN)`. Caractérisation testée :
`Δ_per p = Δ_intérieur u`.

**`translate`.** Translation sous-pixel par déphasage
`exp(-2iπ(ξ_y dy + ξ_x dx))` (interpolation de Shannon, périodique).
À appliquer sur une image (quasi) périodique, sinon les sauts de bord
produisent des oscillations de Gibbs : `perdecomp` d'abord. Pour une
taille paire, la ligne de Nyquist n'a pas de partenaire symétrique : la
partie réelle est gardée.

**`dequantize`.** Translation de (1/2, 1/2) pixel : les valeurs ne sont
plus entières, ce qui supprime la structure artificielle de la
quantification (utile aux indices fondés sur la phase). Ce n'est pas la
multiplication par `(-1)^(x+y)`, qui décale le spectre sans déplacer
l'image.

**`spectrum`.** Spectre centré avec prétraitement `raw`, `perdecomp`
(recommandé) ou `hann` (atténue les bords mais élargit les raies).

**`boundary_jump`.** Saut moyen entre bords opposés : diagnostic de
périodicité, de l'ordre des différences entre voisins pour une image bien
raccordée.

## 3. `edge` — du profil de bord à la MTF

Le module part d'un profil **déjà projeté** sur la normale au bord ; la
détection du bord reste chez l'appelant.

1. **`bin_profile`** : moyenne par bins de `1/oversample` pixel des
   échantillons projetés ; bins vides comblés par interpolation linéaire ;
   `None` si plus de la moitié des bins sont vides.
2. **`esf_to_lsf`** : dérivation (`gradient` ou Savitzky-Golay), puis
   fenêtrage/lissage (`tukey` par défaut, `savgol`, `gaussian`, `none`).
3. **`lsf_to_mtf`** : module de la FFT de la LSF, **normalisé par la
   composante continue** (intégrale de la LSF = saut de l'ESF) ;
   `n_fft` permet un zéro-padding (grille de fréquences plus fine, sans
   information nouvelle).
   Normaliser par le maximum sur les seules fréquences positives — variante
   courante — normalise en fait par la valeur au premier bin
   `1/(N dx)` : la MTF est surestimée partout d'environ
   `1/MTF(1/(N dx))`, soit ~1 % pour une PSF gaussienne de 1 px sur un
   profil de 40 px.
4. **Corrections** : `derivative_correction` divise par `sinc(f dx)`
   (atténuation des différences finies) ; `pixel_aperture_correction`
   divise par `sinc(f p)` pour retirer la MTF du pixel. Toutes deux
   amplifient le bruit près de leurs zéros.
5. **Grandeurs dérivées** : `mtf_at` (MTF@Nyquist avec 0.5),
   `frequency_at` (MTF50 avec 0.5 ; NaN si jamais atteint), `fwhm`, `rer`.

**FWHM.** Prendre le premier et le dernier point au-dessus du
demi-maximum sur tout le domaine casse dès qu'un rebond de bruit isolé,
loin du pic, repasse au-dessus du seuil (observé : 22 px au lieu de
2-3 px). `fwhm` prend la première traversée de part et d'autre du pic.

**RER** (Relative Edge Response, GIQE 5) : `ESF(c + 0.5) - ESF(c - 0.5)`
sur une ESF normalisée dans [0, 1], sans FFT ni fenêtrage.

## 4. `bayer` — motifs de Bayer

Une seule table, `BAYER_PATTERNS`, donne la position de chaque composante
dans le bloc 2x2 pour `RGGB`, `BGGR`, `GRBG`, `GBRG` (nom lu ligne par
ligne depuis le pixel en haut à gauche : `GBRG` = `G B` / `R G`).
`extract_bayer_planes`, `bayer_masks` et `bayer_offsets` en dérivent
toutes, ce qui interdit les incohérences entre fonctions (un test vérifie
que chaque motif correspond à son nom).

Pour une mosaïque de dimension impaire, la dernière ligne ou colonne
(bloc 2x2 incomplet) est ignorée : tous les plans ont la forme
`(H // 2, W // 2)`.

Un plan extrait est à moitié résolution : ses fréquences, exprimées par
pixel du plan, se convertissent par pixel physique en les multipliant par
`BAYER_SAMPLING_FACTOR = 0.5`. Le plan `G` (moyenne de G1 et G2) mélange
deux positions physiques : son décalage `bayer_offsets()["G"]` est une
approximation à une fraction de pixel près.

## 5. `psf` — qualité d'une PSF

- `airy_psf` : PSF de diffraction d'une pupille circulaire non obturée,
  somme 1, échelle donnée en rad/pixel.
- `strehl_ratio` : rapport des pics. N'a de sens que si les deux PSF ont
  même normalisation, même échantillonnage et même bande spectrale ; un
  rapport > 1.02 émet un avertissement (incohérence, pas optique
  « meilleure que la diffraction »).
- `radial_profile` : moyenne par rayon entier.
- `encircled_energy` : fraction de l'énergie dans un disque de rayon `r`
  (somme des pixels dont le centre est dans le disque). Vérifié sur Airy :
  0.838 au premier anneau noir (1.22 λ/D). Le fond doit être retiré
  avant : un fond résiduel fausse la normalisation.

## 6. `detrend` et `tiling`

- `fit_plane_residual` : résidu après ajustement d'un plan par moindres
  carrés, pour ne pas compter un gradient d'éclairement comme du bruit.
- `tile_bounds` / `tile_slices` : tuiles de taille fixe chevauchantes,
  dernière tuile alignée sur le bord ; `grid_slices` : grille `n x n`
  sans recouvrement (reste de la division ignoré).

## 7. Bibliographie

- L. Moisan, « Periodic plus smooth image decomposition », *J. Math.
  Imaging Vision* 39(2), 2011, p. 161-179.
- G. Blanchet, L. Moisan, « An explicit sharpness index related to global
  phase coherence », *ICASSP* 2012, p. 1065-1068 (déquantification).
- ISO 12233:2017, *Photography — Electronic still picture imaging —
  Resolution and spatial frequency responses*.
- P. D. Burns, « Slanted-Edge MTF for Digital Camera and Scanner
  Analysis », *Proc. IS&T PICS*, 2000, p. 135-138.
- L. Harrington et al., « General Image Quality Equation: GIQE Version
  5 », NGA, 2015.
- M. Born, E. Wolf, *Principles of Optics*, Cambridge University Press
  (tache d'Airy, énergie encerclée).
- V. N. Mahajan, « Strehl ratio for primary aberrations: some analytical
  results for circular and annular pupils », *JOSA* 72(9), 1982,
  p. 1258-1266.
