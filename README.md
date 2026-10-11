<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-core

Briques numériques communes aux modules EOQual : transformée de Fourier
d'image sans artefacts de bord, chaîne ESF → LSF → MTF, motifs de Bayer,
PSF (Strehl, énergie encerclée), détrend, découpage en tuiles.

Voir **[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le
principe et les limites de chaque brique.

Pour la transformée de Fourier d'image, une note détaillée
([docs/NOTE_TF_image.md](docs/NOTE_TF_image.md)) et un notebook
d'illustrations ([examples/TF_image_demo.ipynb](examples/TF_image_demo.ipynb) ;
nécessite matplotlib et eoqual-metrics pour la partie indice de netteté)
complètent la référence technique.

## Sommaire

1. [Installation](#installation)
2. [Contenu](#contenu)
3. [Règles](#règles)
4. [Utilisateurs](#utilisateurs)
5. [Licence](#licence)

---

## Installation

```bash
pip install git+https://github.com/EOQual/eoqual-core.git
```

Dépendances : numpy et scipy, rien d'autre.

## Contenu

| Module | Fonctions | Usage typique |
|---|---|---|
| `eoqual.core.fourier` | `perdecomp`, `translate`, `dequantize`, `frequencies`, `spectrum`, `boundary_jump` | Spectre sans « croix » de bord, translation sous-pixel, prétraitement des indices de phase |
| `eoqual.core.edge` | `bin_profile`, `esf_to_lsf`, `lsf_to_mtf`, `derivative_correction`, `pixel_aperture_correction`, `mtf_at`, `frequency_at`, `fwhm`, `rer` | MTF par bord (mire ou arête naturelle) une fois le profil projeté |
| `eoqual.core.bayer` | `BAYER_PATTERNS`, `BAYER_SAMPLING_FACTOR`, `bayer_offsets`, `bayer_masks`, `extract_bayer_planes` | Mosaïques RGGB/BGGR/GRBG/GBRG |
| `eoqual.core.psf` | `airy_psf`, `strehl_ratio`, `radial_profile`, `encircled_energy` | Qualité d'une PSF mesurée |
| `eoqual.core.detrend` | `fit_plane_residual` | Retrait d'un gradient d'éclairement sur zone uniforme |
| `eoqual.core.tiling` | `tile_bounds`, `tile_slices`, `grid_slices` | Calcul par tuiles |

```python
from eoqual.core.fourier import perdecomp, spectrum
from eoqual.core.edge import esf_to_lsf, lsf_to_mtf, frequency_at

p, s = perdecomp(image)                       # composante périodique + lisse
A, fy, fx = spectrum(image)                   # spectre centré, sans croix

lsf = esf_to_lsf(esf, dx=0.25)
f, mtf = lsf_to_mtf(lsf, dx=0.25)
frequency_at(f, mtf, 0.5)                     # MTF50
```

## Règles

Ce paquet est une dépendance des autres modules : il doit rester léger et
stable.

- **numpy et scipy seulement** — pas d'OpenCV, de scikit-image, de torch
  ni de bibliothèque de tracé ;
- **fonctions pures** sur tableaux : pas d'entrées/sorties, de CLI, de
  registre de méthodes ni de journalisation ;
- une brique n'entre ici que si **deux modules au moins** l'utilisent ;
- **API stable** : les modules consommateurs fixent une plage de version
  (`eoqual-core = ">=0.1,<0.2"`) ; toute rupture d'API change la version
  mineure tant que la version majeure vaut 0.

Les indicateurs (indice de netteté SI, scores) n'ont pas leur place ici :
ils relèvent d'``eoqual-metrics``, qui peut en revanche s'appuyer sur ces
briques.

## Utilisateurs

| Module | Briques utilisées |
|---|---|
| eoqual-measure | `edge`, `bayer`, `detrend`, `tiling`, `psf`, `fourier` (recentrage sous-pixel) |
| eoqual-metrics | `fourier` (indice SI), `edge` (AEM, SaSbEM), `tiling` |
| eoqual-demosaicing | `bayer` |
| eoqual-phaseretrieval | `psf` (Airy, Strehl, profil radial) |

## Licence

MIT — voir `license.txt` et `THIRD_PARTY_LICENSES.md`.
