# Licences tierces et provenance du code

**État actuel** : aucun code tiers n'est inclus ; les deux dépendances
sont permissives ; tout le code a été écrit par l'auteur, à partir de
méthodes publiées (voir la bibliographie de `REFERENCE_TECHNIQUE.md`).

## Dépendances

Licences relevées dans les métadonnées des paquets installés
(`importlib.metadata`, 2026-10-11).

| Paquet | Licence | Statut |
|---|---|---|
| numpy | BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0 pour des composants) | 🟢 |
| scipy | BSD-3-Clause | 🟢 |

## Provenance du code

| Module | Origine | Méthodes publiées |
|---|---|---|
| `fourier` | écrit pour ce paquet | Moisan (2011) ; Blanchet & Moisan (2012) |
| `edge` | eoqual-measure, eoqual-metrics (auteur) | ISO 12233 ; Burns (2000) ; Harrington et al. (2015) |
| `bayer` | écrit pour ce paquet | — |
| `psf` | écrit pour ce paquet, sur le modèle d'eoqual-phaseretrieval (auteur) | Born & Wolf ; Mahajan (1982) |
| `detrend`, `tiling` | eoqual-measure, eoqual-metrics (auteur) | — |
