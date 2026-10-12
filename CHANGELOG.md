# Change Log

## Version History

Non publié

:   -   `license.txt` : titulaire des droits (Olivier Amram) au lieu du
        nom du module.

0.1.0 (2026-10-11)

:   -   Première version : `fourier` (TFD d'image, avec note technique
        et notebook d'illustrations), `edge` (extrait d'eoqual-measure, `bin_profile` repris
        d'eoqual-metrics), `bayer` (quatre motifs, table unique), `psf`
        (Airy, Strehl, profil radial, énergie encerclée), `detrend`,
        `tiling` (extrait d'eoqual-metrics).
    -   `lsf_to_mtf` normalise par la composante continue (et non par le
        maximum sur les fréquences positives, variante courante qui
        surestime la MTF d'environ 1 %) ; option `n_fft` (zéro-padding).
    -   `extract_bayer_planes` ignore la dernière ligne ou colonne d'une
        mosaïque de dimension impaire.
