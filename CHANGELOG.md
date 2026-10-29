# Change Log

## Version History

Non publié (0.2.0)

:   -   `license.txt` : titulaire des droits (Olivier Amram) au lieu du
        nom du module.
    -   `kernel` : `psf2otf` et `convolve_circular` (repris
        d'eoqual-measure, `synthetic.processing`) ; utilisés par
        eoqual-measure et eoqual-deconvolution.
    -   `psf.integrated_profile` : profils gaussien (exact, par `erf`),
        Moffat et Airy intégrés sur le pixel (repris d'eoqual-phaseretrieval,
        `simulation.star_profiles.integrated_star_profile`) ; ajout de la
        rotation de la gaussienne ; même convention de centre pour les
        trois profils (centre du pixel `i` en `i`) — l'original décale
        Moffat et Airy d'un demi-pixel par rapport à la gaussienne.
    -   `noise` : `estimate_noise_sigma` (Donoho & Johnstone), en numpy
        seul, identique à `skimage.restoration.estimate_sigma` (écart
        ≤ 1e-15) ; utilisé par eoqual-denoising et eoqual-deconvolution.

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
