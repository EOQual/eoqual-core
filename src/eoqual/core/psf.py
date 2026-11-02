"""
PSF : référence de diffraction (Airy), profils intégrés sur le pixel,
rapport de Strehl, profil radial et énergie encerclée.

Toutes les fonctions supposent une PSF centrée en ``(shape - 1) / 2``.

Références : M. Born, E. Wolf, *Principles of Optics* (tache d'Airy,
énergie encerclée) ; V. N. Mahajan, "Strehl ratio for primary
aberrations: some analytical results for circular and annular pupils",
JOSA 72(9), 1982.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "airy_psf", "airy_radius_px", "integrated_profile", "supersampled_coverage",
    "strehl_ratio", "radial_profile", "encircled_energy",
]


def _radii(shape: tuple[int, ...]) -> NDArray[np.float64]:
    y, x = np.indices(shape)
    cy, cx = (np.array(shape) - 1) / 2.0
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)


def airy_psf(
    shape: tuple[int, int],
    wavelength: float,
    diameter: float,
    pixel_scale: float,
) -> NDArray[np.float64]:
    """
    PSF de diffraction d'une pupille circulaire non obturée, de somme 1.

    Parameters
    ----------
    shape :
        ``(ny, nx)``.
    wavelength :
        Longueur d'onde [m].
    diameter :
        Diamètre de pupille [m].
    pixel_scale :
        Angle sous-tendu par un pixel [rad/pixel] ; pour une PSF
        sur-échantillonnée d'un facteur ``k``, passer ``pixel_scale / k``.
    """
    from scipy.special import j1

    kr = (np.pi * diameter / wavelength) * _radii(shape) * pixel_scale
    psf = np.ones_like(kr)
    nonzero = kr != 0
    psf[nonzero] = (2 * j1(kr[nonzero]) / kr[nonzero]) ** 2
    return psf / psf.sum()


#: Premier zéro de J1(πx)/x : rayon du premier anneau noir, en λ/D.
AIRY_FIRST_ZERO = 1.2196698912665045


def airy_radius_px(
    wavelength: float,
    diameter: float,
    focal_length: float,
    pixel_pitch: float,
) -> float:
    """
    Rayon du premier anneau noir de la tache d'Airy, en pixels détecteur :
    ``1.22 λ f / (D p)``.

    Parameters
    ----------
    wavelength, diameter, focal_length, pixel_pitch :
        Longueur d'onde, diamètre de pupille, focale et pas pixel, dans une
        même unité de longueur (mètres par exemple).

    Notes
    -----
    Le système est échantillonné à Nyquist pour la fréquence de coupure
    optique ``D / (λ f)`` quand ``λ f / (D p) = 2``, soit un rayon de 2.44
    pixels.
    """
    for name, v in (("wavelength", wavelength), ("diameter", diameter),
                    ("focal_length", focal_length), ("pixel_pitch", pixel_pitch)):
        if v <= 0:
            raise ValueError(f"{name} doit être strictement positif.")
    return AIRY_FIRST_ZERO * wavelength * focal_length / (diameter * pixel_pitch)


def integrated_profile(
    shape: tuple[int, int],
    center: tuple[float, float],
    profile: str,
    params: dict | None = None,
    oversample: int = 5,
) -> NDArray[np.float64]:
    """
    Profil de PSF empirique **intégré sur la surface de chaque pixel**
    (et non échantillonné au centre des pixels).

    Parameters
    ----------
    shape :
        ``(H, W)``.
    center :
        ``(y0, x0)`` centre sous-pixel, en coordonnées pixel (centre du
        pixel ``(i, j)`` en ``(i, j)``).
    profile : {"gaussian", "moffat", "airy"}
        - ``"gaussian"`` : intégration **exacte** par la fonction d'erreur
          (sauf si ``theta`` ≠ 0 : sur-échantillonnage) ;
        - ``"moffat"``, ``"airy"`` : sur-échantillonnage (grille
          ``oversample`` fois plus fine, moyennée par pixel).
    params : dict, optional
        - ``"gaussian"`` : ``sigma=(sigma_y, sigma_x)`` [px], défaut
          ``(1.5, 1.5)`` ; ``theta`` [rad], rotation, défaut 0 ;
        - ``"moffat"`` : ``alpha`` [px], défaut 2.0 ; ``beta``, défaut
          4.765 (turbulence atmosphérique) ;
        - ``"airy"`` : ``radius_zero`` [px], rayon du premier anneau noir
          (``1.22 λ/D`` ramené aux pixels), défaut 2.0 ;
        - tous : ``amplitude``, défaut 1.
    oversample :
        Facteur de sur-échantillonnage des profils sans primitive.

    Returns
    -------
    NDArray
        Image ``(H, W)``, non normalisée (``amplitude`` × profil).
    """
    from scipy.special import erf, j1

    params = params or {}
    h, w = shape
    y0, x0 = center
    amp = params.get("amplitude", 1.0)

    if profile == "gaussian":
        sigma_y, sigma_x = params.get("sigma", (1.5, 1.5))
        theta = params.get("theta", 0.0)
        if theta == 0.0:
            y_grid, x_grid = np.indices((h, w))
            val_x = erf((x_grid + 0.5 - x0) / (sigma_x * np.sqrt(2))) - erf(
                (x_grid - 0.5 - x0) / (sigma_x * np.sqrt(2))
            )
            val_y = erf((y_grid + 0.5 - y0) / (sigma_y * np.sqrt(2))) - erf(
                (y_grid - 0.5 - y0) / (sigma_y * np.sqrt(2))
            )
            return amp * 0.25 * val_x * val_y

    step = 1.0 / oversample
    sub = np.arange(oversample) * step - 0.5 + step / 2
    y_sub = (np.arange(h)[:, None] + sub[None, :]).ravel()[:, None] - y0
    x_sub = (np.arange(w)[:, None] + sub[None, :]).ravel()[None, :] - x0

    if profile == "gaussian":
        c, s = np.cos(theta), np.sin(theta)
        u = c * x_sub + s * y_sub
        v = -s * x_sub + c * y_sub
        high_res = np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2)) / (2 * np.pi * sigma_x * sigma_y)
    elif profile == "moffat":
        alpha = params.get("alpha", 2.0)
        beta = params.get("beta", 4.765)
        high_res = (1 + (x_sub**2 + y_sub**2) / alpha**2) ** (-beta)
    elif profile == "airy":
        r_zero = params.get("radius_zero", 2.0)
        k = 3.8317059702075125 / r_zero  # premier zéro de J1 au rayon r_zero
        x_val = k * np.sqrt(x_sub**2 + y_sub**2) + 1e-12
        high_res = (2 * j1(x_val) / x_val) ** 2
    else:
        raise ValueError(f"profile inconnu : {profile!r} (attendu gaussian/moffat/airy)")

    return amp * high_res.reshape(h, oversample, w, oversample).mean(axis=(1, 3))


def supersampled_coverage(
    shape_func,
    shape: tuple[int, int],
    center: tuple[float, float],
    supersample: int = 8,
) -> NDArray[np.float64]:
    """
    Fraction de couverture sous-pixel d'une forme géométrique quelconque.

    Évalue ``shape_func(y, x) -> bool | float`` sur une grille
    ``supersample`` fois plus fine par pixel de sortie (coordonnées pixel
    relatives à ``center``), puis moyenne chaque bloc ``supersample x
    supersample`` — un filtre boîte, équivalent à un redimensionnement
    anti-crénelé pour un facteur de décimation entier, mais en pur numpy
    (reshape + moyenne), sans dépendance supplémentaire.

    Généralise la fonction privée ``_supersampled`` d'
    ``eoqual.deconvolution.psf`` (disque de défocalisation, filé
    linéaire) ; utilisée aussi par ``eoqual.phaseretrieval.pupil`` pour
    adoucir les bras d'araignée d'une pupille générée analytiquement (un
    seuillage binaire produit, pour un bras fin, une largeur/transmission
    qui dépend de façon arbitraire de l'alignement sous-pixel de la grille
    — un artefact numérique, pas un effet physique).

    Parameters
    ----------
    shape_func : callable
        ``shape_func(y, x) -> ndarray`` — vrai/valeur à l'intérieur de la
        forme, évalué en coordonnées pixel ``(y, x)`` relatives à
        ``center`` (mêmes tableaux larges ``(h, supersample, w,
        supersample)`` en entrée que renvoyés en sortie par
        ``np.broadcast``).
    shape : tuple[int, int]
        Taille de sortie ``(H, W)``.
    center : tuple[float, float]
        Centre ``(y0, x0)`` en coordonnées pixel (pixel ``(0, 0)`` en haut
        à gauche) — à la charge de l'appelant pour rester cohérent avec sa
        propre convention de centrage (p. ex. ``(n - 1) / 2`` ou ``n //
        2``, voir ``eoqual.core.kernel`` sur ce point).
    supersample : int, default 8
        Facteur de sous-échantillonnage par dimension (``supersample**2``
        sous-pixels moyennés par pixel de sortie).

    Returns
    -------
    NDArray
        ``(H, W)``, valeurs dans ``[0, 1]`` pour un ``shape_func``
        booléen (fraction de couverture) — **non normalisé** (pas de
        somme 1) : à la charge de l'appelant pour un noyau de PSF.
    """
    h, w = shape
    y0, x0 = center
    s = np.arange(supersample) / supersample + 0.5 / supersample - 0.5
    y = np.arange(h) - y0
    x = np.arange(w) - x0
    yy = y[:, None, None, None] + s[None, None, :, None]
    xx = x[None, :, None, None] + s[None, None, None, :]
    return np.asarray(shape_func(yy, xx), dtype=np.float64).mean(axis=(2, 3))


def strehl_ratio(psf_measured: NDArray[np.floating], psf_reference: NDArray[np.floating]) -> float:
    """
    Rapport de Strehl ``max(psf_measured) / max(psf_reference)``.

    Les deux PSF doivent partager la même normalisation (somme 1 par
    exemple), le même échantillonnage et la même bande spectrale. Un
    rapport nettement supérieur à 1 signale qu'une de ces conditions
    n'est pas remplie : un avertissement est émis.
    """
    strehl = float(np.max(psf_measured) / np.max(psf_reference))
    if strehl > 1.02:
        warnings.warn(
            f"Strehl = {strehl:.3f} > 1 : référence probablement incohérente avec la PSF "
            "mesurée (échantillonnage, normalisation ou bande spectrale).",
            stacklevel=2,
        )
    return strehl


def radial_profile(psf: NDArray[np.floating]) -> NDArray[np.float64]:
    """Profil radial moyen, une valeur par rayon entier (pixels) depuis le centre."""
    r = _radii(psf.shape).astype(int)
    total = np.bincount(r.ravel(), psf.ravel())
    count = np.bincount(r.ravel())
    return total / np.maximum(count, 1)


def encircled_energy(
    psf: NDArray[np.floating],
    radii: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Énergie encerclée EE(r) : fraction de l'énergie totale contenue dans
    le disque de rayon ``r`` centré sur la PSF.

    Parameters
    ----------
    psf :
        PSF 2D centrée (le fond doit avoir été retiré : un fond résiduel
        fausse la normalisation).
    radii :
        Rayons (pixels) où évaluer EE. Défaut : 0.5, 1.5, ... jusqu'au
        bord du plus grand cercle inscrit.

    Returns
    -------
    tuple
        ``(radii, ee)``, ``ee`` croissante, égale à 1 si le disque couvre
        toute la PSF.
    """
    r = _radii(psf.shape).ravel()
    values = np.asarray(psf, dtype=np.float64).ravel()
    total = values.sum()
    if total <= 0:
        raise ValueError("La somme de la PSF doit être positive.")
    if radii is None:
        radii = np.arange(0.5, (min(psf.shape) - 1) / 2.0 + 0.5, 1.0)
    radii = np.asarray(radii, dtype=np.float64)

    order = np.argsort(r)
    cumulative = np.cumsum(values[order])
    counts = np.searchsorted(r[order], radii, side="right")
    ee = np.where(counts > 0, cumulative[np.maximum(counts - 1, 0)], 0.0) / total
    return radii, ee


# EOF
