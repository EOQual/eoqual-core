"""
PSF : référence de diffraction (Airy), rapport de Strehl, profil radial
et énergie encerclée.

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

__all__ = ["airy_psf", "strehl_ratio", "radial_profile", "encircled_energy"]


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
