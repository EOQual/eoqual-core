"""
Estimation de l'écart-type d'un bruit blanc additif.

Estimateur robuste de Donoho & Johnstone (1994) : la sous-bande
d'ondelettes diagonale la plus fine contient surtout du bruit, et la
médiane des valeurs absolues de ses coefficients, divisée par
``Φ⁻¹(3/4) ≈ 0.6745``, estime l'écart-type du bruit.

Ondelette de Daubechies à 2 moments nuls (``db2``) et prolongement
symétrique des bords : mêmes conventions que ``pywt.dwtn(..., "db2")``,
donc même résultat que ``skimage.restoration.estimate_sigma`` (écart
≤ 1e-15), sans dépendre de PyWavelets.

Référence : D. L. Donoho, I. M. Johnstone, "Ideal spatial adaptation by
wavelet shrinkage", Biometrika 81(3), 1994.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

__all__ = ["finest_diagonal_detail", "estimate_noise_sigma"]

_S3 = np.sqrt(3.0)
#: Filtre passe-haut d'analyse db2 (ordre de ``pywt.Wavelet("db2").dec_hi``).
#: Valeurs : [-0.483, 0.837, -0.224, -0.129].
_DB2_DEC_HI = np.array([-(1 + _S3), 3 + _S3, -(3 - _S3), 1 - _S3]) / (4 * np.sqrt(2.0))


def _highpass_decimate(x: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Filtrage passe-haut db2 puis décimation d'un facteur 2 selon ``axis``."""
    x = np.moveaxis(x, axis, -1)
    n_taps = _DB2_DEC_HI.size
    ext = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(n_taps - 1, n_taps - 1)], mode="symmetric")
    # Convolution « valid » vectorisée sur le dernier axe.
    m = ext.shape[-1] - n_taps + 1
    out = np.zeros(ext.shape[:-1] + (m,))
    for j, tap in enumerate(_DB2_DEC_HI):
        out += tap * ext[..., n_taps - 1 - j : n_taps - 1 - j + m]
    return np.moveaxis(out[..., 1::2], -1, axis)


def finest_diagonal_detail(image: ArrayLike) -> NDArray[np.float64]:
    """Sous-bande d'ondelettes ``db2`` diagonale (``dd``) du premier niveau."""
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Image 2D attendue, reçu ndim={x.ndim}.")
    return _highpass_decimate(_highpass_decimate(x, 0), 1)


def estimate_noise_sigma(image: ArrayLike) -> float:
    """
    Écart-type d'un bruit blanc gaussien additif, dans les unités de
    l'image.

    Fiable pour un bruit blanc sur une image modérément texturée ; il
    **surestime** le bruit sur une image à fort contenu haute fréquence,
    et le **sous-estime** sur un bruit corrélé (image filtrée, dématricée,
    rééchantillonnée).

    Raises
    ------
    ValueError
        Si l'image n'est pas 2D ou contient des valeurs non finies.
    """
    x = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("L'image contient des valeurs non finies (NaN/inf).")
    detail = finest_diagonal_detail(x)
    return float(np.median(np.abs(detail)) / norm.ppf(0.75))


# EOF
