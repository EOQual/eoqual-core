"""
Chaîne de calcul d'une réponse de bord : profil projeté → ESF → LSF → MTF,
et grandeurs dérivées (MTF à une fréquence, MTF50, FWHM, RER).

Ce module ne détecte aucun bord et ne choisit aucune zone : il part d'un
profil déjà projeté sur la normale au bord. La détection (mire, bord
choisi, arêtes naturelles) reste dans le module qui l'utilise
(``eoqual-measure`` pour la mesure contrôlée, ``eoqual-metrics`` pour
l'indicateur sur arêtes naturelles).

Conventions : les fréquences sont en cycles par unité de l'abscisse du
profil (cycles/pixel pour un profil en pixels, 0.5 = Nyquist) ; les MTF
sont normalisées à 1 en fréquence nulle.

Références : ISO 12233:2017 (edge SFR) ; P. D. Burns, "Slanted-Edge MTF
for Digital Camera and Scanner Analysis", Proc. IS&T PICS, 2000 ;
L. Harrington et al., "General Image Quality Equation: GIQE Version 5",
NGA, 2015 (RER).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "bin_profile",
    "esf_to_lsf",
    "lsf_to_mtf",
    "derivative_correction",
    "pixel_aperture_correction",
    "mtf_at",
    "frequency_at",
    "fwhm",
    "rer",
]


# --------------------------------------------------------------------------
# Profil projeté -> ESF sur-échantillonnée
# --------------------------------------------------------------------------


def bin_profile(
    offsets: NDArray[np.floating],
    values: NDArray[np.floating],
    half_width: float,
    oversample: int,
    min_filled_fraction: float = 0.5,
) -> NDArray[np.float64] | None:
    """
    Moyenne par bins d'échantillons projetés sur la normale au bord.

    Parameters
    ----------
    offsets :
        Distance signée de chaque pixel à la droite de bord (pixels).
    values :
        Intensité de chaque pixel (même ordre).
    half_width :
        Demi-largeur de la grille (pixels) ; la grille couvre
        ``[-half_width, half_width]``.
    oversample :
        Nombre de bins par pixel.
    min_filled_fraction :
        Fraction minimale de bins non vides ; en dessous, renvoie ``None``.

    Returns
    -------
    NDArray[np.float64] | None
        ESF sur-échantillonnée (bins vides comblés par interpolation
        linéaire), ou ``None`` si la grille est trop lacunaire.
    """
    bin_width = 1.0 / oversample
    bin_edges = np.arange(-half_width, half_width + bin_width, bin_width)
    bin_idx = np.clip(np.digitize(offsets, bin_edges) - 1, 0, len(bin_edges) - 2)

    esf = np.full(len(bin_edges) - 1, np.nan)
    for b in np.unique(bin_idx):
        esf[b] = values[bin_idx == b].mean()

    valid = ~np.isnan(esf)
    if valid.sum() < min_filled_fraction * len(esf):
        return None
    return np.interp(np.arange(len(esf)), np.flatnonzero(valid), esf[valid])


# --------------------------------------------------------------------------
# ESF -> LSF -> MTF
# --------------------------------------------------------------------------


def esf_to_lsf(
    esf: NDArray[np.float64],
    dx: float,
    derivative_method: str = "gradient",
    filter_type: str = "tukey",
    window_length: int = 15,
    polyorder: int = 3,
    tukey_alpha: float = 0.15,
) -> NDArray[np.float64]:
    """
    LSF par dérivation numérique de l'ESF, puis filtrage/fenêtrage.

    Parameters
    ----------
    esf :
        Edge Spread Function, échantillonnée au pas ``dx``.
    dx :
        Pas d'échantillonnage.
    derivative_method :
        ``"gradient"`` (différences centrées) ou ``"savgol_deriv"``
        (dérivée de Savitzky-Golay, lisse et dérive en une étape).
    filter_type :
        ``"tukey"`` (apodisation des extrémités), ``"savgol"``,
        ``"gaussian"`` (écart-type ``window_length / 6``) ou ``"none"``.
    window_length, polyorder :
        Paramètres de Savitzky-Golay (``window_length`` impair).
    tukey_alpha :
        Paramètre de la fenêtre de Tukey.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    from scipy.signal.windows import tukey

    if derivative_method == "gradient":
        lsf = np.gradient(esf, dx)
    elif derivative_method == "savgol_deriv":
        if len(esf) > window_length:
            lsf = savgol_filter(esf, window_length, polyorder, deriv=1, delta=dx)
        else:
            lsf = np.gradient(esf, dx)  # fenêtre trop grande pour ce profil
    else:
        raise ValueError(
            f"Unknown derivative_method: {derivative_method!r} "
            "(expected 'gradient' or 'savgol_deriv')."
        )

    if filter_type == "tukey":
        lsf = lsf * tukey(len(lsf), alpha=tukey_alpha)
    elif filter_type == "savgol":
        if len(lsf) > window_length:
            lsf = savgol_filter(lsf, window_length, polyorder)
    elif filter_type == "gaussian":
        lsf = gaussian_filter1d(lsf, window_length / 6.0)
    elif filter_type != "none":
        raise ValueError(
            f"Unknown filter_type: {filter_type!r} "
            "(expected 'tukey', 'savgol', 'gaussian' or 'none')."
        )
    return lsf


def lsf_to_mtf(
    lsf: NDArray[np.float64],
    dx: float,
    n_fft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    MTF : module de la FFT de la LSF, normalisé par sa valeur en fréquence
    nulle (l'intégrale de la LSF, c'est-à-dire le saut de l'ESF).

    Normaliser par le maximum sur les seules fréquences positives, variante
    courante, revient à normaliser par la valeur au premier bin
    (``1 / (N dx)``) : la MTF est alors surestimée partout de
    ``1 / MTF(1 / (N dx))`` — environ 1 % pour une PSF gaussienne de 1 px
    sur un profil de 40 px.

    Parameters
    ----------
    lsf :
        Line Spread Function, échantillonnée au pas ``dx``.
    dx :
        Pas d'échantillonnage.
    n_fft :
        Taille de FFT ; ``> len(lsf)`` complète la LSF par des zéros
        (grille de fréquences plus fine, sans information nouvelle).
        Défaut : ``len(lsf)``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``(freqs, mtf)`` aux fréquences strictement positives (cycles par
        unité de ``dx``).
    """
    n = len(lsf) if n_fft is None else n_fft
    spectrum = np.abs(np.fft.fft(lsf, n=n))
    freqs = np.fft.fftfreq(n, d=dx)

    dc = spectrum[0]
    mask = freqs > 0
    freqs, mtf = freqs[mask], spectrum[mask]

    if dc < 1e-10:
        return freqs, np.zeros_like(mtf)
    return freqs, mtf / dc


def derivative_correction(
    freqs: NDArray[np.float64],
    mtf: NDArray[np.float64],
    dx: float,
) -> NDArray[np.float64]:
    """Corrige l'atténuation ``sinc(f*dx)`` due à la dérivation par différences finies."""
    sinc = np.sinc(freqs * dx)
    sinc[sinc == 0] = 1e-8
    return mtf / sinc


def pixel_aperture_correction(
    freqs: NDArray[np.float64],
    mtf: NDArray[np.float64],
    pixel_pitch: float,
) -> NDArray[np.float64]:
    """Retire la MTF de l'intégration sur le pixel, ``sinc(f*pixel_pitch)``."""
    mtf_pixel = np.sinc(freqs * pixel_pitch)
    mtf_pixel[mtf_pixel == 0] = 1e-8
    return mtf / mtf_pixel


# --------------------------------------------------------------------------
# Grandeurs dérivées
# --------------------------------------------------------------------------


def mtf_at(
    freqs: NDArray[np.float64],
    mtf: NDArray[np.floating | np.complexfloating],
    frequency: float,
) -> float:
    """Module de la MTF interpolé à ``frequency`` (0.5 = Nyquist en cycles/pixel)."""
    return float(np.interp(frequency, freqs, np.abs(mtf)))


def frequency_at(
    freqs: NDArray[np.float64],
    mtf: NDArray[np.floating | np.complexfloating],
    level: float = 0.5,
) -> float:
    """
    Fréquence du premier passage du module de la MTF sous ``level``
    (MTF50 pour ``level=0.5``), par interpolation linéaire.

    NaN si la MTF ne descend jamais sous ``level`` sur la plage fournie.
    """
    mtf = np.abs(mtf)
    below = np.flatnonzero(mtf < level)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    f1, f2, m1, m2 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
    return float(f1 + (level - m1) * (f2 - f1) / (m2 - m1))


def fwhm(x: NDArray[np.float64], lsf: NDArray[np.float64]) -> float:
    """
    Largeur à mi-hauteur de la LSF.

    Prend la PREMIÈRE traversée du demi-maximum de part et d'autre du pic
    (interpolation sous-pixel), et non le premier et le dernier point
    au-dessus du seuil sur tout le domaine : cette seconde définition
    casse dès qu'un rebond de bruit isolé, loin du pic, repasse au-dessus
    du seuil.

    Returns
    -------
    float
        FWHM en unités de ``x`` ; 0.0 si la LSF ne redescend pas sous la
        moitié de son maximum des deux côtés du pic.
    """
    peak_idx = int(np.argmax(lsf))
    peak = lsf[peak_idx]
    if peak <= 0:
        return 0.0
    half = peak / 2.0

    def _first_crossing(indices: range) -> float | None:
        prev = peak_idx
        for i in indices:
            if lsf[i] < half:
                t = (half - lsf[prev]) / (lsf[i] - lsf[prev])
                return float(x[prev] + t * (x[i] - x[prev]))
            prev = i
        return None

    x_left = _first_crossing(range(peak_idx - 1, -1, -1))
    x_right = _first_crossing(range(peak_idx + 1, lsf.size))

    if x_left is None or x_right is None:
        return 0.0
    return float(abs(x_right - x_left))


def rer(
    x_esf: NDArray[np.float64],
    esf: NDArray[np.float64],
    edge_center: float = 0.0,
) -> float:
    """
    Relative Edge Response : ``ESF(centre + 0.5) - ESF(centre - 0.5)`` sur
    une ESF normalisée dans [0, 1] (paramètre de la GIQE).
    """
    esf_plus = float(np.interp(edge_center + 0.5, x_esf, esf))
    esf_minus = float(np.interp(edge_center - 0.5, x_esf, esf))
    return esf_plus - esf_minus


# EOF
