"""
Motifs de Bayer : position de chaque composante dans le bloc 2x2,
extraction des plans sous-échantillonnés, masques d'échantillonnage.

Une seule table (:data:`BAYER_PATTERNS`) décrit les quatre motifs ; toutes
les fonctions en dérivent, ce qui garantit leur cohérence. Le nom du
motif décrit le bloc 2x2 en partant du pixel en haut à gauche, ligne par
ligne : ``GBRG`` = ``G B`` puis ``R G``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BAYER_PATTERNS",
    "BAYER_SAMPLING_FACTOR",
    "bayer_offsets",
    "bayer_masks",
    "extract_bayer_planes",
]

#: Position (ligne, colonne) de chaque composante dans le bloc 2x2.
#: ``G1`` est le premier vert rencontré en lisant le bloc ligne par
#: ligne, ``G2`` le second.
BAYER_PATTERNS: dict[str, dict[str, tuple[int, int]]] = {
    "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
    "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
    "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
}

#: Un pixel d'un plan extrait vaut deux pixels physiques dans chaque
#: dimension : sa fréquence de Nyquist (0.5 cycle/pixel du plan) vaut
#: 0.25 cycle/pixel physique. Facteur à appliquer aux fréquences
#: mesurées sur un plan pour les exprimer par pixel physique.
BAYER_SAMPLING_FACTOR = 0.5


def _check_pattern(pattern: str) -> dict[str, tuple[int, int]]:
    try:
        return BAYER_PATTERNS[pattern.upper()]
    except KeyError:
        raise ValueError(
            f"Motif de Bayer inconnu : {pattern!r} (attendu : {', '.join(BAYER_PATTERNS)})."
        ) from None


def bayer_offsets(pattern: str = "RGGB") -> dict[str, tuple[float, float]]:
    """
    Décalage (ligne, colonne) de chaque plan dans la grille pleine
    résolution : le pixel ``(i, j)`` du plan correspond au pixel physique
    ``(2*i + d_ligne, 2*j + d_colonne)``.

    Inclut ``"G"`` (moyenne de G1 et G2), dont le décalage est la moyenne
    des deux — une approximation à une fraction de pixel près, puisque
    chaque pixel de ce plan moyenne deux positions physiques.
    """
    offsets = {k: (float(r), float(c)) for k, (r, c) in _check_pattern(pattern).items()}
    g1, g2 = offsets["G1"], offsets["G2"]
    offsets["G"] = ((g1[0] + g2[0]) / 2.0, (g1[1] + g2[1]) / 2.0)
    return offsets


def extract_bayer_planes(
    image: NDArray[np.floating],
    pattern: str = "RGGB",
) -> dict[str, NDArray[np.floating]]:
    """
    Sépare une mosaïque de Bayer brute en ses plans sous-échantillonnés.

    Parameters
    ----------
    image :
        Mosaïque RAW 2D (une image déjà dématricée à 3 canaux n'est pas
        acceptée).
    pattern :
        ``"RGGB"``, ``"BGGR"``, ``"GRBG"`` ou ``"GBRG"``.

    Returns
    -------
    dict
        Plans ``"R"``, ``"G1"``, ``"G2"``, ``"B"`` et ``"G"`` (moyenne de
        G1 et G2), tous de forme ``(H // 2, W // 2)`` : pour une dimension
        impaire, la dernière ligne ou colonne (bloc 2x2 incomplet) est
        ignorée.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Bayer image must be a 2D array (raw mosaic), got ndim={image.ndim}. "
            "A 3-channel image is already demosaiced and cannot be treated as raw Bayer."
        )
    if image.shape[0] < 4 or image.shape[1] < 4:
        raise ValueError("Image too small to extract Bayer planes (need >= 4x4).")

    h, w = 2 * (image.shape[0] // 2), 2 * (image.shape[1] // 2)
    even = image[:h, :w]
    planes = {name: even[r::2, c::2] for name, (r, c) in _check_pattern(pattern).items()}
    planes["G"] = (planes["G1"] + planes["G2"]) / 2.0
    return {k: planes[k] for k in ("R", "G1", "G2", "B", "G")}


def bayer_masks(shape: tuple[int, int], pattern: str = "RGGB") -> dict[str, NDArray[np.bool_]]:
    """
    Masques booléens pleine résolution des pixels échantillonnés par
    chaque composante : ``"R"``, ``"G"`` (G1 et G2), ``"B"``, ``"G1"``,
    ``"G2"``.
    """
    masks: dict[str, NDArray[np.bool_]] = {}
    for name, (r, c) in _check_pattern(pattern).items():
        m = np.zeros(shape, dtype=bool)
        m[r::2, c::2] = True
        masks[name] = m
    masks["G"] = masks["G1"] | masks["G2"]
    return {k: masks[k] for k in ("R", "G", "B", "G1", "G2")}


# EOF
