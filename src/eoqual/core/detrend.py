"""
Retrait d'une tendance radiométrique basse fréquence (plan) avant mesure.

Utile à toute mesure sur zones uniformes (variance du bruit, courbe de
transfert photonique, bruit d'égalisation, contraste sur bruit) : un
gradient d'éclairement non retiré y est compté à tort comme du bruit.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["fit_plane_residual"]


def fit_plane_residual(
    values: NDArray[np.floating],
    coords: NDArray[np.floating],
) -> NDArray[np.float64]:
    """
    Ajuste un plan ``a*x + b*y + c`` par moindres carrés et renvoie le résidu.

    Parameters
    ----------
    values :
        Valeurs 1D (patch ou segment aplati).
    coords :
        Coordonnées ``(x, y)`` de chaque valeur, forme ``(N, 2)``.

    Returns
    -------
    NDArray[np.float64]
        ``values`` moins le plan ajusté, même forme que ``values``.
    """
    values = np.asarray(values, dtype=np.float64)
    design = np.column_stack([coords[:, 0], coords[:, 1], np.ones(len(values))])
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coeffs


# EOF
