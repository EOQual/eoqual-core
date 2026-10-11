"""
Transformée de Fourier discrète d'une image, sans artefacts de bord.

- :func:`perdecomp`   : décomposition périodique + lisse u = p + s (Moisan,
  2011), supprime la « croix » d'artefacts de bord dans le spectre.
- :func:`translate`   : translation sous-pixellique par interpolation de
  Shannon (déphasage dans le domaine de Fourier).
- :func:`dequantize`  : translation de (1/2, 1/2) pixel (Blanchet & Moisan),
  casse la structure entière des niveaux de gris.
- :func:`frequencies` : axes de fréquences en cycles/pixel (Nyquist = 0.5).
- :func:`spectrum`    : spectre centré (amplitude ou complexe), avec
  prétraitement au choix.
- :func:`boundary_jump` : saut moyen entre bords opposés (diagnostic de
  périodicité).

L'indice de netteté SI (Blanchet & Moisan, 2012), construit sur ces
briques, est un indicateur de qualité : il relève d'``eoqual-metrics``.

Références
----------
L. Moisan, "Periodic plus smooth image decomposition",
    J. Math. Imaging Vision 39(2), 2011, p. 161-179.
G. Blanchet, L. Moisan, "An explicit sharpness index related to global phase
    coherence", ICASSP 2012, p. 1065-1068.
"""
from __future__ import annotations

import numpy as np

__all__ = ["perdecomp", "translate", "dequantize", "frequencies", "spectrum", "boundary_jump"]


def _check_2d(image):
    u = np.asarray(image, dtype=float)
    if u.ndim != 2:
        raise ValueError(f"image 2D attendue, reçu ndim={u.ndim} (traiter les canaux séparément)")
    return u


def perdecomp(image):
    """
    Décomposition périodique + lisse : u = p + s.

    ``s`` est la solution (de moyenne nulle) de l'équation de Poisson périodique
    Δs = v, où v ne vaut non nul que sur le bord et contient les sauts
    « de l'autre côté » du tore :

        v[0, :]  = u[-1, :] - u[0, :]     v[-1, :] += u[0, :] - u[-1, :]
        v[:, 0] += u[:, -1] - u[:, 0]     v[:, -1] += u[:, 0] - u[:, -1]

    Le laplacien discret périodique est diagonal en Fourier, de valeurs propres
    2 cos(2πq/M) + 2 cos(2πr/N) - 4, d'où une résolution en O(MN log MN).

    Parameters
    ----------
    image : array_like (M, N)

    Returns
    -------
    p : ndarray (M, N)  composante périodique (même moyenne que u)
    s : ndarray (M, N)  composante lisse (moyenne nulle)
    """
    u = _check_2d(image)
    M, N = u.shape

    v = np.zeros_like(u)
    dv = u[-1, :] - u[0, :]
    v[0, :] += dv
    v[-1, :] -= dv
    dh = u[:, -1] - u[:, 0]
    v[:, 0] += dh
    v[:, -1] -= dh

    q = np.arange(M).reshape(-1, 1)
    r = np.arange(N).reshape(1, -1)
    denom = 2 * np.cos(2 * np.pi * q / M) + 2 * np.cos(2 * np.pi * r / N) - 4
    denom[0, 0] = 1.0  # fréquence nulle traitée à part

    s_hat = np.fft.fft2(v) / denom
    s_hat[0, 0] = 0.0  # moyenne de s nulle
    s = np.real(np.fft.ifft2(s_hat))
    return u - s, s


def translate(image, dy, dx):
    """
    Translation sous-pixellique par interpolation de Shannon (périodique) :
    w(y, x) = u(y - dy, x - dx), réalisée par le déphasage
    exp(-2iπ (ξ_y dy + ξ_x dx)) sur la TFD.

    Pour une taille paire, la fréquence de Nyquist n'a pas de partenaire
    symétrique : on garde la partie réelle, ce qui revient à la symétriser.
    L'image doit être (quasi) périodique, sinon les sauts de bord produisent
    des oscillations de Gibbs : appliquer ``perdecomp`` avant.
    """
    u = _check_2d(image)
    fy, fx = frequencies(u.shape, centered=False)
    phase = np.exp(-2j * np.pi * (fy[:, None] * dy + fx[None, :] * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(u) * phase))


def dequantize(image):
    """
    Déquantification : translation de (1/2, 1/2) pixel par interpolation de Shannon.

    Après translation, les valeurs ne sont plus des entiers : on supprime la
    structure artificielle introduite par la quantification des niveaux de gris
    (utile notamment pour les indices de netteté fondés sur la phase).

    Attention : ce n'est PAS la multiplication par (-1)^(x+y), qui translate le
    spectre de (M/2, N/2) (équivalent d'un fftshift) sans déplacer l'image.
    """
    return translate(image, 0.5, 0.5)


def frequencies(shape, centered=True):
    """
    Axes de fréquences (fy, fx) en cycles/pixel, dans [-0.5, 0.5[.
    La fréquence de Nyquist vaut 0.5 cycle/pixel quelle que soit la taille.
    ``centered=True`` donne l'ordre de ``np.fft.fftshift``.
    """
    M, N = shape[:2]
    fy, fx = np.fft.fftfreq(M), np.fft.fftfreq(N)
    if centered:
        fy, fx = np.fft.fftshift(fy), np.fft.fftshift(fx)
    return fy, fx


def spectrum(image, method="perdecomp", dequant=False, output="log"):
    """
    Spectre centré d'une image.

    Parameters
    ----------
    image : array_like (M, N)
    method : {"raw", "perdecomp", "hann"}
        Prétraitement contre les artefacts de bord :
        - "raw"       : TFD directe (croix d'artefacts sur les axes),
        - "perdecomp" : composante périodique p (recommandé, conserve le contenu),
        - "hann"      : fenêtre de Hann 2D (atténue les bords, élargit les raies).
    dequant : bool
        Applique ``dequantize`` après le prétraitement. Une translation ne
        modifie que la phase : avec output="complex", tout le spectre change ;
        avec "amplitude" ou "log", l'effet se limite à la ligne/colonne de
        Nyquist des tailles paires (coefficients atténués, voire annulés).
    output : {"log", "amplitude", "complex"}
        log(1 + |U|), |U|, ou U complexe.

    Returns
    -------
    S : ndarray (M, N)  spectre centré (fréquence nulle en [M//2, N//2])
    fy, fx : ndarray    axes de fréquences en cycles/pixel
    """
    if output not in ("log", "amplitude", "complex"):
        raise ValueError(f"output inconnu : {output!r} (log, amplitude, complex)")
    u = _check_2d(image)
    if method == "perdecomp":
        u, _ = perdecomp(u)
    elif method == "hann":
        M, N = u.shape
        u = (u - u.mean()) * np.outer(np.hanning(M), np.hanning(N)) + u.mean()
    elif method != "raw":
        raise ValueError(f"method inconnue : {method!r} (raw, perdecomp, hann)")
    if dequant:
        u = dequantize(u)
    U = np.fft.fftshift(np.fft.fft2(u))
    fy, fx = frequencies(u.shape, centered=True)
    if output == "complex":
        return U, fy, fx
    A = np.abs(U)
    return (np.log1p(A) if output == "log" else A), fy, fx


def boundary_jump(image):
    """
    Saut moyen entre bords opposés (|u[0]-u[-1]| et |u[:,0]-u[:,-1]|).
    Pour une image bien raccordée sur le tore, il est de l'ordre des
    différences entre pixels voisins à l'intérieur de l'image.
    """
    u = _check_2d(image)
    return 0.5 * (np.abs(u[0, :] - u[-1, :]).mean() + np.abs(u[:, 0] - u[:, -1]).mean())


# EOF
