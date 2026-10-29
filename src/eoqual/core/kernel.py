"""
Noyaux de convolution : fonction de transfert sur une grille FFT,
convolution circulaire. Pour construire un noyau, voir
:func:`eoqual.core.psf.integrated_profile` et :func:`eoqual.core.psf.airy_psf`.

Convention de centrage : le centre d'un noyau de forme ``(kh, kw)`` est
le pixel ``(kh // 2, kw // 2)`` — le pixel central pour une taille
impaire, et la convention de ``np.fft.fftshift`` pour une taille paire.
:func:`psf2otf` ramène ce pixel à l'origine ``(0, 0)`` avant la FFT : la
fonction de transfert d'un noyau symétrique est alors réelle (pas de
décalage de phase).

La convolution circulaire suppose l'image périodique. C'est exact pour
des données synthétiques construites ainsi (et la relation spectrale
``Y = H·X`` est alors exacte) ; sur une image réelle, traiter les bords
avant (prolongement miroir, ou :func:`eoqual.core.fourier.perdecomp`).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["psf2otf", "convolve_circular"]


def psf2otf(kernel: ArrayLike, shape: tuple[int, int]) -> NDArray[np.complex128]:
    """
    Fonction de transfert d'un noyau sur la grille FFT de ``shape``
    (ordre ``np.fft.fftfreq`` : fréquence nulle en ``[0, 0]``).

    Le noyau est complété par des zéros jusqu'à ``shape`` puis décalé
    circulairement pour placer son centre ``(kh // 2, kw // 2)`` à
    l'origine.

    Raises
    ------
    ValueError
        Si le noyau est plus grand que ``shape``.
    """
    k = np.atleast_2d(np.asarray(kernel, dtype=np.float64))
    kh, kw = k.shape
    if kh > shape[0] or kw > shape[1]:
        raise ValueError(f"Noyau {k.shape} plus grand que la grille {tuple(shape)}.")
    pad = np.zeros(shape, dtype=np.float64)
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def convolve_circular(image: ArrayLike, kernel: ArrayLike) -> NDArray[np.float64]:
    """
    Convolution circulaire (FFT) d'une image ``(H, W)`` — ou de chaque
    image d'une pile ``(..., H, W)`` — par un noyau centré.
    """
    x = np.asarray(image, dtype=np.float64)
    otf = psf2otf(kernel, x.shape[-2:])
    return np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * otf, axes=(-2, -1)).real


# EOF
