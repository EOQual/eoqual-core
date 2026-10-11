"""
Découpage d'une image en tuiles : grille régulière sans recouvrement, ou
tuiles de taille fixe chevauchantes alignées sur les bords.

Seul le découpage est commun ; ce qu'on calcule sur chaque tuile et la
façon d'agréger les résultats restent dans le module appelant.
"""
from __future__ import annotations

__all__ = ["tile_bounds", "tile_slices", "grid_slices"]


def tile_bounds(size: int, tile_size: int, overlap: float = 0.0) -> list[tuple[int, int]]:
    """
    Bornes ``[début, fin)`` d'un découpage 1D en tuiles chevauchantes.

    Le pas vaut ``round(tile_size * (1 - overlap))`` ; la dernière tuile
    est alignée sur le bord. Si ``size <= tile_size``, une seule tuile
    ``[0, size)``.
    """
    if size <= tile_size:
        return [(0, size)]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)  # dernière tuile alignée sur le bord
    return [(s, s + tile_size) for s in starts]


def tile_slices(
    shape: tuple[int, int],
    tile_size: int,
    overlap: float = 0.0,
) -> list[tuple[slice, slice]]:
    """Tuiles 2D chevauchantes de côté ``tile_size`` (voir :func:`tile_bounds`)."""
    rows = tile_bounds(shape[0], tile_size, overlap)
    cols = tile_bounds(shape[1], tile_size, overlap)
    return [(slice(r0, r1), slice(c0, c1)) for r0, r1 in rows for c0, c1 in cols]


def grid_slices(shape: tuple[int, int], n_tiles_per_side: int) -> list[tuple[slice, slice]]:
    """
    Grille régulière ``n x n`` sans recouvrement, tuiles de taille
    ``(h // n, w // n)`` ; le reste de la division est ignoré.
    """
    th, tw = shape[0] // n_tiles_per_side, shape[1] // n_tiles_per_side
    return [
        (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
        for i in range(n_tiles_per_side)
        for j in range(n_tiles_per_side)
    ]


# EOF
