"""Tests de eoqual.core.bayer : cohérence des quatre motifs."""
import numpy as np
import pytest

from eoqual.core.bayer import BAYER_PATTERNS, bayer_masks, bayer_offsets, extract_bayer_planes

@pytest.mark.parametrize("pattern", list(BAYER_PATTERNS))
def test_pattern_matches_its_name(pattern):
    block = [[None, None], [None, None]]
    for name, (r, c) in BAYER_PATTERNS[pattern].items():
        block[r][c] = name[0]
    assert "".join(block[0] + block[1]) == pattern


@pytest.mark.parametrize("pattern", list(BAYER_PATTERNS))
def test_planes_and_masks_agree(pattern):
    rgb = {"R": 1.0, "G": 2.0, "B": 3.0}
    masks = bayer_masks((8, 10), pattern)
    mosaic = sum(masks[c] * v for c, v in rgb.items())
    planes = extract_bayer_planes(mosaic, pattern)
    for c in ("R", "G1", "G2", "B", "G"):
        assert planes[c].shape == (4, 5)
        assert np.all(planes[c] == rgb[c[0]])
    assert masks["R"].sum() == masks["B"].sum() == masks["G"].sum() / 2 == 20


def test_odd_shape_drops_incomplete_block():
    planes = extract_bayer_planes(np.arange(7 * 9, dtype=float).reshape(7, 9), "GBRG")
    assert all(p.shape == (3, 4) for p in planes.values())


def test_offsets_rggb():
    off = bayer_offsets("rggb")
    assert off["R"] == (0.0, 0.0) and off["B"] == (1.0, 1.0) and off["G"] == (0.5, 0.5)


def test_errors():
    with pytest.raises(ValueError):
        bayer_offsets("XXXX")
    with pytest.raises(ValueError):
        extract_bayer_planes(np.zeros((8, 8, 3)))
