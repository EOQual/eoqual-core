"""Tests de eoqual.core.kernel, eoqual.core.noise et eoqual.core.psf.integrated_profile."""
import numpy as np
import pytest

from eoqual.core.kernel import convolve_circular, psf2otf
from eoqual.core.psf import integrated_profile
from eoqual.core.noise import estimate_noise_sigma, finest_diagonal_detail


def test_psf2otf_symmetric_kernel_is_real_and_dc_one():
    g = integrated_profile((17, 17), (8, 8), "gaussian", {"sigma": (2.0, 2.0)})
    otf = psf2otf(g / g.sum(), (64, 48))
    assert otf[0, 0] == pytest.approx(1.0)
    assert np.abs(otf.imag).max() < 1e-12
    with pytest.raises(ValueError):
        psf2otf(np.ones((5, 5)), (4, 4))


def test_convolve_circular_delta_and_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 40))
    delta = np.zeros((5, 5))
    delta[2, 2] = 1
    np.testing.assert_allclose(convolve_circular(x, delta), x, atol=1e-12)
    shift = np.zeros((3, 3))
    shift[1, 0] = 1  # pixel à gauche du centre : y[i, j] = x[i, j + 1]
    np.testing.assert_allclose(convolve_circular(x, shift), np.roll(x, -1, axis=1), atol=1e-12)
    stack = np.stack([x, 2 * x])
    out = convolve_circular(stack, integrated_profile((7, 7), (3, 3), "gaussian", {"sigma": (1.0, 1.0)}))
    np.testing.assert_allclose(out[1], 2 * out[0], atol=1e-12)


@pytest.mark.parametrize("shape", [(64, 64), (63, 50), (17, 9)])
def test_noise_matches_skimage(shape):
    skr = pytest.importorskip("skimage.restoration")
    rng = np.random.default_rng(1)
    x = rng.normal(size=shape) + np.linspace(0, 5, shape[1])
    assert estimate_noise_sigma(x) == pytest.approx(skr.estimate_sigma(x, channel_axis=None), abs=1e-12)


def test_noise_estimate_on_smooth_scene():
    rng = np.random.default_rng(2)
    y, x = np.mgrid[:256, :256] / 256
    scene = np.sin(6 * x) + np.cos(4 * y)
    assert estimate_noise_sigma(scene + 0.05 * rng.normal(size=scene.shape)) == pytest.approx(0.05, rel=0.05)
    assert finest_diagonal_detail(scene).shape == (129, 129)
    with pytest.raises(ValueError):
        estimate_noise_sigma(np.full((8, 8), np.nan))


def test_integrated_gaussian_exact_and_rotated():
    g = integrated_profile((41, 41), (20.0, 20.3), "gaussian", {"sigma": (1.0, 3.0)})
    assert g.sum() == pytest.approx(1.0, abs=1e-6)
    cy = (g.sum(1) * np.arange(41)).sum() / g.sum()
    cx = (g.sum(0) * np.arange(41)).sum() / g.sum()
    assert (cy, cx) == (pytest.approx(20.0, abs=1e-9), pytest.approx(20.3, abs=1e-6))
    # Rotation de 90° : intégration par sur-échantillonnage, proche de l'exact transposé.
    exact = integrated_profile((41, 41), (20, 20), "gaussian", {"sigma": (1.0, 3.0)})
    rot = integrated_profile((41, 41), (20, 20), "gaussian", {"sigma": (1.0, 3.0), "theta": np.pi / 2}, oversample=9)
    np.testing.assert_allclose(rot, exact.T, atol=2e-4)


def test_integrated_airy_and_moffat():
    a = integrated_profile((33, 33), (16, 16), "airy", {"radius_zero": 3.0})
    assert np.unravel_index(a.argmax(), a.shape) == (16, 16)
    assert a[16, 19] < 0.01 * a[16, 16]  # premier anneau noir à 3 px
    m = integrated_profile((33, 33), (16, 16), "moffat", {"alpha": 2.0, "beta": 2.5, "amplitude": 3.0})
    assert m.max() == pytest.approx(3.0, rel=0.1)
    with pytest.raises(ValueError):
        integrated_profile((5, 5), (2, 2), "lorentz")
