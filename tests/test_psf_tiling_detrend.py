"""Tests de eoqual.core.psf, tiling et detrend."""
import numpy as np
import pytest

from eoqual.core.detrend import fit_plane_residual
from eoqual.core.psf import airy_psf, encircled_energy, radial_profile, strehl_ratio
from eoqual.core.tiling import grid_slices, tile_bounds, tile_slices

WAVELENGTH, DIAMETER = 0.5e-6, 0.1
LAMBDA_OVER_D = WAVELENGTH / DIAMETER


@pytest.fixture(scope="module")
def airy():
    # 8 pixels par lambda/D : PSF bien échantillonnée
    return airy_psf((257, 257), WAVELENGTH, DIAMETER, LAMBDA_OVER_D / 8)


def test_airy_normalised_and_strehl(airy):
    assert airy.sum() == pytest.approx(1.0)
    assert strehl_ratio(airy, airy) == pytest.approx(1.0)
    with pytest.warns(UserWarning):
        strehl_ratio(airy * 2, airy)


def test_encircled_energy_first_dark_ring(airy):
    # Premier anneau noir à 1.22 lambda/D : EE ~ 0.838 (Born & Wolf)
    radii, ee = encircled_energy(airy, radii=np.array([1.22 * 8]))
    assert ee[0] == pytest.approx(0.838, abs=0.01)
    radii, ee = encircled_energy(airy)
    assert np.all(np.diff(ee) >= 0) and ee[-1] <= 1.0 + 1e-12


def test_radial_profile_peak_at_center(airy):
    prof = radial_profile(airy)
    assert np.argmax(prof) == 0


def test_tiling():
    assert tile_bounds(10, 20) == [(0, 10)]
    bounds = tile_bounds(100, 32, overlap=0.25)
    assert bounds[0] == (0, 32) and bounds[-1] == (68, 100)
    assert len(tile_slices((100, 50), 32, 0.25)) == len(bounds) * len(tile_bounds(50, 32, 0.25))
    grid = grid_slices((90, 60), 3)
    assert len(grid) == 9 and grid[-1] == (slice(60, 90), slice(40, 60))


def test_detrend_removes_plane():
    yy, xx = np.mgrid[0:8, 0:8]
    values = (3 * xx - 2 * yy + 5).ravel().astype(float)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    assert np.allclose(fit_plane_residual(values, coords), 0)
