"""Tests de eoqual.core.edge sur une ESF gaussienne à MTF analytique."""
import numpy as np
import pytest
from scipy.special import erf

from eoqual.core.edge import (
    bin_profile, esf_to_lsf, frequency_at, fwhm, lsf_to_mtf, mtf_at,
    derivative_correction, rer,
)

SIGMA = 1.0
DX = 0.1


@pytest.fixture
def gaussian_edge():
    x = np.arange(-20, 20, DX)
    esf = 0.5 * (1 + erf(x / (SIGMA * np.sqrt(2))))
    return x, esf


def test_lsf_mtf_matches_gaussian(gaussian_edge):
    x, esf = gaussian_edge
    lsf = esf_to_lsf(esf, DX, filter_type="none")
    freqs, mtf = lsf_to_mtf(lsf, DX)
    mtf = derivative_correction(freqs, mtf, DX)
    expected = np.exp(-2 * (np.pi * SIGMA * freqs) ** 2)
    keep = freqs <= 0.5
    assert np.max(np.abs(mtf[keep] - expected[keep])) < 2e-3
    assert mtf_at(freqs, mtf, 0.25) == pytest.approx(np.exp(-2 * (np.pi * 0.25) ** 2), abs=2e-3)


def test_lsf_to_mtf_zero_padding(gaussian_edge):
    _, esf = gaussian_edge
    lsf = np.gradient(esf, DX)
    f1, m1 = lsf_to_mtf(lsf, DX)
    f4, m4 = lsf_to_mtf(lsf, DX, n_fft=4 * lsf.size)
    assert f4.size > 3 * f1.size
    assert mtf_at(f4, m4, 0.3) == pytest.approx(mtf_at(f1, m1, 0.3), abs=1e-3)


def test_frequency_at_mtf50():
    f = np.linspace(0, 0.5, 501)
    mtf = np.exp(-2 * (np.pi * SIGMA * f) ** 2)
    f50 = np.sqrt(np.log(2) / 2) / (np.pi * SIGMA)
    assert frequency_at(f, mtf, 0.5) == pytest.approx(f50, abs=1e-4)
    assert np.isnan(frequency_at(f, mtf, 1e-9))


def test_fwhm_gaussian_and_robust_to_isolated_bounce(gaussian_edge):
    x, esf = gaussian_edge
    lsf = np.gradient(esf, DX)
    expected = 2 * np.sqrt(2 * np.log(2)) * SIGMA
    assert fwhm(x, lsf) == pytest.approx(expected, rel=5e-3)  # interpolation linéaire au pas DX
    bounced = lsf.copy()
    bounced[np.argmin(np.abs(x - 15))] = lsf.max()  # rebond isolé loin du pic
    assert fwhm(x, bounced) == pytest.approx(expected, rel=5e-3)


def test_rer(gaussian_edge):
    x, esf = gaussian_edge
    assert rer(x, esf) == pytest.approx(erf(0.5 / (SIGMA * np.sqrt(2))), abs=1e-3)


def test_bin_profile():
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-5, 5, 20000)
    values = (offsets > 0).astype(float)
    esf = bin_profile(offsets, values, half_width=5, oversample=4)
    assert esf.shape == (40,)
    assert esf[0] == 0 and esf[-1] == 1
    assert bin_profile(np.zeros(3), np.ones(3), 5, 4) is None  # grille trop lacunaire


def test_bad_options(gaussian_edge):
    _, esf = gaussian_edge
    with pytest.raises(ValueError):
        esf_to_lsf(esf, DX, derivative_method="nope")
    with pytest.raises(ValueError):
        esf_to_lsf(esf, DX, filter_type="nope")
