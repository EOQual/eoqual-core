"""Tests de eoqual.core.fourier."""
import numpy as np
import pytest

from eoqual.core.fourier import boundary_jump, dequantize, frequencies, perdecomp, spectrum, translate


def _periodic_laplacian(a):
    return sum(np.roll(a, k, axis=ax) for k in (1, -1) for ax in (0, 1)) - 4 * a


def _interior_laplacian(a):
    lap = _periodic_laplacian(a)
    lap[0, :] -= a[-1, :] - a[0, :]
    lap[-1, :] -= a[0, :] - a[-1, :]
    lap[:, 0] -= a[:, -1] - a[:, 0]
    lap[:, -1] -= a[:, 0] - a[:, -1]
    return lap


@pytest.mark.parametrize("shape", [(64, 64), (37, 50)])
def test_perdecomp_properties(shape):
    rng = np.random.default_rng(0)
    u = rng.random(shape) + np.linspace(0, 3, shape[1])[None, :]
    p, s = perdecomp(u)
    assert np.allclose(p + s, u)
    assert abs(s.mean()) < 1e-12
    # Caractérisation de Moisan : Δ_per p = Δ_interieur u
    assert np.allclose(_periodic_laplacian(p), _interior_laplacian(u))


def test_perdecomp_reduces_boundary_jump():
    y, x = np.mgrid[0:64, 0:64] / 63
    u = x + (y > 0.5) * 0.5
    p, _ = perdecomp(u)
    assert boundary_jump(p) < 0.05 * boundary_jump(u)


def test_perdecomp_unchanged_when_borders_match():
    # s = 0 ssi les bords opposés sont égaux (v = 0) ; une image périodique
    # échantillonnée ne vérifie pas cela en général, s y est seulement petit.
    u = np.random.default_rng(4).random((32, 48))
    u[-1, :] = u[0, :]
    u[:, -1] = u[:, 0]
    p, s = perdecomp(u)
    assert np.allclose(s, 0) and np.allclose(p, u)


def test_translate_integer_is_roll():
    u = np.random.default_rng(1).random((40, 30))
    assert np.allclose(translate(u, 3, -2), np.roll(u, (3, -2), axis=(0, 1)))


def test_dequantize_twice_is_unit_shift_odd_size():
    u = np.random.default_rng(2).random((33, 41))
    assert np.allclose(dequantize(dequantize(u)), np.roll(u, (1, 1), axis=(0, 1)))


def test_dequantize_preserves_mean_and_breaks_integers():
    u = np.random.default_rng(3).integers(0, 8, (32, 32)).astype(float)
    w = dequantize(u)
    assert np.isclose(w.mean(), u.mean())
    assert not np.allclose(w, np.round(w))


def test_frequencies_nyquist():
    fy, fx = frequencies((64, 51))
    assert fy.min() == -0.5 and fx.max() < 0.5
    assert fy[32] == 0 and fx[25] == 0


def test_spectrum_shapes_and_errors():
    u = np.zeros((16, 20))
    for m in ("raw", "perdecomp", "hann"):
        A, fy, fx = spectrum(u, method=m, dequant=True)
        assert A.shape == (16, 20) and fy.size == 16 and fx.size == 20
    with pytest.raises(ValueError):
        spectrum(u, method="foo")
    with pytest.raises(ValueError):
        spectrum(u, output="foo")
    with pytest.raises(ValueError):
        perdecomp(np.zeros((4, 4, 3)))


def test_dequant_only_changes_phase_except_nyquist():
    u = np.random.default_rng(5).random((33, 41))
    A0 = spectrum(u, output="amplitude")[0]
    A1 = spectrum(u, output="amplitude", dequant=True)[0]
    assert np.allclose(A0, A1)  # taille impaire : amplitude identique
    U0 = spectrum(u, output="complex")[0]
    U1 = spectrum(u, output="complex", dequant=True)[0]
    assert not np.allclose(U0, U1)  # mais la phase change
