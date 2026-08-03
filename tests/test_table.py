"""Test foilselector.openmcextension.table."""

import numpy as np
import openmc
from scipy import integrate

from foilselector.openmcextension.table import Integral, Tab1DExtended

sample_points = np.array([[0, 1, 2, 3, 4], [1, 1, 2, 0, 6]])
x, y = sample_points
nx = np.clip(x, 0.1, np.inf)
ny = np.clip(y, 0.1, np.inf)


def test_area_scheme_1():
    """Confirm integration of area scheme 1 is correct."""
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [1])
    i = Integral(Tab1DExtended.from_openmc(tab))
    np.testing.assert_array_equal(i._area, [1, 1, 2, 0])


def test_area_scheme_2():
    """Confirm integration of area scheme 2 is correct."""
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Integral(Tab1DExtended.from_openmc(tab))
    np.testing.assert_array_equal(i._area, [1, 1.5, 1, 3])


def test_area_scheme_3():
    """Confirm integration of area scheme 3 is correct."""
    tab = openmc.data.Tabulated1D(nx, y, [len(x)], [3])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(nx[:-1], nx[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [0.9]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i._area, model_answer, rtol=1 / 50, atol=0)


def test_area_scheme_4():
    """Confirm integration of area scheme 4 is correct."""
    tab = openmc.data.Tabulated1D(x, ny, [len(x)], [4])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(x[:-1], x[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [1]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i._area, model_answer, rtol=1 / 50, atol=0)


def test_area_scheme_5():
    """Confirm integration of area scheme 5 is correct."""
    tab = openmc.data.Tabulated1D(nx, ny, [len(x)], [5])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(nx[:-1], nx[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [0.9]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i._area, model_answer, rtol=1 / 50, atol=0)
