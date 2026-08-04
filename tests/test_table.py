"""Test foilselector.openmcextension.table."""

import numpy as np
import openmc
from scipy import integrate

from foilselector.openmcextension.table import Integral, Tab1DExtended

sample_points = np.array([[0, 1, 2, 3, 4], [1, 1, 2, 0, 6]])
x, y = sample_points
nx = np.clip(x, 0.1, np.inf)
ny = np.clip(y, 0.1, np.inf)


def test_table_call():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [1])
    Tab1DExtended.from_openmc(tab)(1)


def test_area_scheme_1():
    """Confirm integration of area scheme 1 is correct."""
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [1])
    i = Integral(Tab1DExtended.from_openmc(tab))
    np.testing.assert_array_equal(i.areas, [1, 1, 2, 0])


def test_area_scheme_2():
    """Confirm integration of area scheme 2 is correct."""
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Integral(Tab1DExtended.from_openmc(tab))
    np.testing.assert_array_equal(i.areas, [1, 1.5, 1, 3])


def test_area_scheme_3():
    """Confirm integration of area scheme 3 is correct."""
    tab = openmc.data.Tabulated1D(nx, y, [len(nx)], [3])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(nx[:-1], nx[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [0.9]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i.areas, model_answer, rtol=1 / 50, atol=0)


def test_area_scheme_4():
    """Confirm integration of area scheme 4 is correct."""
    tab = openmc.data.Tabulated1D(x, ny, [len(x)], [4])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(x[:-1], x[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [1]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i.areas, model_answer, rtol=1 / 50, atol=0)


def test_area_scheme_5():
    """Confirm integration of area scheme 5 is correct."""
    tab = openmc.data.Tabulated1D(nx, ny, [len(nx)], [5])
    i = Integral(Tab1DExtended.from_openmc(tab))
    smooth_x = np.linspace(nx[:-1], nx[1:]).T
    smooth_y = i.func(smooth_x)
    model_answer = [0.9]
    for yi, xi in zip(smooth_y[1:], smooth_x[1:], strict=False):
        model_answer.append(integrate.trapz(yi, xi))
    assert np.allclose(i.areas, model_answer, rtol=1 / 50, atol=0)


def test_conversion_idempotency():
    """Converting back-and-forth between openmc.data.Tabulated1D and Tab1DExtended should
    preserve the data.
    """
    tab = openmc.data.Tabulated1D(nx, ny, [len(nx)], [5])
    i = Tab1DExtended.from_openmc(tab)
    tab2 = i.restore_openmc_copy()
    j = Tab1DExtended.from_openmc(tab2)
    np.testing.assert_array_equal(tab.x, tab2.x)
    np.testing.assert_array_equal(tab.y, tab2.y)
    np.testing.assert_array_equal(tab.breakpoints, tab2.breakpoints)
    np.testing.assert_array_equal(tab.interpolation, tab2.interpolation)
    np.testing.assert_array_equal(i.x, j.x)
    np.testing.assert_array_equal(i.y, j.y)
    np.testing.assert_array_equal(i.interpolation, j.interpolation)


def test_table_plot():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [1])
    Tab1DExtended.from_openmc(tab).plot()

    tab = openmc.data.Tabulated1D(nx, ny, [len(nx)], [5])
    Tab1DExtended.from_openmc(tab).plot()


def test_table_copy():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    Tab1DExtended.from_openmc(tab).copy()


def test_table_repr():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    name = str(i)
    assert type(i).__name__ in name
    assert f"{len(x) - 1} cells" in name
    assert f"{min(x)} - {max(x)}" in name or f"{min(x)}-{max(x)}" in name
    assert f"{min(y)} - {max(y)}" in name or f"{min(y)}-{max(y)}" in name


def test_hash():
    """Make sure that the hash depends solely on the current state of the data stored
    inside Tab1DExtended.
    """
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    j = i.copy()
    assert hash(i) == hash(j), "Identical data should give the same hash."

    tab2 = openmc.data.Tabulated1D(nx, y, [len(nx)], [2])
    i2 = Tab1DExtended.from_openmc(tab2)
    tab3 = openmc.data.Tabulated1D(x, y, [len(x)], [1])
    i3 = Tab1DExtended.from_openmc(tab3)
    assert hash(i2) != hash(i), "Similar data should give not identical"
    assert hash(i3) != hash(i), "Similar data should give not identical"
    recorded_hash = hash(i)
    i.x = i.x[:-1]
    i.y = i.y[:-1]
    i.interpolation = i.interpolation[:-1]
    assert hash(i) != recorded_hash, "Changing the data should change the hash too."


def test_table_asdict():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    Tab1DExtended.from_openmc(tab)._asdict()


def test_offset_x():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    j = i.offset_x(1)
    np.testing.assert_array_equal(j.x, x + 1)


def test_offset_y():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    j = i + 1
    np.testing.assert_array_equal(j.y, y + 1)


def test_scale_y():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    j = i * 1.5
    np.testing.assert_array_equal(j.y, y * 1.5)


def test_apply_scaling():
    tab = openmc.data.Tabulated1D(x, y, [len(x)], [2])
    i = Tab1DExtended.from_openmc(tab)
    tab2 = openmc.data.Tabulated1D(nx, ny, [len(nx)], [5])
    j = Tab1DExtended.from_openmc(tab2)
    k = j.apply_scaling(i)

    k.plot()
