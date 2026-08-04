"""Extend the functionalities of openmc classes Tabulate(openmc.data.Tabulated1D related
classes).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt

# numpy stuff
import numpy as np
import openmc
from numpy import array as ary
from numpy import log as ln
from numpy import typing as npt
from openmc.data import INTERPOLATION_SCHEME


def plot_tab(tab: openmc.data.Tabulated1D | Tab1DExtended, *args, **kwargs) -> plt.Axes:  # noqa: ANN002, ANN003
    """Quick function to plot the curve described by the Tab1D."""
    return plt.plot(tab.x, tab.y, *args, **kwargs)  # noqa: DOC201


def expand_interpolation_regions(
    interpolation: Iterable[float],
    breakpoints: Iterable[int],
    length_of_table: int,
) -> npt.NDArray[float]:
    """Convert the openmc.data.Tabulated1D interpolation scheme data into a format
    that is more verbose, but easier to manage.

    Parameters
    ----------
    inteprolation:
        The .interpolation list attribute of an instance of Tabulated1D

    Returns
    -------
    new_interpolation:
        A list with len==length_of_table-1, each element is the interpolation scheme
        for that particular bin.
    """
    # n cells, with n+1 boundaries
    new_interpolation = np.zeros(length_of_table - 1, dtype=int)
    for point, scheme_number in list(zip(breakpoints, interpolation, strict=False))[
        ::-1
    ]:
        # use an offset of -1 to describe the cell *before* it,
        new_interpolation[: point - 1] = scheme_number
    return new_interpolation


def detabulate(openmc_tab1d: openmc.data.Tabulated1D | Tab1DExtended) -> dict:
    """Convert a openmc.data.openmc_tab1d into something json serialize-able.

    Returns
    -------
    :
        A dictionary of the data described by the Tab1D.
    """
    if isinstance(openmc_tab1d, Tab1DExtended):
        openmc_tab1d = openmc_tab1d.restore_openmc_copy()
    scheme = expand_interpolation_regions(
        openmc_tab1d.interpolation,
        openmc_tab1d.breakpoints,
        len(openmc_tab1d.x),
    )

    return dict(
        x=openmc_tab1d.x.tolist(),
        y=openmc_tab1d.y.tolist(),
        interpolation=scheme.tolist(),
    )


def tabulate(detabulated_dict: dict) -> openmc.data.Tabulated1D:
    """
    Parameters
    ----------
    detabulated_dict:
        Should have 3 columns: "x" (list of len=n+1), "y" (list of len=n+1),
        "interpolation" (list of len=n).

    Returns
    -------
    :
        An openmc.data.Tabulated1D reconstructed from the dictionary of data.
    """
    interpolation_long = list(detabulated_dict["interpolation"])  # make a copy
    # pad the interpolation_long to the lenght of n+1
    interpolation_long.append(0)
    # the interpolation scheme = N/A; use zero as placeholder for N/A.
    breakpoints = sorted(np.argwhere(np.diff(interpolation_long)).flatten() + 2)
    interpolation = ary(interpolation_long)[ary(breakpoints) - 2].tolist()
    return openmc.data.Tabulated1D(
        detabulated_dict["x"],
        detabulated_dict["y"],
        breakpoints,
        interpolation,
    )


class Integral:
    """The integral of a Tabulated function."""

    __slots__ = [
        "_interpolation",
        "areas",
        "func",
        "verbose",
    ]  # for memory management, in case we want to create a lot of instances of Integral.

    def __init__(
        self,
        func: openmc.data.Tabulated1D | Tab1DExtended,
        *,
        verbose: bool = False,
    ):
        """
        Attributes
        ----------
        interpolation[i]:
            describes the interpolation scheme between self.x[i-1] to self.x[i],
            using the scheme specified by INTERPOLATION_SCHEME[self.interpolation[i]].
        areas[i]:
            describes the area from  self.func(self.func.x[i-1]) to
            self.func(self.func.x[i]).

        Raises
        ------
        ValueError
            If the x value is not monotonically increasing.
        TypeError
            If the func is not within the allowed types.
        """
        if not (np.diff(func.x) >= 0).all():
            raise ValueError(
                "The data points must be stored in a manner so that the x "
                "values are monotonically increasing.",
            )
        self.func = func  # pointer to the actual function, so that it can be used later.
        if isinstance(func, openmc.data.Tabulated1D):
            self._interpolation = expand_interpolation_regions(
                func.interpolation,
                func.breakpoints,
                len(func.x),
            )
        elif isinstance(func, Tab1DExtended):
            self._interpolation = func.interpolation
        else:
            raise TypeError("Must be a Tabulated1D/Tab1DExtended object!")
        self.areas = self._calculate_area_of_each_cell(
            self.func.x,
            self.func.y,
            self._interpolation,
        )
        self.verbose = verbose

    def definite_integral(
        self,
        a: npt.NDArray[float] | float,
        b: npt.NDArray[float] | float,
    ) -> npt.NDArray[float] | float:
        """
        Definite integral that handles either an array pair of (a, b) or
        a scalar pair (a, b).

        Parameters
        ----------
        a:
            The integration lower limit(s).
        b:
            The integration upper limit(s).

        Returns
        -------
        :
            The function integrated between a and b.

        Raises
        ------
        ValueError
            When the integration limits aren't in the right shape (1D array or scalar),
            or is disordered (i.e. not (a<=b).all()).
        """
        if np.shape(a) != np.shape(b):
            raise ValueError("The dimension of (a) must match that of (b)")
        if not isinstance(a, Iterable):
            return self._definite_integral_array(ary([a]), ary([b]))[0]
        if ary(a).ndim != 1:
            raise ValueError(f"{a} must be a flat 1D array")
        if not (np.diff([a, b], axis=0) >= 0).all():
            raise ValueError("Can only integrate in the positive direction.")
        if np.not_equal(np.clip(a, self.func.x.min(), self.func.x.max()), a).any():
            if self.verbose:
                print(
                    f"Lower integration limit {a} is beyond (likely below) recorded "
                    "range of x values! Clipping it back into range...",
                )
            a = np.clip(a, self.func.x.min(), self.func.x.max())
        if np.not_equal(np.clip(b, self.func.x.min(), self.func.x.max()), b).any():
            if self.verbose:
                print(
                    f"Integration limit {b} is beyond (likely above) recorded "
                    "range of x values! Clipping it back into range...",
                )
            b = np.clip(b, self.func.x.min(), self.func.x.max())

        return self._definite_integral_array(ary(a), ary(b))

    def _definite_integral_array(
        self,
        a: npt.NDArray[float],
        b: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        # finding the completely enveloped cells using l_bounds and u_bounds.
        n = len(self.areas)
        l_bounds = np.broadcast_to(
            self.func.x[:-1],
            [len(a), n],
        ).T  # we don't care whether or not a is larger than the last x
        u_bounds = np.broadcast_to(
            self.func.x[1:],
            [len(b), n],
        ).T  # we dont' care whether or not b is smaller than the first x.

        # calculate area.
        ge_a = np.greater_equal(
            l_bounds,
            a,
        ).T  # 2D array of bin upper bounds which are >= a.
        le_b = np.less_equal(
            u_bounds,
            b,
        ).T  # 2D array of bin lower bounds which are <= b.
        # use <= and >= instead of < and > to allow the left-edge and right-edge to
        # have zero dx.

        area_2d = np.broadcast_to(self.areas, [len(a), n])
        central_area = (area_2d * (-1 + ge_a + le_b)).sum(axis=1)
        # -1 + False + False = -1 (cell envelope entire [a, b] interval);
        # -1 + True + False = 0,
        # -1 + False - True = 0.
        # (cell to the right/left of the entire [a, b] interval respectively);
        # -1 + True + True = +1 ([a, b] interval envelopes entire cell).

        # left-edge half-cell
        l_ind = n - ge_a.sum(axis=1)
        l_edge_x = ary([a, self.func.x[l_ind]])
        l_edge_y = ary([self.func(a), self.func.y[l_ind]])
        l_edge_scheme = self._interpolation[
            np.clip(l_ind - 1, 0, None, dtype=int)
        ]  # make sure it doesn't go below zero when ge_a sums to equal n
        # (i.e. a is less than the second x).

        # right-edge half-cell.
        r_ind = le_b.sum(axis=1)
        r_edge_x = ary([self.func.x[r_ind], b])
        r_edge_y = ary([self.func.y[r_ind], self.func(b)])

        # calculate the left-edge half cell and right-edge half cell areas.
        l_edge_area, r_edge_area = np.zeros(len(a)), np.zeros(len(a))
        for scheme_number in INTERPOLATION_SCHEME:
            # loop 5 times (x2 area calculations per loop) to get the l/r edges areas
            matching_l = l_edge_scheme == scheme_number
            matching_r = l_edge_scheme == scheme_number
            l_edge_area[matching_l] = getattr(self, "area_scheme_" + str(scheme_number))(
                *l_edge_x.T[matching_l].T,
                *l_edge_y.T[matching_l].T,
            )
            r_edge_area[matching_r] = getattr(self, "area_scheme_" + str(scheme_number))(
                *r_edge_x.T[matching_r].T,
                *r_edge_y.T[matching_r].T,
            )

        return l_edge_area + central_area + r_edge_area

    @classmethod
    def _calculate_area_of_each_cell(
        cls,
        x: npt.NDArray[float],
        y: npt.NDArray[float],
        interpolation: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """
        Parameters
        ----------
        x : the list of x coordinates where the boundaries of the cells are. Length = n+1
        y : the list of y coordinates, denoting the values of the. Length = n+1
        interpolation : the list of interpolation schemes for each cell. Length = n.

        Returns
        -------
        area : Area of each cell. Length = n.
        """
        # for interpoloation_scheme in
        _areas = np.zeros(len(x[:-1]))
        for scheme_number in INTERPOLATION_SCHEME:
            # loop through each type of interpolation scheme
            matching_cells = (
                interpolation == scheme_number
            )  # matching_cells is a boolean mask of len = n.
            if (
                matching_cells.sum() > 0
            ):  # save time by avoiding unnecessary method calls.
                # Needs tests to confirm if it saves time.
                _areas[matching_cells] = getattr(
                    cls,
                    "area_scheme_" + str(scheme_number),
                )(
                    x[:-1][matching_cells],
                    x[1:][matching_cells],
                    y[:-1][matching_cells],
                    y[1:][matching_cells],
                )  # x-left, x-right, y-left, y-right
        return _areas

    @staticmethod
    def area_scheme_1(
        x1: npt.NDArray[float],
        x2: npt.NDArray[float],
        y1: npt.NDArray[float],
        y2: npt.NDArray[float],  # noqa: ARG004
    ) -> npt.NDArray[float]:
        """Integrate the area under a curve that uses interpolation scheme 1.

        Returns
        -------
        :
            area from x1 to x2.
        """
        dx = x2 - x1
        return y1 * dx

    @staticmethod
    def area_scheme_2(
        x1: npt.NDArray[float],
        x2: npt.NDArray[float],
        y1: npt.NDArray[float],
        y2: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Integrate the area under a curve that uses interpolation scheme 2.

        Returns
        -------
        :
            area from x1 to x2.
        """
        dx, dy = x2 - x1, y2 - y1
        return dy * dx / 2 + y1 * dx

    @staticmethod
    def area_scheme_3(
        x1: npt.NDArray[float],
        x2: npt.NDArray[float],
        y1: npt.NDArray[float],
        y2: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Integrate the area under a curve that uses interpolation scheme 3.

        Returns
        -------
        :
            area from x1 to x2.
        """
        dx, dy = x2 - x1, y2 - y1
        dx_dlnx = x1.copy()
        valid = np.logical_and(x2 != x1, x1 > 0.0)
        dx_dlnx[valid] = dx[valid] / (ln(x2[valid]) - ln(x1[valid]))

        return y1 * dx + dy * x2 - dy * dx_dlnx

    @staticmethod
    def area_scheme_4(
        x1: npt.NDArray[float],
        x2: npt.NDArray[float],
        y1: npt.NDArray[float],
        y2: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Integrate the area under a curve that uses interpolation scheme 4.

        Returns
        -------
        :
            area from x1 to x2.
        """
        dx, dy = x2 - x1, y2 - y1
        dy_dlny = y1.copy()
        valid = np.logical_and(y2 != y1, np.logical_and(y1 > 0.0, y2 > 0.0))
        dy_dlny[valid] = dy[valid] / (ln(y2[valid]) - ln(y1[valid]))
        return dy_dlny * dx

    @staticmethod
    def area_scheme_5(
        x1: npt.NDArray[float],
        x2: npt.NDArray[float],
        y1: npt.NDArray[float],
        y2: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Integrate the area under a curve that uses interpolation scheme 5.

        if m==-1:
            should give y1/x1**m *(dlnx)

            if x1==0: should give dx*y2 = x2*y2.
            if y1==0: should give 0 = y1.
            if dx ==0: should give 0 = dx.
        then:
            # dlnx -> inf is covered by x1==0
            dlny -> inf is covered by y1==0
            m -> inf is covered by dlny -> inf (y1==0)

        inv_x1_m = 1/x1**m
        diff_over_expo = (x2**(m+1) - x1**2(m+1)) / (m+1)

        Returns
        -------
        :
            area from x1 to x2.
        """
        resulting_area = np.zeros_like(y1)
        dx = x2 - x1
        zero_width = x2 == x1
        zero_height = y2 == y1

        dlnx = np.zeros_like(x1)
        x0 = np.logical_or(x1 > 0.0, x2 > 0.0)
        valid_x = np.logical_and(~zero_width, x0)
        dlnx[valid_x] = ln(x2[valid_x]) - ln(x1[valid_x])

        dlny = np.zeros_like(x1)
        y0 = np.logical_or(y1 > 0.0, y2 > 0.0)
        valid_y = np.logical_and(~zero_height, y0)
        dlny[valid_y] = ln(y2[valid_y]) - ln(y1[valid_y])

        # zero_width area = 0.0
        resulting_area[zero_height] = y1[zero_height] * dx[zero_height]
        resulting_area[np.logical_or(x1 > 0.0, x2)] = dx * y2

        # find the special cases: dlnx==0; x^-1.
        m_1_case = np.logical_and(
            np.isclose(dlnx, -dlny),
            np.logical_and(valid_x, valid_y),
        )
        normal = np.logical_and(~m_1_case, np.logical_and(valid_x, valid_y))
        # normal case: no dlnx==0 and no slope ==-1

        # x^-1 case
        resulting_area[m_1_case] = y1[m_1_case] * dlnx[m_1_case] * x1[m_1_case]

        # normal case
        m = dlny[normal] / dlnx[normal]
        # problematic if any(x1==0, y1==0, dx==0)
        quotient = (x2[normal] / x1[normal]) ** (m + 1)
        diff_over_expo = (quotient - 1) / (m + 1)
        resulting_area[normal] = y1[normal] * x1[normal] * diff_over_expo
        return resulting_area


class Tab1DExtended:
    """Tabulated1D class extended so that it can be used for higher level functions."""

    _fields = ("x", "y", "interpolation")

    def __init__(
        self,
        x: npt.NDArray[float],
        y: npt.NDArray[float],
        interpolation: npt.NDArray[float],
    ):
        if (
            (np.shape(x) != np.shape(y))
            or (np.ndim(x) != 1)
            or (len(interpolation) != len(x) - 1)
        ):
            raise ValueError("Expected 1D array of the correct shape.")
        self.x = ary(x)
        self.y = ary(y)
        self.interpolation = ary(interpolation)

    def _asdict(self) -> dict:
        return dict(x=self.x, y=self.y, interpolation=self.interpolation)

    def restore_openmc_copy(self) -> openmc.data.Tabulated1D:
        """Create a copy as a openmc.data.Tabulated1D table.

        Returns
        -------
        :
            a reconstructed openmc.data.Tabulated1D object, which is more compact.
        """
        return tabulate(dict(x=self.x, y=self.y, interpolation=self.interpolation))

    def offset_x(self, x_offset: float | npt.NDArray) -> Tab1DExtended:
        """Create a copy of itself, but with the x data points offset horizontally.

        Returns
        -------
        :
            a copy of the function with an offset applied on all values of x.
        """
        return self.__class__(self.x + x_offset, self.y, self.interpolation)

    def __call__(self, x: float | npt.NDArray) -> float | npt.NDArray:
        """
        Create a copy of this Tab1DExtended on the fly in openmc.data.Tabulated1D, then
        direct all calls to that function.

        A fairly compute-intensive implementation, but one that is guaranteed to work.

        Returns
        -------
        :
            the function interpolated at the required x value(s).
        """
        return self.restore_openmc_copy()(x)

    def __add__(self, y_offset: float | npt.NDArray) -> Tab1DExtended:
        """Create a copy of itself, but with the y data points offset vertically.

        Returns
        -------
        :
            A copy of the function with an offset applied on all values of y.
        """
        return self.__class__(self.x, self.y + y_offset, self.interpolation)

    def __mul__(self, scale_factor: float) -> Tab1DExtended:
        """Scale all y-vales by the scalar scale_factor.

        Returns
        -------
        :
            A copy of the function with an offset applied on all values of y.

        Raises
        ------
        NotImplementedError:
            Currently does not support direct multiplication onto another function.
        """
        if isinstance(scale_factor, Tab1DExtended | openmc.data.Tabulated1D):
            raise NotImplementedError("Use apply_scaling instead!")
        return self.__class__(self.x, self.y * scale_factor, self.interpolation)

    def apply_scaling(
        self,
        other_curve: Callable,
        subdivision: int = 10,
    ) -> Tab1DExtended:
        """More finely divide the current curve into a number of subdivisions,
        then scale each of these finely divided points up or down by the scale factor
        specified by the other curve. i.e.
            scale_factor = other_curve(finely_divided_x)
            finely_divided_y = self.y * scale_factor
        Depending on what interpolation scheme was used in the other_curve and
        whether the two curves have the same set of x-values or not, the resultant
        function may not produce the accurately interpolated data. However, it will do
        for the time being for calcaulating the continuous gamma distribution
        * absolute efficiency of the detector setup.

        Parameters
        ----------
        other_curve: Tab1DExtended | openmc.data.Tabulated1D | Callable
            A function that returns a scale_factor for a given x.


        Returns
        -------
        :
            Linearized curve. While each point's y-value was obtained by scaling the
            y-values appropriately, the interpolation scheme is no longer preserved,
            and is chosen to be linear.
        """
        new_x = np.linspace(
            self.x[:-1],
            self.x[1:],
            subdivision,
            endpoint=False,
        ).T.flatten()
        new_y = self(new_x) * other_curve(new_x)
        new_interp = np.ones((len(self.x) - 1) * subdivision, dtype=int) * 2
        new_x = np.append(new_x, self.x[-1])
        new_y = np.append(new_y, self([self.x[-1]])[0] * other_curve([self.x[-1]])[0])
        return self.__class__(
            x=new_x,
            y=new_y,
            interpolation=new_interp,
        )

    def __hash__(self) -> int:
        """Turn its data into tuples, then hash the resulting 3-tuple.

        Returns
        -------
        :
            The hash created by the tuple of all of its data (x, y, interpolation).
        """
        return hash((tuple(self.x), tuple(self.y), tuple(self.interpolation)))

    def __repr__(self) -> str:
        """Include the min and max x and y ranges in the repr text."""
        return (  # noqa: DOC201
            f"<{self.__class__!s} with {len(self.interpolation)} cells, where x is "
            f"between {np.min(self.x)}-{np.max(self.x)} and "
            f"y is between {np.min(self.y)}-{np.max(self.y)}>"
        )

    def copy(self) -> Tab1DExtended:
        """Create a new instance of Tab1DExtended using the same underlying data."""
        return self.__class__(self.x, self.y, self.interpolation)  # noqa: DOC201

    @classmethod
    def from_openmc(cls, openmc_instance: openmc.data.Tabulated1D):
        """Create an instance of Tab1DExtended from an instance of
        openmc.data.Tabulated1D.
        """
        return cls(**detabulate(openmc_instance))  # noqa: DOC201

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        """Create a plot according to the underlying data.

        Parameters
        ----------
        ax:
            the plt.Axes object on which the function shall be plotted.

        Returns
        -------
        ax:
            the plt.Axes object on which the function has been plotted.
        """
        ax = ax or plt.axes()
        continuous_x = np.linspace(
            self.x[:-1],
            np.nextafter(self.x[1:], -1),
            endpoint=True,
        ).T.flatten()
        continuous_x = np.append(continuous_x, self.x[-1])
        continuous_y = self(continuous_x)
        ax.plot(continuous_x, continuous_y)
        ax.scatter(self.x, self.y)
        return ax
