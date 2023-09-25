"""extend the functionalities of openmc classes Tabulate(openmc.data.Tabulated1D related classes)"""

# numpy stuff
import numpy as np
from numpy import array as ary
from numpy import log as ln
import contextlib # to silence numpy error

import openmc
from openmc.data import INTERPOLATION_SCHEME
from collections.abc import Iterable # to check type

import matplotlib.pyplot as plt
from foilselector.openmcextension.extended_io import detabulate

plot_tab = lambda tab, *args, **kwargs: plt.plot(tab.x, tab.y, *args, **kwargs)

class SilenceNumpyDivisionError(contextlib.ContextDecorator):
    def __enter__(self):
        self.prev_divide_error_state = np.geterr()["divide"] # record current state of error handling style
        np.seterr(divide="ignore") # force ignore all division errors
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.seterr(divide=self.prev_divide_error_state) # undo error silencing
        if exc_type is None:
            return True
        else: # any type of error
            return False

class Integrate():
    __slots__ = ["_area", "func", "_interpolation", "verbose"] # for memory management, in case we want to create a lot of instances of Integrate.
    def __init__(self, func, verbose=False):
        """
        self.interpolation[i] describes the interpolation scheme between self.x[i-1] to self.x[i], using the scheme specified by
            INTERPOLATION_SCHEME[self.interpolation[i]]
        """
        assert isinstance(func, openmc.data.Tabulated1D), "Must be a Tabulated1D object"
        assert all(np.diff(func.x)>=0), "The data points must be stored in a manner so that the x values are monotonically increasing."
        # there are n+1 boundaries, but only n cells. And whenever we use x1, we'll also use x2.
        # Therefore the best way to store x and y is to store them as above: x1, x2, y1, y2.
        self.func = func # pointer to the actual function, so that it can be used later.
        self._interpolation = np.zeros(len(self.func.x[:-1]), dtype=int) # n cells
        for point, scheme_number in list(zip(func.breakpoints, func.interpolation))[::-1]:
            self._interpolation[:point-1] = scheme_number # use an offset of -1 to describe the cell *before* it,
        self._area = self._calculate_area_of_each_cell(self.func.x, self.func.y, self._interpolation)
        self.verbose = verbose

    @SilenceNumpyDivisionError()
    def definite_integral(self, a, b):
        """
        Definite integral that handles an array of (a, b) vs a scalar pair of (a, b) in different manners.
        The main difference is that (a, b) will be clipped back into range if it exceeds the recorded x-values' range in the array case;
        while such treatment won't happen in the scalar case.
        We might change this later to remove the problem of havin g
        """
        assert (np.diff([a,b], axis=0)>=0).all(), "Can only integrate in the positive direction."
        # assert False, "How the fuck are you not catching this shit?"
        if np.not_equal(np.clip(a, self.func.x.min(), self.func.x.max()), a).any():
            if self.verbose:
                print("Integration limit is below recorded range of x values! Clipping it back into range...")
            a = np.clip(a, self.func.x.min(), self.func.x.max())
        if np.not_equal(np.clip(b, self.func.x.min(), self.func.x.max()), b).any():
            if self.verbose:
                print("Integration limit is above recorded range of x values! Clipping it back into range...")
            b = np.clip(b, self.func.x.min(), self.func.x.max())

        if isinstance(a, Iterable):
            assert np.shape(a)==np.shape(b), "The dimension of (a) must match that of (b)"
            assert ary(a).ndim==1, "Must be a flat 1D array"
            return self._definite_integral_array(ary(a), ary(b))
        else:
            return self._definite_integral_array(ary([a]), ary([b]))[0]

    def _definite_integral_array(self, a, b):
        n = len(self._area)

        x_array_2d = np.broadcast_to(self.func.x, [len(a), n+1]).T
        # finding the completely enveloped cells using l_bounds and u_bounds.
        l_bounds = np.broadcast_to(self.func.x[:-1], [len(a), n]).T # we don't care whether or not a is larger than the last x
        u_bounds = np.broadcast_to(self.func.x[1:], [len(b),n]).T # we dont' care whether or not b is smaller than the first x.

        # calculate area.
        ge_a = np.greater_equal(l_bounds, a).T # 2D array of bin upper bounds which are >= a.
        le_b = np.less_equal(u_bounds, b).T # 2D array of bin lower bounds which are <= b.
        # use <= and >= instead of < and > to allow the left-edge and right-edge to have zero dx.

        area_2d = np.broadcast_to(self._area, [len(a), n])
        central_area = (area_2d * (-1 + ge_a + le_b)).sum(axis=1)
        # -1 + False + False = -1 (cell envelope entire [a, b] interval);
        # -1 + True + False = 0, -1 + False - True = 0 (cell to the right/left of the entire [a, b] interval respectively);
        # -1 + True + True = +1 ([a, b] interval envelopes entire cell).

        # left-edge half-cell
        l_ind = n - ge_a.sum(axis=1)
        l_edge_x = ary([a, self.func.x[l_ind]])
        l_edge_y = ary([self.func(a), self.func.y[l_ind]])
        l_edge_scheme = self._interpolation[np.clip(l_ind - 1, 0, None, dtype=int)] # make sure it doesn't go below zero when ge_a sums to equal n (i.e. a is less than the second x).

        # right-edge half-cell.
        r_ind = le_b.sum(axis=1)
        r_edge_x = ary([self.func.x[r_ind], b])
        r_edge_y = ary([self.func.y[r_ind], self.func(b)])
        r_edge_scheme = self._interpolation[np.clip(r_ind, None, n-1, dtype=int)]

        # calculate the left-edge half cell and right-edge half cell areas.
        l_edge_area, r_edge_area = np.zeros(len(a)), np.zeros(len(a))
        for scheme_number in INTERPOLATION_SCHEME.keys(): # loop 5 times (x2 area calculations per loop) to get the l/r edges areas
            matching_l = l_edge_scheme==scheme_number
            matching_r = l_edge_scheme==scheme_number
            l_edge_area[matching_l] = getattr(self, "area_scheme_"+str(scheme_number))(*l_edge_x.T[matching_l].T, *l_edge_y.T[matching_l].T)
            r_edge_area[matching_r] = getattr(self, "area_scheme_"+str(scheme_number))(*r_edge_x.T[matching_r].T, *r_edge_y.T[matching_r].T)

        return l_edge_area+central_area+r_edge_area

    @classmethod
    def _calculate_area_of_each_cell(cls, x, y, interpolation):
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
        areas = np.zeros(len(x[:-1]))
        for scheme_number in INTERPOLATION_SCHEME.keys(): # loop through each type of interpolation scheme
            matching_cells = interpolation==scheme_number # matching_cells is a boolean mask of len = n.
            if matching_cells.sum()>0: # save time by avoiding unnecessary method calls. Don't know if this helps or not, need to test.
                areas[matching_cells] = getattr(cls,
                    "area_scheme_"+str(scheme_number))(x[:-1][matching_cells],
                                                        x[1:][matching_cells],
                                                        y[:-1][matching_cells],
                                                        y[1:][matching_cells]) # x-left, x-right, y-left, y-right
        return areas

    @staticmethod
    def area_scheme_1(x1, x2, y1, y2):
        dx = x2 - x1
        return y1 * dx

    @staticmethod
    def area_scheme_2(x1, x2, y1, y2):
        dx, dy = x2-x1, y2-y1
        return dy*dx/2 + y1*dx

    @staticmethod
    def area_scheme_3(x1, x2, y1, y2):
        """
        0<=x1<=x2
        """
        dx, dy = x2-x1, y2-y1
        dlnx = ln(x2) - ln(x1)
        # m = dy/dlnx
        return y1*dx + dy*x2 - dy*np.nan_to_num(dx/dlnx, nan=x1, posinf=x1, neginf=x1)

    @staticmethod
    def area_scheme_4(x1, x2, y1, y2):
        dx, dy = x2-x1, y2-y1
        dlny = ln(y2) - ln(y1)
        # m = dlny/dx
        return np.nan_to_num(dy/dlny, nan=y1, posinf=y1, neginf=y1) * dx # nan_to_num is needed to take care of y2 = y1,
        # which makes dy/dlny appraoch the value of y1. (since dlny-> 0 one order of magnitude faster than 1 dy->0)
        # if dy is exactly zero, dy/dlny would've returned nan.
        # if dy is slightly less, dy/dlny would've returned neginf.
        # if dy is slightly more, dy/dlny would've returned posinf.

    @staticmethod
    def area_scheme_5(x1, x2, y1, y2):
        """
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
        """
        resulting_area = np.empty(x1.shape)
        dx, dy = x2-x1, y2-y1
        dlnx, dlny = ln(x2) - ln(x1), ln(y2) - ln(y1)
        # shouldn't raise any warnings so far unless x1==0 or y1==0, which isn't well defined (but we can extend the definition to cover it)

        # find the special cases: dlnx==0; x^-1; all others.
        dlnx_0 = dlnx==0
        inverse_special_case = np.logical_and(np.isclose(dlnx, -dlny), ~dlnx_0) # not already included by dlnx==0 case
        normal = ~np.logical_or(inverse_special_case, dlnx_0) # no dlnx==0 and no slope ==-1

        # dlnx==0 case
        resulting_area[dlnx_0] = 0

        # x^-1 case
        resulting_area[inverse_special_case] = y1[inverse_special_case] * dlnx[inverse_special_case] * x1[inverse_special_case] # y1*dlnx*x1

        # normal case
        m = dlny[normal]/dlnx[normal]
        #   problematic if any(x1==0, y1==0, dx==0)
        inv_x1_m = y1[normal]/(x1[normal]**m)
        diff_over_expo = (x2[normal]**(m+1) - x1[normal]**(m+1)) / (m+1)

        resulting_area[normal] = np.nan_to_num(inv_x1_m * diff_over_expo, nan=dx[normal], posinf=dx[normal], neginf=dx[normal])
        return resulting_area
        # m is problematic if
        # any(x1==0, y1==0, dx==0)
        # which respectively gives:
        # T, F, F:  x2*y2 =       = dx*(y1+dy)
        # F, T, F: 0 = y1 = dx*y1
        # F, F, T: 0 = dx = dx*y1 = dx*(y1+dy)
        # T, T, F: 0 = y1 = dx*y1
        # T, F, T: 0 = dx = dx*y1 = dx*(y1+dy)
        # F, T, T: 0 =      dx*y1 = y2 * (y1/y2)
        # T, T, T: 0 = y2 = dx*y1 = dx*(y1+dy)
        # if we loosen the constraints a bit by "blaming it on the user",
        # and say that if x1==0 then (y1 must== y2), ("otherwise it's your fault it integrated wrongly"),
        # then expression (dx*y1) becomes valid for all of the above edge cases listed.

class Tab1DExtended:
    """
    To be finished later
    """
    def __init__(self, x, y, interpolation):
        assert np.shape(x)==np.shape(y)
        assert np.ndim(x)==1
        assert len(interpolation)==len(x)-1
        self.x = ary(x)
        self.y = ary(y)
        self.interpolation = ary(interpolation)

    def __call__(self, x_new):
        x_new = ary(x_new)
        x_new

    def __add__(self, tab2):
        raise NotImplementedError

    def __mul__(self, tab):
        raise NotImplementedError

    @classmethod
    def from_openmc(cls, openmc_instance):
        return cls(**detabulate(openmc_instance))
