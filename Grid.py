import numpy as np
"""
Grid code was copied from Zingale's
https://github.com/python-hydro/hydro_examples/blob/master/advection/advection.py
"""


class Grid1d(object):

    def __init__(self, nx, ng, x_min=0.0, x_max=1.0):
        self.ng = ng
        self.nx = nx

        self.x_min = x_min
        self.x_max = x_max

        # python is zero-based.  Make easy integers to know where the
        # actual data lives
        self.ilo = ng
        self.ihi = ng + nx - 1

        # physical coords -- cell-centered, left and right edges
        self.dx = (x_max - x_min) / nx
        self.x = x_min + (np.arange(nx + 2 * ng) - ng + 0.5) * self.dx
        self.xl = x_min + (np.arange(nx + 2 * ng) - ng) * self.dx
        self.xr = x_min + (np.arange(nx + 2 * ng) + 1.0) * self.dx
        self.x_actual = self.x[self.ilo: self.ihi + 1]
        # storage for the solution
        self.a = self.zeros_array()

    def zeros_array(self):
        """ return a new array dimensioned for our grid """
        return np.zeros((self.nx + 2 * self.ng), dtype=np.float64)

    def fill_BCs(self):
        """ fill all single ghost cells with periodic boundary conditions """

        for gc in range(self.ng):
            # left boundary
            self.a[self.ilo - 1 - gc] = self.a[self.ihi - gc]

            # right boundary
            self.a[self.ihi + 1 + gc] = self.a[self.ilo + gc]

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if len(e) != 2 * self.ng + self.nx:
            return None

        return np.sqrt(self.dx*np.sum(e[self.ilo:self.ihi+1]**2))
        # return np.max(abs(e[self.ilo:self.ihi + 1]))
