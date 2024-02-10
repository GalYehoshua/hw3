import encodings.zlib_codec

import numpy as np
from matplotlib import pyplot as plt
from Grid import Grid1d

CFL: float = 0.5
N = 64  # number of cells


def REA_and_err(num_of_cells):
    grid = Grid1d(num_of_cells, 2)
    grid.a[np.logical_and(grid.x >= 0.333, grid.x <= 0.666)] = 1.0
    initial_condition = grid.a[grid.ilo: grid.ihi + 1]
    h = grid.dx
    dt = CFL * h  # if we want a none unit U, need to change this

    # print(dt, int(1 / dt))

    def delta_a(a):
        g = grid
        dx = g.dx
        return np.array([min(abs(a[i + 1] - a[i - 1]) / 2 * dx,
                             abs(a[i + 1] - a[i]) / 2 * dx,
                             abs(a[i] - a[i - 1]) / 2 * dx) *
                         np.sign(a[i + 1] - a[i - 1]) if (a[i + 1] - a[i]) * (a[i] - a[i - 1]) > 0
                         else 0 for i in range(1, len(a) - 1)])

    def update_a():
        """ conservative update """
        g = grid
        anew = g.zeros_array()
        da = delta_a(g.a)
        anew[g.ilo:g.ihi + 1] = g.a[g.ilo:g.ihi + 1] + \
                                CFL * (g.a[g.ilo - 1: g.ihi] + 0.5 * (1 - CFL) * da[g.ilo - 1: g.ihi]
                                       - g.a[g.ilo: g.ihi + 1] + 0.5 * (1 - CFL) * da[g.ilo: g.ihi + 1])

        return anew

    for n in range(int(1 / dt)):
        grid.a = update_a()
        grid.fill_BCs()

    # plotting the initial wave, and state after one cycle.
    plt.plot(grid.x_actual, initial_condition, color='b')
    plt.plot(grid.x_actual, grid.a[grid.ilo: grid.ihi + 1], color='orange')
    plt.show()


REA_and_err(64)
# nums = np.linspace(64, 564, 4).astype(int)
# plt.plot(nums ** -0.5, [eps(x) for x in nums])
# plt.show()

# in the case of Riemann problem, we saw in class it was equivalent to upwind.
# for now i skip it.
