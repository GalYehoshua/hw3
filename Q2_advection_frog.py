import numpy as np
from matplotlib import pyplot as plt

CFL: float = 0.5
N = 64  # number of cells
O: int = 2 * N // 3  # number of zeros in IC
from Grid import Grid1d

# shifting cells according to period BC, for leap frog we need to take into account both positive and negative shifts
def neg_shift_adv(adv):
    return np.concatenate([[adv[-2]], adv[:-1]])


def pos_shift_adv(adv):
    return np.concatenate([adv[1:], [adv[1]]])


assert (neg_shift_adv(np.array([0, 1, 2, 1, 0])) == np.array([1, 0, 1, 2, 1])).all()
assert (pos_shift_adv(np.array([0, 1, 2, 1, 0])) == np.array([1, 2, 1, 0, 1])).all()


# mean sqr. over 1 period.
def err(adv_f, adv_i, dx):
    assert len(adv_f) == len(adv_i), f"Something wrong with lengths {adv_f}, {adv_i}, {len(adv_f) - len(adv_i)}"
    return dx * sum([(adv_f[i] - adv_i[i]) ** 2 for i in range(len(adv_i))])


def eps(num_of_cells):
    x_range = np.linspace(0, 1, num_of_cells)
    # We use here U = 1 for the velocity of the advection
    h = x_range[1] - x_range[0]
    dt = CFL * h  # if we want a none unit U, need to change this
    # print(dt, int(1 / dt))

    # TODO : check other initial conditions from Garcia, Lax ICs
    grid = Grid1d(num_of_cells, 0)
    grid.a[np.logical_and(grid.x >= 0.333, grid.x <= 0.666)] = 1.0
    initial_condition = grid.a[grid.ilo: grid.ihi + 1]
    grid.a[np.logical_and(grid.x >= 0.333 + 1/num_of_cells, grid.x <= 0.666 + 1/num_of_cells)] = 1.0
    initial_condition1 = grid.a[grid.ilo: grid.ihi + 1]
    advection = [initial_condition, initial_condition1]

    other_dt = False
    if other_dt:
        dt = 1 * h

    for n in range(int(1 / dt)):
        # dt Other than that of CFL...
        if other_dt:
            advection.append(advection[-2] - dt / h * (pos_shift_adv(advection[-1]) - neg_shift_adv(advection[-1])))
            continue

        advection.append(advection[-2] - CFL * (pos_shift_adv(advection[-1]) - neg_shift_adv(advection[-1])))

    plt.plot(x_range, initial_condition, color='b')
    plt.plot(x_range, advection[-1], color='orange')
    plt.show()

    return err(advection[-1], initial_condition, h)


nums = np.linspace(64, 564, 10).astype(int)
plt.plot(nums ** -0.5, [eps(x) for x in nums])
plt.xlabel("1 / Sqrt(N)")
plt.ylabel("err")
plt.show()
