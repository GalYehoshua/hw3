import numpy as np
from matplotlib import pyplot as plt

CFL: float = 0.5
N = 64  # number of cells
O: int = 2 * N // 3  # number of zeros in IC


def neg_shift_adv(adv):
    return np.concatenate([[adv[-2]], adv[:-1]])


assert (neg_shift_adv(np.array([0, 1, 2, 1, 0])) == np.array([1, 0, 1, 2, 1])).all()


def err(adv_f, adv_i, dx):
    assert len(adv_f) == len(adv_i), f"Something wrong with lengths {adv_f}, {adv_i}, {len(adv_f) - len(adv_i)}"
    return dx * sum([(adv_f[i] - adv_i[i]) ** 2 for i in range(len(adv_i))])


def eps(num_of_cells):
    x_range = np.linspace(0, 1, num_of_cells)
    num_of_zeros: int = 2 * num_of_cells // 3  # number of zeros in IC
    # We use here U = 1 for the velocity of the advection
    initial_condition = np.concatenate(
        [np.zeros(num_of_zeros // 2), np.ones(num_of_cells - num_of_zeros + num_of_zeros % 2),
         np.zeros(num_of_zeros // 2)])
    h = x_range[1] - x_range[0]
    dt = CFL * h  # if we want a none unit U, need to change this
    # print(dt, int(1 / dt))
    advection = [initial_condition]

    other_dt = False
    if other_dt:
        dt = 1.01 * h

    for n in range(int(1 / dt)):
        # dt Other than that of CFL...
        if other_dt:
            advection.append(advection[-1] - dt / h * (advection[-1] - neg_shift_adv(advection[-1])))
            continue

        advection.append(advection[-1] - CFL * (advection[-1] - neg_shift_adv(advection[-1])))

    plt.plot(x_range, initial_condition, color='b')
    plt.plot(x_range, advection[-1], color='orange')
    plt.show()

    return err(advection[-1], initial_condition, h)


nums = np.linspace(64, 564, 2).astype(int)
plt.plot(nums ** -0.5, [eps(x) for x in nums])
plt.xlabel("1 / Sqrt(N)")
plt.ylabel("err")
plt.show()
