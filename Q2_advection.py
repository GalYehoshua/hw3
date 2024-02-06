import numpy as np
from matplotlib import pyplot as plt

CFL: float = 0.5
N = 64  # number of cells
O: int = 2 * N // 3  # number of zeros in IC
x_range = np.linspace(0, 1, N)

# We use here U = 1 for the velocity of the advection
initial_condition = np.concatenate([np.zeros(O // 2), np.ones(N - O), np.zeros(O // 2)])
h = x_range[1] - x_range[0]
dt = CFL * h  # if we want a none unit U, need to change this
print(dt, int(1 / dt))
advection = [initial_condition]


def neg_shift_adv(adv):
    return np.concatenate([[adv[-2]], adv[:-1]])


assert (neg_shift_adv(np.array([0, 1, 2, 1, 0])) == np.array([1, 0, 1, 2, 1])).all()

for n in range(int(1 / dt)):
    advection.append(advection[-1] - CFL * (advection[-1] - neg_shift_adv(advection[-1])))

plt.plot(x_range, initial_condition, color='b')
plt.plot(x_range, advection[-100], color='orange')
plt.show()
