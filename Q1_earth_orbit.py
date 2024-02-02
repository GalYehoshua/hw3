import numpy as np


def initial_conditions(a, e):
    # x0 = 0, y0 = a * (1 - e), vx (see in Ex3), vy = 0
    gm = 4 * np.pi
    return 0, a * (1 - e), -(gm / a * (1 + e) / (1 - e)), 0


# not depend on time explicitly.

def f(location_velocity_vector):
    assert len(location_velocity_vector) == 4, f"bad vector, {location_velocity_vector}"
    x, y, vx, vy = location_velocity_vector
    gm = 4 * np.pi
    one_over_r3 = (x ** 2 + y ** 2) ** -1.5
    return np.array(
        [vx,
         vy,
         - gm * x * one_over_r3,
         - gm * y * one_over_r3
         ]
    )


def compute_single_step_euler(x, y, vx, vy, dt):
    gm = 4 * np.pi
    one_over_r3 = (x ** 2 + y ** 2) ** -1.5
    return (x + dt * vx,
            y + dt * vy,
            vx - dt * gm * x * one_over_r3,
            vy - dt * gm * y * one_over_r3
            )


def compute_single_step_rk4(x, y, vx, vy, dt):
    location_velocity_vector = [x, y, vx, vy]
    k1 = dt * f(location_velocity_vector)
    k2 = dt * f(np.array(location_velocity_vector) + 0.5 * k1)
    k3 = dt * f(np.array(location_velocity_vector) + 0.5 * k2)
    k4 = dt * f(np.array(location_velocity_vector) + k3)
    return np.array(location_velocity_vector) + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def earth_orbit_euler():
    steps = 1000
    dt = 1.0 / steps
    a, e = 1, 0.1
    x, y, vx, vy = initial_conditions(a, e)
    xt, yt, vxt, vyt = [], [], [], []
    for i in range(steps):
        xt.append(x)
        yt.append(y)
        vxt.append(vx)
        vyt.append(vyt)
        x, y, vx, vy = compute_single_step_euler(x, y, vx, vy, dt)

    return x, y, vx, vy


def earth_orbit_rk4():
    steps = 1000
    dt = 1.0 / steps
    a, e = 1, 0.1
    x, y, vx, vy = initial_conditions(a, e)
    xt, yt, vxt, vyt = [], [], [], []
    for i in range(steps):
        xt.append(x)
        yt.append(y)
        vxt.append(vx)
        vyt.append(vyt)
        x, y, vx, vy = compute_single_step_rk4(x, y, vx, vy, dt)

    return x, y, vx, vy


if __name__ == "__main__":
    print(earth_orbit_euler())
    print(earth_orbit_rk4())
