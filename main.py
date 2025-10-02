import random
import numpy as np
# System parameters (global)
m = 1.0  # mass of the particle
phi = lambda y: 0.5 * y**2  # potential field function (example: quadratic potential)

# Simulation parameters (global)
time_steps = 10
f_i = 0.5  # constant applied force

# Initial conditions (global)
y_0 = 0.0  # initial position
v_0 = 0.0  # initial velocity
v_max = 10.0  # maximum velocity
p_c = 0.1  # crash probability factor


class State:
    def __init__(self, y: float, v: float):
        self.y = y
        self.v = v

    def __str__(self):
        return f"State(y={self.y:.3g}, v={self.v:.3g})"


def update_state(state: State, f_i: float):
    """Update the state of the particle based on the state equations.

    Args:
        state: Current state (y, v)
        f_i: Applied force (input)
    """
    f_net = compute_net_force(f_i, state.y)
    state.y = update_position(state.y, state.v)
    state.v = update_velocity(state.v, f_net)


def update_position(y: float, v: float) -> float:
    mu = 0
    sigma = 0.5 * y
    return y + v + random.gauss(mu, sigma)


def update_velocity(v: float, f_net: float) -> float:
    """Update velocity: v'(t) = v(t) + (1/m) * f_net(t)

    For pure particle without potential field: f_net(t) = f_i(t)
    """
    crash_prob = p_c * abs(v)/v_max
    normal_dist = np.random.uniform()
    if normal_dist <= crash_prob:
        v_n = v + 1/m * f_net + random.gauss(0, 0.1 * v)
    else :
        v_n = 0
    return v_n


def compute_net_force(f_i: float, y: float) -> float:
    """Compute net force: f_net(t) = f_i(t) - d/dy(phi(y))

    Args:
        f_i: Applied force (input)
        y: Current position

    Returns:
        Net force acting on the particle
    """
    # Numerical derivative of phi with respect to y
    epsilon = 1e-8
    d_phi_dy = (phi(y + epsilon) - phi(y - epsilon)) / (2 * epsilon)
    return f_i - d_phi_dy


def main():
    # Initialize state
    state = State(y_0, v_0)

    print("Pure Particle Dynamics Simulation")
    print("=" * 50)
    print(f"Initial state: {state}")
    print(f"Mass: {m:.3g}, Applied force: {f_i:.3g}")
    print(f"Potential field: phi(y) = {phi(y_0):.3g} * y^2")
    print("=" * 50)

    # Simulation loop
    for t in range(time_steps):
        print(f"t={t}: {state}")
        update_state(state, f_i)

    print(f"t={time_steps}: {state}")


if __name__ == "__main__":
    main()
