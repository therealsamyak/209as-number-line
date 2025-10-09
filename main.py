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
y_max = 100
p_c = 0.1  # crash probability factor
p_w = 0.1

A = 1 # Amplitude of f_phi

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
    f_net = net_force(f_i, state.y)
    state.y = update_position(state.y, state.v)
    state.v = update_velocity(state.v, f_net)

def update_position(y: float, v: float) -> float:
    return y + v

def noisy_position(y: float) -> float:
    mu = 0
    sigma = 0.5 * y
    return y + random.gauss(mu, sigma)

def print_state(t, x: State):
    y_noisy = noisy_position(x.y)
    print(t, f"State(y={x.y:.3g}, v={x.v:.3g}, y_noisy={y_noisy:.3g})")

def wobble_prob(v):
    return (v/v_max) * (p_w/2)

def speed_wobble(v):
    p = wobble_prob(v) # Probability of adding speed due to wobble
    r = random.uniform(0,1)
    if r <= p:
        return 1
    elif r > 1-(p):
        return -1
    else:
        return 0

def crash_wobble(v):
    return p_c * abs(v)/v_max

def update_velocity(v: float, f_net: float) -> float:
    p = crash_wobble(v)
    r = random.uniform()
    if r <= p:
        return 0
    else:
        return v + f_net/m + speed_wobble(v)

def f_phi(y):
    return int(A*np.sin(2*np.pi*y/y_max))

def net_force(f_i: float, y: float) -> float:
    return f_i + f_phi(y)

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
        print_state(t, state)
        update_state(state, f_i)

    print(f"t={time_steps}: {state}")


if __name__ == "__main__":
    main()
