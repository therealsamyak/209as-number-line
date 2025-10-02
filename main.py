class State:
    def __init__(self, y: float, v: float):
        self.y = y
        self.v = v

    def __str__(self):
        return f"State(y={self.y}, v={self.v})"


def update_state(state: State, f_i: float, m: float):
    """Update the state of the particle based on the state equations.

    Args:
        state: Current state (y, v)
        f_i: Applied force (input)
        m: Mass of the particle
    """
    state.y = update_position(state.y, state.v)
    state.v = update_velocity(state.v, f_i, m)


def update_position(y: float, v: float) -> float:
    """Update position: y'(t) = y(t) + v(t)"""
    return y + v


def update_velocity(v: float, f_i: float, m: float) -> float:
    """Update velocity: v'(t) = v(t) + (1/m) * f_net(t)

    For pure particle without potential field: f_net(t) = f_i(t)
    """
    return v + (f_i / m)


def main():
    # System parameters
    m = 1.0  # mass of the particle
    y_0 = 0.0  # initial position
    v_0 = 0.0  # initial velocity

    # Initialize state
    state = State(y_0, v_0)

    # Simulation parameters
    time_steps = 10
    f_i = 0.5  # constant applied force for demonstration

    print("Pure Particle Dynamics Simulation")
    print("=" * 50)
    print(f"Initial state: {state}")
    print(f"Mass: {m}, Applied force: {f_i}")
    print("=" * 50)

    # Simulation loop
    for t in range(time_steps):
        print(f"t={t}: {state}")
        update_state(state, f_i, m)

    print(f"t={time_steps}: {state}")


if __name__ == "__main__":
    main()
