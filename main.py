import numpy as np
from collections import deque

"""
Pure particle on a numberline dynamics

System parameters:
* m ∈ ℝ - mass of the particle
* y₀ ∈ ℝ - initial position of the particle  
* v₀ ∈ ℝ - initial velocity of the particle
* φ(y) - force-exerting external potential field, based purely on position
* v_max - maximum particle velocity in the system
* p_c - probability of crashing, need to multiply by |v|/v_max

State: x(t) = [y(t), v(t)]ᵀ ∈ ℝ²
Input: u(t) = f_i(t) ∈ [-1, 1] - applied force
Output: z(t) = [y_s(t), v_s(t)]ᵀ = [y(t), v(t)]ᵀ - sensor reading of position and velocity

State equation: x'(t) = x(t) + [v(t), (1/m)*f_net(t)]ᵀ
where f_net(t) = f_i(t) - d/dy(φ(y))
"""

# System parameters
m = 1.0  # mass of the particle
A = 1.0  # amplitude for potential field derivative


def phi(y):
    """Potential field function φ(y) = A*cos(y)"""
    return A * np.cos(y)


def d_phi_dy(y):
    """Derivative of potential field: d/dy(φ(y)) = -A*sin(y)"""
    return -A * np.sin(y)


# Simulation parameters
time_steps = 10
f_i = 0.5  # constant applied force

# Initial conditions
y_0 = 1.0  # initial position (can be changed)
v_0 = 1.0  # initial velocity (can be changed)
v_max = 2.0  # maximum velocity
y_max = 2.0  # maximum position

# Goal state
y_goal = 0.0  # goal position
v_goal = 0.0  # goal velocity

# Numerical precision (global)
PRECISION_DIGITS = 6  # round calculations to this many decimal places
EPS = 10 ** (-PRECISION_DIGITS)


def r(x: float) -> float:
    """Round helper respecting global precision."""
    return float(np.round(x, PRECISION_DIGITS))


# Global PRM graph
V = set()  # vertices (states)
E = set()  # edges (reachable connections)
G = {"V": V, "E": E}  # graph structure


class State:
    def __init__(self, y: float, v: float):
        self.y = y
        self.v = v

    def __str__(self):
        return f"State(y={self.y:.3g}, v={self.v:.3g})"


def update_state(state: State, f_i: float):
    """Update the state of the particle based on the state equations.

    State equation: x'(t) = x(t) + [v(t), (1/m)*f_net(t)]ᵀ
    where f_net(t) = f_i(t) - d/dy(φ(y))

    Args:
        state: Current state (y, v)
        f_i: Applied force (input) ∈ [-1, 1]
    """
    f_net = compute_net_force(f_i, state.y)
    state.y = update_position(state.y, state.v)
    state.v = update_velocity(state.v, f_net)


def update_position(y: float, v: float) -> float:
    return r(y + v)


def sensor_output(state: State) -> tuple[float, float]:
    """Sensor output: z(t) = [y_s(t), v_s(t)]ᵀ = [y(t), v(t)]ᵀ

    Full knowledge sensor with no noise - returns exact position and velocity.

    Args:
        state: Current state (y, v)

    Returns:
        Tuple of (sensor_position, sensor_velocity)
    """
    return (state.y, state.v)


def print_state(t, x: State):
    """Print state with sensor output"""
    y_sensor, v_sensor = sensor_output(x)
    print(
        t,
        f"State(y={x.y:.3g}, v={x.v:.3g}) | Sensor: y_s={y_sensor:.3g}, v_s={v_sensor:.3g}",
    )


def update_velocity(v: float, f_net: float) -> float:
    """Update velocity: v'(t) = v(t) + (1/m) * f_net(t)

    Args:
        v: Current velocity
        f_net: Net force

    Returns:
        New velocity
    """
    v_n = v + 1 / m * f_net
    return r(v_n)


def compute_net_force(f_i: float, y: float) -> float:
    """Compute net force ignoring potential field: f_net(t) = f_i(t)

    Args:
        f_i: Applied force (input) ∈ [-1, 1]
        y: Current position (unused)

    Returns:
        Net force acting on the particle
    """
    return f_i


def is_reachable(s_new: State, s_nearest: State) -> bool:
    """Check if s_new is reachable from s_nearest using 3-timestep approach.

    Ignoring potential field:
    - Apply f1 for first 2 timesteps to land exactly at target position
    - Apply f2 on the 3rd timestep to set target velocity (position does not move)

    Args:
        s_new: Target state
        s_nearest: Source state

    Returns:
        True if reachable (f1, f2 ∈ [-1, 1]), False otherwise
    """
    # Calculate required f1 and f2 to reach target in 3 timesteps

    y0 = s_nearest.y
    v0 = s_nearest.v
    y_target = s_new.y
    v_target = s_new.v

    # After two timesteps with f1:
    # y2 = y0 + 2*v0 + (f1/m)
    # v2 = v0 + 2*(f1/m)
    # Enforce y2 = y_target to solve for f1
    f1 = m * (y_target - y0 - 2 * v0)

    # Then choose f2 so that v3 = v_target, where v3 = v2 + f2/m
    v2 = v0 + 2 * (f1 / m)
    f2 = m * (v_target - v2)

    # Check if both forces are within valid range [-1, 1] with tolerance
    return (-1.0 - EPS) <= f1 <= (1.0 + EPS) and (-1.0 - EPS) <= f2 <= (1.0 + EPS)


def graph_build(itr):
    """Build PRM graph by sampling random states and checking reachability.

    For each new state s_new:
    - Add s_new to V
    - For every other state s_nearest in V:
      - Check if s_nearest is reachable from s_new via input force F ∈ [-1, 1]
      - If reachable, add edge {s_new, s_nearest} to E
    """
    for i in range(itr):
        # Pick state s_new by random
        # Random numbers rounded to 2 decimal places for precision
        y_new = np.random.uniform(-y_max, y_max)
        v_new = np.random.uniform(-v_max, v_max)
        s_new = State(y_new, v_new)

        # Add s_new to V
        V.add(s_new)

        # For every other state s_nearest in V
        for s_nearest in V:
            if s_nearest is s_new:  # Skip self
                continue

            # Check if s_nearest is reachable from s_new via application of input_force = F
            if is_reachable(
                s_nearest, s_new
            ):  # Note: checking if s_nearest is reachable FROM s_new
                E.add((s_new, s_nearest))


def extract_forces_from_path(path: list[State]) -> list[tuple[float, float]]:
    """Extract the sequence of forces needed to follow a path of states.

    Uses 3-timestep approach (ignoring potential):
    - f1 for 2 timesteps to reach target position
    - f2 on 3rd timestep to set target velocity

    Args:
        path: List of states from start to goal

    Returns:
        List of (f1, f2) tuples to apply between consecutive states
    """
    forces = []

    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]

        # Calculate required forces to go from current_state to next_state
        y0 = current_state.y
        v0 = current_state.v
        y_target = next_state.y
        v_target = next_state.v

        # f1 enforces y2 = y_target after two timesteps
        f1 = m * (y_target - y0 - 2 * v0)

        # f2 sets velocity on the third timestep
        v2 = v0 + 2 * (f1 / m)
        f2 = m * (v_target - v2)

        # Clamp to valid range [-1, 1] then round
        f1 = r(max(-1.0, min(1.0, f1)))
        f2 = r(max(-1.0, min(1.0, f2)))
        forces.append((f1, f2))

    return forces


def graph_search(s_i: State, s_f: State):
    """Breadth-First Search from s_i to s_f over the directed graph (V, E).

    Uses the global sets V and E where E contains tuples (u, v) representing
    a directed edge u -> v. Returns a list of States representing the path
    from s_i to s_f (inclusive) if found, otherwise returns None.
    """

    # Build adjacency list from the global edge set E
    adj = {}
    for u, v in E:
        adj.setdefault(u, []).append(v)

    # Trivial case: start is goal (within tolerance)
    if s_i is s_f or (abs(s_i.y - s_f.y) <= EPS and abs(s_i.v - s_f.v) <= EPS):
        return [s_i]

    # Standard BFS
    visited = set()
    parent = {}
    q = deque()

    if s_i not in V:
        # If the start isn't in the graph, there's no path under the current (V, E)
        return None

    visited.add(s_i)
    q.append(s_i)

    found = None
    while q:
        u = q.popleft()
        for nxt in adj.get(u, []):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = u
                # Check goal by identity first, then by state match (within tolerance)
                if nxt is s_f or (
                    abs(nxt.y - s_f.y) <= EPS and abs(nxt.v - s_f.v) <= EPS
                ):
                    found = nxt
                    q.clear()
                    break
                q.append(nxt)

    if found is None:
        return None

    # Reconstruct path from s_i to found
    path = [found]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def main():
    # Initialize start and goal states
    start_state = State(y_0, v_0)
    goal_state = State(y_goal, v_goal)

    # Initialize graph with start state (G = {V = {s_1}, E})
    V.add(start_state)
    V.add(goal_state)

    print("Pure Particle Dynamics - PRM Force Sequence Generation")
    print("=" * 60)
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")
    print(f"Mass: {m:.3g}")
    print("Potential field: ignored in this run")
    print("=" * 60)

    # PRM algorithm as per pseudocode
    itr = 1000
    max_iterations = 10  # Prevent infinite loop
    iteration = 0

    while iteration < max_iterations:
        print(f"\nPRM Iteration {iteration + 1}")

        # Build graph with itr samples
        graph_build(itr)
        print(f"Graph now has {len(V)} vertices and {len(E)} edges")

        # Search for path from start to goal
        path = graph_search(start_state, goal_state)

        if path is not None:
            print(f"Found path with {len(path)} states!")

            # Extract force sequence from path
            force_sequence = extract_forces_from_path(path)

            print(f"\nForce sequence ({len(force_sequence)} steps):")
            print("Step | f1    | f2    | Position | Velocity")
            print("-" * 50)

            # Simulate with the generated forces to verify the path
            current_state = State(start_state.y, start_state.v)
            for i, (f1, f2) in enumerate(force_sequence):
                print(
                    f"{i:4d} | {f1:5.3f} | {f2:5.3f} | {current_state.y:8.3f} | {current_state.v:8.3f}"
                )
                # Apply f1 for 2 timesteps (y and v update)
                update_state(current_state, f1)
                update_state(current_state, f1)
                # Apply f2 for 1 timestep (velocity-only update; position remains fixed)
                current_state.v = update_velocity(current_state.v, f2)

            print(f"\nFinal state: {current_state}")
            reached = (
                abs(current_state.y - goal_state.y) <= EPS
                and abs(current_state.v - goal_state.v) <= EPS
            )
            print(f"Goal reached: {reached}")

            # Output the force sequence for external use
            print(f"\nForce sequence (for external use): {force_sequence}")
            print("Each step: (f1_applied_2_timesteps, f2_applied_1_timestep)")
            break
        else:
            print("No path found, continuing to build graph...")
            iteration += 1

    if iteration >= max_iterations:
        print("Failed to find a path after maximum iterations!")


if __name__ == "__main__":
    main()
