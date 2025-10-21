import random
import numpy as np
from collections import deque
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
y_max = 10000.0  # maximum position
#p_c = 0.1  # base crash probability

G = set()
V = set()
E = set()

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
    #f_net = compute_net_force(f_i, state.y)
    f_net = f_i
    state.y = update_position(state.y, state.v)
    state.v = update_velocity(state.v, f_net)

    


def update_position(y: float, v: float) -> float:
    return y + v

def print_state(t, x: State):
    #y_noisy = noisy_position(x.y)
    print(t, f"State(y={x.y:.3g}, v={x.v:.3g})")


def update_velocity(v: float, f_net: float) -> float:
    """Update velocity: v'(t) = v(t) + (1/m) * f_net(t)

    For pure particle without potential field: f_net(t) = f_i(t)
    """
    # crash_prob = p_c * abs(v)/v_max
    # normal_dist = np.random.uniform()
    # if normal_dist <= crash_prob:
    #     v_n = v + 1/m * f_net + random.gauss(0, 0.1 * v)
    # else :
    #     v_n = 0
    v_n = v + 1/m * f_net
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

def is_reachable(s_new: State, s_nearest: State) -> bool:
    delta_v = s_new.v - s_nearest.v
    delta_y = s_new.y - s_nearest.y
    if abs(delta_v) <= 0.01 and abs(delta_y) <= 0.01:
        return True
    return False

def graph_build(itr):
    for i in range(itr):
        y_new = np.random.uniform(-y_max, y_max)
        v_new = np.random.uniform(-v_max, v_max)
        s_new = State(y_new, v_new)
        V.add(s_new)
        
        for s_nearest in V:
            if is_reachable(s_new, s_nearest):
                E.add((s_new, s_nearest))

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

    # Trivial case: start is goal
    if s_i is s_f or (abs(s_i.y - s_f.y) <= 1e-12 and abs(s_i.v - s_f.v) <= 1e-12):
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
                # Check goal by identity first, then by state match
                if nxt is s_f or (abs(nxt.y - s_f.y) <= 1e-12 and abs(nxt.v - s_f.v) <= 1e-12):
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

    itr = 1000
    while 


if __name__ == "__main__":
    main()
