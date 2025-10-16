import numpy as np
from scipy.stats import norm
from collections import defaultdict
import matplotlib.pyplot as plt


class ContinuousParticleSystem:
    """The TRUE system with continuous state space"""

    def __init__(
        self, m, y_max, v_max, A, p_c, goal_y=0.0, goal_v=0.0, goal_tolerance=0.1
    ):
        self.m = m
        self.y_max = y_max
        self.v_max = v_max
        self.A = A
        self.p_c = p_c
        self.goal_y = goal_y
        self.goal_v = goal_v
        self.goal_tolerance = goal_tolerance

    def phi(self, y):
        """Potential field force"""
        return self.A * np.sin(2 * np.pi * y / self.y_max)

    def step(self, state, action):
        """
        Take one step in continuous system (Week 1 dynamics)

        Args:
            state: (y, v) continuous floats
            action: float in [-1, 1] or discrete {-1, 0, 1}

        Returns:
            next_state: (y', v') continuous
            reward: float
            done: bool
        """
        y, v = state

        # Net force: f_net = f_i + f_φ(y)
        f_net = action + self.phi(y)

        # Check crash probability: p_c * |v| / v_max
        p_crash = self.p_c * abs(v) / self.v_max
        if np.random.rand() < p_crash:
            # Crash: velocity goes to 0
            v_next = 0.0
            y_next = y  # Position unchanged on crash
        else:
            # Week 1 speed wobble: Gaussian noise N(0, (0.1*v)^2)
            speed_wobble = np.random.normal(0, 0.1 * abs(v))

            # Update velocity: v[t+1] = v[t] + (1/m)*f_net[t] + wobble
            v_next = v + (f_net / self.m) + speed_wobble
            v_next = np.clip(v_next, -self.v_max, self.v_max)

            # Update position: y[t+1] = y[t] + v[t]
            y_next = y + v
            y_next = np.clip(y_next, -self.y_max, self.y_max)

        # Reward: 1 if near goal, 0 otherwise
        reward = (
            1.0
            if (
                abs(y_next - self.goal_y) < self.goal_tolerance
                and abs(v_next - self.goal_v) < self.goal_tolerance
            )
            else 0.0
        )
        done = reward == 1.0

        return (y_next, v_next), reward, done

    def reset(self, y0=None, v0=None):
        """Reset to initial state"""
        if y0 is None:
            y0 = np.random.uniform(-self.y_max * 0.8, self.y_max * 0.8)
        if v0 is None:
            v0 = np.random.uniform(-self.v_max * 0.5, self.v_max * 0.5)
        return (y0, v0)


class GridParticleSystem:
    """Discretized approximation with ANALYTICAL transition computation"""

    def __init__(self, continuous_system, delta_y, delta_v, goal_y=0.0, goal_v=0.0):
        """
        Args:
            continuous_system: ContinuousParticleSystem instance
            delta_y: position grid resolution
            delta_v: velocity grid resolution
            goal_y: target position (default: 0.0)
            goal_v: target velocity (default: 0.0)
        """
        self.continuous = continuous_system
        self.delta_y = delta_y
        self.delta_v = delta_v
        self.goal_y = goal_y
        self.goal_v = goal_v

        # Create grid
        self.y_grid = np.arange(
            -continuous_system.y_max, continuous_system.y_max + delta_y, delta_y
        )
        self.v_grid = np.arange(
            -continuous_system.v_max, continuous_system.v_max + delta_v, delta_v
        )

        self.n_y = len(self.y_grid)
        self.n_v = len(self.v_grid)

        print(f"Grid size: {self.n_y} x {self.n_v} = {self.n_y * self.n_v} states")

        # Precompute transition probabilities ANALYTICALLY
        self.transitions = self._compute_transitions_analytical()

    def discretize(self, y, v):
        """Map continuous state to discrete grid indices"""
        i = int(np.clip((y + self.continuous.y_max) / self.delta_y, 0, self.n_y - 1))
        j = int(np.clip((v + self.continuous.v_max) / self.delta_v, 0, self.n_v - 1))
        return i, j

    def continuous_state(self, i, j):
        """Get continuous state from grid indices (cell center)"""
        y = self.y_grid[i]
        v = self.v_grid[j]
        return y, v

    def _compute_transitions_analytical(self):
        """
        Compute transition probabilities ANALYTICALLY
        P(s' | s, a) using known probability distributions

        Returns:
            dict: transitions[(i, j, a)] = {(i', j'): probability}
        """
        print("Computing transitions analytically...")
        transitions = {}
        actions = [-1, 0, 1]

        total = self.n_y * self.n_v * len(actions)
        count = 0

        for i in range(self.n_y):
            for j in range(self.n_v):
                y, v = self.continuous_state(i, j)

                for a in actions:
                    trans_probs = self._compute_single_transition(i, j, y, v, a)
                    transitions[(i, j, a)] = trans_probs

                    count += 1
                    if count % 500 == 0:
                        print(f"Progress: {count}/{total} ({100 * count / total:.1f}%)")

        print("Transition computation complete!")
        return transitions

    def _compute_single_transition(self, i, j, y, v, a):
        """
        Analytically compute P(s' | s=(y,v), a) for Week 1 dynamics

        Week 1 Dynamics:
            v' = v + (a + φ(y))/m + N(0, (0.1*v)^2)  [if no crash]
            v' = 0                                     [if crash]
            y' = y + v

        Where:
            Speed wobble: Gaussian N(0, (0.1*v)^2)
            Crash happens with probability p_c * |v| / v_max
        """
        transitions = defaultdict(float)

        # 1. Compute deterministic components (no dt, discrete time)
        f_phi = self.continuous.phi(y)
        f_net = a + f_phi
        v_deterministic = v + (f_net / self.continuous.m)
        y_next_deterministic = y + v  # Position updates with current velocity

        # Clip y to bounds
        y_next_deterministic = np.clip(
            y_next_deterministic, -self.continuous.y_max, self.continuous.y_max
        )

        # 2. Crash probability
        p_crash = self.continuous.p_c * abs(v) / self.continuous.v_max
        p_crash = min(p_crash, 1.0)  # Cap at 1
        p_no_crash = 1 - p_crash

        # 3. Gaussian speed wobble: N(0, (0.1*v)^2)
        wobble_std = 0.1 * abs(v)

        # 4. Handle crash case
        if p_crash > 0:
            # After crash: v=0, y unchanged
            i_crash, j_crash = self.discretize(y, 0.0)
            transitions[(i_crash, j_crash)] += p_crash

        # 5. Handle no-crash case
        if p_no_crash > 0:
            # Velocity mean after deterministic update
            v_mean = v_deterministic

            # Integrate Gaussian noise over each velocity grid cell
            for j_next in range(self.n_v):
                v_cell_low = self.v_grid[j_next]
                v_cell_high = v_cell_low + self.delta_v

                # Handle boundary cells specially to capture clipped probability mass
                # For boundary cells, integrate from cell edge to infinity (or -infinity)
                if j_next == self.n_v - 1:
                    # Highest velocity cell: capture all v >= v_cell_low
                    # (after clipping, any v > v_max ends up in this cell)
                    v_cell_high_clipped = float("inf")
                    v_cell_low_clipped = v_cell_low
                elif j_next == 0:
                    # Lowest velocity cell: capture all v <= v_cell_high
                    # (after clipping, any v < -v_max ends up in this cell)
                    v_cell_low_clipped = float("-inf")
                    v_cell_high_clipped = v_cell_high
                else:
                    # Interior cells: normal clipping
                    v_cell_low_clipped = max(v_cell_low, -self.continuous.v_max)
                    v_cell_high_clipped = min(v_cell_high, self.continuous.v_max)

                if wobble_std < 1e-6:
                    # No noise case (when v ≈ 0)
                    if v_cell_low_clipped <= v_mean < v_cell_high_clipped:
                        p_v_in_cell = 1.0
                    else:
                        p_v_in_cell = 0.0
                else:
                    # Probability that v' lands in this cell under Gaussian wobble
                    p_v_in_cell = norm.cdf(
                        v_cell_high_clipped, v_mean, wobble_std
                    ) - norm.cdf(v_cell_low_clipped, v_mean, wobble_std)

                if p_v_in_cell > 1e-8:  # Ignore negligible probabilities
                    # Position transition (deterministic given current v)
                    i_next = self.discretize(y_next_deterministic, 0)[0]

                    # Add to transition probability
                    key = (i_next, j_next)
                    transitions[key] += p_no_crash * p_v_in_cell

        # Normalize (should sum to ~1, but numerical errors possible)
        total_prob = sum(transitions.values())
        if total_prob > 0:
            transitions = {k: v / total_prob for k, v in transitions.items()}

        return dict(transitions)

    def reward(self, i, j):
        """Reward function on grid"""
        y, v = self.continuous_state(i, j)
        # Goal: reach target position and velocity (within one grid cell)
        if (
            abs(y - self.goal_y) <= self.delta_y
            and abs(v - self.goal_v) <= self.delta_v
        ):
            return 1.0
        return 0.0


class ValueIteration:
    """Solve discretized MDP using value iteration"""

    def __init__(self, grid_system, gamma=0.95):
        self.grid = grid_system
        self.gamma = gamma
        self.actions = [-1, 0, 1]

        # Initialize value function and policy
        self.V = np.zeros((grid_system.n_y, grid_system.n_v))
        self.policy = np.zeros((grid_system.n_y, grid_system.n_v), dtype=int)

    def solve(self, max_iterations=1000, threshold=1e-4):
        """Run value iteration until convergence"""
        print(f"Running value iteration (gamma={self.gamma})...")

        for iteration in range(max_iterations):
            V_old = self.V.copy()
            delta = 0

            # Update each state
            for i in range(self.grid.n_y):
                for j in range(self.grid.n_v):
                    # Compute Q-values for all actions
                    q_values = []
                    for a_idx, a in enumerate(self.actions):
                        q = self._compute_q_value(i, j, a)
                        q_values.append(q)

                    # Bellman update
                    self.V[i, j] = max(q_values)
                    self.policy[i, j] = np.argmax(q_values)

                    # Track convergence
                    delta = max(delta, abs(self.V[i, j] - V_old[i, j]))

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: delta = {delta:.6f}")

            if delta < threshold:
                print(f"Converged after {iteration + 1} iterations!")
                break

        return self.V, self.policy

    def _compute_q_value(self, i, j, a):
        """
        Compute Q(s, a) = Σ P(s'|s,a) * [R(s,a,s') + γ*V(s')]
        """
        transitions = self.grid.transitions.get((i, j, a), {})

        q_value = 0.0
        for (i_next, j_next), prob in transitions.items():
            reward = self.grid.reward(i_next, j_next)
            q_value += prob * (reward + self.gamma * self.V[i_next, j_next])

        return q_value

    def get_action(self, y, v):
        """Get action for continuous state using learned policy"""
        i, j = self.grid.discretize(y, v)
        action_idx = self.policy[i, j]
        return self.actions[action_idx]


class PolicyEvaluator:
    """Test learned policy on continuous system"""

    def __init__(self, continuous_system, value_iteration):
        self.continuous = continuous_system
        self.vi = value_iteration

    def run_episode(self, y0=None, v0=None, max_steps=500, render=False):
        """
        Run one episode using learned policy

        Returns:
            total_reward: float
            steps: int
            trajectory: list of (y, v) tuples
            success: bool
        """
        state = self.continuous.reset(y0, v0)
        trajectory = [state]
        total_reward = 0

        for step in range(max_steps):
            # Get action from policy (trained on grid)
            action = self.vi.get_action(*state)

            # Execute in continuous system
            next_state, reward, done = self.continuous.step(state, action)

            total_reward += reward
            trajectory.append(next_state)

            if render and step % 20 == 0:
                y, v = state
                print(f"  Step {step:3d}: y={y:6.2f}, v={v:6.2f}, a={action:2d}")

            if done:
                if render:
                    print(f"  ✓ Goal reached in {step + 1} steps!")
                return total_reward, step + 1, trajectory, True

            state = next_state

        if render:
            print(f"  ✗ Episode ended after {max_steps} steps (no goal)")
        return total_reward, max_steps, trajectory, False

    def evaluate(self, n_episodes=100, verbose=True):
        """Run multiple episodes and compute statistics"""
        results = {"successes": 0, "steps_list": [], "rewards_list": []}

        if verbose:
            print(f"\nEvaluating policy over {n_episodes} episodes...")

        for ep in range(n_episodes):
            reward, steps, _, success = self.run_episode()
            results["rewards_list"].append(reward)
            results["steps_list"].append(steps)
            if success:
                results["successes"] += 1

            if verbose and (ep + 1) % 20 == 0:
                print(f"  Completed {ep + 1}/{n_episodes} episodes...")

        # Compute statistics
        results["success_rate"] = results["successes"] / n_episodes
        results["avg_reward"] = np.mean(results["rewards_list"])
        results["avg_steps"] = np.mean(results["steps_list"])
        results["avg_steps_success"] = (
            np.mean(
                [
                    s
                    for s, r in zip(results["steps_list"], results["rewards_list"])
                    if r > 0
                ]
            )
            if results["successes"] > 0
            else float("inf")
        )

        return results


def visualize_policy(vi, grid_sys, save_path="policy_visualization.png"):
    """Visualize value function and policy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Value function heatmap
    im1 = ax1.imshow(
        vi.V.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            -grid_sys.continuous.y_max,
            grid_sys.continuous.y_max,
            -grid_sys.continuous.v_max,
            grid_sys.continuous.v_max,
        ],
    )
    ax1.set_xlabel("Position y", fontsize=12)
    ax1.set_ylabel("Velocity v", fontsize=12)
    ax1.set_title("Value Function V*(y, v)", fontsize=14, fontweight="bold")
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.plot(0, 0, "r*", markersize=15, label="Goal")
    ax1.legend()
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Value", fontsize=11)

    # Policy visualization with arrows
    skip = max(1, grid_sys.n_y // 20)  # Subsample for clarity
    Y_indices = np.arange(0, grid_sys.n_y, skip)
    V_indices = np.arange(0, grid_sys.n_v, skip)

    Y_pos = []
    V_pos = []
    U_force = []

    for i in Y_indices:
        for j in V_indices:
            y, v = grid_sys.continuous_state(i, j)
            action_idx = vi.policy[i, j]
            action = vi.actions[action_idx]

            Y_pos.append(y)
            V_pos.append(v)
            U_force.append(action)

    # Color code by action
    colors = np.array(U_force)
    scatter = ax2.scatter(
        Y_pos, V_pos, c=colors, cmap="RdYlGn", s=50, vmin=-1, vmax=1, alpha=0.7
    )

    ax2.set_xlabel("Position y", fontsize=12)
    ax2.set_ylabel("Velocity v", fontsize=12)
    ax2.set_title("Policy π*(y, v)", fontsize=14, fontweight="bold")
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.plot(0, 0, "r*", markersize=15, label="Goal")
    ax2.set_xlim([-grid_sys.continuous.y_max, grid_sys.continuous.y_max])
    ax2.set_ylim([-grid_sys.continuous.v_max, grid_sys.continuous.v_max])
    cbar2 = plt.colorbar(scatter, ax=ax2)
    cbar2.set_label("Action (force)", fontsize=11)
    cbar2.set_ticks([-1, 0, 1])
    ax2.legend()

    # Add parameter information as text box
    param_text = (
        f"System Parameters:\n"
        f"m={grid_sys.continuous.m}, A={grid_sys.continuous.A}, p_c={grid_sys.continuous.p_c}\n"
        f"y_max={grid_sys.continuous.y_max}, v_max={grid_sys.continuous.v_max}\n"
        f"Grid: Δy={grid_sys.delta_y}, Δv={grid_sys.delta_v}\n"
        f"Goal: ({grid_sys.goal_y}, {grid_sys.goal_v}), γ={vi.gamma}"
    )
    fig.text(
        0.5,
        0.02,
        param_text,
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for parameter text
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Policy visualization saved to {save_path}")
    plt.show()


def plot_trajectory(
    trajectory, continuous_sys=None, grid_sys=None, save_path="trajectory.png"
):
    """Plot state trajectory over time"""
    y_vals = [s[0] for s in trajectory]
    v_vals = [s[1] for s in trajectory]
    times = list(range(len(trajectory)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Position over time
    axes[0].plot(times, y_vals, "b-", linewidth=2)
    axes[0].axhline(0, color="r", linestyle="--", label="Goal y=0")
    axes[0].set_ylabel("Position y", fontsize=12)
    axes[0].set_title("Trajectory", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Velocity over time
    axes[1].plot(times, v_vals, "g-", linewidth=2)
    axes[1].axhline(0, color="r", linestyle="--", label="Goal v=0")
    axes[1].set_ylabel("Velocity v", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Phase space (y vs v)
    axes[2].plot(y_vals, v_vals, "purple", linewidth=2, alpha=0.6)
    axes[2].plot(y_vals[0], v_vals[0], "go", markersize=10, label="Start")
    axes[2].plot(y_vals[-1], v_vals[-1], "bs", markersize=10, label="End")
    axes[2].plot(0, 0, "r*", markersize=15, label="Goal")
    axes[2].set_xlabel("Position y", fontsize=12)
    axes[2].set_ylabel("Velocity v", fontsize=12)
    axes[2].set_title("Phase Space", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axis("equal")

    # Add parameter information if provided
    if continuous_sys is not None:
        param_text = (
            f"System Parameters: m={continuous_sys.m}, A={continuous_sys.A}, "
            f"p_c={continuous_sys.p_c}, y_max={continuous_sys.y_max}, v_max={continuous_sys.v_max}"
        )
        if grid_sys is not None:
            param_text += f"\nGrid: Δy={grid_sys.delta_y}, Δv={grid_sys.delta_v}"
        fig.text(
            0.5,
            0.02,
            param_text,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved to {save_path}")
    plt.show()


def plot_convergence(results_by_resolution, continuous_sys=None, gamma=None):
    """Plot how performance varies with grid resolution"""
    resolutions = sorted(results_by_resolution.keys())
    success_rates = [results_by_resolution[r]["success_rate"] for r in resolutions]
    avg_steps = [results_by_resolution[r]["avg_steps_success"] for r in resolutions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(resolutions, success_rates, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Grid Resolution Δ", fontsize=12)
    ax1.set_ylabel("Success Rate", fontsize=12)
    ax1.set_title("Success Rate vs Resolution", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2.plot(resolutions, avg_steps, "s-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Grid Resolution Δ", fontsize=12)
    ax2.set_ylabel("Avg Steps to Goal", fontsize=12)
    ax2.set_title("Efficiency vs Resolution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add parameter information if provided
    if continuous_sys is not None:
        param_text = (
            f"System Parameters: m={continuous_sys.m}, A={continuous_sys.A}, "
            f"p_c={continuous_sys.p_c}, y_max={continuous_sys.y_max}, v_max={continuous_sys.v_max}"
        )
        if gamma is not None:
            param_text += f", γ={gamma}"
        fig.text(
            0.5,
            0.02,
            param_text,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        plt.tight_layout()

    plt.savefig("resolution_analysis.png", dpi=150, bbox_inches="tight")
    print("Resolution analysis saved to resolution_analysis.png")
    plt.show()


def main():
    """Complete pipeline: create systems, solve, evaluate, visualize"""

    print("=" * 60)
    print("GRID DECOMPOSITION MDP SOLVER")
    print("=" * 60)

    # ============================================================
    # 1. SYSTEM PARAMETERS (Week 1)
    # ============================================================
    # All parameters are easily configurable for arbitrary environments/tasks:
    # - Change m, y_max, v_max for different dynamics
    # - Change A to adjust potential field amplitude (A=0 disables field)
    # - Change p_c to adjust crash probability
    # - Change goal_y, goal_v to test different target states
    params = {
        "m": 1.0,
        "y_max": 10.0,
        "v_max": 5.0,
        "A": 2.0,
        "p_c": 0.1,
        "goal_y": 0.0,  # Target position (try 5.0, -3.0, etc.)
        "goal_v": 0.0,  # Target velocity (try 2.0, -1.0, etc.)
        "goal_tolerance": 0.1,  # Goal tolerance for continuous system
    }

    print("\nSystem Parameters (Week 1 Dynamics):")
    for key, val in params.items():
        print(f"  {key:16s} = {val}")

    # ============================================================
    # 2. CREATE CONTINUOUS (REAL) SYSTEM
    # ============================================================
    print("\n" + "=" * 60)
    print("Creating continuous system...")
    continuous_sys = ContinuousParticleSystem(**params)
    print("✓ Continuous system created")

    # ============================================================
    # 3. CREATE DISCRETIZED GRID SYSTEM
    # ============================================================
    print("\n" + "=" * 60)
    print("Creating grid approximation...")

    delta_y = 0.25  # Position resolution (smaller = finer grid, better accuracy)
    delta_v = 0.25  # Velocity resolution (smaller = finer grid, better accuracy)

    print(f"  Resolution: Δy={delta_y}, Δv={delta_v}")
    print(f"  Goal: y={params['goal_y']}, v={params['goal_v']}")

    grid_sys = GridParticleSystem(
        continuous_sys,
        delta_y=delta_y,
        delta_v=delta_v,
        goal_y=params["goal_y"],
        goal_v=params["goal_v"],
    )
    print("✓ Grid system created with analytical transitions")

    # ============================================================
    # 4. SOLVE WITH VALUE ITERATION
    # ============================================================
    print("\n" + "=" * 60)
    print("Solving MDP with value iteration...")

    vi = ValueIteration(grid_sys, gamma=0.95)
    V, policy = vi.solve(max_iterations=200, threshold=1e-4)

    print("✓ Policy computed")
    print(f"  Max value: {np.max(V):.4f}")
    print(f"  Min value: {np.min(V):.4f}")

    # ============================================================
    # 5. TEST POLICY ON CONTINUOUS SYSTEM
    # ============================================================
    print("\n" + "=" * 60)
    print("Testing policy on continuous system...")

    evaluator = PolicyEvaluator(continuous_sys, vi)

    # Single episode with visualization
    print("\n--- Single Episode (verbose) ---")
    reward, steps, trajectory, success = evaluator.run_episode(
        y0=-8.0, v0=0.0, render=True
    )

    # Multiple episodes for statistics
    print("\n--- Evaluation over 100 episodes ---")
    results = evaluator.evaluate(n_episodes=100, verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate:        {results['success_rate'] * 100:.1f}%")
    print(f"Avg Reward:          {results['avg_reward']:.3f}")
    print(f"Avg Steps (all):     {results['avg_steps']:.1f}")
    print(f"Avg Steps (success): {results['avg_steps_success']:.1f}")

    # ============================================================
    # 6. VISUALIZATIONS
    # ============================================================
    print("\n" + "=" * 60)
    print("Creating visualizations...")

    visualize_policy(vi, grid_sys)
    plot_trajectory(trajectory, continuous_sys, grid_sys)

    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)

    return continuous_sys, grid_sys, vi, results


def resolution_experiment():
    """Experiment with different grid resolutions"""

    print("\n" + "=" * 60)
    print("RESOLUTION EXPERIMENT")
    print("=" * 60)

    params = {
        "m": 1.0,
        "y_max": 10.0,
        "v_max": 5.0,
        "A": 2.0,
        "p_c": 0.1,
        "goal_y": 0.0,
        "goal_v": 0.0,
        "goal_tolerance": 0.1,
    }

    continuous_sys = ContinuousParticleSystem(**params)

    resolutions = [1.0, 0.5, 0.25, 0.2, 0.15]
    results_by_res = {}

    for delta in resolutions:
        print(f"\n{'=' * 60}")
        print(f"Testing resolution Δ = {delta}")
        print(f"{'=' * 60}")

        grid_sys = GridParticleSystem(
            continuous_sys,
            delta_y=delta,
            delta_v=delta,
            goal_y=params["goal_y"],
            goal_v=params["goal_v"],
        )
        vi = ValueIteration(grid_sys, gamma=0.95)
        vi.solve(max_iterations=200, threshold=1e-4)

        evaluator = PolicyEvaluator(continuous_sys, vi)
        results = evaluator.evaluate(n_episodes=100, verbose=False)

        results_by_res[delta] = results

        print(f"\nResults for Δ={delta}:")
        print(f"  Success Rate: {results['success_rate'] * 100:.1f}%")
        print(f"  Avg Steps (success): {results['avg_steps_success']:.1f}")

    plot_convergence(results_by_res, continuous_sys, gamma=0.95)

    return results_by_res


if __name__ == "__main__":
    # Run main pipeline
    continuous_sys, grid_sys, vi, results = main()

    # Optional: run resolution experiment
    results_by_res = resolution_experiment()
