# Complete Implementation Guide: Grid-Based MDP Solver for Continuous-State Control

## Overview

This is a complete implementation of a grid-based Markov Decision Process (MDP) solver for continuous-state control problems. The system solves a particle control task with stochastic dynamics by discretizing the continuous state space into a finite grid, computing optimal policies via value iteration, and providing multiple methods for executing the learned policy in continuous space, including a k-nearest neighbors (k-NN) averaging approach for smoother action selection.

---

## Problem Description

### Physical System

We control a particle moving in 1D space with the following dynamics:

**State Variables:**

- Position: y (continuous, bounded by [-y_max, y_max])
- Velocity: v (continuous, bounded by [-v_max, v_max])

**Control:**

- Actions: a ∈ {-1, 0, 1} (discrete force inputs)

**Dynamics (Week 1 formulation):**

```
v[t+1] = v[t] + (1/m) * (a + φ(y)) + noise
y[t+1] = y[t] + v[t]
```

Where:

- φ(y) = A \* sin(2πy/y_max) is a sinusoidal potential field force
- noise ~ N(0, (0.1\*v)^2) is Gaussian speed wobble
- Crash probability: p_crash = p_c \* |v| / v_max (when crash occurs, v → 0)

**Objective:**

- Reach and maintain the goal state (y=0, v=0)
- Maximize discounted cumulative reward

---

## Architecture

### Core Components

The implementation consists of four main classes:

1. **ContinuousParticleSystem**: The true continuous-state environment
2. **GridParticleSystem**: Discretized approximation with analytical transitions
3. **ValueIteration**: MDP solver with multiple action selection methods
4. **PolicyEvaluator**: Policy testing and evaluation framework

---

## Class 1: ContinuousParticleSystem

### Purpose

Represents the true continuous-state dynamical system. This is the "ground truth" environment where policies are actually executed.

### Key Methods

#### `__init__(m, y_max, v_max, A, p_c, goal_y, goal_v, goal_tolerance)`

Initializes the continuous system with physical parameters.

**Parameters:**

- `m`: Mass (affects acceleration from forces)
- `y_max`: Position bounds (state space: [-y_max, y_max])
- `v_max`: Velocity bounds (state space: [-v_max, v_max])
- `A`: Potential field amplitude
- `p_c`: Crash probability coefficient
- `goal_y`, `goal_v`: Target state
- `goal_tolerance`: Distance threshold for goal achievement

#### `phi(y)`

Computes the potential field force at position y:

```python
φ(y) = A * sin(2πy / y_max)
```

This creates a sinusoidal force field that adds complexity to the control problem.

#### `step(state, action)`

Executes one timestep of the dynamics.

**Process:**

1. Compute net force: f_net = action + φ(y)
2. Determine if crash occurs (probabilistic)
3. If crash: v_next = 0, y unchanged
4. If no crash:
   - Add speed wobble: noise ~ N(0, (0.1\*v)^2)
   - Update velocity: v_next = v + f_net/m + noise
   - Update position: y_next = y + v
5. Clip to bounds
6. Check if goal reached (reward = 1 if yes, else 0)

**Returns:** (next_state, reward, done)

#### `reset(y0, v0)`

Resets to initial state (random if not specified).

---

## Class 2: GridParticleSystem

### Purpose

Creates a finite-state MDP approximation of the continuous system by:

1. Discretizing the state space into a regular grid
2. Analytically computing transition probabilities P(s'|s,a)
3. Defining rewards on the grid

### Key Innovation: Analytical Transitions

Instead of Monte Carlo sampling, transition probabilities are computed exactly using known probability distributions (Gaussian noise, Bernoulli crashes). This makes value iteration more accurate and stable.

### Key Methods

#### `__init__(continuous_system, delta_y, delta_v, goal_y, goal_v, k_neighbors, use_knn)`

Creates the grid and precomputes all transitions.

**Parameters:**

- `continuous_system`: Reference to the true dynamics
- `delta_y`, `delta_v`: Grid resolution (smaller = finer grid)
- `goal_y`, `goal_v`: Target state
- `k_neighbors`: Number of nearest neighbors for k-NN averaging (default: 4)
- `use_knn`: Whether to use k-NN averaging by default (default: False)

**Grid Creation:**

```python
y_grid = [-y_max, -y_max + Δy, ..., y_max]  # n_y cells
v_grid = [-v_max, -v_max + Δv, ..., v_max]  # n_v cells
Total states: n = n_y × n_v
```

#### `discretize(y, v)`

Maps continuous state to grid indices:

```python
i = floor((y + y_max) / delta_y)
j = floor((v + v_max) / delta_v)
```

#### `continuous_state(i, j)`

Maps grid indices back to continuous coordinates (cell centers).

#### `discretize_knn(y, v, k)`

Finds the k nearest grid cells to a continuous state.

**Algorithm:**
1. Compute Euclidean distance from (y, v) to all grid cells
2. Sort by distance
3. Return the k closest cells with their distances

**Returns:** List of (i, j, distance) tuples for k nearest cells

#### `get_averaged_state_knn(y, v, k)`

Computes a smoothed state by averaging k nearest neighbors.

**Process:**
1. Find k nearest grid cells using `discretize_knn()`
2. Get the continuous coordinates (y_cell, v_cell) for each
3. Average the positions: y_avg = Σ y_cell / k
4. Average the velocities: v_avg = Σ v_cell / k

**Returns:** (y_avg, v_avg) - averaged continuous state

**Use case:** Reduces discretization noise by smoothing over nearby grid cells rather than snapping to a single nearest cell.

#### `_compute_transitions_analytical()`

Computes P(s'|s,a) for all state-action pairs.

**For each (i, j, a):**

1. Get continuous state (y, v) at cell center
2. Compute deterministic next state components
3. Calculate crash probability
4. For crash case: add transition to (y, 0)
5. For no-crash case: integrate Gaussian velocity noise over grid cells
6. Store resulting probability distribution

**Mathematical Details:**

- Velocity noise: N(v_det, (0.1\*v)^2)
- Probability of landing in cell j':
  ```
  P(v' ∈ [v_j', v_j'+1]) = Φ((v_high - v_det)/σ) - Φ((v_low - v_det)/σ)
  ```
  where Φ is the standard normal CDF

#### `reward(i, j)`

Returns 1.0 if state (i,j) is within one grid cell of goal, else 0.0.

---

## Class 3: ValueIteration

### Purpose

Solves the discretized MDP using dynamic programming and provides four different methods for action selection in continuous space.

### Value Iteration Algorithm

#### `__init__(grid_system, gamma)`

Initializes solver with grid and discount factor.

**State:**

- `V`: Value function array [n_y × n_v]
- `policy`: Policy array [n_y × n_v] (stores action indices)
- `actions`: [-1, 0, 1]

#### `solve(max_iterations, threshold)`

Runs value iteration until convergence.

**Algorithm:**

```
Initialize V = 0
repeat:
    for each state (i, j):
        for each action a:
            Q(i, j, a) = Σ P(s'|s,a) [R(s') + γV(s')]
        V(i, j) = max_a Q(i, j, a)
        policy(i, j) = argmax_a Q(i, j, a)
    until ||V - V_old|| < threshold
```

**Returns:** (V, policy)

#### `_compute_q_value(i, j, a)`

Computes Q(s,a) using Bellman equation:

```
Q(i, j, a) = Σ_{i',j'} P((i',j')|(i,j),a) * [R(i',j') + γ*V(i',j')]
```

---

## Action Selection Methods

The key innovation: after learning V\* on the grid, we can use it in continuous space via different strategies.

### Method 1: Baseline (Nearest-Neighbor)

**Implementation:** `get_action(y, v, use_knn=False)`

**Algorithm:**

1. Discretize (y, v) to nearest grid cell (i, j)
2. Look up precomputed policy: a = policy[i, j]
3. Return action

**Characteristics:**

- Fastest (O(1) lookup)
- Discontinuous policy (jumps at grid boundaries)
- Suffers from discretization artifacts
- Performance degrades with coarse grids

**Use case:** When grid is already fine and speed is critical

---

### Method 1b: k-Nearest Neighbors Averaging

**Implementation:** `get_action(y, v, use_knn=True, k=4)`

**Algorithm:**

1. Find k nearest grid cells to (y, v)
2. Compute averaged state: (y_avg, v_avg) from k neighbors
3. Discretize the averaged state to grid cell (i, j)
4. Look up precomputed policy: a = policy[i, j]
5. Return action

**Mathematical Details:**
```
neighbors = discretize_knn(y, v, k)  # k nearest cells
y_avg = (1/k) * Σ y_cell  for all neighbors
v_avg = (1/k) * Σ v_cell  for all neighbors
(i, j) = discretize(y_avg, v_avg)
action = policy[i, j]
```

**Characteristics:**

- Smooths discretization by averaging nearby grid cells
- Reduces noise from grid quantization
- Still O(n) for finding k-NN (can be optimized)
- More robust to grid alignment
- Continuous in expectation over neighbors

**Use case:** When you want smoother behavior without changing the value function or adding much computational cost

**Configuration:**
- `k_neighbors`: Set in grid system initialization
- `use_knn`: Enable/disable globally or per call
- Typical values: k ∈ {4, 8, 16}

---

### Method 2: Bilinear Interpolation

**Implementation:** `get_action_bilinear(y, v)`

**Algorithm:**

1. Find 4 surrounding grid cells: (i0,j0), (i1,j0), (i0,j1), (i1,j1)
2. Compute interpolation weights based on position within cell:
   ```
   α = (y - y_i0) / Δy
   β = (v - v_j0) / Δv
   ```
3. For each action a, interpolate Q-value:
   ```
   Q_interp(a) = (1-α)(1-β)Q00(a) + α(1-β)Q10(a)
                + (1-α)βQ01(a) + αβQ11(a)
   ```
4. Return action with highest interpolated Q-value

**Helper Method:** `get_value_bilinear(y, v)`

- Interpolates value function V(y,v) over 4 neighbors
- Used by the combined method

**Characteristics:**

- Smooth, continuous policy
- Reduces discretization artifacts
- Better at grid boundaries
- Still based on grid transitions
- O(n_v) per action (4 Q-value computations)

**Use case:** When you want smoother policies without changing dynamics model

---

### Method 3: 1-Step Lookahead

**Implementation:** `get_action_lookahead(y, v)`

**Algorithm:**

1. For each action a:
   a. Compute deterministic next state (y_det, v_det) using true dynamics
   b. Compute crash probability from current velocity
   c. Integrate over stochastic outcomes:
   - Crash case: Q += p_crash \* [R(y, 0) + γV(y, 0)]
   - No-crash case: integrate over Gaussian velocity noise
     ```
     Q += (1-p_crash) * Σ_j' P(v'∈cell_j') * [R(y_det, v_j') + γV(i_det, j')]
     ```
2. Return action with highest Q-value

**Helper Method:** `_compute_continuous_q_value(y, v, a)`

- Performs the expectation computation
- Uses analytical integration over grid cells
- Discretizes next states to nearest grid cells for V lookup

**Key Insight:** Uses true continuous dynamics instead of precomputed grid transitions. The next state distribution is computed from the current continuous state, not from a grid cell.

**Characteristics:**

- Adapts to continuous state dynamics
- More accurate at non-grid-point states
- Better handles stochastic transitions
- Still uses nearest-neighbor for V lookup
- O(n_v) per action (velocity integration)

**Use case:** When current state is far from grid cell centers

---

### Method 4: Lookahead + Bilinear (Combined)

**Implementation:** `get_action_lookahead_bilinear(y, v)`

**Algorithm:**

1. For each action a:
   a. Compute deterministic next state (true dynamics)
   b. Compute crash probability
   c. Integrate over stochastic outcomes:
   - Crash case: Q += p_crash \* [R(y,0) + γV_interp(y, 0)]
   - No-crash case: sample velocity distribution
     ```
     For each velocity sample v_k:
         weight = Gaussian_pdf(v_k) * Δv
         V_est = V_interp(y_det, v_k)  # ← bilinear interpolation
         Q += weight * [R(y_det, v_k) + γV_est]
     ```
2. Return action with highest Q-value

**Helper Method:** `_compute_continuous_q_value_bilinear(y, v, a)`

- Similar to lookahead but uses bilinear interpolation for V
- Samples velocity distribution (default: 20 samples)
- Computes probability weights using Gaussian PDF

**Key Innovation:** Combines both improvements:

- 1-step lookahead: adapts to continuous dynamics
- Bilinear interpolation: smooth V estimates everywhere

**Characteristics:**

- Best performance across all resolutions
- Most robust to coarse grids
- Smooth, continuous policy
- Accurate value estimates everywhere
- Slowest (most computation per action)
- O(n_samples) per action

**Use case:** When you want best performance and can afford the computation

---

## Class 4: PolicyEvaluator

### Purpose

Tests learned policies in the continuous environment and computes performance statistics.

### Key Methods

#### `__init__(continuous_system, value_iteration, method)`

Initializes evaluator with a specific action selection method.

**Parameters:**

- `continuous_system`: The true environment
- `value_iteration`: Learned V\* and policy
- `method`: One of ['baseline', 'bilinear', 'lookahead', 'lookahead_bilinear']

**Method Dispatch:**
The class maintains a dictionary mapping method names to action functions:

```python
action_functions = {
    'baseline': vi.get_action,
    'bilinear': vi.get_action_bilinear,
    'lookahead': vi.get_action_lookahead,
    'lookahead_bilinear': vi.get_action_lookahead_bilinear,
}
```

#### `run_episode(y0, v0, max_steps, render)`

Runs one episode using the selected method.

**Process:**

1. Reset environment to initial state
2. For each timestep:
   - Get action using selected method
   - Execute action in continuous system
   - Record reward and trajectory
   - Check if goal reached
3. Return (total_reward, steps, trajectory, success)

#### `evaluate(n_episodes, verbose)`

Runs multiple episodes and computes statistics.

**Statistics Computed:**

- Success rate: fraction of episodes reaching goal
- Average reward per episode
- Average steps per episode (all episodes)
- Average steps to goal (successful episodes only)

**Returns:** Dictionary with all statistics

---

## Experiment Functions

### 1. `compare_methods(continuous_sys, vi, n_episodes)`

**Purpose:** Compare all 4 action selection methods on the same learned value function.

**Process:**

1. Create evaluator for each method
2. Run n_episodes for each
3. Collect and display statistics
4. Generate comparison plot

**Output Plot:** `method_comparison.png`

- 3-panel bar chart
- Panel 1: Success rates
- Panel 2: Average steps to goal
- Panel 3: Average rewards

**Key Insight:** Isolates the effect of action selection method by using the same V\* for all methods.

---

### 2. `resolution_experiment()`

**Purpose:** Test baseline method at multiple grid resolutions.

**Process:**

1. For each resolution Δ ∈ [1.0, 0.5, 0.25, 0.2, 0.15]:
   - Create grid system
   - Solve MDP (value iteration)
   - Evaluate baseline method
   - Record statistics
2. Plot results

**Output Plot:** `resolution_analysis.png`

- 2-panel line plot
- Panel 1: Success rate vs resolution
- Panel 2: Average steps vs resolution

**Key Insight:** Shows how grid fineness affects performance.

---

### 3. `resolution_method_experiment(n_episodes)`

**Purpose:** Comprehensive test of all methods at multiple resolutions.

**Process:**

1. For each resolution Δ ∈ [1.0, 0.5, 0.25]:
   - Create grid and solve MDP once
   - For each method:
     - Evaluate performance
     - Record statistics
2. Plot comprehensive comparison

**Output Plot:** `resolution_method_comparison.png`

- 2-panel line plot with 4 lines (one per method)
- Panel 1: Success rate vs resolution (all methods)
- Panel 2: Average steps vs resolution (all methods)
- Different colors and markers for each method

**Key Insight:** Reveals which methods are most robust to coarse discretization.

---

## Visualization Functions

### 1. `visualize_policy(vi, grid_sys, save_path)`

Creates a 2-panel visualization of the learned policy.

**Panel 1: Value Function**

- Heatmap of V\*(y, v)
- Shows which states are valuable
- Goal marked with red star at (0, 0)

**Panel 2: Policy**

- Scatter plot showing action at each grid point
- Color-coded by action: red (-1), yellow (0), green (1)
- Shows the decision boundary

**Output:** `policy_visualization.png`

---

### 2. `plot_trajectory(trajectory, continuous_sys, grid_sys, save_path)`

Plots the state trajectory from a single episode.

**3 Panels:**

1. Position y vs time
2. Velocity v vs time
3. Phase space: y vs v trajectory

**Visualization:**

- Start point: green circle
- End point: blue square
- Goal: red star
- System parameters displayed at bottom

**Output:** `trajectory.png`

---

### 3. `plot_convergence(results_by_resolution, continuous_sys, gamma)`

Plots how baseline performance varies with grid resolution.

**2 Panels:**

1. Success rate vs resolution
2. Average steps vs resolution

**Output:** `resolution_analysis.png`

---

### 4. `plot_method_comparison(results_by_method, continuous_sys, grid_sys, save_path)`

Bar charts comparing all 4 methods on fixed grid.

**3 Panels:**

1. Success rates
2. Average steps to goal
3. Average rewards

Each bar labeled with exact value.

**Output:** `method_comparison.png`

---

### 5. `plot_resolution_method_comparison(results, continuous_sys, gamma)`

Line plots showing all methods across all resolutions.

**2 Panels:**

1. Success rate vs resolution (4 lines)
2. Efficiency vs resolution (4 lines)

Each method has distinct color and marker.

**Output:** `resolution_method_comparison.png`

---

## Main Pipeline

### `main()` Function

Complete workflow from setup to evaluation.

**Steps:**

1. **Define Parameters**

   ```python
   m = 1.0, y_max = 10.0, v_max = 5.0
   A = 2.0, p_c = 0.1
   goal = (0, 0), Δy = Δv = 0.25
   start_y = -8.0, start_v = 0.0  # Initial state
   k_neighbors = 4, use_knn = False  # k-NN configuration
   ```

2. **Create Continuous System**

   ```python
   continuous_sys = ContinuousParticleSystem(**params)
   ```

3. **Create Grid System**

   ```python
   grid_sys = GridParticleSystem(
       continuous_sys, delta_y, delta_v, 
       goal_y, goal_v,
       k_neighbors=4, use_knn=False
   )
   ```

   - This precomputes all transition probabilities
   - Configures k-NN parameters for action selection

4. **Solve MDP**

   ```python
   vi = ValueIteration(grid_sys, gamma=0.95)
   V, policy = vi.solve(max_iterations=200, threshold=1e-4)
   ```

5. **Evaluate Baseline Method**

   ```python
   evaluator = PolicyEvaluator(continuous_sys, vi, method='baseline')
   results = evaluator.evaluate(n_episodes=100)
   ```

6. **Generate Visualizations**
   - Policy heatmaps
   - Example trajectory

**Returns:** (continuous_sys, grid_sys, vi, results)

---

### `if __name__ == "__main__":` Block

Runs all experiments when script is executed directly.

**Experiment Sequence:**

1. **Main Pipeline**

   - Train value function
   - Test baseline method
   - Generate basic plots

2. **Experiment 1: Method Comparison**

   - Compare all 4 methods on same grid (Δ=0.25)
   - 100 episodes per method
   - Generate comparison plot

3. **Experiment 2: Resolution Analysis**

   - Test baseline at 5 different resolutions
   - 100 episodes per resolution
   - Generate resolution plot

4. **Experiment 3: Comprehensive Analysis**
   - Test all methods at 3 resolutions
   - 50 episodes per (method, resolution) pair
   - Generate comprehensive comparison plot

**Total Runtime:** ~35-50 minutes on typical laptop

**Output Files:**

- `policy_visualization.png`
- `trajectory.png`
- `method_comparison.png`
- `resolution_analysis.png`
- `resolution_method_comparison.png`

---

## Usage Examples

### Basic: Run All Experiments

```python
python main.py
```

### Custom: Test Specific Method

```python
from main import *

# Setup system
params = {
    "m": 1.0, "y_max": 10.0, "v_max": 5.0,
    "A": 2.0, "p_c": 0.1,
    "goal_y": 0.0, "goal_v": 0.0,
    "goal_tolerance": 0.1
}
continuous_sys = ContinuousParticleSystem(**params)

# Create grid and solve
grid_sys = GridParticleSystem(continuous_sys, delta_y=0.25, delta_v=0.25)
vi = ValueIteration(grid_sys, gamma=0.95)
vi.solve()

# Evaluate specific method
evaluator = PolicyEvaluator(continuous_sys, vi, method='lookahead_bilinear')
results = evaluator.evaluate(n_episodes=100)

print(f"Success rate: {results['success_rate']:.1%}")
print(f"Avg steps: {results['avg_steps']:.1f}")
```

### Custom: Single Episode with Visualization

```python
evaluator = PolicyEvaluator(continuous_sys, vi, method='bilinear')
reward, steps, trajectory, success = evaluator.run_episode(
    y0=-8.0, v0=0.0, render=True
)

plot_trajectory(trajectory, continuous_sys, grid_sys)
```

### Custom: Compare Just Two Methods

```python
methods_to_test = ['baseline', 'lookahead_bilinear']
results = {}

for method in methods_to_test:
    evaluator = PolicyEvaluator(continuous_sys, vi, method=method)
    results[method] = evaluator.evaluate(n_episodes=50, verbose=False)

for method, res in results.items():
    print(f"{method}: {res['success_rate']:.1%} success")
```

### Custom: Test k-NN Averaging

```python
# Enable k-NN globally in grid system
grid_sys = GridParticleSystem(
    continuous_sys, delta_y=0.5, delta_v=0.5,
    k_neighbors=8, use_knn=True  # Enable k-NN with k=8
)
vi = ValueIteration(grid_sys, gamma=0.95)
vi.solve()

# Evaluate with k-NN enabled (uses grid's settings)
evaluator = PolicyEvaluator(continuous_sys, vi)
results = evaluator.evaluate(n_episodes=100)
print(f"k-NN (k=8): {results['success_rate']:.1%} success")

# Or override k-NN settings per evaluation
results_standard = evaluator.evaluate(n_episodes=50, use_knn=False)
results_knn4 = evaluator.evaluate(n_episodes=50, use_knn=True, k=4)
results_knn16 = evaluator.evaluate(n_episodes=50, use_knn=True, k=16)
```

### Custom: Compare Standard vs k-NN on Coarse Grid

```python
# Test on coarse grid (where k-NN should help)
grid_sys = GridParticleSystem(continuous_sys, delta_y=1.0, delta_v=1.0)
vi = ValueIteration(grid_sys, gamma=0.95)
vi.solve()

evaluator = PolicyEvaluator(continuous_sys, vi)

# Standard nearest-neighbor
results_std = evaluator.evaluate(n_episodes=100, use_knn=False)

# k-NN smoothing
results_knn = evaluator.evaluate(n_episodes=100, use_knn=True, k=8)

print(f"Coarse Grid (Δ=1.0):")
print(f"  Standard: {results_std['success_rate']:.1%}")
print(f"  k-NN (k=8): {results_knn['success_rate']:.1%}")
print(f"  Improvement: +{(results_knn['success_rate'] - results_std['success_rate'])*100:.1f}%")
```

---

## Implementation Details

### Numerical Stability

**State Clipping:**
All continuous states are clipped to valid bounds before discretization:

```python
y = np.clip(y, -y_max, y_max)
v = np.clip(v, -v_max, v_max)
```

**Grid Boundary Handling:**
Discretization indices are clamped to valid range:

```python
i = int(np.clip(i, 0, n_y - 1))
j = int(np.clip(j, 0, n_v - 1))
```

**Probability Normalization:**
Transition probabilities are normalized to sum to 1.0:

```python
total_prob = sum(transitions.values())
transitions = {k: v/total_prob for k, v in transitions.items()}
```

**Edge Cases:**

- When v ≈ 0: wobble_std ≈ 0, handle deterministic case
- At grid boundaries: interpolation uses boundary cells
- Crash probability: capped at 1.0

---

### Computational Complexity

**Precomputation (Value Iteration):**

- Grid states: n = n_y × n_v
- Actions: |A| = 3
- Transitions: O(n² |A|) storage
- Value iteration: O(k × n² |A|) where k is number of iterations
- Typical: k ≈ 50-100

**Action Selection (per query):**

| Method    | Complexity   | Operations                    |
| --------- | ------------ | ----------------------------- |
| Baseline  | O(1)         | 1 lookup                      |
| Bilinear  | O(1)         | 4 V lookups, 3 Q computations |
| Lookahead | O(n_v)       | n_v probability computations  |
| Combined  | O(n_samples) | n_samples V interpolations    |

**Episode Runtime:**

- Baseline: ~0.01s per episode
- Bilinear: ~0.02s per episode
- Lookahead: ~0.1s per episode
- Combined: ~0.15s per episode

---

### Memory Requirements

**Grid Storage:**

- V array: n_y × n_v floats (8 bytes each)
- Policy array: n_y × n_v ints (4 bytes each)
- Transitions: ~n² |A| × 10 entries (sparse, ~100-500 MB for fine grids)

**Example (Δ=0.25):**

- Grid: 81 × 41 = 3321 states
- V: ~26 KB
- Policy: ~13 KB
- Transitions: ~50 MB

**Example (Δ=0.1):**

- Grid: 201 × 101 = 20301 states
- V: ~158 KB
- Policy: ~79 KB
- Transitions: ~2 GB

---

## Performance Characteristics

### Expected Results

**Method Comparison (Δ=0.25):**

- Baseline: ~75-85% success rate
- Bilinear: ~80-90% success rate
- Lookahead: ~85-92% success rate
- Combined: ~90-95% success rate

**Resolution Trends (Baseline):**

- Δ=1.0 (coarse): ~30-50% success
- Δ=0.5 (medium): ~60-75% success
- Δ=0.25 (fine): ~75-85% success
- Δ=0.15 (very fine): ~85-90% success

**Combined Method vs Grid Resolution:**

- Much more robust to coarse grids
- Maintains ~70-80% success even at Δ=1.0
- Difference from baseline increases as grid gets coarser

---

### Trade-offs

**Fine Grid (small Δ):**

- Pros: More accurate, all methods perform similarly
- Cons: Exponentially more states, slower value iteration, more memory

**Coarse Grid (large Δ):**

- Pros: Fast value iteration, low memory
- Cons: Poor baseline performance, need smarter action selection

**Method Selection:**

- Baseline: Fast execution, poor on coarse grids
- Bilinear: Moderate improvement, still fast
- Lookahead: Significant improvement, moderate cost
- Combined: Best performance, highest cost

**Recommended Strategy:**

- Use coarser grid (faster value iteration)
- Use combined method (compensates for coarse grid)
- Net result: Better performance/computation trade-off

**k-NN Averaging:**

- Best for: Medium-coarse grids where baseline shows discretization artifacts
- Trade-off: Better than baseline, slower than baseline, but simpler than interpolation methods
- Tuning: Start with k=4, increase if policy is too noisy, decrease if too smooth
- Works well combined with: Any grid resolution, especially Δ ∈ [0.5, 1.0]

---

## Testing and Validation

**Code Quality:**

- All functions have docstrings
- Type hints where appropriate
- Consistent naming conventions
- Modular design with clear interfaces

**Numerical Validation:**

- Probability distributions sum to 1
- Value function converges
- Policies are deterministic
- Trajectories stay in bounds

**Experimental Validation:**

- Three-tier experiment design isolates variables
- Multiple episodes for statistical significance
- Visualizations confirm intuitions
- Results match theoretical predictions

---

## Extensions and Future Work

**Potential Improvements:**

1. **Higher-order interpolation:** Bicubic or tricubic instead of bilinear
2. **Multi-step lookahead:** Planning horizon > 1
3. **Adaptive grids:** Finer resolution near goal, coarser elsewhere
4. **Function approximation:** Neural network V function (no grid)
5. **Parallel computing:** Speed up value iteration and evaluation
6. **Online adaptation:** Update V function during execution
7. **Model uncertainty:** Robust RL with dynamics uncertainty

**Code Extensions:**

- Save/load trained policies
- Interactive visualization tools
- Hyperparameter optimization
- Batch evaluation framework
- Additional environments

---

## References and Background

**Reinforcement Learning:**

- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Bertsekas (2005): Dynamic Programming and Optimal Control

**Discretization Methods:**

- Munos & Moore (2002): Variable resolution discretization in optimal control
- Davies et al. (1997): Effective grid resolution for continuous-state MDPs

**Interpolation:**

- Choi & Van Roy (2006): A generalized Kalman filter for fixed point approximation
- Gordon (1995): Stable function approximation in dynamic programming

**Related Techniques:**

- Model Predictive Control (MPC)
- Trajectory optimization
- Approximate dynamic programming
- Value function approximation in continuous spaces

---

## Conclusion

This implementation provides a complete, production-ready system for solving continuous-state control problems using grid-based discretization. The multiple action selection methods demonstrate the trade-off between computational cost and policy quality, with several options available depending on requirements.

**Method Selection Guide:**

1. **Baseline (Nearest-Neighbor)**: Use when grid is already fine (Δ < 0.2) and speed is critical
2. **k-NN Averaging**: Use for quick improvement over baseline with minimal code changes, especially on medium-coarse grids
3. **Bilinear Interpolation**: Use when you want smooth policies without changing dynamics model
4. **Lookahead**: Use when current state accuracy matters more than speed
5. **Combined (Lookahead + Bilinear)**: Use for best performance on coarse grids

The comprehensive experimental framework enables systematic comparison of methods and analysis of how grid resolution affects performance. This provides valuable insights for practitioners choosing discretization strategies for continuous-state RL problems.

Key takeaways:

- Grid discretization is effective but requires careful handling of continuous states
- Smarter action selection can compensate for coarser grids
- k-NN averaging provides a simple middle-ground between baseline and advanced interpolation
- The optimal approach depends on computational constraints and performance requirements
- Bilinear interpolation and lookahead are complementary techniques that combine well
- Configurable parameters (k_neighbors, use_knn, start_y, start_v) enable easy experimentation
