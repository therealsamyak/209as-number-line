# Mathematical Derivations for Action Selection Methods

This document provides detailed mathematical derivations for the four action selection methods implemented in `main.py`.

## Problem Setup

We have a continuous-state MDP with:

- **State space**: $(y, v) \in \mathbb{R}^2$ where $y$ is position and $v$ is velocity
- **Action space**: $a \in \{-1, 0, 1\}$ (discrete force inputs)
- **Dynamics** (Week 1):
  $$v_{t+1} = v_t + \frac{f_\text{net}}{m} + \mathcal{N}(0, (0.1v_t)^2)$$
  $$y_{t+1} = y_t + v_t$$
  where $f_\text{net} = a + \phi(y)$ and $\phi(y) = A\sin(2\pi y / y_\text{max})$

- **Crash mechanics**: With probability $p_\text{crash} = p_c \cdot |v| / v_\text{max}$, the velocity resets to 0
- **Goal**: Reach and maintain $(y, v) = (0, 0)$

## Grid Discretization

We discretize the continuous state space into a grid:

- Position cells: $y_i = -y_\text{max} + i \cdot \Delta_y$ for $i = 0, 1, \ldots, n_y-1$
- Velocity cells: $v_j = -v_\text{max} + j \cdot \Delta_v$ for $j = 0, 1, \ldots, n_v-1$

The discretized MDP has:

- Finite state space: $\mathcal{S} = \{(i, j) : i \in [0, n_y), j \in [0, n_v)\}$
- Transition probabilities: $P((i', j') | (i, j), a)$ computed analytically
- Value function: $V^*(i, j)$ learned via value iteration
- Policy: $\pi^*(i, j) = \arg\max_a Q^*(i, j, a)$

---

## Method 1: Baseline (Nearest-Neighbor)

### Description

The simplest approach: discretize the continuous state to the nearest grid cell and look up the precomputed policy.

### Mathematical Formulation

Given continuous state $(y, v)$:

1. **Discretization**: Find nearest grid cell
   $$i = \left\lfloor \frac{y + y_\text{max}}{\Delta_y} \right\rfloor, \quad j = \left\lfloor \frac{v + v_\text{max}}{\Delta_v} \right\rfloor$$

2. **Action selection**: Look up policy
   $$\pi_\text{baseline}(y, v) = \pi^*(i, j)$$

### Pros and Cons

**Pros**:

- Extremely fast (single lookup)
- No additional computation after value iteration

**Cons**:

- Discontinuous policy (jumps at grid boundaries)
- Discretization artifacts can lead to suboptimal behavior
- Performance degrades significantly with coarse grids

---

## Method 1b: k-Nearest Neighbors (k-NN) Averaging

### Description

An extension of the baseline that smooths the state by averaging k nearest grid cells before discretization.

### Mathematical Formulation

Given continuous state $(y, v)$ and parameter $k$:

1. **Find k-nearest neighbors**: Compute Euclidean distances to all grid cells
   $$d(i, j) = \sqrt{(y - y_i)^2 + (v - v_j)^2}$$
   
   Let $\mathcal{N}_k(y, v) = \{(i_1, j_1), \ldots, (i_k, j_k)\}$ be the k cells with smallest distances.

2. **Compute averaged state**:
   $$\bar{y} = \frac{1}{k} \sum_{(i, j) \in \mathcal{N}_k} y_i$$
   $$\bar{v} = \frac{1}{k} \sum_{(i, j) \in \mathcal{N}_k} v_j$$

3. **Discretize averaged state**:
   $$i^* = \left\lfloor \frac{\bar{y} + y_\text{max}}{\Delta_y} \right\rfloor, \quad j^* = \left\lfloor \frac{\bar{v} + v_\text{max}}{\Delta_v} \right\rfloor$$

4. **Action selection**: Look up policy at discretized averaged state
   $$\pi_\text{knn}(y, v) = \pi^*(i^*, j^*)$$

### Geometric Interpretation

The k-NN method creates a "smoothed" version of the state by:
- Taking a local neighborhood around the true state
- Averaging their coordinates (acts as a low-pass filter)
- Looking up the policy at this smoothed location

This reduces quantization noise from grid discretization.

### Relationship to Nearest-Neighbor

- When $k=1$: Reduces to standard nearest-neighbor baseline
- As $k \to \infty$: Averages over all grid cells (converges to grid center)
- Typical values: $k \in \{4, 8, 16\}$

### Pros and Cons

**Pros**:

- Smoother state-to-action mapping
- Reduces discretization artifacts
- More robust to grid alignment
- Simple modification to baseline
- Can improve performance on coarse grids

**Cons**:

- Slower than baseline: O(n_y × n_v) for finding neighbors
- Still uses single policy lookup (not interpolating actions)
- Choice of k requires tuning
- Computational cost increases with grid size

---

## Method 2: Bilinear Interpolation

### Description

Interpolate Q-values over the 4 nearest neighbor cells, then select the action with highest interpolated Q-value.

### Mathematical Formulation

Given continuous state $(y, v)$:

1. **Find surrounding cells**: Identify the 4 corners $(i_0, j_0), (i_1, j_0), (i_0, j_1), (i_1, j_1)$ where:
   $$i_0 = \left\lfloor \frac{y + y_\text{max}}{\Delta_y} \right\rfloor, \quad i_1 = i_0 + 1$$
   $$j_0 = \left\lfloor \frac{v + v_\text{max}}{\Delta_v} \right\rfloor, \quad j_1 = j_0 + 1$$

2. **Compute interpolation weights**:
   $$\alpha = \frac{y - y_{i_0}}{\Delta_y} \in [0, 1], \quad \beta = \frac{v - v_{j_0}}{\Delta_v} \in [0, 1]$$

3. **Interpolate Q-values** for each action $a$:
   $$Q_\text{interp}(y, v, a) = (1-\alpha)(1-\beta) Q^*(i_0, j_0, a) + \alpha(1-\beta) Q^*(i_1, j_0, a)$$
   $$+ (1-\alpha)\beta Q^*(i_0, j_1, a) + \alpha\beta Q^*(i_1, j_1, a)$$

4. **Action selection**:
   $$\pi_\text{bilinear}(y, v) = \arg\max_a Q_\text{interp}(y, v, a)$$

### Geometric Interpretation

The weights $\alpha$ and $\beta$ represent the normalized distance from the lower-left corner:

- $(\alpha, \beta) = (0, 0)$ → at $(i_0, j_0)$ (lower-left)
- $(\alpha, \beta) = (1, 0)$ → at $(i_1, j_0)$ (lower-right)
- $(\alpha, \beta) = (0, 1)$ → at $(i_0, j_1)$ (upper-left)
- $(\alpha, \beta) = (1, 1)$ → at $(i_1, j_1)$ (upper-right)

### Pros and Cons

**Pros**:

- Smooth policy (continuous action selection)
- Reduces discretization artifacts
- Better performance near grid boundaries

**Cons**:

- Still fundamentally based on grid transitions
- Doesn't adapt to true continuous dynamics
- Requires 4 Q-value computations per action

---

## Method 3: 1-Step Lookahead

### Description

For each action, compute the expected value by simulating one step in the continuous system and evaluating the expected next state using the learned value function.

### Mathematical Formulation

Given continuous state $(y, v)$ and action $a$:

1. **Compute deterministic next state**:
   $$f_\text{net} = a + \phi(y)$$
   $$v_\text{det} = v + \frac{f_\text{net}}{m}$$
   $$y_\text{det} = y + v$$

2. **Compute crash probability**:
   $$p_\text{crash} = \min\left(p_c \cdot \frac{|v|}{v_\text{max}}, 1\right)$$

3. **Compute expected Q-value**:
   $$Q_\text{lookahead}(y, v, a) = p_\text{crash} \cdot \mathbb{E}[R + \gamma V^* | \text{crash}] + (1 - p_\text{crash}) \cdot \mathbb{E}[R + \gamma V^* | \text{no crash}]$$

4. **Crash case**: Velocity becomes 0, position unchanged
   $$\mathbb{E}[R + \gamma V^* | \text{crash}] = R(y, 0) + \gamma V^*(y, 0)$$
   where $V^*(y, 0)$ is obtained by discretizing $(y, 0)$ to nearest grid cell.

5. **No-crash case**: Integrate over Gaussian velocity noise
   $$\mathbb{E}[R + \gamma V^* | \text{no crash}] = \int_{v'} p(v') \cdot [R(y_\text{det}, v') + \gamma V^*(y_\text{det}, v')] \, dv'$$
   where $v' \sim \mathcal{N}(v_\text{det}, (0.1v)^2)$ clipped to $[-v_\text{max}, v_\text{max}]$

6. **Numerical integration**: We integrate analytically over velocity grid cells:
   $$\mathbb{E}[R + \gamma V^* | \text{no crash}] = \sum_{j'=0}^{n_v-1} P(v' \in [v_{j'}, v_{j'+1}]) \cdot [R(i_\text{next}, j') + \gamma V^*(i_\text{next}, j')]$$
   where:
   $$P(v' \in [v_{j'}, v_{j'+1}]) = \Phi\left(\frac{v_{j'+1} - v_\text{det}}{\sigma}\right) - \Phi\left(\frac{v_{j'} - v_\text{det}}{\sigma}\right)$$
   and $\Phi$ is the CDF of the standard normal distribution, $\sigma = 0.1|v|$

7. **Action selection**:
   $$\pi_\text{lookahead}(y, v) = \arg\max_a Q_\text{lookahead}(y, v, a)$$

### Key Insight

This method uses the **true continuous dynamics** to compute next state distributions, rather than relying on precomputed grid transitions. This makes it more accurate, especially when the continuous state is far from grid cell centers.

### Pros and Cons

**Pros**:

- Adapts to continuous state dynamics
- More accurate at non-grid-point states
- Better handles stochastic transitions

**Cons**:

- Slower (requires expectation computation)
- Still uses nearest-neighbor for V function lookup
- Computational cost scales with grid resolution

---

## Method 4: Lookahead + Bilinear (Combined)

### Description

The most sophisticated method: combines 1-step lookahead with bilinear interpolation of the value function.

### Mathematical Formulation

Similar to Method 3, but with a key difference:

**Instead of**: $V^*(y, v) \approx V^*(i, j)$ (nearest-neighbor)

**Use**: Bilinear interpolation over 4 neighbors:
$$V^*_\text{interp}(y, v) = (1-\alpha)(1-\beta) V^*(i_0, j_0) + \alpha(1-\beta) V^*(i_1, j_0)$$
$$+ (1-\alpha)\beta V^*(i_0, j_1) + \alpha\beta V^*(i_1, j_1)$$

### Complete Algorithm

1. Compute deterministic dynamics (same as Method 3)

2. **Crash case**:
   $$\mathbb{E}[R + \gamma V^* | \text{crash}] = R(y, 0) + \gamma V^*_\text{interp}(y, 0)$$

3. **No-crash case**: Sample-based approximation with interpolation
   $$\mathbb{E}[R + \gamma V^* | \text{no crash}] \approx \sum_{k=1}^{n_\text{samples}} w_k \cdot [R(y_\text{det}, v_k) + \gamma V^*_\text{interp}(y_\text{det}, v_k)]$$
   where:

   - $v_k$ are sampled from the velocity distribution (e.g., uniformly spaced within $[\mu - 3\sigma, \mu + 3\sigma]$)
   - $w_k$ are probability weights: $w_k \propto \phi(v_k; \mu, \sigma) \cdot \Delta v$
   - $\phi(v; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(v-\mu)^2}{2\sigma^2}\right)$

4. Combine crash and no-crash cases:
   $$Q_\text{combined}(y, v, a) = p_\text{crash} \cdot \mathbb{E}[\cdot | \text{crash}] + (1-p_\text{crash}) \cdot \mathbb{E}[\cdot | \text{no crash}]$$

5. Action selection:
   $$\pi_\text{combined}(y, v) = \arg\max_a Q_\text{combined}(y, v, a)$$

### Why This Works Best

This method combines the advantages of both approaches:

1. **1-step lookahead** adapts to true continuous dynamics
2. **Bilinear interpolation** provides smooth value estimates at arbitrary states

The result is a policy that:

- Is smooth and continuous
- Accurately reflects the underlying dynamics
- Performs well even on coarse grids
- Reduces discretization errors at multiple levels

### Pros and Cons

**Pros**:

- Best performance across all grid resolutions
- Most robust to coarse discretization
- Smooth, continuous policy
- Accurate value estimates everywhere

**Cons**:

- Slowest method (most computation per action)
- Requires careful numerical integration
- Implementation complexity

---

## Computational Complexity Comparison

Let $n = n_y \times n_v$ be the total number of grid states.

| Method       | Precomputation | Per-Action Query      | Total per State         |
| ------------ | -------------- | --------------------- | ----------------------- |
| Baseline     | $O(1)$         | $O(1)$                | $O(1)$                  |
| k-NN         | $O(1)$         | $O(n)$ (find k-NN)    | $O(n)$                  |
| Bilinear     | $O(1)$         | $O(n_v)$ (4 Q-values) | $O(n_v)$                |
| Lookahead    | $O(1)$         | $O(n_v)$ (integrate)  | $O(n_v)$                |
| Combined     | $O(1)$         | $O(n_\text{samples})$ | $O(n_\text{samples})$   |

All methods use the same value iteration for precomputation: $O(n^2 \cdot |\mathcal{A}|)$

**Note on k-NN complexity**: The naive implementation is O(n) to find k neighbors. This can be optimized to O(log n) using spatial data structures (KD-trees, ball trees), but for typical grid sizes (< 10,000 states), the naive approach is sufficient.

---

## Expected Performance Trends

### As grid resolution increases ($\Delta_y, \Delta_v \to 0$):

- All methods converge to optimal continuous policy
- Baseline improves significantly (discretization error decreases)
- Interpolation/lookahead benefits diminish
- **Trade-off**: Finer grids = exponentially more states = slower value iteration

### As grid resolution decreases (coarse grids):

- Baseline degrades significantly
- Bilinear interpolation helps moderately
- Lookahead helps significantly
- Combined method performs best
- **Trade-off**: Coarser grids = faster value iteration but need smarter action selection

### Practical recommendation:

Use **lookahead + bilinear** when:

- Grid is relatively coarse (large $\Delta$)
- Computational budget for value iteration is limited
- High performance is critical

Use **baseline** when:

- Grid is already fine
- Real-time performance is critical
- Value iteration budget is large

---

## Implementation Notes

### Numerical Stability

- Always clip states to bounds before discretization
- Use CDF differences (not PDF integrals) for Gaussian probabilities
- Normalize interpolation weights to sum to 1
- Handle edge cases (v ≈ 0, near boundaries)

### Computational Optimizations

- Cache grid cell lookups
- Vectorize probability computations
- Use sparse storage for transitions
- Precompute Gaussian CDF values

### Testing and Validation

The code includes three experiments to validate the methods:

1. **Fixed-grid comparison**: Isolate method differences
2. **Resolution sweep**: Test sensitivity to discretization
3. **Comprehensive analysis**: Full picture of method × resolution interactions

---

## References

- **Bilinear Interpolation**: Classic computer graphics technique adapted for RL
- **1-Step Lookahead**: Related to model-based RL and trajectory optimization
- **Grid Discretization**: Munos & Moore (2002) on variable resolution discretization
- **Continuous-State MDPs**: Bertsekas (2005) on approximate dynamic programming
