grid decomposition of continuous space problem

### Week 1

#### Pure particle on a numberline dynamics without ϕ, speed wobble, crashes, or sensor.

The following are system parameters (used for programming/simulation purposes):

- $m \in \mathbb{R}$ - mass of the particle
- $y_0 \in \mathbb{R}$ - initial position of the particle
- $v_0 \in \mathbb{R}$ - initial velocity of the particle
- $\phi(y)$ - force-exerting external potential field, based purely on position of the particle in the system
- $v_{max}$ - maximum particle velocity in the system (independent constant for now, might change later)
- $p_c$ - probability of crashing, need to multiply by $\frac{|v|}{v_{max}}$
  <br/>
  $
x(t) = \begin{pmatrix}
y(t) \\
v(t)
\end{pmatrix}
\in \mathbb{R}^2
$
  The state of the system is based on the following:
- $y(t) \in \mathbb{R}$ - position of the particle at time $t$
- $v(t) \in \mathbb{R}$ - velocity of the particle at time $t$
  <br/>
  $
u(t) = f_i(t) \in [-1, 1]
$
  The user input to the system is:
- $f_i(t) \in [-1, 1]$ - applied force
  <br/>
  $
  x'(t) = x(t)

* \begin{pmatrix}
  v(t) \\
  \frac{1}{m}f\_{net}(t)
  \end{pmatrix}
  =
  \begin{pmatrix}
  y(t) \\
  v(t)
  \end{pmatrix}
* \begin{pmatrix}
  v(t) \\
  \frac{1}{m}f\_{net}(t)
  \end{pmatrix}
  $
  The state equation is from Newton's laws of motions in discrete time, as mentioned in the document.

- $f_{net}(t) = f_i(t) - \frac{d}{dy}\phi(y)$
  <br/>
  $
z(t) = y_s(t) = y(t)
$
  The output of the system is the current position of the particle **based on the sensor data** at the time the position was taken.
  <br/>
  **Notes:**
- _noisy sensor affects the output_
- _the crash probability and speed wobble affect the state_
  <br/>

#### Adding noise and other criteria

Adding in the speed wobbles, we get the new velocity equation to be as follows:
$ v[t+1] = v[t] + \frac{1}{m}f*{net}(t) + \mathcal{N}(0, 0.01v[t]^2)$
Adding the sensor noise, affects the output, but not the position update so y(t) is the same but z(t) is as follows: 
$z[t] = y[t] + \mathcal{N}(0, 0.25v[t]^2)$
Adding crash probability. Suppose we draw probability $p\in U([0,1])$ where $U$ is the uniform distribution. Then, we have:
$v[t+1] = \begin{cases}
0 & p \leq p_c \cdot \frac{|v[t]|}{v*{\max}} \\
v[t] + \frac{1}{m}f*{\text{net}}(t) + \mathcal{N}(0, 0.01v[t]^2) & p > p_c \cdot \frac{|v[t]|}{v*{\max}}
\end{cases}$
<br/>

### Week 2

The input is now:
$$a[t] =  f_i[t] \in \{-1, 0, 1\}$$
And the force field is defined as:
$$f_\phi(y) = int(A\sin(2\pi y/y_{max}))$$
This makes the net force:
$$f_{net}[t] = f_i[t] + int(A\sin(2\pi \cdot y[t]/y_{max}))$$
The position becomes:

$$
y[t + 1] = \begin{cases}
	-y_{max} & y[t] + v[t] < -y_{max}\\
	y[t] + v[t] & y[t] + v[t] \leq y_{max}\\
    y_{max} & y[t] + v[t] > y_{max}
\end{cases}
$$

The new v(t) will now include additive speed wobble d(v) as defined in the writeup

$$
d(v) = \begin{cases}
1 & w.p. \quad \frac{v}{v_{max}} \cdot \frac{p_w}{2}\\
0 & w.p. \quad (1 - \frac{v}{v_{max}} \cdot p_w)\\
-1 & w.p \quad \frac{v}{v_{max}} \cdot \frac{p_w}{2}
\end{cases}
$$

$$
v[t+1] = \begin{cases}
0 & w.p. \quad p_c \cdot \frac{|v[t]|}{v_{\max}} \\
v[t] + \frac{1}{m}f_{net}[t] +  d(v[t]) & w.p.\quad 1 - p_c \cdot \frac{|v[t]|}{v_{\max}}\\
\end{cases}
$$

Like before, our state is defined as

$$
s[t] = \begin{pmatrix}
y[t] \\
v[t]
\end{pmatrix}
$$

but this time they are in the integer space, instead of all real numbers.
The robot knows its exact current state at all times, making the output just the state at time t:
$$o[t] = s[t]$$

### Brainstorming Algorithms:

\begin{enumerate}
\item Have a starting point $-y_0$. Accelerate until we hit the midpoint $-y_0/2$. Once we hit the midpoint, we decelerate
\item If we crash at point $y_1$, we consider that our new starting position
\item If our future state will **cross the midpoint**, input force during that transition should be 0.
\end{enumerate}
We define reward as:

$$
R(x,f_i,x') = \begin{cases}
1 & x' = \begin{pmatrix} 0 \\ 0 \end{pmatrix}\\
0 & otherwise
\end{cases}
$$

Our horizon is an infinite horizon.
$H = \{0, 1,...,\infty\}$
For $v'\in \{v[t] - (int(A\sin(2\pi \cdot y[t]/y_{max})) - 2), v[t] + (int(A\sin(2\pi \cdot y[t]/y_{max})))\}:\\$
$\quad \mathbb{P}(v', -1, v[t]) = \mathbb{P}(d(v[t]) = v' + int(A\sin(2\pi \cdot y[t]/y_{max}) + 1)\\$
For $v'\in \{v[t] - (int(A\sin(2\pi \cdot y[t]/y_{max})) - 1), v[t] + (int(A\sin(2\pi \cdot y[t]/y_{max})) + 1)\}:\\$
$\quad \mathbb{P}(v', 0, v[t]) = \mathbb{P}(d(v[t]) = v' + int(A\sin(2\pi \cdot y[t]/y_{max}))\\$
For $v'\in \{v[t] - (int(A\sin(2\pi \cdot y[t]/y_{max}))), v[t] + (int(A\sin(2\pi \cdot y[t]/y_{max})) + 2)\}:\\$
$\quad \mathbb{P}(v', 1, v[t]) = \mathbb{P}(d(v[t]) = v' + int(A\sin(2\pi \cdot y[t]/y_{max}) - 1)\\$
If the action is to give a -1 force, then we can never get to a velocity = $v[t] - (int(A\sin(2\pi \cdot y[t]/y_{max})) + 2$ or $v[t] - (int(A\sin(2\pi \cdot y[t]/y_{max})) + 1$

$$
V^*_{i+1} = max_a \sum_{s'} P((v, y), a, (v', y'))[R((v, y), a, (v', y')) + \gamma V^*_i]
$$

for this problem specifically

how would i code this such that we have our original system but want to implement a 'mock system' with the grid decomposition to get a policy to then plug in to the original system to see how well it solves it etc.

Formulate then solve an approximate MDP problem on this continuous space dynamical numberline
system, finding a continuous space policy for a variety of task definitions.
Try to do so in a way that allows for arbitrary environments and tasks to be loaded, giving you a
general approximate dynamical MDP solver that you can apply to these problems.
You may want to consider problems in the following order:
• Discretize the state and action spaces using grid decomposition (with the resolution being a configurable
hyperparameter), and compute an optimal policy on the discretized problem.
• Using a nearest-neighbor (NN), 0-step lookahead process: derive, simulate, and evaluate the resulting continuous
space policies for varying grid resolutions.

you are misunderstanding, its more about numbers not about the type, like its a sin function but can we change the amplitude easily? or its a reward function but can it support other reward states besides 0,0 at speed 0

# Understanding the meaning of trajectory.png

_Exported on 10/14/2025 at 11:40:56 PDT from Cursor (1.7.38)_

---

**User**

what is trajectory.png represent

---

**Cursor**

I'll help you understand what `trajectory.png` represents. Let me first look at the image and the code that likely generates it.

Based on the image and code, **`trajectory.png`** represents the behavior of a **dynamical system on a number line** tracking an agent's movement over time. It contains three subplots:

## 1. **Position over Time** (Top, Blue)

- Shows how the agent's position `y` changes over time steps
- The red dashed line at `y=0` marks the goal position
- The agent oscillates around the goal, trying to reach and stay at position 0

## 2. **Velocity over Time** (Middle, Green)

- Shows how the agent's velocity `v` changes over time steps
- The red dashed line at `v=0` marks the goal velocity (stationary)
- The agent's velocity fluctuates as it tries to come to rest

## 3. **Phase Space** (Bottom, Purple)

- Plots velocity versus position, showing the system's state trajectory
- **Green circle**: Starting state
- **Blue square**: Ending state
- **Red star**: Goal state (position=0, velocity=0)
- This visualization shows the path the system takes through its state space

This appears to be from a **reinforcement learning or control problem** where an agent learns to navigate to position 0 with zero velocity, similar to a pendulum or cart-pole balancing problem. The oscillatory behavior suggests the agent is learning to dampen its motion and converge to the goal state.

---

**User**

look at the code in here how is it generated and why is it important

---

**Cursor**

Based on the code you've provided, here's how `trajectory.png` is generated and why it's important:

## How It's Generated

The trajectory is created through this process:

### 1. **Data Collection** (lines 621-623 in `main()`)

```python
reward, steps, trajectory, success = evaluator.run_episode(
    y0=-8.0, v0=0.0, render=True
)
```

- A single episode is run on the **continuous (real) system** starting from position y=-8.0 and velocity v=0.0
- The agent uses the **policy learned from the discretized grid** via Value Iteration
- At each timestep, the agent's continuous state `(y, v)` is recorded in the trajectory list

### 2. **Visualization** (lines 471-509, called at line 644)

The `plot_trajectory()` function creates three subplots:

**Top Plot - Position vs Time:**

```python
axes[0].plot(times, y_vals, "b-", linewidth=2)
axes[0].axhline(0, color="r", linestyle="--", label="Goal y=0")
```

Shows how position evolves over timesteps

**Middle Plot - Velocity vs Time:**

```python
axes[1].plot(times, v_vals, "g-", linewidth=2)
axes[1].axhline(0, color="r", linestyle="--", label="Goal v=0")
```

Shows how velocity evolves over timesteps

**Bottom Plot - Phase Space:**

```python
axes[2].plot(y_vals, v_vals, "purple", linewidth=2, alpha=0.6)
axes[2].plot(y_vals[0], v_vals[0], "go", markersize=10, label="Start")
axes[2].plot(y_vals[-1], v_vals[-1], "bs", markersize=10, label="End")
axes[2].plot(0, 0, "r*", markersize=15, label="Goal")
```

Shows the state-space trajectory (velocity vs position)

## Why It's Important

### 1. **Validates the Grid Approximation**

This is the **key insight** of your project:

- The policy is learned on a **discretized grid** (GridParticleSystem)
- But it's executed on the **continuous system** (ContinuousParticleSystem)
- The trajectory proves that discrete planning can successfully control continuous dynamics

### 2. **Demonstrates Convergence Behavior**

The oscillating pattern in your trajectory shows:

- The agent is **actively controlling** to reach the goal at (y=0, v=0)
- The system has **stochastic dynamics** (speed wobble, crashes) causing variations
- The agent learns to **dampen oscillations** and stabilize near the goal

### 3. **Phase Space Reveals Control Strategy**

The bottom plot is particularly insightful:

- Shows the **closed-loop behavior** - how position and velocity interact
- Reveals if the policy creates stable/unstable dynamics
- In your image, the trajectory spirals toward the goal, indicating a stabilizing controller

### 4. **Debugging and Analysis**

- If the trajectory diverges or fails to reach the goal, it indicates issues with:
  - Grid resolution (delta_y, delta_v)
  - Transition probability computation
  - Value iteration convergence
  - System parameters (A, p_c, m)

### 5. **Visual Proof of Concept**

For your Week 3 assignment (based on the prompts folder), this demonstrates:

- **Week 1 dynamics** are correctly implemented (speed wobble, crashes, potential field)
- **Grid decomposition** successfully approximates continuous state space
- **Value iteration** finds an optimal policy despite discretization errors

The trajectory is essentially the "output" that validates your entire MDP pipeline: continuous system → grid discretization → analytical transitions → value iteration → policy execution → goal achievement.

```python
reward, steps, trajectory, success = evaluator.run_episode(
    y0=-8.0, v0=0.0, render=True
)
```

```python
axes[0].plot(times, y_vals, "b-", linewidth=2)
axes[0].axhline(0, color="r", linestyle="--", label="Goal y=0")
```

```python
axes[1].plot(times, v_vals, "g-", linewidth=2)
axes[1].axhline(0, color="r", linestyle="--", label="Goal v=0")
```

```python
axes[2].plot(y_vals, v_vals, "purple", linewidth=2, alpha=0.6)
axes[2].plot(y_vals[0], v_vals[0], "go", markersize=10, label="Start")
axes[2].plot(y_vals[-1], v_vals[-1], "bs", markersize=10, label="End")
axes[2].plot(0, 0, "r*", markersize=15, label="Goal")
```


