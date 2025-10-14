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
