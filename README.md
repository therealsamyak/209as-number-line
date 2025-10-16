# 209as-number-line

## Setup

You need Python 3.13 or higher.

Do the following

MacOS / Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows

```sh
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Usage

Run the main simulation with all experiments:

```sh
python3 main.py # Mac / Linux
python main.py # Windows
```

This will run three comprehensive experiments:

1. **Method Comparison**: Compare all 4 action selection methods on the same grid
2. **Resolution Analysis**: Test how grid resolution affects the baseline method
3. **Comprehensive Analysis**: Test all methods at multiple resolutions

### Action Selection Methods

The code implements **4 different approaches** for selecting actions in continuous space:

#### 1. Baseline (Nearest-Neighbor)

- Maps continuous state to nearest grid cell
- Looks up precomputed policy at that cell
- Fast but suffers from discretization artifacts

#### 2. Bilinear Interpolation

- Computes Q-values for 4 nearest neighbor cells
- Interpolates based on distance weights
- Smoother policy, better at grid boundaries

#### 3. 1-Step Lookahead

- Simulates one step forward for each action
- Evaluates expected value using learned V function
- Adapts to continuous dynamics, not just grid transitions

#### 4. Lookahead + Bilinear (Combined)

- Uses 1-step lookahead with bilinear interpolation of V
- Most sophisticated: combines advantages of both
- Best performance, especially at coarse resolutions

### Generated Outputs

The program generates the following visualizations:

- `policy_visualization.png`: Value function V*(y,v) and policy π*(y,v) heatmaps
- `trajectory.png`: Example episode trajectory in position, velocity, and phase space
- `method_comparison.png`: Bar charts comparing success rate and efficiency of all 4 methods
- `resolution_analysis.png`: How baseline method performs at different grid resolutions
- `resolution_method_comparison.png`: Comprehensive comparison showing which methods are most robust to coarse grids

## Autoformat

Install the 'Ruff' VSCode Extension here: https://marketplace.cursorapi.com/items/?itemName=charliermarsh.ruff for autoformatting.

## Disclaimer

AI was used to generate code in this repository. See `prompts/` folder for the full list of prompts used for coding.
