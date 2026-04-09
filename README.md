# Mobile Robotics Final Project

**EECE 5550 Mobile Robotics — Final Project**
Sofia Makowska & Camille Ferrell | Northeastern University | Spring 2026

---

## Overview

This project conducts a rigorous empirical evaluation of the **Hybrid-RRT*** path planning algorithm (Ganesan et al., 2024), which proposes a hybrid sampling strategy combining uniform and goal-directed non-uniform sampling within the RRT* framework. We evaluate whether the paper's claimed performance improvements hold under two challenging conditions not fully characterized in the original work: **narrow passage environments** and **high obstacle density environments**.

We implement and compare four sampling-based planners from scratch in Python:

| Algorithm | Sampling Strategy |
|---|---|
| RRT* | Pure uniform |
| Informed RRT* | Uniform, then ellipse-restricted after first solution |
| RRT*-N | Pure normal distribution along start-goal line |
| Hybrid RRT* | Mixture of uniform and normal (controlled by α) |

---

## Repository Structure

```
.
├── rrt_star.py             # RRT* baseline planner
├── informed_rrt_star.py    # Informed RRT* planner
├── rrt_star_n.py           # RRT*-N planner
├── hybrid_rrt_star.py      # Hybrid RRT* planner (algorithm under evaluation)
├── utils.py                # Shared utilities: Map, samplers, collision checking,
│                           #   nearest neighbor, rewiring, path extraction
├── run_experiments.py      # Environment generators, batch trial runner,
│                           #   and summary statistics
└── experiments.ipynb       # Jupyter notebook: single-run demos, convergence
                            #   plots, and full experiment results
```

---

## Requirements

```
python >= 3.9
numpy
matplotlib
jupyter
```

Install dependencies with:

```bash
pip install numpy matplotlib jupyter
```

---

## Usage

### Running the notebook

```bash
jupyter notebook experiments.ipynb
```

The notebook is organized as follows:
1. **RRT\*** — single-run demo and convergence plot
2. **Informed RRT\*** — single-run demo and comparison with RRT*
3. **RRT\*-N** — single-run demo and comparison with RRT* and Informed RRT*
4. **Hybrid RRT\*** — single-run demo and comparison of all four planners
5. **Experiments** — batch evaluation across narrow passage and cluttered environments

### Running experiments directly

```python
from run_experiments import run_narrow_passage_experiment, run_clutter_experiment, summarise

params = dict(max_iter=2000, step_size=0.05, gamma=1.0, goal_bias=0.05, alpha=0.5)

# Experiment 1: narrow passages
results_np = run_narrow_passage_experiment(
    passage_widths=[0.02, 0.05, 0.10, 0.20, 0.40],
    n_trials=50,
    params=params
)

# Experiment 2: obstacle density
results_cl = run_clutter_experiment(
    densities=[0.10, 0.20, 0.30, 0.40, 0.50],
    n_trials=50,
    params=params
)

# Print summary for a condition
for key, trials in results_np.items():
    alg, w = key
    s = summarise(trials)
    print(f"{alg:20s}  w={w:.2f}  success={s['success_rate']:.0%}  "
          f"cost={s['cost_med']:.3f}  t_init={s['t_init_med']:.2f}s")
```

---

## Experiments

### Experiment 1 — Narrow Passage Environments

A vertical wall divides the configuration space, leaving a passage of width `w` at the top. We vary `w ∈ {0.02, 0.05, 0.10, 0.20, 0.40}` and measure how each planner's success rate and convergence degrade as the passage narrows.

### Experiment 2 — High Obstacle Density Environments

Randomly placed rectangular obstacles cover a fraction `ρ` of the map. We vary `ρ ∈ {0.10, 0.20, 0.30, 0.40, 0.50}` and measure whether the advantage of Hybrid-RRT* over RRT* diminishes in cluttered environments.

### Metrics

- **Success rate** — fraction of trials finding a solution within the iteration budget
- **Time to first solution** `t_init` — wall-clock time to first feasible path
- **Nodes at first solution** `n_init` — tree size at first solution
- **Final path cost** `c*` — best path cost at end of all iterations
- **Convergence curve** `c*(n)` — best cost as a function of samples drawn

All metrics reported as **median ± IQR** over 50 independent trials.

---

## Reference

S. Ganesan, B. Ramalingam, and R. E. Mohan, "A hybrid sampling-based RRT* path planning algorithm for autonomous mobile robot navigation," *Expert Systems with Applications*, vol. 258, p. 125206, 2024. doi: [10.1016/j.eswa.2024.125206](https://doi.org/10.1016/j.eswa.2024.125206)
