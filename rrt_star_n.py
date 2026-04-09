import time
import numpy as np
from utils import (Map, steer, nearest_neighbor,
                   rangesearch, choose_parent, rewire, extract_path,
                   collision_free)


# ── Non-uniform sampler ───────────────────────────────────────────────────

def sample_normal(map_: Map, q_start, q_goal, goal_bias=0.05):
    """Sample from a normal distribution centred on the start-goal line.

    This is the pure non-uniform sampler used by RRT*-N (Mohammed et al.,
    2021). Unlike Hybrid-RRT*, there is no uniform fallback — all samples
    are drawn from the normal distribution, giving faster convergence in
    open environments but limited exploration in complex ones.

    The distribution is centred at the midpoint of the start-goal line,
    with sigma scaled to roughly cover the corridor between start and goal.
    Samples are clipped to map bounds.
    """
    if np.random.rand() < goal_bias:
        return q_goal.copy()

    mid       = 0.5 * (q_start + q_goal)
    direction = q_goal - q_start
    length    = np.linalg.norm(direction)
    sigma     = length / 4.0

    # perturb in both x and y so the sampler can reach off-axis regions
    q = mid + np.random.randn(2) * sigma
    q = np.clip(q,
                [map_.x_lim[0], map_.y_lim[0]],
                [map_.x_lim[1], map_.y_lim[1]])
    return q


# ── RRT*-N ────────────────────────────────────────────────────────────────

def rrt_star_n(map_: Map, q_start, q_goal, goal_radius, params):
    """RRT*-N planner (Mohammed et al., 2021).

    Identical to RRT* except that all samples are drawn from a normal
    distribution centred on the start-goal line (pure non-uniform
    sampling, no uniform fallback).

    Parameters
    ----------
    map_        : Map
    q_start     : array-like (2,)
    q_goal      : array-like (2,)
    goal_radius : float
    params      : dict with keys
                    max_iter   - int
                    step_size  - float
                    gamma      - float  rewire constant
                    goal_bias  - float

    Returns
    -------
    path         : (M,2) ndarray or None
    cost         : float
    t_init       : float  time to first solution (inf if none)
    n_init       : int    nodes at first solution (inf if none)
    cost_history : (max_iter,) ndarray
    """
    q_start = np.array(q_start, dtype=float)
    q_goal  = np.array(q_goal,  dtype=float)

    max_n   = params['max_iter'] + 1
    nodes   = np.empty((max_n, 2))
    parents = np.full(max_n, -1, dtype=int)
    costs   = np.full(max_n, np.inf)

    nodes[0]   = q_start
    parents[0] = -1
    costs[0]   = 0.0
    n_nodes    = 1

    cost_history = np.full(params['max_iter'], np.inf)
    best_cost    = np.inf
    best_leaf    = -1
    t_init       = np.inf
    n_init       = np.inf
    t_start      = time.perf_counter()

    for i in range(params['max_iter']):

        # ── KEY DIFFERENCE FROM RRT*: pure normal sampler ─────────────────
        q_rand = sample_normal(map_, q_start, q_goal, params['goal_bias'])

        # identical to RRT* from here on ───────────────────────────────────
        near_idx = nearest_neighbor(nodes[:n_nodes], q_rand)
        q_near   = nodes[near_idx]
        q_new    = steer(q_near, q_rand, params['step_size'])

        if not collision_free(q_near, q_new, map_):
            cost_history[i] = best_cost
            continue

        r = min(
            params['gamma'] * np.sqrt(np.log(n_nodes) / n_nodes),
            params['step_size']
        )
        near = rangesearch(nodes[:n_nodes], q_new, r)

        best_p, c_new = choose_parent(
            nodes[:n_nodes], costs[:n_nodes], near, near_idx, q_new, map_
        )

        idx          = n_nodes
        nodes[idx]   = q_new
        parents[idx] = best_p
        costs[idx]   = c_new
        n_nodes     += 1

        rewire(nodes[:n_nodes], parents, costs, near, idx, map_)

        if np.linalg.norm(q_new - q_goal) <= goal_radius:
            if costs[idx] < best_cost:
                best_cost = costs[idx]
                best_leaf = idx
                if np.isinf(t_init):
                    t_init = time.perf_counter() - t_start
                    n_init = n_nodes

        cost_history[i] = best_cost

    if best_leaf == -1:
        return None, np.inf, t_init, n_init, cost_history

    path = extract_path(nodes[:n_nodes], parents[:n_nodes], best_leaf)
    return path, best_cost, t_init, n_init, cost_history