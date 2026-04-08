import time
import numpy as np
from utils import (Map, steer, nearest_neighbor,
                   rangesearch, choose_parent, rewire, extract_path,
                   collision_free, sample_hybrid)


def hybrid_rrt_star(map_: Map, q_start, q_goal, goal_radius, params):
    """Hybrid-RRT* planner (Ganesan et al., 2024).

    Identical to RRT* except that the sampling distribution mixes
    uniform sampling and a goal-directed non-uniform sampler, controlled
    by the parameter alpha:

        With probability alpha:     uniform sample over Q_free
        With probability 1-alpha:   draw one uniform and one normal sample
                                    (centred on the start-goal line) and
                                    keep whichever is closer to the goal

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
                    alpha      - float in [0,1], mixing parameter
                                 (alpha=1 reduces to pure uniform / RRT*)

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

        # ── KEY DIFFERENCE FROM RRT*: hybrid sampler ──────────────────────
        q_rand = sample_hybrid(map_, q_start, q_goal,
                               params['goal_bias'], params['alpha'])

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