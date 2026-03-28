import time
import numpy as np
from utils import (Map, sample_uniform, steer, nearest_neighbor,
                   rangesearch, choose_parent, rewire, extract_path)


def rrt_star(map_: Map, q_start, q_goal, goal_radius, params):
    """RRT* planner.

    Parameters
    ----------
    map_        : Map
    q_start     : array-like (2,)
    q_goal      : array-like (2,)
    goal_radius : float
    params      : dict with keys
                    max_iter   - int
                    step_size  - float  (eta)
                    gamma      - float  rewire constant, e.g. 1.0
                    goal_bias  - float  e.g. 0.05

    Returns
    -------
    path         : (M,2) ndarray or None
    cost         : float  (inf if no solution found)
    t_init       : float  time-to-first-solution in seconds (inf if none)
    n_init       : int    node count at first solution     (inf if none)
    cost_history : (max_iter,) ndarray  best cost at each iteration
    """
    q_start = np.array(q_start, dtype=float)
    q_goal  = np.array(q_goal,  dtype=float)

    # ── tree storage (pre-allocated for speed) ────────────────────────────
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

    # ── main loop ─────────────────────────────────────────────────────────
    for i in range(params['max_iter']):

        # 1. sample
        q_rand = sample_uniform(map_, q_goal, params['goal_bias'])

        # 2. nearest neighbour
        near_idx = nearest_neighbor(nodes[:n_nodes], q_rand)
        q_near   = nodes[near_idx]

        # 3. steer
        q_new = steer(q_near, q_rand, params['step_size'])

        # 4. collision check on new edge
        if not map_.obstacles.shape[0] == 0:
            from utils import collision_free
            if not collision_free(q_near, q_new, map_):
                cost_history[i] = best_cost
                continue

        # 5. rewire radius (RRT* formula)
        r = min(
            params['gamma'] * np.sqrt(np.log(n_nodes) / n_nodes),
            params['step_size']
        )

        # 6. near nodes
        near = rangesearch(nodes[:n_nodes], q_new, r)

        # 7. choose best parent
        best_p, c_new = choose_parent(
            nodes[:n_nodes], costs[:n_nodes], near, near_idx, q_new, map_
        )

        # 8. add node
        idx            = n_nodes
        nodes[idx]     = q_new
        parents[idx]   = best_p
        costs[idx]     = c_new
        n_nodes       += 1

        # 9. rewire
        rewire(nodes[:n_nodes], parents, costs, near, idx, map_)

        # 10. goal check
        if np.linalg.norm(q_new - q_goal) <= goal_radius:
            if costs[idx] < best_cost:
                best_cost = costs[idx]
                best_leaf = idx
                if np.isinf(t_init):
                    t_init = time.perf_counter() - t_start
                    n_init = n_nodes

        cost_history[i] = best_cost

    # ── extract path ──────────────────────────────────────────────────────
    if best_leaf == -1:
        return None, np.inf, t_init, n_init, cost_history

    path = extract_path(nodes[:n_nodes], parents[:n_nodes], best_leaf)
    return path, best_cost, t_init, n_init, cost_history