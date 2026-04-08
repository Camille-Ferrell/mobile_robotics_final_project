import time
import numpy as np
from utils import (Map, steer, nearest_neighbor,
                   rangesearch, choose_parent, rewire, extract_path,
                   sample_uniform, collision_free)


# ── Informed sampler ──────────────────────────────────────────────────────

def sample_informed(q_start, q_goal, c_best, map_: Map, goal_bias=0.05):
    """Sample uniformly from the prolate hyperspheroid (ellipse in 2D)
    defined by the current best path cost c_best.

    The ellipse has:
      - foci at q_start and q_goal
      - transverse semi-axis  a = c_best / 2
      - conjugate semi-axis   b = sqrt(c_best^2 - c_min^2) / 2
    where c_min = ||q_goal - q_start|| is the straight-line distance.

    Samples that fall outside the map bounds are rejected and resampled.
    Falls back to uniform sampling until a first solution is found (c_best=inf).
    """
    if np.isinf(c_best):
        return sample_uniform(map_, q_goal, goal_bias)

    if np.random.rand() < goal_bias:
        return q_goal.copy()

    c_min = np.linalg.norm(q_goal - q_start)
    if c_best <= c_min:          # numerical guard
        return sample_uniform(map_, q_goal, goal_bias=0.0)

    # ellipse centre and rotation matrix
    centre    = 0.5 * (q_start + q_goal)
    a         = c_best / 2.0
    b         = np.sqrt(max(c_best**2 - c_min**2, 0.0)) / 2.0

    # rotation: align major axis with q_start -> q_goal
    diff      = q_goal - q_start
    theta     = np.arctan2(diff[1], diff[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R         = np.array([[cos_t, -sin_t],
                           [sin_t,  cos_t]])

    # rejection sample until we land inside map bounds
    for _ in range(200):
        # uniform sample from unit ball, scaled to ellipse
        r     = np.sqrt(np.random.rand())
        angle = np.random.rand() * 2 * np.pi
        unit  = np.array([r * np.cos(angle), r * np.sin(angle)])
        q     = centre + R @ np.array([a * unit[0], b * unit[1]])

        if (map_.x_lim[0] <= q[0] <= map_.x_lim[1] and
                map_.y_lim[0] <= q[1] <= map_.y_lim[1]):
            return q

    # fallback if rejection sampling fails (very tight ellipse)
    return sample_uniform(map_, q_goal, goal_bias=0.0)


# ── Informed RRT* ─────────────────────────────────────────────────────────

def informed_rrt_star(map_: Map, q_start, q_goal, goal_radius, params):
    """Informed RRT* planner (Gammell et al., 2014).

    Identical to RRT* except that once an initial solution is found,
    all subsequent samples are drawn from the informed ellipsoidal set
    rather than the full configuration space.

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

        # ── KEY DIFFERENCE FROM RRT*: informed sampler ────────────────────
        q_rand = sample_informed(q_start, q_goal, best_cost,
                                 map_, params['goal_bias'])

        # identical to RRT* from here on ──────────────────────────────────
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