import numpy as np
import time
from utils import Map
from rrt_star import rrt_star
from informed_rrt_star import informed_rrt_star
from hybrid_rrt_star import hybrid_rrt_star


# ═════════════════════════════════════════════════════════════════════════
#  Environment generators
# ═════════════════════════════════════════════════════════════════════════

def make_narrow_passage(w, x_lim=(0,1), y_lim=(0,1)):
    """Single vertical wall with a gap of width w at the top.

    The wall runs from y=0 to y=(1-w), leaving a passage of width w
    at the top of the map. The wall is centred horizontally.

    Parameters
    ----------
    w : float  passage width as fraction of map height

    Returns
    -------
    map_    : Map
    q_start : (2,) ndarray
    q_goal  : (2,) ndarray
    """
    wall_x     = 0.45
    wall_width = 0.10
    wall_h     = 1.0 - w           # wall height leaves gap of w at top

    obs     = [[wall_x, 0.0, wall_width, wall_h]]
    map_    = Map(x_lim, y_lim, obs)
    q_start = np.array([0.1, 0.5])
    q_goal  = np.array([0.9, 0.5])
    return map_, q_start, q_goal


def make_cluttered(rho, seed, x_lim=(0,1), y_lim=(0,1),
                   obs_w=0.08, obs_h=0.08, max_attempts=5000):
    """Randomly placed rectangular obstacles with approximate density rho.

    Obstacles are placed until the covered fraction of the map reaches
    rho. A connectivity check (straight-line clearance) ensures a path
    between start and goal is not trivially blocked; if placement fails,
    the last valid map is returned.

    Parameters
    ----------
    rho  : float  target obstacle coverage fraction
    seed : int    random seed for reproducibility

    Returns
    -------
    map_    : Map
    q_start : (2,) ndarray
    q_goal  : (2,) ndarray
    """
    rng     = np.random.default_rng(seed)
    q_start = np.array([0.05, 0.5])
    q_goal  = np.array([0.95, 0.5])

    area_per_obs = obs_w * obs_h
    map_area     = np.diff(x_lim)[0] * np.diff(y_lim)[0]
    target_n_obs = int(rho * map_area / area_per_obs)

    obs      = []
    coverage = 0.0

    for _ in range(max_attempts):
        if len(obs) >= target_n_obs:
            break

        x = rng.uniform(x_lim[0], x_lim[1] - obs_w)
        y = rng.uniform(y_lim[0], y_lim[1] - obs_h)
        candidate = [x, y, obs_w, obs_h]

        # skip if obstacle covers start or goal
        def blocks_point(q, o):
            return (o[0] <= q[0] <= o[0]+o[2]) and (o[1] <= q[1] <= o[1]+o[3])

        if blocks_point(q_start, candidate) or blocks_point(q_goal, candidate):
            continue

        obs.append(candidate)
        coverage += area_per_obs / map_area

    map_ = Map(x_lim, y_lim, obs if obs else None)
    return map_, q_start, q_goal


# ═════════════════════════════════════════════════════════════════════════
#  Single trial runner
# ═════════════════════════════════════════════════════════════════════════

def run_trial(algorithm, map_, q_start, q_goal, goal_radius, params):
    """Run one trial of an algorithm and return a results dict."""
    t0 = time.perf_counter()
    path, cost, t_init, n_init, cost_hist = algorithm(
        map_, q_start, q_goal, goal_radius, params
    )
    t_total = time.perf_counter() - t0

    return {
        'success':      path is not None,
        'cost':         cost,
        't_init':       t_init,
        'n_init':       n_init,
        't_total':      t_total,
        'cost_history': cost_hist,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Batch experiment runners
# ═════════════════════════════════════════════════════════════════════════

def run_narrow_passage_experiment(passage_widths, n_trials, params, goal_radius=0.05):
    """Experiment 1: vary narrow passage width w.

    Returns
    -------
    results : dict  keyed by (algorithm_name, w), each value is a list
              of per-trial result dicts (length n_trials)
    """
    algorithms = {
        'RRT*':          rrt_star,
        'Informed RRT*': informed_rrt_star,
        'Hybrid RRT*':   hybrid_rrt_star,
    }

    results = {}
    total   = len(passage_widths) * len(algorithms) * n_trials
    done    = 0

    for w in passage_widths:
        map_, q_start, q_goal = make_narrow_passage(w)
        for alg_name, alg_fn in algorithms.items():
            trials = []
            for t in range(n_trials):
                np.random.seed(t)          # reproducible across algorithms
                r = run_trial(alg_fn, map_, q_start, q_goal, goal_radius, params)
                trials.append(r)
                done += 1
                if done % 10 == 0:
                    print(f'  [{done}/{total}] w={w:.2f}  {alg_name}  trial {t+1}')
            results[(alg_name, w)] = trials

    return results


def run_clutter_experiment(densities, n_trials, params, goal_radius=0.05):
    """Experiment 2: vary obstacle density rho.

    Each (density, trial) pair uses a fixed random seed so all three
    algorithms see the same obstacle map.

    Returns
    -------
    results : dict  keyed by (algorithm_name, rho)
    """
    algorithms = {
        'RRT*':          rrt_star,
        'Informed RRT*': informed_rrt_star,
        'Hybrid RRT*':   hybrid_rrt_star,
    }

    results = {}
    total   = len(densities) * len(algorithms) * n_trials
    done    = 0

    for rho in densities:
        for alg_name, alg_fn in algorithms.items():
            trials = []
            for t in range(n_trials):
                map_, q_start, q_goal = make_cluttered(rho, seed=t)
                np.random.seed(t)
                r = run_trial(alg_fn, map_, q_start, q_goal, goal_radius, params)
                trials.append(r)
                done += 1
                if done % 10 == 0:
                    print(f'  [{done}/{total}] rho={rho:.2f}  {alg_name}  trial {t+1}')
            results[(alg_name, rho)] = trials

    return results


# ═════════════════════════════════════════════════════════════════════════
#  Summary statistics
# ═════════════════════════════════════════════════════════════════════════

def summarise(trials):
    """Compute summary statistics over a list of trial result dicts.

    Returns a dict with median and IQR for each key metric.
    """
    success_rate = np.mean([r['success'] for r in trials])

    def med_iqr(key):
        vals = np.array([r[key] for r in trials if r['success']])
        if len(vals) == 0:
            return np.nan, np.nan
        return np.median(vals), np.subtract(*np.percentile(vals, [75, 25]))

    cost_med,   cost_iqr   = med_iqr('cost')
    t_init_med, t_init_iqr = med_iqr('t_init')
    n_init_med, n_init_iqr = med_iqr('n_init')

    return {
        'success_rate': success_rate,
        'cost_med':     cost_med,   'cost_iqr':     cost_iqr,
        't_init_med':   t_init_med, 't_init_iqr':   t_init_iqr,
        'n_init_med':   n_init_med, 'n_init_iqr':   n_init_iqr,
    }