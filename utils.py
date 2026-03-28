import numpy as np


# ── Map / environment ─────────────────────────────────────────────────────

class Map:
    """2D axis-aligned environment.

    Parameters
    ----------
    x_lim, y_lim : (float, float)
        Bounding box of the configuration space.
    obstacles : list of (x, y, w, h)
        Each obstacle is an axis-aligned rectangle defined by its
        bottom-left corner (x, y) and its width/height (w, h).
    """
    def __init__(self, x_lim, y_lim, obstacles=None):
        self.x_lim = np.array(x_lim, dtype=float)
        self.y_lim = np.array(y_lim, dtype=float)
        self.obstacles = np.array(obstacles, dtype=float) if obstacles else np.empty((0, 4))

    @property
    def bounds(self):
        return self.x_lim, self.y_lim


# ── Collision checking ────────────────────────────────────────────────────

def point_in_obs(q, obstacles):
    """True if point q=[x,y] is inside any obstacle rectangle."""
    if obstacles.shape[0] == 0:
        return False
    x, y = q
    in_x = (x >= obstacles[:, 0]) & (x <= obstacles[:, 0] + obstacles[:, 2])
    in_y = (y >= obstacles[:, 1]) & (y <= obstacles[:, 1] + obstacles[:, 3])
    return bool(np.any(in_x & in_y))


def collision_free(q1, q2, map_: Map, n_checks=20):
    """True if the straight-line segment q1->q2 is obstacle-free.

    Uses uniform point sampling along the segment — simple and effective
    for axis-aligned rectangular obstacles at typical planning scales.
    """
    if map_.obstacles.shape[0] == 0:
        return True
    for t in np.linspace(0, 1, n_checks):
        q = q1 + t * (q2 - q1)
        if point_in_obs(q, map_.obstacles):
            return False
    return True


# ── Tree primitives ───────────────────────────────────────────────────────

def nearest_neighbor(nodes, q):
    """Index of the node in *nodes* (Nx2 array) closest to q."""
    diffs = nodes - q
    dists = np.einsum('ij,ij->i', diffs, diffs)   # squared distances, no sqrt needed
    return int(np.argmin(dists))


def rangesearch(nodes, q, r):
    """Indices of all nodes within Euclidean distance r of q."""
    diffs = nodes - q
    dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
    return list(np.where(dists <= r)[0])


def steer(q_near, q_rand, eta):
    """Move from q_near toward q_rand by at most eta."""
    d = np.linalg.norm(q_rand - q_near)
    if d <= eta:
        return q_rand.copy()
    return q_near + eta * (q_rand - q_near) / d


# ── Parent selection & rewiring ───────────────────────────────────────────

def choose_parent(nodes, costs, near_idx, default_idx, q_new, map_):
    """Return (best_parent_index, cost_through_that_parent) for q_new."""
    best_idx  = default_idx
    best_cost = costs[default_idx] + np.linalg.norm(q_new - nodes[default_idx])

    for k in near_idx:
        if not collision_free(nodes[k], q_new, map_):
            continue
        c = costs[k] + np.linalg.norm(q_new - nodes[k])
        if c < best_cost:
            best_cost = c
            best_idx  = k

    return best_idx, best_cost


def rewire(nodes, parents, costs, near_idx, new_idx, map_):
    """Rewire near nodes through new_idx if it lowers their cost-to-root."""
    q_new  = nodes[new_idx]
    c_new  = costs[new_idx]

    for k in near_idx:
        if k == parents[new_idx]:
            continue
        if not collision_free(q_new, nodes[k], map_):
            continue
        c_through = c_new + np.linalg.norm(nodes[k] - q_new)
        if c_through < costs[k]:
            parents[k] = new_idx
            costs[k]   = c_through


# ── Path extraction ───────────────────────────────────────────────────────

def extract_path(nodes, parents, leaf_idx):
    """Walk parent pointers from leaf back to root; return (Mx2) path."""
    path = []
    idx  = leaf_idx
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    return np.array(path[::-1])


# ── Samplers ──────────────────────────────────────────────────────────────

def sample_uniform(map_: Map, q_goal, goal_bias=0.05):
    """Uniform random sample with occasional goal bias."""
    if np.random.rand() < goal_bias:
        return q_goal.copy()
    x = map_.x_lim[0] + np.random.rand() * np.diff(map_.x_lim)[0]
    y = map_.y_lim[0] + np.random.rand() * np.diff(map_.y_lim)[0]
    return np.array([x, y])


def sample_hybrid(map_: Map, q_start, q_goal, goal_bias=0.05, alpha=0.5):
    """Hybrid sampler used by Hybrid-RRT*.

    With probability alpha    -> uniform sample over Q_free
    With probability 1-alpha  -> draw one uniform and one normal sample
                                 (centred on the start-goal line) and
                                 keep whichever is closer to the goal.

    Parameters
    ----------
    alpha : float in [0, 1]
        Mixing weight. alpha=1 reduces to pure uniform sampling.
    """
    if np.random.rand() < goal_bias:
        return q_goal.copy()

    if np.random.rand() < alpha:
        return sample_uniform(map_, q_goal, goal_bias=0.0)

    # non-uniform branch: normal distribution along start-goal line
    mid       = 0.5 * (q_start + q_goal)
    direction = q_goal - q_start
    length    = np.linalg.norm(direction)
    sigma     = length / 4.0            # spread roughly covers the corridor

    q_normal  = mid + np.random.randn() * sigma * (direction / (length + 1e-9))
    q_normal  = np.clip(q_normal,
                        [map_.x_lim[0], map_.y_lim[0]],
                        [map_.x_lim[1], map_.y_lim[1]])

    q_uniform = sample_uniform(map_, q_goal, goal_bias=0.0)

    # keep whichever is closer to goal
    if np.linalg.norm(q_normal - q_goal) < np.linalg.norm(q_uniform - q_goal):
        return q_normal
    return q_uniform


# ── Visualisation helper ──────────────────────────────────────────────────

def plot_map(ax, map_: Map, q_start, q_goal):
    """Draw obstacles, start, and goal on a matplotlib Axes."""
    import matplotlib.patches as patches

    ax.set_xlim(map_.x_lim)
    ax.set_ylim(map_.y_lim)
    ax.set_aspect('equal')

    for obs in map_.obstacles:
        rect = patches.Rectangle(
            (obs[0], obs[1]), obs[2], obs[3],
            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.6
        )
        ax.add_patch(rect)

    ax.plot(*q_start, 'go', markersize=8, label='Start')
    ax.plot(*q_goal,  'r*', markersize=10, label='Goal')
    ax.legend(fontsize=8)