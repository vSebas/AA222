import numpy as np
from pathlib import Path

ZONE_1 = np.array([5.0, 5.0])
INNER_RADIUS = 1.5
OUTER_RADIUS = 4.0
ZONE_2 = np.array([7.5, 2.5])
ZONE_3 = np.array([2.5, 7.5])
EXCLUSION_RADIUS = 1.2

def feasible_points(points):
    d_zone_1 = np.linalg.norm(points - ZONE_1, axis=1)
    d_zone_2 = np.linalg.norm(points - ZONE_2, axis=1)
    d_zone_3 = np.linalg.norm(points - ZONE_3, axis=1)

    return (
        (d_zone_1 >= INNER_RADIUS)
        & (d_zone_1 <= OUTER_RADIUS)
        & (d_zone_2 >= EXCLUSION_RADIUS)
        & (d_zone_3 >= EXCLUSION_RADIUS)
    )

def sample_feasible_points(rng, count):
    points = []
    while len(points) < count:
        candidates = rng.uniform(1.0, 9.0, size=(4 * count, 2))
        valid = candidates[feasible_points(candidates)]
        points.extend(valid[: count - len(points)])

    return np.array(points)

def random_feasible_layout(rng, n):
    return sample_feasible_points(rng, n).reshape(2 * n)

def repair_layout(x, n, rng):
    points = x.reshape(n, 2).copy()
    valid = feasible_points(points)
    if not np.all(valid):
        points[~valid] = sample_feasible_points(rng, np.sum(~valid))
    return points.reshape(2 * n)

def min_pairwise_distance(x, n=2):
    pts = x.reshape(n, 2)
    best = np.inf

    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, np.linalg.norm(pts[i] - pts[j]))

    return best

def layout_score(x, n):
    points = x.reshape(n, 2)
    if not np.all(feasible_points(points)):
        return -np.inf
    return min_pairwise_distance(x, n)

def cross_entropy(f, n, k_max, m=100, m_elite=10, rng=None):
    """
    f: objective function
    P: proposal distribution
    m: sample size
    m_elite: number of samples to use when refitting
    """
    # if not 0 < m_elite <= m:
    #     raise ValueError("m_elite must be between 1 and m")
    # rng = np.random.default_rng() if rng is None else rng

    dim = 2 * n     # beta, HML coords

    initial_samples = np.array([random_feasible_layout(rng, n) for _ in range(m)])
    initial_scores = np.array([f(sample) for sample in initial_samples])
    elite_samples = initial_samples[np.argsort(initial_scores)[-m_elite:]]

    mu = np.mean(elite_samples, axis=0)
    sigma = np.cov(elite_samples, rowvar=False) + 1e-4 * np.eye(dim)
    
    best_x = None
    best_score = -np.inf

    for _ in range(k_max):
        samples = rng.multivariate_normal(mu, sigma, m)
        samples = np.array([repair_layout(sample, n, rng) for sample in samples])
        scores = np.array([f(sample) for sample in samples])

        i_best = np.argmax(scores)
        if scores[i_best] > best_score:
            best_score = scores[i_best]
            best_x = samples[i_best].copy()

        elite_idx = np.argsort(scores)[-m_elite:]
        elite_samples = samples[elite_idx]

        mu = np.mean(elite_samples, axis=0)
        sigma = np.cov(elite_samples, rowvar=False) + 1e-4 * np.eye(dim)

    return best_x, best_score

def optimize_layout(n, restarts=5, k_max=30, m=350, m_elite=50, seed=0):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = -np.inf
    f = lambda x: layout_score(x, n)

    for _ in range(restarts):
        x, score = cross_entropy(f, n, k_max, m, m_elite, rng)
        if score > best_score:
            best_x = x
            best_score = score

    return best_x, best_score

def solve_task_1(output_csv=None):
    if output_csv is None:
        output_csv = Path(__file__).with_name("portfolio_layout_results.csv")

    results = []
    layouts = {}

    for n in range(2, 11):
        x, score = optimize_layout(n, seed=100 + n)
        results.append((n, score))
        layouts[n] = x
        print(f"n={n}, p*={score:.6f}", flush=True)

    np.savetxt(output_csv, np.array(results), delimiter=",", fmt=["%d", "%.8f"])
    return results, layouts

def main():
    solve_task_1()

if __name__ == "__main__":
    main()
