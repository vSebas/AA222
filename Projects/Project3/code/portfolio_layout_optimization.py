import numpy as np
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optimistix as optx
from slsqp_jax import SLSQP, SLSQPConfig, ToleranceConfig

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

def jax_ineq_constraints(z, args):
    n = args
    points = z[:-1].reshape((n, 2))
    p = z[-1]

    d1 = jnp.linalg.norm(points - jnp.array(ZONE_1), axis=1)
    d2 = jnp.linalg.norm(points - jnp.array(ZONE_2), axis=1)
    d3 = jnp.linalg.norm(points - jnp.array(ZONE_3), axis=1)

    constraints = [
        d1 - INNER_RADIUS,
        OUTER_RADIUS - d1,
        d2 - EXCLUSION_RADIUS,
        d3 - EXCLUSION_RADIUS,
    ]

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(jnp.array([jnp.linalg.norm(points[i] - points[j]) - p]))

    return jnp.concatenate(constraints)

def jax_obj(z, args):
    return -z[-1], None

def refine_layout_jax(x0, n):
    z = jnp.array(np.concatenate([x0.copy(), [min_pairwise_distance(x0, n)]]))
    bounds = jnp.array([[1.0, 9.0]] * (2 * n) + [[0.0, 8.0]])
    n_ineq = 4 * n + n * (n - 1) // 2

    solver = SLSQP(
        ineq_constraint_fn=jax_ineq_constraints,
        n_ineq_constraints=n_ineq,
        bounds=bounds,
        config=SLSQPConfig(tolerance=ToleranceConfig(rtol=1e-8, atol=1e-8, max_steps=200)),
    )

    try:
        sol = optx.minimise(jax_obj, solver, z, args=n, has_aux=True, max_steps=200)
    except Exception:
        return x0, min_pairwise_distance(x0, n)

    x = np.array(sol.value[:-1])
    score = layout_score(x, n)
    if score == -np.inf:
        return x0, min_pairwise_distance(x0, n)

    return x, score

def outer_circle_layout(n, phase):
    angles = phase + np.arange(n) * 2.0 * np.pi / n
    points = ZONE_1 + OUTER_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])
    return points.reshape(2 * n)

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

def optimize_layout(n, restarts=5, k_max=30, m=350, m_elite=50, seed=0, boundary_starts=24, verbose=True):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = -np.inf
    f = lambda x: layout_score(x, n)

    if verbose:
        print(f"n={n}: optimizing layout", flush=True)

    feasible_boundary_starts = 0
    for i, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, boundary_starts, endpoint=False)):
        x0 = outer_circle_layout(n, phase)
        if not np.all(feasible_points(x0.reshape(n, 2))):
            continue

        feasible_boundary_starts += 1
        x, score = refine_layout_jax(x0, n)
        if score > best_score:
            best_x = x
            best_score = score

        if verbose:
            print(
                f"n={n}: boundary start {i + 1}/{boundary_starts}, "
                f"feasible={feasible_boundary_starts}, best={best_score:.6f}",
                flush=True,
            )

    if verbose:
        print(
            f"n={n}: boundary starts done "
            f"({feasible_boundary_starts}/{boundary_starts} feasible), best={best_score:.6f}",
            flush=True,
        )

    for i in range(restarts):
        if verbose:
            print(f"n={n}: CEM restart {i + 1}/{restarts} running", flush=True)
        x, score = cross_entropy(f, n, k_max, m, m_elite, rng)
        x, score = refine_layout_jax(x, n)
        if score > best_score:
            best_x = x
            best_score = score
        if verbose:
            print(f"n={n}: CEM restart {i + 1}/{restarts} done, best={best_score:.6f}", flush=True)

    return best_x, best_score

def solve_task_1(output_csv=None):
    if output_csv is None:
        output_csv = Path(__file__).with_name("portfolio_layout_results.csv")

    results = []
    layouts = {}

    for n in range(2, 11):
        print(f"starting n={n}", flush=True)
        x, score = optimize_layout(n, seed=100 + n)
        results.append((n, score))
        layouts[n] = x
        print(f"finished n={n}, p*={score:.6f}", flush=True)

    np.savetxt(output_csv, np.array(results), delimiter=",", fmt=["%d", "%.8f"])
    print(f"saved results to {output_csv}", flush=True)
    return results, layouts

def main():
    solve_task_1()

if __name__ == "__main__":
    main()
