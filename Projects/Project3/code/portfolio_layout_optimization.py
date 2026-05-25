import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scipy.optimize import minimize

ZONE_1 = np.array([5.0, 5.0])
INNER_RADIUS = 1.5
OUTER_RADIUS = 4.0
ZONE_2 = np.array([7.5, 2.5])
ZONE_3 = np.array([2.5, 7.5])
EXCLUSION_RADIUS = 1.2
LOG_FILE = Path(__file__).with_name("portfolio_layout_optimization.log")

def log(message):
    print(message, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(message + "\n")

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

def project_point(point, n_passes=5):
    point = point.copy()

    for _ in range(n_passes):
        for center, lower, upper in (
            (ZONE_1, INNER_RADIUS, OUTER_RADIUS),
            (ZONE_2, EXCLUSION_RADIUS, np.inf),
            (ZONE_3, EXCLUSION_RADIUS, np.inf),
        ):
            direction = point - center
            distance = np.linalg.norm(direction)
            if distance < 1e-12:
                direction = np.array([1.0, 0.0])
                distance = 1.0
            if distance < lower:
                point = center + direction / distance * lower
            elif distance > upper:
                point = center + direction / distance * upper

    return point

def project_layout(x, n):
    points = x.reshape(n, 2).copy()
    valid = feasible_points(points)
    for i in np.where(~valid)[0]:
        points[i] = project_point(points[i])
    return points.reshape(2 * n)

def repair_layout(x, n, rng):
    points = x.reshape(n, 2).copy()
    valid = feasible_points(points)
    if not np.all(valid):
        points[~valid] = sample_feasible_points(rng, np.sum(~valid))
    return points.reshape(2 * n)

def min_pairwise_distance(x, n=2):
    points = x.reshape(n, 2)
    i, j = np.triu_indices(n, k=1)
    return np.min(np.linalg.norm(points[i] - points[j], axis=1))

def layout_score(x, n):
    points = x.reshape(n, 2)
    if not np.all(feasible_points(points)):
        return -np.inf
    return min_pairwise_distance(x, n)

def slsqp_constraints(z, n):
    points = z[:-1].reshape((n, 2))
    p = z[-1]

    d1 = np.linalg.norm(points - ZONE_1, axis=1)
    d2 = np.linalg.norm(points - ZONE_2, axis=1)
    d3 = np.linalg.norm(points - ZONE_3, axis=1)

    constraints = [
        d1 - INNER_RADIUS,
        OUTER_RADIUS - d1,
        d2 - EXCLUSION_RADIUS,
        d3 - EXCLUSION_RADIUS,
    ]

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append([np.linalg.norm(points[i] - points[j]) - p])

    return np.concatenate(constraints)

def slsqp_obj(z):
    return -z[-1]

def refine_layout_slsqp(x0, n, max_steps=100):
    start_score = layout_score(x0, n)
    if start_score == -np.inf:
        return x0, -np.inf

    z0 = np.concatenate([x0.copy(), [start_score]])
    bounds = [(1.0, 9.0)] * (2 * n) + [(0.0, 8.0)]
    result = minimize(
        slsqp_obj,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": lambda z: slsqp_constraints(z, n)},
        options={"maxiter": max_steps, "ftol": 1e-9, "disp": False},
    )

    if not result.success:
        log(f"SLSQP failed: {result.message}; min_constraint={np.min(slsqp_constraints(result.x, n)):.3e}")
        return x0, start_score

    x = result.x[:-1]
    score = layout_score(x, n)
    if score == -np.inf:
        log(f"SLSQP returned infeasible layout; min_constraint={np.min(slsqp_constraints(result.x, n)):.3e}")
        return x0, start_score

    return x, score

def circle_points(center, radius, count):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

def annulus_ring_points(n_rings=6, points_per_ring=240):
    radii = np.linspace(INNER_RADIUS, OUTER_RADIUS, n_rings + 2)[1:-1]
    return np.vstack([circle_points(ZONE_1, radius, points_per_ring) for radius in radii])

def boundary_candidate_points(rng, points_per_circle=1000, interior_count=5000, n_rings=6, points_per_ring=240):
    candidates = np.vstack([
        circle_points(ZONE_1, OUTER_RADIUS, points_per_circle),
        circle_points(ZONE_1, INNER_RADIUS, points_per_circle),
        annulus_ring_points(n_rings, points_per_ring),
        circle_points(ZONE_2, EXCLUSION_RADIUS, points_per_circle),
        circle_points(ZONE_3, EXCLUSION_RADIUS, points_per_circle),
        sample_feasible_points(rng, interior_count),
    ])
    return candidates[feasible_points(candidates)]

def greedy_maxmin_layout(candidates, n, start_idx):
    selected = [candidates[start_idx]]
    available = np.ones(len(candidates), dtype=bool)
    available[start_idx] = False
    nearest_dist = np.linalg.norm(candidates - selected[0], axis=1)

    while len(selected) < n:
        masked_dist = np.where(available, nearest_dist, -np.inf)
        next_idx = int(np.argmax(masked_dist))
        selected.append(candidates[next_idx])
        available[next_idx] = False
        new_dist = np.linalg.norm(candidates - candidates[next_idx], axis=1)
        nearest_dist = np.minimum(nearest_dist, new_dist)

    return np.array(selected).reshape(2 * n)

def greedy_candidate_layouts(rng, n, n_starts=8, interior_anchor_fraction=0.8):
    candidates = boundary_candidate_points(rng)
    interior = sample_feasible_points(rng, max(4 * n_starts, 1000))
    interior_start = len(candidates)
    candidates = np.vstack([candidates, interior])

    if len(candidates) < n:
        return []

    anchors = [
        int(np.argmax(candidates[:, 0])),
        int(np.argmin(candidates[:, 0])),
        int(np.argmax(candidates[:, 1])),
        int(np.argmin(candidates[:, 1])),
    ]
    n_extra = max(0, n_starts - len(anchors))
    n_interior = int(np.ceil(interior_anchor_fraction * n_extra))
    n_global = n_extra - n_interior
    interior_idx = np.arange(interior_start, len(candidates))

    if n_interior:
        anchors.extend(rng.choice(interior_idx, size=n_interior, replace=False))
    if n_global:
        anchors.extend(rng.choice(len(candidates), size=n_global, replace=False))

    layouts = []
    for start_idx in anchors[:n_starts]:
        layout = greedy_maxmin_layout(candidates, n, int(start_idx))
        if np.all(feasible_points(layout.reshape(n, 2))):
            layouts.append(layout)

    return layouts

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

def cma_es(f, n, k_max, m=100, m_elite=None, rng=None, sigma0=2.0, x0=None):
    dim = 2 * n
    m_elite = m // 2 if m_elite is None else m_elite

    weights = np.log(m_elite + 0.5) - np.log(np.arange(1, m_elite + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff),
    )

    mean = random_feasible_layout(rng, n) if x0 is None else x0.copy()
    sigma = sigma0
    C = np.eye(dim)
    p_c = np.zeros(dim)
    p_sigma = np.zeros(dim)
    expected_norm = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    best_x = mean.copy()
    best_score = f(best_x)

    for generation in range(k_max):
        C = 0.5 * (C + C.T)
        eigenvalues, B = np.linalg.eigh(C)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        C_sqrt = B @ np.diag(np.sqrt(eigenvalues))
        C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(eigenvalues)) @ B.T

        z = rng.standard_normal((m, dim))
        y = z @ C_sqrt.T
        samples = mean + sigma * y
        samples = np.array([project_layout(sample, n) for sample in samples])
        scores = np.array([f(sample) for sample in samples])

        i_best = np.argmax(scores)
        if scores[i_best] > best_score:
            best_score = scores[i_best]
            best_x = samples[i_best].copy()

        elite_idx = np.argsort(scores)[-m_elite:][::-1]
        elite_samples = samples[elite_idx]
        old_mean = mean.copy()
        mean = weights @ elite_samples
        y_w = (mean - old_mean) / sigma

        p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
        norm_p_sigma = np.linalg.norm(p_sigma)
        h_sigma = norm_p_sigma / np.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (generation + 1))) < (
            1.4 + 2.0 / (dim + 1.0)
        ) * expected_norm
        p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w

        elite_y = (elite_samples - old_mean) / sigma
        rank_mu = sum(weights[i] * np.outer(elite_y[i], elite_y[i]) for i in range(m_elite))
        C = (
            (1.0 - c1 - c_mu + (1.0 - h_sigma) * c1 * c_c * (2.0 - c_c)) * C
            + c1 * np.outer(p_c, p_c)
            + c_mu * rank_mu
        )
        sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / expected_norm - 1.0))

    return best_x, best_score

def refine_all_layouts(layouts, n, best_x, best_score, label, slsqp_steps, verbose=True):
    for i, x0 in enumerate(layouts):
        start_score = layout_score(x0, n)
        if start_score == -np.inf:
            continue

        x, score = refine_layout_slsqp(x0, n, slsqp_steps)
        if score > best_score:
            best_x = x
            best_score = score

        if verbose:
            log(
                f"n={n}: {label} refine {i + 1}/{len(layouts)}, "
                f"start={start_score:.6f}, refined={score:.6f}, best={best_score:.6f}"
            )

    return best_x, best_score

def optimize_layout(n,restarts,k_max,m,m_elite,seed,greedy_starts,slsqp_steps,
                    verbose=True):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = -np.inf
    f = lambda x: layout_score(x, n)
    candidate_layouts = []
    cma_layouts = []

    if verbose:
        log(f"n={n}: optimizing layout")

    # Greedy max-min candidates cover outer, inner, exclusion, and interior samples.
    if verbose:
        log(f"n={n}: starting greedy candidate generation ({greedy_starts} starts)")
    greedy_layouts = greedy_candidate_layouts(rng, n, greedy_starts)
    candidate_layouts.extend(greedy_layouts)
    if verbose:
        log(f"n={n}: greedy candidate starts generated ({len(greedy_layouts)})")

    # Use the strongest deterministic candidates as CMA-ES starting means.
    ranked_starts = sorted(
        ((layout_score(x, n), x) for x in candidate_layouts),
        key=lambda item: item[0],
        reverse=True,
    )

    for i in range(restarts):
        x0 = ranked_starts[i][1] if i < len(ranked_starts) else None
        if verbose:
            source = "seeded" if x0 is not None else "random"
            log(f"n={n}: CMA-ES restart {i + 1}/{restarts} running ({source})")
        x, score = cma_es(f, n, k_max, m, m_elite, rng, x0=x0)
        candidate_layouts.append(x)
        cma_layouts.append(x)
        if score > best_score:
            best_x = x
            best_score = score
        if verbose:
            log(f"n={n}: CMA-ES restart {i + 1}/{restarts} done, best={best_score:.6f}")

    best_x, best_score = refine_all_layouts(
        cma_layouts,
        n,
        best_x,
        best_score,
        "CMA-ES SLSQP",
        slsqp_steps,
        verbose=verbose,
    )

    return best_x, best_score

def efficiency(distance):
    return 1.0 / (1.0 + 1.0 / distance)

def draw_constraints(ax):
    ax.add_patch(Circle(ZONE_1, OUTER_RADIUS, fill=False, color="green", linewidth=2.0, label="Outer boundary"))
    ax.add_patch(Circle(ZONE_1, INNER_RADIUS, fill=True, color="gray", alpha=0.25, label="Inner exclusion"))
    ax.add_patch(Circle(ZONE_2, EXCLUSION_RADIUS, fill=True, color="red", alpha=0.25, label="High-beta growth exclusion"))
    ax.add_patch(Circle(ZONE_3, EXCLUSION_RADIUS, fill=True, color="orange", alpha=0.25, label="Defensive value exclusion"))
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Market Beta Score")
    ax.set_ylabel("Value HML Score")
    ax.grid(True, alpha=0.25)

def save_layout_plot(n, x, score, output_dir):
    points = x.reshape(n, 2)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_constraints(ax)
    ax.scatter(points[:, 0], points[:, 1], s=70, color="black", zorder=5, label="Stocks")
    for i, point in enumerate(points, start=1):
        ax.annotate(str(i), point + np.array([0.08, 0.08]), fontsize=9)
    ax.set_title(f"Optimized Portfolio Layout, n={n}, p*={score:.4f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"layout_n{n}.png", dpi=200)
    plt.close(fig)

def save_distance_plot(results, output_dir):
    values = np.array(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(values[:, 0], values[:, 1], marker="o", color="black")
    ax.set_xlabel("Number of Stocks n")
    ax.set_ylabel("Minimum Pairwise Distance p*")
    ax.set_title("Minimum Pairwise Distance vs Portfolio Size")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(values[:, 0].astype(int))
    fig.tight_layout()
    fig.savefig(output_dir / "minimum_distance_vs_n.png", dpi=200)
    plt.close(fig)

def save_efficiency_csv(results, output_dir):
    values = np.array([(n, score, efficiency(score)) for n, score in results])
    np.savetxt(
        output_dir / "portfolio_efficiency_results.csv",
        values,
        delimiter=",",
        fmt=["%d", "%.8f", "%.8f"],
    )

def save_part1_outputs(results, layouts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for n in (3, 5, 7):
        if n in layouts:
            score = dict(results)[n]
            save_layout_plot(n, layouts[n], score, output_dir)
    save_distance_plot(results, output_dir)
    save_efficiency_csv(results, output_dir)

def solve_task_1():
    output_csv = Path(__file__).with_name("portfolio_layout_results.csv")
    output_dir = Path(__file__).with_name("part1_outputs")
    output_dir = Path(output_dir)

    existing = {
        int(row[0]): float(row[1])
        for row in np.atleast_2d(np.loadtxt(output_csv, delimiter=","))
    } if output_csv.exists() else {}

    # target_ns = ((9,10))
    # target_ns = range(2, 8)
    target_ns = (10,)
    results = [(n, existing[n]) for n in range(2, 11) if n in existing and n not in target_ns]
    layouts = {}
    
    args = dict(
        restarts=15,
        k_max=51,
        m=500,
        m_elite=100,
        greedy_starts=2000,
        slsqp_steps=200,
    )

    for n in target_ns:
        log(f"starting n={n}")
        x, score = optimize_layout(n, seed=100 + n, **args)
        results.append((n, score))
        layouts[n] = x
        log(f"finished n={n}, p*={score:.6f}")

    results.sort(key=lambda item: item[0])
    np.savetxt(output_csv, np.array(results), delimiter=",", fmt=["%d", "%.8f"])
    log(f"saved results to {output_csv}")
    save_part1_outputs(results, layouts, output_dir)
    log(f"saved plots and efficiency results to {output_dir}")
    return results, layouts

def main():
    LOG_FILE.write_text("")
    solve_task_1()

if __name__ == "__main__":
    main()
