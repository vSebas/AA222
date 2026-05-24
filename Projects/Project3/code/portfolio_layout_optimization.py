import numpy as np
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def project_point(point, n_passes=5):
    point = point.copy()
    centers = (ZONE_2, ZONE_3)

    for _ in range(n_passes):
        direction = point - ZONE_1
        distance = np.linalg.norm(direction)
        if distance < 1e-12:
            direction = np.array([1.0, 0.0])
            distance = 1.0

        if distance < INNER_RADIUS:
            point = ZONE_1 + direction / distance * INNER_RADIUS
        elif distance > OUTER_RADIUS:
            point = ZONE_1 + direction / distance * OUTER_RADIUS

        for center in centers:
            direction = point - center
            distance = np.linalg.norm(direction)
            if distance < 1e-12:
                direction = np.array([1.0, 0.0])
                distance = 1.0
            if distance < EXCLUSION_RADIUS:
                point = center + direction / distance * EXCLUSION_RADIUS

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

def refine_layout_jax(x0, n, max_steps=100):
    z = jnp.array(np.concatenate([x0.copy(), [min_pairwise_distance(x0, n)]]))
    bounds = jnp.array([[1.0, 9.0]] * (2 * n) + [[0.0, 8.0]])
    n_ineq = 4 * n + n * (n - 1) // 2

    solver = SLSQP(
        ineq_constraint_fn=jax_ineq_constraints,
        n_ineq_constraints=n_ineq,
        bounds=bounds,
        config=SLSQPConfig(tolerance=ToleranceConfig(rtol=1e-8, atol=1e-8, max_steps=max_steps)),
    )

    try:
        sol = optx.minimise(jax_obj, solver, z, args=n, has_aux=True, max_steps=max_steps)
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

def circle_points(center, radius, count):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])

def boundary_candidate_points(rng, points_per_circle=1000, interior_count=5000):
    candidates = [
        circle_points(ZONE_1, OUTER_RADIUS, points_per_circle),
        circle_points(ZONE_1, INNER_RADIUS, points_per_circle),
        circle_points(ZONE_2, EXCLUSION_RADIUS, points_per_circle),
        circle_points(ZONE_3, EXCLUSION_RADIUS, points_per_circle),
        sample_feasible_points(rng, interior_count),
    ]

    candidates = np.vstack(candidates)
    candidates = candidates[feasible_points(candidates)]
    return candidates

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

def greedy_candidate_layouts(rng, n, n_starts=8):
    candidates = boundary_candidate_points(rng)
    if len(candidates) < n:
        return []

    anchors = [
        int(np.argmax(candidates[:, 0])),
        int(np.argmin(candidates[:, 0])),
        int(np.argmax(candidates[:, 1])),
        int(np.argmin(candidates[:, 1])),
    ]
    if n_starts > len(anchors):
        anchors.extend(rng.choice(len(candidates), size=n_starts - len(anchors), replace=False))

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

def cma_es(f, n, k_max, m=100, m_elite=None, rng=None, sigma0=2.0):
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

    mean = random_feasible_layout(rng, n)
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

def refine_top_layouts(layouts, n, best_x, best_score, max_refines, label, slsqp_steps, verbose=True):
    if not layouts or max_refines <= 0:
        return best_x, best_score

    scored = [(layout_score(x, n), x) for x in layouts]
    scored = [(score, x) for score, x in scored if score != -np.inf]
    scored.sort(key=lambda item: item[0], reverse=True)

    for i, (_, x0) in enumerate(scored[:max_refines]):
        x, score = refine_layout_jax(x0, n, slsqp_steps)
        if score > best_score:
            best_x = x
            best_score = score
        if verbose:
            print(f"n={n}: {label} refine {i + 1}/{min(max_refines, len(scored))}, best={best_score:.6f}", flush=True)

    return best_x, best_score

def optimize_layout(
    n,
    restarts=2,
    k_max=15,
    m=120,
    m_elite=30,
    seed=0,
    boundary_starts=8,
    greedy_starts=4,
    final_refines=2,
    slsqp_steps=100,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = -np.inf
    f = lambda x: layout_score(x, n)
    candidate_layouts = []

    if verbose:
        print(f"n={n}: optimizing layout", flush=True)

    # Deterministic boundary candidates are cheap to score and often near active constraints.
    feasible_boundary_starts = 0
    projected_boundary_starts = 0
    for i, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, boundary_starts, endpoint=False)):
        x0 = outer_circle_layout(n, phase)
        was_feasible = np.all(feasible_points(x0.reshape(n, 2)))
        if not was_feasible:
            x0 = project_layout(x0, n)
            if not np.all(feasible_points(x0.reshape(n, 2))):
                continue
            projected_boundary_starts += 1

        if not np.all(feasible_points(x0.reshape(n, 2))):
            continue

        feasible_boundary_starts += 1
        candidate_layouts.append(x0)

        if verbose:
            status = "feasible" if was_feasible else "projected"
            print(
                f"n={n}: boundary start {i + 1}/{boundary_starts}, "
                f"{status}, accepted={feasible_boundary_starts}",
                flush=True,
            )

    if verbose:
        print(
            f"n={n}: boundary starts done "
            f"({feasible_boundary_starts}/{boundary_starts} accepted, "
            f"{projected_boundary_starts} projected)",
            flush=True,
        )

    # Greedy max-min candidates cover outer, inner, exclusion, and interior samples.
    greedy_layouts = greedy_candidate_layouts(rng, n, greedy_starts)
    candidate_layouts.extend(greedy_layouts)
    if verbose:
        print(f"n={n}: greedy candidate starts generated ({len(greedy_layouts)})", flush=True)

    # CMA-ES provides stochastic global search; SLSQP is applied only to top candidates.
    for i in range(restarts):
        if verbose:
            print(f"n={n}: CMA-ES restart {i + 1}/{restarts} running", flush=True)
        x, score = cma_es(f, n, k_max, m, m_elite, rng)
        candidate_layouts.append(x)
        if score > best_score:
            best_x = x
            best_score = score
        if verbose:
            print(f"n={n}: CMA-ES restart {i + 1}/{restarts} done, best={best_score:.6f}", flush=True)

    best_x, best_score = refine_top_layouts(
        candidate_layouts,
        n,
        best_x,
        best_score,
        final_refines,
        "final SLSQP",
        slsqp_steps,
        verbose,
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
    ax.set_ylabel("Optimized Minimum Pairwise Distance p*")
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

def solve_task_1(output_csv=None, output_dir=None, optimizer_kwargs=None):
    if output_csv is None:
        output_csv = Path(__file__).with_name("portfolio_layout_results.csv")
    if output_dir is None:
        output_dir = Path(__file__).with_name("part1_outputs")
    output_dir = Path(output_dir)

    results = []
    layouts = {}
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    for n in range(2, 11):
        print(f"starting n={n}", flush=True)
        x, score = optimize_layout(n, seed=100 + n, **optimizer_kwargs)
        results.append((n, score))
        layouts[n] = x
        print(f"finished n={n}, p*={score:.6f}", flush=True)

    np.savetxt(output_csv, np.array(results), delimiter=",", fmt=["%d", "%.8f"])
    print(f"saved results to {output_csv}", flush=True)
    save_part1_outputs(results, layouts, output_dir)
    print(f"saved plots and efficiency results to {output_dir}", flush=True)
    return results, layouts

def main():
    optimizer_kwargs = dict(
        restarts=5,
        k_max=30,
        m=350,
        m_elite=50,
        boundary_starts=24,
        greedy_starts=8,
        final_refines=4,
        slsqp_steps=150,
    )
    solve_task_1(optimizer_kwargs=optimizer_kwargs)

if __name__ == "__main__":
    main()
