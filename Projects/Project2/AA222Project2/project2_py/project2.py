#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

def c_flat(c, x):
    return np.asarray(c(x), dtype=float).reshape(-1)

def descent_step(f, g, x, n, count):
    if count() >= n:
        return x, False

    f_x = f(x)
    g_x = g(x)

    g_norm = np.linalg.norm(g_x)

    d = -g_x / g_norm

    alpha = 1.0
    p = 0.5
    beta = 1e-4

    while count() < n:
        x_trial = x + alpha * d
        f_trial = f(x_trial)
        if f_trial <= f_x + beta * alpha * (g_x.T @ d):
            return x_trial, True
        alpha *= p

    return x, False

def minimize(x0, f, g, n, count,
             max_steps=None,record_history=False,
             record_values=False,record_counts=False):
    x_best = x0.copy()
    x_history = [x_best.copy()] if record_history else None
    f_history = [f(x_best)] if record_values else None
    count_history = [count()] if record_counts else None
    steps = 0

    while count() < n:
        if max_steps is not None and steps >= max_steps:
            break

        x_new, moved = descent_step(f, g, x_best, n, count)
        if not moved:
            break

        x_best = x_new
        steps += 1

        if record_history:
            x_history.append(x_best.copy())
        if record_values:
            f_history.append(f(x_best))
        if record_counts:
            count_history.append(count())

    return x_best, x_history, f_history, count_history

def objective(x, f, c, rho1, rho2, n, count):
    if count() + 2 > n:
        return np.inf
    
    c_x = c_flat(c, x)
    p_count = np.sum(c_x > 0.0)
    p_quad = np.sum(np.maximum(c_x, 0.0) ** 2)
    return f(x) + rho1 * p_count + rho2 * p_quad

def grad_objective(x, f, g, c, rho2, n, count):
    if count() + 3 + len(x) > n:
        return None

    g_x = g(x)
    c_x = c_flat(c, x)
    m = np.maximum(c_x, 0.0)

    eps = 1e-6
    grad_c = np.zeros((len(c_x), len(x)))
    for j in range(len(x)):
        x_eps = x.copy()
        x_eps[j] += eps
        grad_c[:, j] = (c_flat(c, x_eps) - c_x) / eps

    return g_x + 2.0 * rho2 * (grad_c.T @ m)

def penalty_method(x0, f, g, c, n, count, rho1=1.0, rho2=10.0, gamma=4.0):
    x = x0.copy()
    best_x = x.copy()
    best_violation = np.inf

    while count() + 8 + len(x) <= n:
        x, _, _, _ = minimize(x, lambda z:objective(z, f, c, rho1, rho2, n, count),
                                 lambda z:grad_objective(z, f, g, c, rho2, n, count),
                                 n, count, max_steps=5)

        if count() + 1 > n:
            break

        c_x = c_flat(c, x)
        max_violation = np.max(np.maximum(c_x, 0.0))
        if max_violation < best_violation:
            best_x = x.copy()
            best_violation = max_violation

        if max_violation <= 0.0:
            return x

        rho1 *= gamma
        rho2 *= gamma

    return best_x

def optimize_with_history(f, g, c, x0, n, count, prob, max_steps=None):
    x_best = penalty_method(x0, f, g, c, n, count)
    return x_best, np.array([x0.copy(), x_best.copy()]), np.array([]), np.array([])

def optimize(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_best = x0
    
    x_best = penalty_method(x0,f,g,c,n,count)
    
    return x_best


def _plot_run(problem_cls, x0, rho1, rho2, gamma, outer_steps=10, inner_steps=1):
    p = problem_cls()
    p.nolimit()
    x = x0.astype(float).copy()
    x_history = [x.copy()]
    f_history = [p.f(x)]
    violation_history = [np.max(np.maximum(c_flat(p.c, x), 0.0))]

    for _ in range(outer_steps):
        x, _, _, _ = minimize(
            x,
            lambda z, rr1=rho1, rr2=rho2: objective(z, p.f, p.c, rr1, rr2, p.n, p.count),
            lambda z, rr2=rho2: grad_objective(z, p.f, p.g, p.c, rr2, p.n, p.count),
            p.n,
            p.count,
            max_steps=inner_steps,
            record_history=True,
        )

        x_history.append(x.copy())
        f_history.append(p.f(x))
        violation_history.append(np.max(np.maximum(c_flat(p.c, x), 0.0)))

        rho1 *= gamma
        rho2 *= gamma

    return np.array(x_history), np.array(f_history), np.array(violation_history)


def _plot_grid(prob):
    xs = np.linspace(-3.0, 3.0, 350)
    ys = np.linspace(-3.0, 3.0, 350)
    X, Y = np.meshgrid(xs, ys)

    if prob == "simple1":
        F = -X * Y + 2.0 / (3.0 * np.sqrt(3.0))
        C1 = X + Y**2 - 1.0
        C2 = -X - Y
    elif prob == "simple2":
        F = 100.0 * (Y - X**2) ** 2 + (1.0 - X) ** 2
        C1 = (X - 1.0) ** 3 - Y + 1.0
        C2 = X + Y - 2.0
    else:
        raise ValueError("Only simple1 and simple2 have 2D plotting grids.")

    feasible = (C1 <= 0.0) & (C2 <= 0.0)
    return X, Y, F, feasible


def generate_plots(output_dir="readme_plots"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from project2_py.helpers import Simple1, Simple2

    os.makedirs(output_dir, exist_ok=True)

    starts = {
        "simple1": [
            np.array([1.6, 1.6]),
            np.array([0.2, 1.4]),
            np.array([1.2, 0.1]),
        ],
        "simple2": [
            np.array([-1.5, 1.5]),
            np.array([0.0, 0.0]),
            np.array([1.6, -1.0]),
        ],
    }
    problems = {
        "simple1": Simple1,
        "simple2": Simple2,
    }
    algorithms = {
        "mixed": {
            "label": "Mixed penalty",
            "rho1": 1.0,
            "rho2": 10.0,
            "gamma": 4.0,
        },
        "quadratic": {
            "label": "Quadratic penalty",
            "rho1": 0.0,
            "rho2": 10.0,
            "gamma": 4.0,
        },
    }

    all_paths = {}
    for prob, problem_cls in problems.items():
        X, Y, F, feasible = _plot_grid(prob)
        all_paths[prob] = {}

        for alg_key, alg in algorithms.items():
            paths = []
            f_values = []
            violations = []

            for x0 in starts[prob]:
                x_history, f_history, violation_history = _plot_run(
                    problem_cls,
                    x0,
                    alg["rho1"],
                    alg["rho2"],
                    alg["gamma"],
                )
                paths.append(x_history)
                f_values.append(f_history)
                violations.append(violation_history)

            all_paths[prob][alg_key] = (paths, f_values, violations)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.contourf(
                X,
                Y,
                feasible.astype(float),
                levels=[-0.1, 0.5, 1.1],
                colors=["white", "#d9ead3"],
                alpha=0.75,
            )
            levels = np.linspace(np.nanpercentile(F, 5), np.nanpercentile(F, 90), 18)
            ax.contour(X, Y, F, levels=levels, colors="0.35", linewidths=0.6)

            for idx, x_history in enumerate(paths, start=1):
                ax.plot(
                    x_history[:, 0],
                    x_history[:, 1],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.2,
                    label=f"start {idx}",
                )
                ax.scatter(x_history[0, 0], x_history[0, 1], s=35, marker="s")
                ax.scatter(x_history[-1, 0], x_history[-1, 1], s=45, marker="*")

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_title(f"{prob}: {alg['label']}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{prob}_{alg_key}_path.png", dpi=200)
            plt.close(fig)

    simple2_xlim = (0, 10)

    for alg_key, alg in algorithms.items():
        _, f_values, violations = all_paths["simple2"][alg_key]

        fig, ax = plt.subplots(figsize=(6, 4))
        for idx, f_history in enumerate(f_values, start=1):
            ax.plot(
                np.arange(len(f_history)),
                f_history,
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                label=f"start {idx}",
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("$f(x)$")
        ax.set_yscale("log")
        ax.set_xlim(simple2_xlim)
        ax.set_title(f"simple2 objective: {alg['label']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/simple2_{alg_key}_objective.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for idx, violation_history in enumerate(violations, start=1):
            ax.plot(
                np.arange(len(violation_history)),
                violation_history,
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                label=f"start {idx}",
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("max constraint violation")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlim(simple2_xlim)
        ax.set_title(f"simple2 violation: {alg['label']}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/simple2_{alg_key}_violation.png", dpi=200)
        plt.close(fig)

    return all_paths
