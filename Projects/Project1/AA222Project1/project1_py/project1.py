#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from project1_py.helpers import Simple1, Simple2, Simple3

# Rosebrock: 2D, 20 evals, parabolic valley
# Himmelblau: 2D, 40 evals, multiple minima
# Powell: 4D, 100 evals, singular Hessian at optimum
# Secret 1: 2D, 100 evals
# Secret 2: 2D, 400 evals, hints

def descent_step(f, g, x, n, count):
    if count() + 3 > n:
        return x, False

    g_x = g(x)
    f_x = f(x)
    g_norm = np.linalg.norm(g_x)

    if (not np.isfinite(g_norm)) or g_norm == 0.0:
        return x, False

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

def run_descent(
    f,
    g,
    x0,
    n,
    count,
    max_steps=None,
    record_history=False,
    record_values=False,
    record_counts=False,
):
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

def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    x_best, _, _, _ = run_descent(f, g, x0, n, count)
    return x_best

def optimize_with_history(f, g, x0, n, count, prob, max_steps=None):
    x_best, x_history, f_history, count_history = run_descent(
        f,
        g,
        x0,
        n,
        count,
        max_steps=max_steps,
        record_history=True,
        record_values=True,
        record_counts=True,
    )
    return x_best, np.array(x_history), np.array(f_history), np.array(count_history)

def generate_plots(output_dir="readme_plots", max_steps=None, starts=None, use_nolimit=False):

    os.makedirs(output_dir, exist_ok=True)
    if starts is None:
        starts = {
            "simple1": [
                np.array([-1.5, 2.0]),
                np.array([0.0, -1.0]),
                np.array([2.0, 2.0]),
            ],
            "simple2": [
                np.array([-3.0, 3.0]),
                np.array([3.0, -2.0]),
                np.array([0.5, 0.5]),
            ],
            "simple3": [
                np.array([3.0, -1.0, 0.0, 1.0]),
                np.array([-2.0, 2.0, -1.0, 2.0]),
                np.array([1.5, -2.0, 2.0, -1.0]),
            ],
        }

    problem_titles = {
        "simple1": "Rosenbrock",
        "simple2": "Himmelblau",
        "simple3": "Powell",
    }
    problem_classes = {
        "simple1": Simple1,
        "simple2": Simple2,
        "simple3": Simple3,
    }

    histories = {}
    values = {}

    for prob in ("simple1", "simple2", "simple3"):
        histories[prob] = []
        values[prob] = []
        problem_cls = problem_classes[prob]

        for x0 in starts[prob]:
            problem = problem_cls()
            if use_nolimit:
                problem.nolimit()
            _, x_history, f_history, count_history = optimize_with_history(
                problem.f,
                problem.g,
                np.array(x0, dtype=float),
                problem.n,
                problem.count,
                problem.prob,
                max_steps=max_steps,
            )
            histories[prob].append(x_history)
            values[prob].append((count_history, f_history))

    x1, x2 = np.meshgrid(np.linspace(-2.0, 2.0, 250), np.linspace(-1.0, 3.0, 250))
    z = 100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2
    plt.figure(figsize=(7, 6))
    levels = np.logspace(-1, 3, 18)
    plt.contour(x1, x2, z, levels=levels)
    for idx, x_history in enumerate(histories["simple1"], start=1):
        plt.plot(x_history[:, 0], x_history[:, 1], marker="o", linewidth=1.5, label=f"start {idx}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Rosenbrock path with contours")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rosenbrock_path.png"), dpi=200)
    plt.close()

    for prob in ("simple1", "simple2", "simple3"):
        plt.figure(figsize=(7, 5))
        for idx, (count_history, f_history) in enumerate(values[prob], start=1):
            plt.plot(count_history, f_history, marker="o", linewidth=1.5, label=f"start {idx}")
        plt.xlabel("evaluation count")
        plt.ylabel("f(x)")
        plt.yscale("log")
        plt.title(f"{problem_titles[prob]} convergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prob}_convergence.png"), dpi=200)
        plt.close()

    return histories, values
