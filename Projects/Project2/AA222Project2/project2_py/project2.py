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

def descent_step(f,g,x,n,count):
    if count() + 3 > n:
        return x, False

    g_norm = np.linalg.norm(g)

    if (not np.isfinite(g_norm)) or g_norm == 0.0:
        return x, False

    d = -g / g_norm

    alpha = 1.0
    p = 0.5
    beta = 1e-4

    while count() < n:
        x_trial = x + alpha * d
        f_trial = f(x_trial)
        if f_trial <= f + beta * alpha * (g.T @ d):
            return x_trial, True
        alpha *= p

    return x, False

def minimize(x0,f,g,n,count,
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

def objective(x,f,c,rho1,rho2):
    c_x = c(x)
    p_count = np.sum(c_x > 0.0)
    p_quad = np.sum(np.maximum(c_x, 0.0) ** 2)
    return f(x) + rho1 * p_count + rho2 * p_quad

def grad_objective(x,f,g,c,rho2,n,count):
    g_x = g(x)
    c_x = c(x)

    # if np.all(v == 0.0):
    #     return g_x

    eps = 1e-6
    grad_c = np.zeros((len(c_x), len(x)))
    for j in range(len(x)):
        if count() >= n:
            break
        x_eps = x.copy()
        x_eps[j] += eps
        grad_c[:, j] = (c(x_eps) - c_x) / eps

    return g_x + 2.0*rho2*(grad_c.T @ np.maximum(c_x, 0.0))

def penalty_method(x0,f,g,c,n,count,rho1=1.0,rho2=1.0,gamma=2.0):
    x = x0.copy()

    while count() < n:
        x, _, _, _ = minimize(x0,
                              lambda: objective(x,f,c,rho1,rho2),
                              lambda: grad_objective(x,f,g,c,rho2,n,count),n,count)

        if np.all(c(x) <= 0.0):
            return x

        rho1 *= gamma
        rho2 *= gamma

    return x

def optimize_with_history(x0,f,g,n,count,prob,max_steps=None):
    x_best,x_history,f_history,count_history = penalty_method(f,g,c,x0,
        n,
        count,
        max_steps=max_steps,
        record_history=True,
        record_values=True,
        record_counts=True,
    )
    return x_best, np.array(x_history), np.array(f_history), np.array(count_history)

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