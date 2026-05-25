from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaussian_process import fit_gp_predict, theoretical_efficiency

CAPITAL_PER_STOCK = 25_000_000.0
ALPHA_PER_STOCK = 0.04
COST_PER_STOCK = 550_000.0
X_TRAIN = np.array([2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.0, 8.0])
Y_TRAIN = np.array([0.59, 0.62, 0.82, 0.85, 0.86, 0.88, 0.90, 0.91])


def profit(n, efficiency):
    return n * efficiency * CAPITAL_PER_STOCK * ALPHA_PER_STOCK - n * COST_PER_STOCK


def load_layout_results(path):
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data)
    return data[:, 0].astype(int), data[:, 1]


def save_profit_outputs(output_dir, image_dir=None):
    base_dir = Path(__file__).resolve().parent
    n_values, distances = load_layout_results(base_dir / "portfolio_layout_results.csv")

    gp_mean, gp_std = fit_gp_predict(X_TRAIN, Y_TRAIN, distances)
    gp_lower = gp_mean - 1.96 * gp_std
    gp_upper = gp_mean + 1.96 * gp_std
    theory = theoretical_efficiency(distances)

    theory_profit = profit(n_values, theory)
    mean_profit = profit(n_values, gp_mean)
    lower_profit = profit(n_values, gp_lower)
    upper_profit = profit(n_values, gp_upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        n_values,
        lower_profit / 1e6,
        upper_profit / 1e6,
        color="tab:blue",
        alpha=0.18,
        label="95% confidence region",
    )
    ax.plot(n_values, mean_profit / 1e6, marker="o", color="tab:blue", linewidth=2.0, label="GP mean profit")
    ax.plot(n_values, theory_profit / 1e6, marker="s", color="black", linewidth=1.8, label="Theoretical profit")
    ax.plot(n_values, lower_profit / 1e6, linestyle="--", color="tab:blue", linewidth=1.2, label="95% lower bound")
    ax.plot(n_values, upper_profit / 1e6, linestyle="--", color="tab:orange", linewidth=1.2, label="95% upper bound")
    ax.set_xlabel("Number of stocks n")
    ax.set_ylabel("Expected yearly profit (million USD)")
    ax.set_title("Expected Profit vs. Portfolio Size")
    ax.set_xticks(n_values)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "expected_profit_vs_n.png"
    fig.savefig(plot_path, dpi=200)
    if image_dir is not None:
        fig.savefig(image_dir / "expected_profit_vs_n.png", dpi=200)
    plt.close(fig)

    results = np.column_stack([
        n_values,
        distances,
        theory,
        gp_mean,
        gp_std,
        theory_profit,
        mean_profit,
        lower_profit,
        upper_profit,
    ])
    np.savetxt(
        output_dir / "expected_profit_results.csv",
        results,
        delimiter=",",
        header="n,p_star,theory_eff,gp_mean_eff,gp_std_eff,theory_profit,gp_mean_profit,lower_95_profit,upper_95_profit",
        comments="",
    )

    summary = {
        "theory_n": int(n_values[np.argmax(theory_profit)]),
        "mean_n": int(n_values[np.argmax(mean_profit)]),
        "worst_n": int(n_values[np.argmax(lower_profit)]),
        "best_n": int(n_values[np.argmax(upper_profit)]),
        "theory_profit": float(np.max(theory_profit)),
        "mean_profit": float(np.max(mean_profit)),
        "worst_profit": float(np.max(lower_profit)),
        "best_profit": float(np.max(upper_profit)),
    }
    with (output_dir / "expected_profit_summary.txt").open("w") as f:
        for key, value in summary.items():
            f.write(f"{key},{value}\n")

    return summary


def main():
    base_dir = Path(__file__).resolve().parent
    latex_img_dir = base_dir.parent / "latex" / "img"
    summary = save_profit_outputs(base_dir / "part3_outputs", latex_img_dir)
    print(f"saved Part 3 outputs to {base_dir / 'part3_outputs'}")
    print(summary)


if __name__ == "__main__":
    main()
