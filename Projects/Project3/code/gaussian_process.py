from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LENGTH_SCALE = 4.0
NOISE_VARIANCE = 0.02 ** 2

def theoretical_efficiency(p):
    return 1.0 / (1.0 + 1.0 / p)


def squared_exponential_kernel(x1, x2, length_scale=LENGTH_SCALE):
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    return np.exp(-0.5 * ((x1 - x2) / length_scale) ** 2)

def fit_gp_predict(x_train, y_train, x_test):
    # from book (noisy measurements):
    #     X* -> x_test
    #     X  -> x_train
    #      y -> y_train
    #   we want y_hat, computed from predicted mean and covariance (posteriors)

    mean_train = theoretical_efficiency(x_train)    # m(X)
    mean_test = theoretical_efficiency(x_test)      # m(X*)

    k_pred = squared_exponential_kernel(x_test, x_test)         # K(X*,X*)
    k_pred_train = squared_exponential_kernel(x_test, x_train)  #K(X*,X)
    k_train = squared_exponential_kernel(x_train, x_train)      # K(X,X)

    noisy_covariance = k_train + NOISE_VARIANCE * np.eye(len(x_train))

    k_prod_inv = np.linalg.solve(noisy_covariance.T, k_pred_train.T).T

    posterior_mean = mean_test + k_prod_inv @ (y_train - mean_train)
    posterior_cov = k_pred - k_prod_inv @ k_pred_train.T
    posterior_variance = np.maximum(np.diag(posterior_cov), np.finfo(float).eps)

    return posterior_mean, np.sqrt(posterior_variance)

def save_gp_plot(output_dir):
    x_train = np.array([2.5, 2.5, 4.5, 4.5, 6.5, 6.5, 8.0, 8.0])
    y_train = np.array([0.59, 0.62, 0.82, 0.85, 0.86, 0.88, 0.90, 0.91])
    x_test = np.linspace(1.0, 10.0, 500)

    mean, std = fit_gp_predict(x_train, y_train, x_test)
    theory = theoretical_efficiency(x_test)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        x_test,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="tab:blue",
        alpha=0.18,
        label="95% confidence region",
    )
    ax.plot(x_test, mean, color="tab:blue", linewidth=2.0, label="Predicted GP posterior mean")
    ax.plot(x_test, theory, color="black", linestyle="-", linewidth=2.0, label="Theoretical efficiency over p")
    ax.scatter(x_train, y_train, color="tab:red", s=20, zorder=5, label="Simulation data")
    ax.set_xlabel("Separation distance p")
    ax.set_ylabel("Diversification efficiency")
    ax.set_title("Gaussian Process Fit to Realized Efficiency")
    ax.set_xlim(1.0, 10.0)
    ax.set_ylim(0.45, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gp_efficiency_fit.png", dpi=200)
    plt.close(fig)

    predictions = np.column_stack([x_test, mean, std, mean - 1.96 * std, mean + 1.96 * std])
    np.savetxt(
        output_dir / "gp_efficiency_predictions.csv",
        predictions,
        delimiter=",",
        header="p,mean,std,lower_95,upper_95",
        comments="",
    )


def main():
    output_dir = Path(__file__).with_name("part2_outputs")
    save_gp_plot(output_dir)
    print(f"saved Part 2 GP outputs to {output_dir}")


if __name__ == "__main__":
    main()
