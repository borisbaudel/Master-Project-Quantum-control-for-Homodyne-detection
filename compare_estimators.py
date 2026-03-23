import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

from core.model import (
    build_Aa,
    build_Ca,
    build_continuous_process_covariance,
    discretize_system_van_loan,
    rmse,
)
from core.kalman import DiscreteKalmanFilter

def design_discrete_luenberger_gain(F: np.ndarray, H: np.ndarray, Qe=None, Re=None) -> np.ndarray:
    """
    Design a constant-gain observer using the dual discrete Riccati equation.
    This is a stable Luenberger-type gain, but not the same as the stochastic
    Kalman gain because Qe and Re are designer-chosen weights.
    """
    n = F.shape[0]

    if Qe is None:
        Qe = np.eye(n)

    if Re is None:
        Re = np.array([[1.0]], dtype=float)

    P = solve_discrete_are(F.T, H.T, Qe, Re)
    S = H @ P @ H.T + Re
    L = P @ H.T @ np.linalg.inv(S)
    return L


def simulate_kalman_and_luenberger(
    gamma: float = 1.0,
    omega: float = 0.0,
    g_phi: float = 2.0,
    lam: float = 0.2,
    kappa: float = 1.0,
    theta: float = np.pi / 2,
    dt: float = 0.01,
    T: float = 20.0,
    sigma_q: float = 0.02,
    sigma_p: float = 0.02,
    q_phi: float = 0.01,
    meas_std: float = 0.03,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    A = build_Aa(gamma=gamma, omega=omega, g_phi=g_phi, lam=lam)
    C = build_Ca(kappa=kappa, theta=theta)

    Qc = build_continuous_process_covariance(
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        q_phi=q_phi,
    )

    F, Qd = discretize_system_van_loan(A, Qc, dt)
    R = np.array([[meas_std**2]], dtype=float)

    n_steps = int(T / dt)
    n_state = 3
    t = np.arange(n_steps) * dt

    X_true = np.zeros((n_steps, n_state), dtype=float)
    X_kf = np.zeros((n_steps, n_state), dtype=float)
    X_luen = np.zeros((n_steps, n_state), dtype=float)
    Y = np.zeros(n_steps, dtype=float)

    x_true = np.array([0.0, 0.0, 0.5], dtype=float)

    kf = DiscreteKalmanFilter(
        F=F,
        H=C,
        Qd=Qd,
        R=R,
        x0=np.zeros(n_state, dtype=float),
        P0=10.0 * np.eye(n_state, dtype=float),
    )
    Qe = np.diag([1.0, 2.0, 20.0])   # emphasize phi reconstruction
    Re_design = np.array([[1.0]], dtype=float)
    L = design_discrete_luenberger_gain(F, C, Qe=Qe, Re=Re_design)

    x_luen = np.zeros(n_state, dtype=float)

    for k in range(n_steps):
        w_k = rng.multivariate_normal(np.zeros(n_state), Qd)
        x_true = F @ x_true + w_k

        v_k = rng.normal(0.0, meas_std)
        y_k = float((C @ x_true).item() + v_k)

        # Kalman
        x_kf = kf.step(np.array([y_k]))

        # Stable discrete Luenberger
        innovation_l = y_k - float((C @ x_luen).item())
        x_luen = F @ x_luen + (L.flatten() * innovation_l)

        X_true[k, :] = x_true
        X_kf[k, :] = x_kf
        X_luen[k, :] = x_luen
        Y[k] = y_k

    burn_in = int(0.2 * n_steps)

    results = {
        "t": t,
        "X_true": X_true,
        "X_kf": X_kf,
        "X_luen": X_luen,
        "Y": Y,
        "burn_in": burn_in,
        "L": L,
        "rmse_phi_kf": rmse(X_true[burn_in:, 2], X_kf[burn_in:, 2]),
        "rmse_phi_luen": rmse(X_true[burn_in:, 2], X_luen[burn_in:, 2]),
        "rmse_q_kf": rmse(X_true[burn_in:, 0], X_kf[burn_in:, 0]),
        "rmse_q_luen": rmse(X_true[burn_in:, 0], X_luen[burn_in:, 0]),
        "rmse_p_kf": rmse(X_true[burn_in:, 1], X_kf[burn_in:, 1]),
        "rmse_p_luen": rmse(X_true[burn_in:, 1], X_luen[burn_in:, 1]),
    }
    return results


def plot_comparison(results: dict, save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    t = results["t"]
    burn_in = results["burn_in"]

    X_true = results["X_true"]
    X_kf = results["X_kf"]
    X_luen = results["X_luen"]

    plt.figure(figsize=(10, 4.8))
    plt.plot(t, X_true[:, 2], label="true phi")
    plt.plot(t, X_kf[:, 2], "--", label=f"Kalman, RMSE={results['rmse_phi_kf']:.4f}")
    plt.plot(t, X_luen[:, 2], ":", label=f"Luenberger, RMSE={results['rmse_phi_luen']:.4f}")
    plt.axvline(t[burn_in], linestyle=":", color="k", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(r"$\phi$")
    plt.title("Phase estimation: Kalman vs Luenberger")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_estimators_phi.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "compare_estimators_phi.pdf"))

    plt.figure(figsize=(6.5, 4.2))
    labels = ["Kalman", "Luenberger"]
    values = [results["rmse_phi_kf"], results["rmse_phi_luen"]]
    plt.bar(labels, values)
    plt.ylabel(r"RMSE($\phi$)")
    plt.title("Phase estimation error comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_estimators_bar.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "compare_estimators_bar.pdf"))

    plt.show()


if __name__ == "__main__":
    results = simulate_kalman_and_luenberger(seed=1)

    print(f"Observer gain L:\n{results['L']}")
    print(f"Post-burn-in RMSE phi - Kalman     : {results['rmse_phi_kf']:.6f}")
    print(f"Post-burn-in RMSE phi - Luenberger : {results['rmse_phi_luen']:.6f}")

    plot_comparison(results)