import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from core.model import (
    build_Aa,
    build_Ca,
    build_continuous_process_covariance,
    discretize_system_van_loan,
    rmse,
)
from core.kalman import DiscreteKalmanFilter


def simulate_system(
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
    """
    Simulate:
      - true state trajectory X_k = [q_k, p_k, phi_k]
      - scalar homodyne measurements y_k
      - Kalman estimate Xhat_k

    The phase phi follows an OU-like linear dynamics through the 3rd state:
        phi_dot = -lam * phi + noise
    """
    rng = np.random.default_rng(seed)

    # Continuous-time model
    A = build_Aa(gamma=gamma, omega=omega, g_phi=g_phi, lam=lam)
    C = build_Ca(kappa=kappa, theta=theta)

    # Continuous process covariance density
    Qc = build_continuous_process_covariance(
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        q_phi=q_phi,
    )

    # Exact discrete-time conversion
    F, Qd = discretize_system_van_loan(A, Qc, dt)

    # Measurement covariance
    R = np.array([[meas_std**2]], dtype=float)

    # Simulation lengths
    n_steps = int(T / dt)
    n_state = 3

    # Storage
    X_true = np.zeros((n_steps, n_state), dtype=float)
    X_est = np.zeros((n_steps, n_state), dtype=float)
    Y = np.zeros(n_steps, dtype=float)

    # True initial condition
    x_true = np.array([0.0, 0.0, 0.5], dtype=float)

    # Filter initial condition
    x0_est = np.zeros(n_state, dtype=float)
    P0 = 10.0 * np.eye(n_state, dtype=float)

    kf = DiscreteKalmanFilter(
        F=F,
        H=C,
        Qd=Qd,
        R=R,
        x0=x0_est,
        P0=P0,
    )

    for k in range(n_steps):
        # True process propagation
        w_k = rng.multivariate_normal(mean=np.zeros(n_state), cov=Qd)
        x_true = F @ x_true + w_k

        # Measurement
        v_k = rng.normal(loc=0.0, scale=meas_std)
        y_k = float((C @ x_true).item() + v_k)

        # Kalman estimate
        x_est = kf.step(np.array([y_k]))

        # Store
        X_true[k, :] = x_true
        X_est[k, :] = x_est
        Y[k] = y_k

    t = np.arange(n_steps) * dt

    # Ignore initial transient
    burn_in = int(0.2 * n_steps)

    results = {
        "t": t,
        "X_true": X_true,
        "X_est": X_est,
        "Y": Y,
        "phi_rmse": rmse(X_true[burn_in:, 2], X_est[burn_in:, 2]),
        "q_rmse": rmse(X_true[burn_in:, 0], X_est[burn_in:, 0]),
        "p_rmse": rmse(X_true[burn_in:, 1], X_est[burn_in:, 1]),
        "burn_in": burn_in,
        "params": {
            "gamma": gamma,
            "omega": omega,
            "g_phi": g_phi,
            "lam": lam,
            "kappa": kappa,
            "theta": theta,
            "dt": dt,
            "T": T,
            "sigma_q": sigma_q,
            "sigma_p": sigma_p,
            "q_phi": q_phi,
            "meas_std": meas_std,
            "seed": seed,
        },
    }
    return results


def plot_results(results: dict, save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    t = results["t"]
    X_true = results["X_true"]
    X_est = results["X_est"]
    Y = results["Y"]
    burn_in = results["burn_in"]

    phi_rmse = results["phi_rmse"]
    q_rmse = results["q_rmse"]
    p_rmse = results["p_rmse"]

    # -------------------------
    # Phase plot
    # -------------------------
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, X_true[:, 2], label="true phi")
    plt.plot(t, X_est[:, 2], "--", label="estimated phi")
    plt.axvline(t[burn_in], linestyle=":", label="burn-in end")
    plt.xlabel("Time")
    plt.ylabel(r"$\phi$")
    plt.title(fr"Phase estimation, post-burn-in RMSE = {phi_rmse:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "phi_estimation.png"), dpi=300)

    # -------------------------
    # Quadratures plot
    # -------------------------
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, X_true[:, 0], label="true q")
    plt.plot(t, X_est[:, 0], "--", label="estimated q")
    plt.plot(t, X_true[:, 1], label="true p")
    plt.plot(t, X_est[:, 1], "--", label="estimated p")
    plt.axvline(t[burn_in], linestyle=":", label="burn-in end")
    plt.xlabel("Time")
    plt.ylabel("Quadratures")
    plt.title(fr"Quadrature estimation, q RMSE = {q_rmse:.4f}, p RMSE = {p_rmse:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "quadratures_estimation.png"), dpi=300)

    # -------------------------
    # Measurement plot
    # -------------------------
    plt.figure(figsize=(10, 3.8))
    plt.plot(t, Y)
    plt.axvline(t[burn_in], linestyle=":")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("Homodyne measurement record")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "measurement_record.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    results = simulate_system(
        gamma=1.0,
        omega=0.0,
        g_phi=2.0,
        lam=0.2,
        kappa=1.0,
        theta=np.pi / 2,
        dt=0.01,
        T=20.0,
        sigma_q=0.02,
        sigma_p=0.02,
        q_phi=0.01,
        meas_std=0.03,
        seed=1,
    )

    print("Post-burn-in phi RMSE =", results["phi_rmse"])
    print("Post-burn-in q RMSE   =", results["q_rmse"])
    print("Post-burn-in p RMSE   =", results["p_rmse"])

    plot_results(results)