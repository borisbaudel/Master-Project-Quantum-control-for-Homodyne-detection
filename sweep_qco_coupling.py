import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from core.kalman import DiscreteKalmanFilter
from core.model import (
    build_Aa,
    build_Ca,
    build_continuous_process_covariance,
    discretize_system_van_loan as discretize_plant,
    rmse,
)
from core.qco_model import (
    build_qco_augmented_A,
    build_qco_measurement,
    build_qco_process_covariance,
    discretize_system_van_loan as discretize_qco,
)
from scipy.linalg import solve_discrete_are


def design_discrete_luenberger_gain(F: np.ndarray, H: np.ndarray, Qe=None, Re=None) -> np.ndarray:
    n = F.shape[0]
    if Qe is None:
        Qe = np.eye(n)
    if Re is None:
        Re = np.array([[1.0]], dtype=float)

    P = solve_discrete_are(F.T, H.T, Qe, Re)
    S = H @ P @ H.T + Re
    L = P @ H.T @ np.linalg.inv(S)
    return L


def simulate_linear_observer(F, H, Qd, R, x_true0, x_hat0, P0, T, dt, seed, L=None):
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    n_state = F.shape[0]

    X_true = np.zeros((n_steps, n_state))
    X_est = np.zeros((n_steps, n_state))

    x_true = x_true0.copy()
    x_est = x_hat0.copy()

    kf = None
    if L is None:
        kf = DiscreteKalmanFilter(F=F, H=H, Qd=Qd, R=R, x0=x_hat0, P0=P0)

    for k in range(n_steps):
        w_k = rng.multivariate_normal(np.zeros(n_state), Qd)
        x_true = F @ x_true + w_k

        v_k = rng.normal(0.0, np.sqrt(R[0, 0]))
        y_k = float((H @ x_true).item() + v_k)

        if kf is not None:
            x_est = kf.step(np.array([y_k]))
        else:
            innovation = y_k - float((H @ x_est).item())
            x_est = F @ x_est + L.flatten() * innovation

        X_true[k, :] = x_true
        X_est[k, :] = x_est

    return X_true, X_est


def run_plant_baseline(T=20.0, dt=0.01, seed=1):
    R = np.array([[0.03**2]], dtype=float)

    A_p = build_Aa(gamma=1.0, omega=0.0, g_phi=2.0, lam=0.2)
    C_p = build_Ca(kappa=1.0, theta=np.pi / 2)
    Qc_p = build_continuous_process_covariance(
        sigma_q=0.02,
        sigma_p=0.02,
        q_phi=0.01,
    )
    F_p, Qd_p = discretize_plant(A_p, Qc_p, dt)

    x_true0_p = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0_p = np.zeros(3, dtype=float)
    P0_p = 10.0 * np.eye(3)

    Qe_p = np.diag([1.0, 2.0, 20.0])
    L_p = design_discrete_luenberger_gain(F_p, C_p, Qe=Qe_p, Re=np.array([[1.0]]))

    X_true_pk, X_est_pk = simulate_linear_observer(
        F_p, C_p, Qd_p, R, x_true0_p, x_hat0_p, P0_p, T, dt, seed, L=None
    )
    X_true_pl, X_est_pl = simulate_linear_observer(
        F_p, C_p, Qd_p, R, x_true0_p, x_hat0_p, P0_p, T, dt, seed, L=L_p
    )

    burn = int(0.2 * int(T / dt))
    rmse_pk = rmse(X_true_pk[burn:, 2], X_est_pk[burn:, 2])
    rmse_pl = rmse(X_true_pl[burn:, 2], X_est_pl[burn:, 2])

    return rmse_pk, rmse_pl


def run_qco_case(k_so, k_os, T=20.0, dt=0.01, seed=1):
    R = np.array([[0.03**2]], dtype=float)

    A_q = build_qco_augmented_A(
        gamma_s=1.0,
        omega_s=0.0,
        gamma_o=1.2,
        omega_o=0.4,
        g_phi=2.0,
        lam=0.2,
        k_so=k_so,
        k_os=k_os,
    )
    H_q = build_qco_measurement(
        c_qs=0.0,
        c_ps=1.0,
        c_qo=0.0,
        c_po=0.4,
    )
    Qc_q = build_qco_process_covariance(
        sigma_qs=0.02,
        sigma_ps=0.02,
        sigma_qo=0.02,
        sigma_po=0.02,
        q_phi=0.01,
    )
    F_q, Qd_q = discretize_qco(A_q, Qc_q, dt)

    x_true0_q = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0_q = np.zeros(5, dtype=float)
    P0_q = 10.0 * np.eye(5)

    Qe_q = np.diag([1.0, 2.0, 1.0, 2.0, 20.0])
    L_q = design_discrete_luenberger_gain(F_q, H_q, Qe=Qe_q, Re=np.array([[1.0]]))

    X_true_qk, X_est_qk = simulate_linear_observer(
        F_q, H_q, Qd_q, R, x_true0_q, x_hat0_q, P0_q, T, dt, seed, L=None
    )
    X_true_ql, X_est_ql = simulate_linear_observer(
        F_q, H_q, Qd_q, R, x_true0_q, x_hat0_q, P0_q, T, dt, seed, L=L_q
    )

    burn = int(0.2 * int(T / dt))
    rmse_qk = rmse(X_true_qk[burn:, 4], X_est_qk[burn:, 4])
    rmse_ql = rmse(X_true_ql[burn:, 4], X_est_ql[burn:, 4])

    return rmse_qk, rmse_ql


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    T = 20.0
    dt = 0.01
    seed = 1

    rmse_pk, rmse_pl = run_plant_baseline(T=T, dt=dt, seed=seed)

    k_vals = np.linspace(0.0, 2.0, 21)
    rmse_qk_vals = []
    rmse_ql_vals = []

    for k in k_vals:
        rmse_qk, rmse_ql = run_qco_case(k, k, T=T, dt=dt, seed=seed)
        rmse_qk_vals.append(rmse_qk)
        rmse_ql_vals.append(rmse_ql)

    rmse_qk_vals = np.array(rmse_qk_vals)
    rmse_ql_vals = np.array(rmse_ql_vals)

    print(f"Plant + Kalman baseline     : {rmse_pk:.6f}")
    print(f"Plant + Luenberger baseline : {rmse_pl:.6f}")
    print(f"Best QCO + Kalman           : {rmse_qk_vals.min():.6f} at k={k_vals[np.argmin(rmse_qk_vals)]:.3f}")
    print(f"Best QCO + Luenberger       : {rmse_ql_vals.min():.6f} at k={k_vals[np.argmin(rmse_ql_vals)]:.3f}")

    plt.figure(figsize=(8.4, 4.8))
    plt.plot(k_vals, rmse_qk_vals, linewidth=2.2, label="QCO + Kalman")
    plt.plot(k_vals, rmse_ql_vals, linewidth=2.2, label="QCO + Luenberger")
    plt.axhline(rmse_pk, linestyle="--", linewidth=2, label="Plant + Kalman baseline")
    plt.axhline(rmse_pl, linestyle="--", linewidth=2, label="Plant + Luenberger baseline")
    plt.xlabel(r"QCO coupling $k$  (with $k_{so}=k_{os}=k$)")
    plt.ylabel(r"RMSE($\phi$)")
    plt.title("Effect of QCO coupling on estimation performance")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sweep_qco_coupling.png", dpi=400)
    plt.savefig("plots/sweep_qco_coupling.pdf")
    plt.show()