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

# =========================================================
# Numerics helpers
# =========================================================
def safe_relative_gain(a: float, b: float, eps: float = 1e-12) -> float:
    """
    Relative improvement of b over baseline a:
        (b - a) / a         for observability-like metrics
        (a - b) / a         for error-like metrics
    This helper is used only for "larger is better" metrics.
    """
    denom = max(abs(a), eps)
    return (b - a) / denom


def safe_relative_error_improvement(err_base: float, err_new: float, eps: float = 1e-12) -> float:
    """
    Relative error improvement:
        (err_base - err_new)/err_base
    Positive => new estimator is better.
    """
    denom = max(abs(err_base), eps)
    return (err_base - err_new) / denom


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def discrete_observability_gramian(F: np.ndarray, H: np.ndarray, horizon: int) -> np.ndarray:
    """
    Finite-horizon discrete observability Gramian:
        W_o = sum_{k=0}^{N-1} (F^k)^T H^T H F^k
    """
    n = F.shape[0]
    Wo = np.zeros((n, n), dtype=float)
    Fk = np.eye(n)

    for _ in range(horizon):
        Wo += Fk.T @ H.T @ H @ Fk
        Fk = F @ Fk

    return symmetrize(Wo)


def phase_observability_metric(F: np.ndarray, H: np.ndarray, phase_index: int, horizon: int) -> float:
    """
    Phase-oriented observability metric:
        J_phi = e_phi^T W_o e_phi
              = sum_{k=0}^{N-1} || H F^k e_phi ||^2

    This directly measures how much the phase direction is visible
    in the output over a finite horizon.
    """
    n = F.shape[0]
    e_phi = np.zeros(n, dtype=float)
    e_phi[phase_index] = 1.0

    Fk = np.eye(n)
    J_phi = 0.0

    for _ in range(horizon):
        out = H @ (Fk @ e_phi)
        J_phi += float(np.dot(out.ravel(), out.ravel()))
        Fk = F @ Fk

    return J_phi


def gramian_metrics(F: np.ndarray, H: np.ndarray, horizon: int, phase_index: int):
    """
    Returns generic Gramian metrics + the phase-oriented metric J_phi.
    """
    Wo = discrete_observability_gramian(F, H, horizon)
    eigvals = np.linalg.eigvalsh(Wo)
    eigvals = np.maximum(eigvals, 0.0)

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    cond = np.inf if lam_min < 1e-14 else lam_max / lam_min
    J_phi = phase_observability_metric(F, H, phase_index=phase_index, horizon=horizon)

    return {
        "Wo": Wo,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": cond,
        "trace": float(np.trace(Wo)),
        "J_phi": J_phi,
    }


def effective_noise_proxy(Qd: np.ndarray) -> float:
    """
    Simple effective diffusion proxy.
    """
    return float(np.trace(Qd))


# =========================================================
# Generic simulation helper
# =========================================================
def simulate_kf(F, H, Qd, R, x_true0, x_hat0, P0, T, dt, seed):
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    n_state = F.shape[0]

    X_true = np.zeros((n_steps, n_state))
    X_est = np.zeros((n_steps, n_state))
    Y = np.zeros(n_steps)

    x_true = x_true0.copy()
    kf = DiscreteKalmanFilter(
        F=F,
        H=H,
        Qd=Qd,
        R=R,
        x0=x_hat0.copy(),
        P0=P0.copy(),
    )

    for k in range(n_steps):
        w_k = rng.multivariate_normal(np.zeros(n_state), Qd)
        x_true = F @ x_true + w_k

        v_k = rng.normal(0.0, np.sqrt(R[0, 0]))
        y_k = float((H @ x_true).item() + v_k)

        x_est = kf.step(np.array([y_k]))

        X_true[k, :] = x_true
        X_est[k, :] = x_est
        Y[k] = y_k

    return X_true, X_est, Y


# =========================================================
# Plant baseline
# =========================================================
def plant_metrics(
    theta,
    lam,
    gamma=1.0,
    omega=0.0,
    g_phi=2.0,
    kappa=1.0,
    dt=0.01,
    T=20.0,
    sigma_q=0.02,
    sigma_p=0.02,
    q_phi=0.01,
    meas_std=0.03,
    seed=1,
):
    A_p = build_Aa(gamma=gamma, omega=omega, g_phi=g_phi, lam=lam)
    C_p = build_Ca(kappa=kappa, theta=theta)

    Qc_p = build_continuous_process_covariance(
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        q_phi=q_phi,
    )
    F_p, Qd_p = discretize_plant(A_p, Qc_p, dt)

    R = np.array([[meas_std**2]], dtype=float)

    x_true0 = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(3, dtype=float)
    P0 = 10.0 * np.eye(3)

    X_true, X_est, Y = simulate_kf(F_p, C_p, Qd_p, R, x_true0, x_hat0, P0, T, dt, seed)

    n_steps = int(T / dt)
    burn = int(0.2 * n_steps)
    horizon = max(10, n_steps - burn)

    gram = gramian_metrics(F_p, C_p, horizon=horizon, phase_index=2)

    return {
        "F": F_p,
        "H": C_p,
        "Qd": Qd_p,
        "R": R,
        "X_true": X_true,
        "X_est": X_est,
        "Y": Y,
        "burn": burn,
        "rmse_phi": rmse(X_true[burn:, 2], X_est[burn:, 2]),
        "rmse_q": rmse(X_true[burn:, 0], X_est[burn:, 0]),
        "rmse_p": rmse(X_true[burn:, 1], X_est[burn:, 1]),
        "lambda_min": gram["lambda_min"],
        "lambda_max": gram["lambda_max"],
        "cond": gram["cond"],
        "Wo_trace": gram["trace"],
        "J_phi": gram["J_phi"],
        "noise_proxy": effective_noise_proxy(Qd_p),
    }


# =========================================================
# QCO case
# =========================================================
def qco_metrics(
    theta,
    lam,
    k_so,
    k_os,
    gamma_s=1.0,
    omega_s=0.0,
    gamma_o=1.2,
    omega_o=0.4,
    g_phi=2.0,
    dt=0.01,
    T=20.0,
    sigma_qs=0.02,
    sigma_ps=0.02,
    sigma_qo=0.02,
    sigma_po=0.02,
    q_phi=0.01,
    meas_std=0.03,
    c_qs=0.0,
    c_ps=1.0,
    c_qo=0.0,
    c_po=0.4,
    seed=1,
):
    A_q = build_qco_augmented_A(
        gamma_s=gamma_s,
        omega_s=omega_s,
        gamma_o=gamma_o,
        omega_o=omega_o,
        g_phi=g_phi,
        lam=lam,
        k_so=k_so,
        k_os=k_os,
    )

    H_q = build_qco_measurement(
        c_qs=c_qs,
        c_ps=c_ps,
        c_qo=c_qo,
        c_po=c_po,
    )

    Qc_q = build_qco_process_covariance(
        sigma_qs=sigma_qs,
        sigma_ps=sigma_ps,
        sigma_qo=sigma_qo,
        sigma_po=sigma_po,
        q_phi=q_phi,
    )
    F_q, Qd_q = discretize_qco(A_q, Qc_q, dt)

    R = np.array([[meas_std**2]], dtype=float)

    x_true0 = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(5, dtype=float)
    P0 = 10.0 * np.eye(5)

    X_true, X_est, Y = simulate_kf(F_q, H_q, Qd_q, R, x_true0, x_hat0, P0, T, dt, seed)

    n_steps = int(T / dt)
    burn = int(0.2 * n_steps)
    horizon = max(10, n_steps - burn)

    gram = gramian_metrics(F_q, H_q, horizon=horizon, phase_index=4)

    return {
        "F": F_q,
        "H": H_q,
        "Qd": Qd_q,
        "R": R,
        "X_true": X_true,
        "X_est": X_est,
        "Y": Y,
        "burn": burn,
        "rmse_phi": rmse(X_true[burn:, 4], X_est[burn:, 4]),
        "rmse_qs": rmse(X_true[burn:, 0], X_est[burn:, 0]),
        "rmse_ps": rmse(X_true[burn:, 1], X_est[burn:, 1]),
        "lambda_min": gram["lambda_min"],
        "lambda_max": gram["lambda_max"],
        "cond": gram["cond"],
        "Wo_trace": gram["trace"],
        "J_phi": gram["J_phi"],
        "noise_proxy": effective_noise_proxy(Qd_q),
    }


# =========================================================
# One comparison point
# =========================================================
def compare_plant_vs_qco(
    theta,
    lam,
    k_so,
    k_os,
    alpha_noise=0.2,
    **kwargs,
):
    plant = plant_metrics(theta=theta, lam=lam, **kwargs)
    qco = qco_metrics(theta=theta, lam=lam, k_so=k_so, k_os=k_os, **kwargs)

    delta_rmse = safe_relative_error_improvement(plant["rmse_phi"], qco["rmse_phi"])
    delta_Jphi = safe_relative_gain(plant["J_phi"], qco["J_phi"])

    noise_base = max(plant["noise_proxy"], 1e-12)
    delta_Q = (qco["noise_proxy"] - plant["noise_proxy"]) / noise_base

    score_phi = delta_Jphi - alpha_noise * delta_Q

    return {
        "plant": plant,
        "qco": qco,
        "delta_rmse": delta_rmse,
        "delta_Jphi": delta_Jphi,
        "delta_Q": delta_Q,
        "score_phi": score_phi,
    }


# =========================================================
# Regime map 1: theta vs lambda
# =========================================================
def regime_map_theta_lambda(
    k_so=0.8,
    k_os=0.8,
    theta_vals=None,
    lambda_vals=None,
    alpha_noise=0.2,
    save_dir="plots",
    verbose=True,
):
    os.makedirs(save_dir, exist_ok=True)

    if theta_vals is None:
        theta_vals = np.linspace(0.05, np.pi - 0.05, 50)

    if lambda_vals is None:
        lambda_vals = np.linspace(0.02, 1.5, 45)

    shape = (len(lambda_vals), len(theta_vals))

    delta_rmse_map = np.zeros(shape)
    delta_Jphi_map = np.zeros(shape)
    delta_Q_map = np.zeros(shape)
    score_phi_map = np.zeros(shape)

    rmse_baseline = np.zeros(shape)
    rmse_qco_map = np.zeros(shape)

    Jphi_plant = np.zeros(shape)
    Jphi_qco = np.zeros(shape)

    cond_plant = np.zeros(shape)
    cond_qco = np.zeros(shape)

    for i, lam in enumerate(lambda_vals):
        for j, theta in enumerate(theta_vals):
            comp = compare_plant_vs_qco(
                theta=theta,
                lam=lam,
                k_so=k_so,
                k_os=k_os,
                alpha_noise=alpha_noise,
            )

            delta_rmse_map[i, j] = comp["delta_rmse"]
            delta_Jphi_map[i, j] = comp["delta_Jphi"]
            delta_Q_map[i, j] = comp["delta_Q"]
            score_phi_map[i, j] = comp["score_phi"]

            rmse_baseline[i, j] = comp["plant"]["rmse_phi"]
            rmse_qco_map[i, j] = comp["qco"]["rmse_phi"]

            Jphi_plant[i, j] = comp["plant"]["J_phi"]
            Jphi_qco[i, j] = comp["qco"]["J_phi"]

            cond_plant[i, j] = comp["plant"]["cond"]
            cond_qco[i, j] = comp["qco"]["cond"]

            if verbose:
                print(
                    f"[theta-lambda] lam={lam:.3f}, theta={theta:.3f}, "
                    f"RMSE gain={comp['delta_rmse']:+.3%}, "
                    f"dJphi={comp['delta_Jphi']:+.3%}, "
                    f"dQ={comp['delta_Q']:+.3%}, "
                    f"score_phi={comp['score_phi']:+.3f}"
                )

    TH, LA = np.meshgrid(theta_vals, lambda_vals)

    # --- RMSE map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(TH, LA, delta_rmse_map, shading="auto")
    plt.colorbar(im, label="Relative improvement of QCO over plant KF")
    plt.contour(TH, LA, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"Homodyne angle $\theta$")
    plt.ylabel(r"OU rate $\lambda$")
    plt.title(r"QCO advantage map: $(\mathrm{RMSE}_{plant}-\mathrm{RMSE}_{QCO})/\mathrm{RMSE}_{plant}$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_qco.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_qco.pdf"))

    # --- Phase observability gain map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(TH, LA, delta_Jphi_map, shading="auto")
    plt.colorbar(im, label=r"Relative gain in phase observability $\Delta_{J_\phi}$")
    plt.contour(TH, LA, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"Homodyne angle $\theta$")
    plt.ylabel(r"OU rate $\lambda$")
    plt.title(r"Phase-observability gain map: $(J_\phi^{QCO}-J_\phi^{plant})/J_\phi^{plant}$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_phase_observability_gain.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_phase_observability_gain.pdf"))

    # --- Noise proxy map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(TH, LA, delta_Q_map, shading="auto")
    plt.colorbar(im, label="Relative increase in diffusion proxy")
    plt.contour(TH, LA, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"Homodyne angle $\theta$")
    plt.ylabel(r"OU rate $\lambda$")
    plt.title("QCO added-diffusion proxy map")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_noise_proxy.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_noise_proxy.pdf"))

    # --- Score map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(TH, LA, score_phi_map, shading="auto")
    plt.colorbar(im, label=r"Score $S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$")
    plt.contour(TH, LA, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"Homodyne angle $\theta$")
    plt.ylabel(r"OU rate $\lambda$")
    plt.title("Predictive phase-based QCO score map")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_score_phi.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_theta_lambda_score_phi.pdf"))

    return {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "delta_rmse": delta_rmse_map,
        "delta_Jphi": delta_Jphi_map,
        "delta_Q": delta_Q_map,
        "score_phi": score_phi_map,
        "rmse_baseline": rmse_baseline,
        "rmse_qco": rmse_qco_map,
        "Jphi_plant": Jphi_plant,
        "Jphi_qco": Jphi_qco,
        "cond_plant": cond_plant,
        "cond_qco": cond_qco,
    }


# =========================================================
# Regime map 2: coupling vs measurement noise
# =========================================================
def regime_map_coupling_noise(
    theta=np.pi / 2,
    lam=0.2,
    k_vals=None,
    meas_vals=None,
    alpha_noise=0.2,
    save_dir="plots",
    verbose=True,
):
    os.makedirs(save_dir, exist_ok=True)

    if k_vals is None:
        k_vals = np.linspace(0.0, 2.0, 41)

    if meas_vals is None:
        meas_vals = np.linspace(0.005, 0.12, 40)

    shape = (len(meas_vals), len(k_vals))

    delta_rmse_map = np.zeros(shape)
    delta_Jphi_map = np.zeros(shape)
    delta_Q_map = np.zeros(shape)
    score_phi_map = np.zeros(shape)

    for i, meas_std in enumerate(meas_vals):
        for j, k in enumerate(k_vals):
            comp = compare_plant_vs_qco(
                theta=theta,
                lam=lam,
                k_so=k,
                k_os=k,
                alpha_noise=alpha_noise,
                meas_std=meas_std,
            )

            delta_rmse_map[i, j] = comp["delta_rmse"]
            delta_Jphi_map[i, j] = comp["delta_Jphi"]
            delta_Q_map[i, j] = comp["delta_Q"]
            score_phi_map[i, j] = comp["score_phi"]

            if verbose:
                print(
                    f"[k-noise] meas={meas_std:.4f}, k={k:.3f}, "
                    f"RMSE gain={comp['delta_rmse']:+.3%}, "
                    f"dJphi={comp['delta_Jphi']:+.3%}, "
                    f"dQ={comp['delta_Q']:+.3%}, "
                    f"score_phi={comp['score_phi']:+.3f}"
                )

    KK, MM = np.meshgrid(k_vals, meas_vals)

    # --- RMSE map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(KK, MM, delta_rmse_map, shading="auto")
    plt.colorbar(im, label="Relative improvement of QCO over plant KF")
    plt.contour(KK, MM, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"QCO coupling $k_{so}=k_{os}$")
    plt.ylabel(r"Measurement noise std")
    plt.title("QCO advantage map versus coupling and measurement noise")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_qco.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_qco.pdf"))

    # --- Phase observability gain map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(KK, MM, delta_Jphi_map, shading="auto")
    plt.colorbar(im, label=r"Relative gain in phase observability $\Delta_{J_\phi}$")
    plt.contour(KK, MM, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"QCO coupling $k_{so}=k_{os}$")
    plt.ylabel(r"Measurement noise std")
    plt.title("Phase-observability gain versus coupling and measurement noise")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_phase_observability_gain.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_phase_observability_gain.pdf"))

    # --- Score map
    plt.figure(figsize=(9.2, 5.8))
    im = plt.pcolormesh(KK, MM, score_phi_map, shading="auto")
    plt.colorbar(im, label=r"Score $S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$")
    plt.contour(KK, MM, delta_rmse_map, levels=[0.0], linewidths=2.0, colors="purple")
    plt.xlabel(r"QCO coupling $k_{so}=k_{os}$")
    plt.ylabel(r"Measurement noise std")
    plt.title("Predictive phase-based score versus coupling and measurement noise")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_score_phi.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, "regime_map_coupling_noise_score_phi.pdf"))

    return {
        "k_vals": k_vals,
        "meas_vals": meas_vals,
        "delta_rmse": delta_rmse_map,
        "delta_Jphi": delta_Jphi_map,
        "delta_Q": delta_Q_map,
        "score_phi": score_phi_map,
    }


# =========================================================
# Scatter diagnostics
# =========================================================
def plot_scatter_tradeoff(result_dict, save_dir="plots", prefix="theta_lambda"):
    os.makedirs(save_dir, exist_ok=True)

    delta_rmse = result_dict["delta_rmse"].ravel()
    delta_Jphi = result_dict["delta_Jphi"].ravel()
    delta_Q = result_dict["delta_Q"].ravel()
    score_phi = result_dict["score_phi"].ravel()

    # Scatter 1: phase observability gain vs RMSE gain
    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(delta_Jphi, delta_rmse, s=18, alpha=0.7)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(r"$\Delta_{J_\phi}$")
    plt.ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    plt.title("RMSE gain vs phase observability gain")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_deltaJphi_vs_deltaRMSE.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_deltaJphi_vs_deltaRMSE.pdf"))

    # Scatter 2: noise proxy vs RMSE gain
    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(delta_Q, delta_rmse, s=18, alpha=0.7)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(r"$\Delta_Q$")
    plt.ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    plt.title("RMSE gain vs added-diffusion proxy")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_deltaQ_vs_deltaRMSE.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_deltaQ_vs_deltaRMSE.pdf"))

    # Scatter 3: score vs RMSE gain
    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(score_phi, delta_rmse, s=18, alpha=0.7)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(r"$S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$")
    plt.ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    plt.title("RMSE gain vs predictive phase-based score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_scorephi_vs_deltaRMSE.png"), dpi=400)
    plt.savefig(os.path.join(save_dir, f"{prefix}_scatter_scorephi_vs_deltaRMSE.pdf"))

    corr_dj = np.corrcoef(delta_Jphi, delta_rmse)[0, 1]
    corr_dq = np.corrcoef(delta_Q, delta_rmse)[0, 1]
    corr_score = np.corrcoef(score_phi, delta_rmse)[0, 1]

    print("\n=== Scatter diagnostics ===")
    print(f"corr(delta_Jphi, delta_RMSE) = {corr_dj:+.4f}")
    print(f"corr(delta_Q, delta_RMSE)    = {corr_dq:+.4f}")
    print(f"corr(score_phi, delta_RMSE)  = {corr_score:+.4f}")

    return {
        "corr_deltaJphi_deltaRMSE": corr_dj,
        "corr_deltaQ_deltaRMSE": corr_dq,
        "corr_scorephi_deltaRMSE": corr_score,
    }


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    SAVE_DIR = "plots"
    ALPHA_NOISE = 0.2

    res_theta_lambda = regime_map_theta_lambda(
        k_so=0.8,
        k_os=0.8,
        alpha_noise=ALPHA_NOISE,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    res_coupling_noise = regime_map_coupling_noise(
        theta=np.pi / 2,
        lam=0.2,
        alpha_noise=ALPHA_NOISE,
        save_dir=SAVE_DIR,
        verbose=True,
    )

    plot_scatter_tradeoff(res_theta_lambda, save_dir=SAVE_DIR, prefix="theta_lambda")
    plot_scatter_tradeoff(res_coupling_noise, save_dir=SAVE_DIR, prefix="coupling_noise")

    plt.show()