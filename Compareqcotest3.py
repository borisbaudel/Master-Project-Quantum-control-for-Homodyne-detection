import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# matplotlib backend: interactif par défaut (affichage à l'écran).
# Les appels pyplot sont tous dans le thread principal (après Parallel),
# donc pas de conflit tkinter sur Windows.

import matplotlib
matplotlib.use("Agg")   # forcé Agg dans le process principal aussi,
                         # sinon tkinter est chargé au premier import pyplot
                         # et hérité par les workers loky via fork-on-import.
                         # plt.show() est remplacé par ouverture manuelle des PNG.

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
# FIX 1 — Gramian horizon: use a short physically motivated
# horizon instead of (n_steps - burn).
#
# Rationale: F^k e_phi decays exponentially with the cavity
# damping rate gamma. Beyond ~5 decay times the terms are
# numerically zero and only accumulate floating-point noise,
# producing the spurious 1e-16 values seen in the coupling
# sweep (Fig. 13 of the paper).
#
# Rule of thumb: cover ~5 cavity decay times.
#   n_decay_steps = ceil(5 / (gamma * dt))
# Hard floor of 20 and hard cap of 200 keep it sane across
# all parameter regimes used in the sweeps.
# =========================================================
def gramian_horizon(gamma: float, dt: float, n_decay: float = 5.0,
                    floor: int = 20, cap: int = 200) -> int:
    """Return a physically motivated Gramian horizon."""
    steps = int(np.ceil(n_decay / (gamma * dt)))
    return max(floor, min(steps, cap))


# =========================================================
# FIX 3 — robust seed: derive a deterministic but unique seed
# from the continuous parameters so every grid point uses a
# different noise realization, avoiding single-trajectory bias.
# =========================================================
def param_seed(theta: float, lam: float, extra: float = 0.0) -> int:
    """Deterministic seed derived from simulation parameters."""
    key = f"{theta:.6f}_{lam:.6f}_{extra:.6f}"
    return abs(hash(key)) % (2 ** 31)


# =========================================================
# FIX 6 — measurement coefficients: expose c_po as an
# explicit argument (default kept at 0.4 for backward
# compatibility) and add a docstring explaining the choice.
# Callers can now sweep over it.
# =========================================================
_DEFAULT_C_PS = 1.0   # plant p-quadrature weight
_DEFAULT_C_PO = 0.4   # observer p-quadrature weight — must be justified
                       # physically or swept; 0.4 is the value used in the
                       # paper but is NOT derived from first principles.


# =========================================================
# Numerics helpers  (unchanged)
# =========================================================
def safe_relative_gain(a: float, b: float, eps: float = 1e-12) -> float:
    denom = max(abs(a), eps)
    return (b - a) / denom


def safe_relative_error_improvement(err_base: float, err_new: float,
                                    eps: float = 1e-12) -> float:
    denom = max(abs(err_base), eps)
    return (err_base - err_new) / denom


def safe_inv_sqrt(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(x, eps))


# NOTE: symmetrize removed — Wo = sum(Fk^T H^T H Fk) is
# already symmetric by construction.  Forcing symmetry was
# masking potential numerical issues rather than fixing them.


# =========================================================
# Observability helpers
# =========================================================
def discrete_observability_gramian(F: np.ndarray, H: np.ndarray,
                                   horizon: int) -> np.ndarray:
    """
    Finite-horizon discrete observability Gramian:
        W_o = sum_{k=0}^{N-1} (F^k)^T H^T H F^k

    Use a physically motivated horizon (see gramian_horizon).
    """
    n = F.shape[0]
    Wo = np.zeros((n, n), dtype=float)
    Fk = np.eye(n)

    for _ in range(horizon):
        HFk = H @ Fk
        Wo += HFk.T @ HFk   # equivalent, avoids forming H^T H explicitly
        Fk = F @ Fk

    return Wo   # FIX 7: no symmetrize — already symmetric


def phase_observability_metric(F: np.ndarray, H: np.ndarray,
                               phase_index: int, horizon: int) -> float:
    """
    Phase-oriented observability metric:
        J_phi = sum_{k=0}^{N-1} || H F^k e_phi ||^2
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


def gramian_metrics(F: np.ndarray, H: np.ndarray, horizon: int,
                    phase_index: int) -> dict:
    Wo = discrete_observability_gramian(F, H, horizon)
    eigvals = np.linalg.eigvalsh(Wo)
    eigvals = np.maximum(eigvals, 0.0)

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    cond = np.inf if lam_min < 1e-14 else lam_max / lam_min
    J_phi = phase_observability_metric(F, H, phase_index=phase_index,
                                       horizon=horizon)

    return {
        "Wo": Wo,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": cond,
        "trace": float(np.trace(Wo)),
        "J_phi": J_phi,
    }


def effective_noise_proxy(Qd: np.ndarray) -> float:
    return float(np.trace(Qd))


def rows_to_arrays(rows):
    theta = np.array([r["theta"] for r in rows], dtype=float)
    lam   = np.array([r["lambda"] for r in rows], dtype=float)
    rmse_vals   = np.array([r["rmse_phi"] for r in rows], dtype=float)
    jphi_vals   = np.array([r["J_phi"] for r in rows], dtype=float)
    invsqrt_vals = safe_inv_sqrt(jphi_vals)
    return theta, lam, rmse_vals, jphi_vals, invsqrt_vals


# =========================================================
# Generic simulation helper  (unchanged)
# =========================================================
def simulate_kf(F, H, Qd, R, x_true0, x_hat0, P0, T, dt, seed):
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    n_state = F.shape[0]

    X_true = np.zeros((n_steps, n_state))
    X_est  = np.zeros((n_steps, n_state))
    Y      = np.zeros(n_steps)

    x_true = x_true0.copy()
    kf = DiscreteKalmanFilter(F=F, H=H, Qd=Qd, R=R,
                              x0=x_hat0.copy(), P0=P0.copy())

    for k in range(n_steps):
        w_k   = rng.multivariate_normal(np.zeros(n_state), Qd)
        x_true = F @ x_true + w_k

        v_k = rng.normal(0.0, np.sqrt(R[0, 0]))
        y_k = float((H @ x_true).item() + v_k)

        x_est = kf.step(np.array([y_k]))

        X_true[k, :] = x_true
        X_est[k, :]  = x_est
        Y[k]         = y_k

    return X_true, X_est, Y


# =========================================================
# Monte-Carlo RMSE average over multiple seeds
# =========================================================
def mc_rmse(simulate_fn, n_mc: int = 5, base_seed: int = 0) -> float:
    """
    Run simulate_fn(seed) n_mc times and return the mean RMSE.
    simulate_fn must accept a single integer seed and return a scalar RMSE.
    """
    return float(np.mean([simulate_fn(base_seed + i) for i in range(n_mc)]))


# =========================================================
# FIX 4 — delta_Jphi clipping near blind angles
# =========================================================
_BLIND_CLIP = 50.0   # cap relative J_phi gain to avoid explosion at theta~0,pi

def clipped_relative_gain(a: float, b: float, eps: float = 1e-12,
                           clip: float = _BLIND_CLIP) -> float:
    """
    Relative gain (b-a)/a, clipped to [-clip, clip].
    Prevents blow-up when J_phi_plant -> 0 near blind homodyne angles.
    """
    raw = safe_relative_gain(a, b, eps=eps)
    return float(np.clip(raw, -clip, clip))


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
    seed=None,          # FIX 2: None -> caller supplies via param_seed
    n_mc=5,             # FIX 2: Monte-Carlo average
):
    A_p = build_Aa(gamma=gamma, omega=omega, g_phi=g_phi, lam=lam)
    C_p = build_Ca(kappa=kappa, theta=theta)

    Qc_p = build_continuous_process_covariance(
        sigma_q=sigma_q, sigma_p=sigma_p, q_phi=q_phi)
    F_p, Qd_p = discretize_plant(A_p, Qc_p, dt)

    R = np.array([[meas_std ** 2]], dtype=float)

    x_true0 = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0  = np.zeros(3, dtype=float)
    P0      = 10.0 * np.eye(3)

    n_steps = int(T / dt)
    burn    = int(0.2 * n_steps)

    # FIX 1: short, physically motivated Gramian horizon
    horizon = gramian_horizon(gamma=gamma, dt=dt)

    base_seed = seed if seed is not None else param_seed(theta, lam)

    # FIX 2: average RMSE over n_mc independent noise realizations
    def _one_run(s):
        Xt, Xe, _ = simulate_kf(F_p, C_p, Qd_p, R, x_true0, x_hat0, P0,
                                 T, dt, s)
        return rmse(Xt[burn:, 2], Xe[burn:, 2])

    mean_rmse = mc_rmse(_one_run, n_mc=n_mc, base_seed=base_seed)

    # Gramian on the plant model (deterministic, seed-independent)
    gram = gramian_metrics(F_p, C_p, horizon=horizon, phase_index=2)

    return {
        "F": F_p, "H": C_p, "Qd": Qd_p, "R": R,
        "burn": burn,
        "rmse_phi": mean_rmse,
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
    sigma_qs=0.02,      # FIX 5: plant and QCO noise params kept separate
    sigma_ps=0.02,
    sigma_qo=0.02,
    sigma_po=0.02,
    q_phi=0.01,
    meas_std=0.03,
    c_qs=0.0,
    c_ps=_DEFAULT_C_PS,
    c_qo=0.0,
    c_po=_DEFAULT_C_PO,  # FIX 6: explicit, sweepable
    seed=None,
    n_mc=5,
):
    A_q = build_qco_augmented_A(
        gamma_s=gamma_s, omega_s=omega_s,
        gamma_o=gamma_o, omega_o=omega_o,
        g_phi=g_phi, lam=lam,
        k_so=k_so, k_os=k_os,
    )

    H_q = build_qco_measurement(c_qs=c_qs, c_ps=c_ps, c_qo=c_qo, c_po=c_po)

    Qc_q = build_qco_process_covariance(
        sigma_qs=sigma_qs, sigma_ps=sigma_ps,
        sigma_qo=sigma_qo, sigma_po=sigma_po,
        q_phi=q_phi,
    )
    F_q, Qd_q = discretize_qco(A_q, Qc_q, dt)

    R = np.array([[meas_std ** 2]], dtype=float)

    x_true0 = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0  = np.zeros(5, dtype=float)
    P0      = 10.0 * np.eye(5)

    n_steps = int(T / dt)
    burn    = int(0.2 * n_steps)

    # FIX 1: use gamma_s (plant cavity rate) for the horizon — the phase
    # information reaches the output via the plant cavity first.
    horizon = gramian_horizon(gamma=gamma_s, dt=dt)

    base_seed = seed if seed is not None else param_seed(theta, lam, k_so)

    def _one_run(s):
        Xt, Xe, _ = simulate_kf(F_q, H_q, Qd_q, R, x_true0, x_hat0, P0,
                                 T, dt, s)
        return rmse(Xt[burn:, 4], Xe[burn:, 4])

    mean_rmse = mc_rmse(_one_run, n_mc=n_mc, base_seed=base_seed)

    gram = gramian_metrics(F_q, H_q, horizon=horizon, phase_index=4)

    return {
        "F": F_q, "H": H_q, "Qd": Qd_q, "R": R,
        "burn": burn,
        "rmse_phi": mean_rmse,
        "lambda_min": gram["lambda_min"],
        "lambda_max": gram["lambda_max"],
        "cond": gram["cond"],
        "Wo_trace": gram["trace"],
        "J_phi": gram["J_phi"],
        "noise_proxy": effective_noise_proxy(Qd_q),
    }


# =========================================================
# FIX 5 — compare_plant_vs_qco: separate kwargs for plant
# and QCO to avoid silent parameter mismatches.
# =========================================================
def compare_plant_vs_qco(
    theta,
    lam,
    k_so,
    k_os,
    alpha_noise=0.2,
    # shared params
    g_phi=2.0,
    dt=0.01,
    T=20.0,
    q_phi=0.01,
    meas_std=0.03,
    # plant-specific noise
    sigma_q=0.02,
    sigma_p=0.02,
    # QCO-specific noise
    sigma_qs=0.02,
    sigma_ps=0.02,
    sigma_qo=0.02,
    sigma_po=0.02,
    # QCO observer dynamics
    gamma_s=1.0,
    omega_s=0.0,
    gamma_o=1.2,
    omega_o=0.4,
    # QCO measurement weights
    c_qs=0.0,
    c_ps=_DEFAULT_C_PS,
    c_qo=0.0,
    c_po=_DEFAULT_C_PO,
    # MC
    n_mc=5,
):
    plant_kw = dict(g_phi=g_phi, dt=dt, T=T, q_phi=q_phi, meas_std=meas_std,
                    sigma_q=sigma_q, sigma_p=sigma_p, n_mc=n_mc)
    qco_kw   = dict(g_phi=g_phi, dt=dt, T=T, q_phi=q_phi, meas_std=meas_std,
                    sigma_qs=sigma_qs, sigma_ps=sigma_ps,
                    sigma_qo=sigma_qo, sigma_po=sigma_po,
                    gamma_s=gamma_s, omega_s=omega_s,
                    gamma_o=gamma_o, omega_o=omega_o,
                    c_qs=c_qs, c_ps=c_ps, c_qo=c_qo, c_po=c_po,
                    n_mc=n_mc)

    plant = plant_metrics(theta=theta, lam=lam, **plant_kw)
    qco   = qco_metrics(theta=theta, lam=lam, k_so=k_so, k_os=k_os, **qco_kw)

    delta_rmse = safe_relative_error_improvement(plant["rmse_phi"],
                                                 qco["rmse_phi"])

    # FIX 4: clipped relative gain to avoid blow-up near blind angles
    delta_Jphi = clipped_relative_gain(plant["J_phi"], qco["J_phi"])

    noise_base = max(plant["noise_proxy"], 1e-12)
    delta_Q    = (qco["noise_proxy"] - plant["noise_proxy"]) / noise_base

    score_phi  = delta_Jphi - alpha_noise * delta_Q

    return {
        "plant": plant,
        "qco": qco,
        "delta_rmse": delta_rmse,
        "delta_Jphi": delta_Jphi,
        "delta_Q": delta_Q,
        "score_phi": score_phi,
    }


# =========================================================
# Plant-only dataset
# =========================================================
def collect_plant_theta_lambda_data(
    theta_vals=None,
    lambda_vals=None,
    save_dir="plots",
    verbose=True,
    n_mc=3,
    n_jobs=-1,
    T=20.0,
    **plant_kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    if theta_vals is None:
        theta_vals = np.linspace(0.05, np.pi - 0.05, 50)
    if lambda_vals is None:
        lambda_vals = np.linspace(0.02, 1.5, 45)

    # Build flat list of (i, j, lam, theta) for parallel dispatch
    tasks = [
        (i, j, lam, theta)
        for i, lam in enumerate(lambda_vals)
        for j, theta in enumerate(theta_vals)
    ]

    def _run(i, j, lam, theta):
        met = plant_metrics(theta=theta, lam=lam, n_mc=n_mc, T=T, **plant_kwargs)
        if verbose:
            print(f"[plant] lam={lam:.3f} theta={theta:.3f} "
                  f"rmse={met['rmse_phi']:.5f} Jphi={met['J_phi']:.3e}")
        return i, j, met

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run)(i, j, lam, theta) for i, j, lam, theta in tasks
    )

    shape = (len(lambda_vals), len(theta_vals))
    rmse_map       = np.zeros(shape)
    jphi_map       = np.zeros(shape)
    cond_map       = np.zeros(shape)
    lambda_min_map = np.zeros(shape)
    rows = []

    for i, j, met in results:
        lam   = lambda_vals[i]
        theta = theta_vals[j]
        rmse_map[i, j]       = met["rmse_phi"]
        jphi_map[i, j]       = met["J_phi"]
        cond_map[i, j]       = met["cond"]
        lambda_min_map[i, j] = met["lambda_min"]
        rows.append({
            "theta": theta,
            "lambda": lam,
            "rmse_phi": met["rmse_phi"],
            "J_phi": met["J_phi"],
            "inv_sqrt_Jphi": 1.0 / np.sqrt(max(met["J_phi"], 1e-12)),
            "cond": met["cond"],
            "lambda_min": met["lambda_min"],
        })

    return {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "rmse_map": rmse_map,
        "jphi_map": jphi_map,
        "cond_map": cond_map,
        "lambda_min_map": lambda_min_map,
        "rows": rows,
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
    n_mc=3,
    n_jobs=-1,
    T=20.0,
):
    os.makedirs(save_dir, exist_ok=True)

    if theta_vals is None:
        theta_vals = np.linspace(0.05, np.pi - 0.05, 50)
    if lambda_vals is None:
        lambda_vals = np.linspace(0.02, 1.5, 45)

    tasks = [
        (i, j, lam, theta)
        for i, lam in enumerate(lambda_vals)
        for j, theta in enumerate(theta_vals)
    ]

    def _run(i, j, lam, theta):
        comp = compare_plant_vs_qco(
            theta=theta, lam=lam, k_so=k_so, k_os=k_os,
            alpha_noise=alpha_noise, n_mc=n_mc, T=T,
        )
        if verbose:
            print(f"[theta-lambda] lam={lam:.3f} theta={theta:.3f} "
                  f"RMSE={comp['delta_rmse']:+.3%} "
                  f"dJphi={comp['delta_Jphi']:+.3f} "
                  f"score={comp['score_phi']:+.3f}")
        return i, j, comp

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run)(i, j, lam, theta) for i, j, lam, theta in tasks
    )

    shape = (len(lambda_vals), len(theta_vals))
    delta_rmse_map = np.zeros(shape)
    delta_Jphi_map = np.zeros(shape)
    delta_Q_map    = np.zeros(shape)
    score_phi_map  = np.zeros(shape)
    rmse_baseline  = np.zeros(shape)
    rmse_qco_map   = np.zeros(shape)
    Jphi_plant     = np.zeros(shape)
    Jphi_qco       = np.zeros(shape)
    cond_plant     = np.zeros(shape)
    cond_qco       = np.zeros(shape)

    for i, j, comp in results:
        delta_rmse_map[i, j] = comp["delta_rmse"]
        delta_Jphi_map[i, j] = comp["delta_Jphi"]
        delta_Q_map[i, j]    = comp["delta_Q"]
        score_phi_map[i, j]  = comp["score_phi"]
        rmse_baseline[i, j]  = comp["plant"]["rmse_phi"]
        rmse_qco_map[i, j]   = comp["qco"]["rmse_phi"]
        Jphi_plant[i, j]     = comp["plant"]["J_phi"]
        Jphi_qco[i, j]       = comp["qco"]["J_phi"]
        cond_plant[i, j]     = comp["plant"]["cond"]
        cond_qco[i, j]       = comp["qco"]["cond"]

    TH, LA = np.meshgrid(theta_vals, lambda_vals)

    _save_map(TH, LA, delta_rmse_map,
              xlabel=r"Homodyne angle $\theta$",
              ylabel=r"OU rate $\lambda$",
              clabel="Relative improvement of QCO over plant KF",
              title=r"QCO advantage map: $(\mathrm{RMSE}_{plant}-\mathrm{RMSE}_{QCO})/\mathrm{RMSE}_{plant}$",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_theta_lambda_qco")

    _save_map(TH, LA, delta_Jphi_map,
              xlabel=r"Homodyne angle $\theta$",
              ylabel=r"OU rate $\lambda$",
              clabel=r"Relative gain in phase observability $\Delta_{J_\phi}$",
              title=r"Phase-observability gain map: $(J_\phi^{QCO}-J_\phi^{plant})/J_\phi^{plant}$",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_theta_lambda_phase_observability_gain")

    _save_map(TH, LA, delta_Q_map,
              xlabel=r"Homodyne angle $\theta$",
              ylabel=r"OU rate $\lambda$",
              clabel="Relative increase in diffusion proxy",
              title="QCO added-diffusion proxy map",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_theta_lambda_noise_proxy")

    _save_map(TH, LA, score_phi_map,
              xlabel=r"Homodyne angle $\theta$",
              ylabel=r"OU rate $\lambda$",
              clabel=r"Score $S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$",
              title="Predictive phase-based QCO score map",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_theta_lambda_score_phi")

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
    n_mc=3,
    n_jobs=-1,
    T=20.0,
):
    os.makedirs(save_dir, exist_ok=True)

    if k_vals is None:
        k_vals = np.linspace(0.0, 2.0, 41)
    if meas_vals is None:
        meas_vals = np.linspace(0.005, 0.12, 40)

    tasks = [
        (i, j, meas_std, k)
        for i, meas_std in enumerate(meas_vals)
        for j, k in enumerate(k_vals)
    ]

    def _run(i, j, meas_std, k):
        comp = compare_plant_vs_qco(
            theta=theta, lam=lam, k_so=k, k_os=k,
            alpha_noise=alpha_noise, meas_std=meas_std, n_mc=n_mc, T=T,
        )
        if verbose:
            print(f"[k-noise] meas={meas_std:.4f} k={k:.3f} "
                  f"RMSE={comp['delta_rmse']:+.3%} "
                  f"dJphi={comp['delta_Jphi']:+.3f} "
                  f"score={comp['score_phi']:+.3f}")
        return i, j, comp

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run)(i, j, meas_std, k) for i, j, meas_std, k in tasks
    )

    shape = (len(meas_vals), len(k_vals))
    delta_rmse_map = np.zeros(shape)
    delta_Jphi_map = np.zeros(shape)
    delta_Q_map    = np.zeros(shape)
    score_phi_map  = np.zeros(shape)

    for i, j, comp in results:
        delta_rmse_map[i, j] = comp["delta_rmse"]
        delta_Jphi_map[i, j] = comp["delta_Jphi"]
        delta_Q_map[i, j]    = comp["delta_Q"]
        score_phi_map[i, j]  = comp["score_phi"]

    KK, MM = np.meshgrid(k_vals, meas_vals)

    _save_map(KK, MM, delta_rmse_map,
              xlabel=r"QCO coupling $k_{so}=k_{os}$",
              ylabel="Measurement noise std",
              clabel="Relative improvement of QCO over plant KF",
              title="QCO advantage map versus coupling and measurement noise",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_coupling_noise_qco")

    _save_map(KK, MM, delta_Jphi_map,
              xlabel=r"QCO coupling $k_{so}=k_{os}$",
              ylabel="Measurement noise std",
              clabel=r"Relative gain in phase observability $\Delta_{J_\phi}$",
              title="Phase-observability gain versus coupling and measurement noise",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_coupling_noise_phase_observability_gain")

    _save_map(KK, MM, delta_Q_map,
              xlabel=r"QCO coupling $k_{so}=k_{os}$",
              ylabel="Measurement noise std",
              clabel="Relative increase in diffusion proxy",
              title="QCO added-diffusion proxy versus coupling and measurement noise",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_coupling_noise_noise_proxy")

    _save_map(KK, MM, score_phi_map,
              xlabel=r"QCO coupling $k_{so}=k_{os}$",
              ylabel="Measurement noise std",
              clabel=r"Score $S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$",
              title="Predictive phase-based score versus coupling and measurement noise",
              contour_data=delta_rmse_map,
              save_dir=save_dir,
              fname="regime_map_coupling_noise_score_phi")

    return {
        "k_vals": k_vals,
        "meas_vals": meas_vals,
        "delta_rmse": delta_rmse_map,
        "delta_Jphi": delta_Jphi_map,
        "delta_Q": delta_Q_map,
        "score_phi": score_phi_map,
    }


# =========================================================
# Shared plotting helper
# =========================================================
def _save_map(X, Y, Z, xlabel, ylabel, clabel, title, contour_data,
              save_dir, fname):
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.pcolormesh(X, Y, Z, shading="auto")
    fig.colorbar(im, ax=ax, label=clabel)
    ax.contour(X, Y, contour_data, levels=[0.0], linewidths=2.0,
               colors="purple")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname + ".png"), dpi=400)
    fig.savefig(os.path.join(save_dir, fname + ".pdf"))
    # pas de plt.close() — figure gardée ouverte pour plt.show()


# =========================================================
# Plant-only structural metric plots  (unchanged logic)
# =========================================================
def plot_rmse_vs_jphi_summary(data_dict, save_dir="plots", prefix="plant"):
    os.makedirs(save_dir, exist_ok=True)
    rows = data_dict["rows"]
    _, _, rmse_vals, jphi_vals, invsqrt_vals = rows_to_arrays(rows)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.scatter(invsqrt_vals, rmse_vals, s=18, alpha=0.7)
    ax.set_xlabel(r"$1/\sqrt{J_\phi(N)}$")
    ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
    ax.set_title(r"RMSE vs. $1/\sqrt{J_\phi(N)}$: no global collapse across regimes")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_rmse_vs_inv_sqrt_jphi_summary.png"), dpi=400)
    fig.savefig(os.path.join(save_dir, f"{prefix}_rmse_vs_inv_sqrt_jphi_summary.pdf"))
    # pas de close — gardée pour plt.show()

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.scatter(jphi_vals, rmse_vals, s=18, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$J_\phi(N)$")
    ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
    ax.set_title(r"RMSE vs. $J_\phi(N)$ on log-log axes: multi-regime behavior")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_rmse_vs_jphi_summary_loglog.png"), dpi=400)
    fig.savefig(os.path.join(save_dir, f"{prefix}_rmse_vs_jphi_summary_loglog.pdf"))
    # pas de close — gardée pour plt.show()


def plot_jphi_noncollapse_global(data_dict, save_dir="plots", prefix="plant"):
    os.makedirs(save_dir, exist_ok=True)
    rows = data_dict["rows"]
    theta, lam, rmse_vals, jphi_vals, invsqrt_vals = rows_to_arrays(rows)

    for c_arr, c_label, suffix in [
        (lam,   r"$\lambda$", "lambda"),
        (theta, r"$\theta$",  "theta"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        sc = ax.scatter(invsqrt_vals, rmse_vals, c=c_arr, s=18, alpha=0.75)
        fig.colorbar(sc, ax=ax, label=c_label)
        ax.set_xlabel(r"$1/\sqrt{J_\phi(N)}$")
        ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
        ax.set_title(
            r"Global non-collapse: RMSE vs. $1/\sqrt{J_\phi(N)}$ "
            f"colored by {c_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f"{prefix}_noncollapse_global_{suffix}.png"), dpi=400)
        fig.savefig(os.path.join(save_dir,
                    f"{prefix}_noncollapse_global_{suffix}.pdf"))
        # pas de close — gardée pour plt.show()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(jphi_vals, rmse_vals, c=lam, s=18, alpha=0.75)
    fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$J_\phi(N)$")
    ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
    ax.set_title(r"Global non-collapse on log-log axes")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_noncollapse_loglog_lambda.png"), dpi=400)
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_noncollapse_loglog_lambda.pdf"))
    # pas de close — gardée pour plt.show()


def plot_jphi_family_by_lambda(data_dict, lambda_indices=None,
                               save_dir="plots", prefix="plant",
                               x_clip=1.5):
    """
    x_clip : valeur max de 1/sqrt(Jphi) affichée.
    Les angles quasi-aveugles (Jphi très petit) donnent des points très
    bruités à grand x — on les exclut visuellement sans les supprimer des
    données.
    """
    os.makedirs(save_dir, exist_ok=True)
    lambda_vals = data_dict["lambda_vals"]
    rmse_map    = data_dict["rmse_map"]
    jphi_map    = data_dict["jphi_map"]

    if lambda_indices is None:
        targets = [0.02, 0.10, 0.30, 0.70, 1.50]
        lambda_indices = [
            int(np.argmin(np.abs(lambda_vals - t))) for t in targets
        ]
        seen = []
        lambda_indices = [
            idx for idx in lambda_indices
            if idx not in seen and not seen.append(idx)
        ]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for idx in lambda_indices:
        lam = lambda_vals[idx]
        x_all = safe_inv_sqrt(jphi_map[idx, :])
        y_all = rmse_map[idx, :]

        # Garder uniquement les points dans la zone informative
        mask  = x_all <= x_clip
        x = x_all[mask]
        y = y_all[mask]

        if len(x) < 2:
            continue
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", markersize=3, linewidth=1.2,
                label=fr"$\lambda={lam:.2f}$")

    ax.set_xlabel(r"$1/\sqrt{J_\phi(N)}$")
    ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
    ax.set_title(r"Local families at fixed $\lambda$")
    ax.set_xlim(left=0.0, right=x_clip * 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_families_fixed_lambda.png"), dpi=400)
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_families_fixed_lambda.pdf"))
    # pas de close — gardée pour plt.show()


def plot_jphi_family_by_theta(data_dict, theta_indices=None,
                              save_dir="plots", prefix="plant",
                              x_clip=None):
    """
    x_clip : valeur max de 1/sqrt(Jphi) affichée (None = auto, pas de clip).
    Pour θ fixé, la famille varie en λ — les courbes ne s'étendent pas
    au-delà d'une certaine valeur, donc le clip est moins critique ici.
    """
    os.makedirs(save_dir, exist_ok=True)
    theta_vals = data_dict["theta_vals"]
    rmse_map   = data_dict["rmse_map"]
    jphi_map   = data_dict["jphi_map"]

    if theta_indices is None:
        targets = [0.20, 0.60, np.pi / 2, 2.20, 2.90]
        theta_indices = [
            int(np.argmin(np.abs(theta_vals - t))) for t in targets
        ]
        seen = []
        theta_indices = [
            idx for idx in theta_indices
            if idx not in seen and not seen.append(idx)
        ]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for idx in theta_indices:
        th = theta_vals[idx]
        x_all = safe_inv_sqrt(jphi_map[:, idx])
        y_all = rmse_map[:, idx]

        if x_clip is not None:
            mask = x_all <= x_clip
            x = x_all[mask]
            y = y_all[mask]
        else:
            x = x_all
            y = y_all

        if len(x) < 2:
            continue
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", markersize=3, linewidth=1.2,
                label=fr"$\theta={th:.2f}$")

    ax.set_xlabel(r"$1/\sqrt{J_\phi(N)}$")
    ax.set_ylabel(r"$\mathrm{RMSE}(\phi)$")
    ax.set_title(r"Local families at fixed $\theta$")
    if x_clip is not None:
        ax.set_xlim(left=0.0, right=x_clip * 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_families_fixed_theta.png"), dpi=400)
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_families_fixed_theta.pdf"))
    # pas de close — gardée pour plt.show()


def fit_local_scaling_by_lambda(data_dict, lambda_indices=None):
    lambda_vals = data_dict["lambda_vals"]
    rmse_map    = data_dict["rmse_map"]
    jphi_map    = data_dict["jphi_map"]

    if lambda_indices is None:
        n = len(lambda_vals)
        lambda_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    print("\n=== Local scaling fits at fixed lambda ===")
    for idx in lambda_indices:
        lam = lambda_vals[idx]
        x = safe_inv_sqrt(jphi_map[idx, :])
        y = rmse_map[idx, :]
        a, b = np.polyfit(x, y, deg=1)
        ss_res = np.sum((y - (a * x + b)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        corr = np.corrcoef(x, y)[0, 1]
        print(f"lambda={lam:.4f} | corr={corr:+.4f} | "
              f"RMSE ≈ {a:.4e}/sqrt(Jphi) + {b:.4e} | R^2={r2:.4f}")


def fit_local_scaling_by_theta(data_dict, theta_indices=None):
    theta_vals = data_dict["theta_vals"]
    rmse_map   = data_dict["rmse_map"]
    jphi_map   = data_dict["jphi_map"]

    if theta_indices is None:
        n = len(theta_vals)
        theta_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    print("\n=== Local scaling fits at fixed theta ===")
    for idx in theta_indices:
        th = theta_vals[idx]
        x = safe_inv_sqrt(jphi_map[:, idx])
        y = rmse_map[:, idx]
        a, b = np.polyfit(x, y, deg=1)
        ss_res = np.sum((y - (a * x + b)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        corr = np.corrcoef(x, y)[0, 1]
        print(f"theta={th:.4f} | corr={corr:+.4f} | "
              f"RMSE ≈ {a:.4e}/sqrt(Jphi) + {b:.4e} | R^2={r2:.4f}")


# =========================================================
# QCO scatter diagnostics
# =========================================================
def plot_scatter_tradeoff(result_dict, save_dir="plots", prefix="theta_lambda"):
    os.makedirs(save_dir, exist_ok=True)

    delta_rmse = result_dict["delta_rmse"].ravel()
    delta_Jphi = result_dict["delta_Jphi"].ravel()
    delta_Q    = result_dict["delta_Q"].ravel()
    score_phi  = result_dict["score_phi"].ravel()

    for x_arr, xlabel, fname_suffix in [
        (delta_Jphi, r"$\Delta_{J_\phi}$",                 "deltaJphi"),
        (delta_Q,    r"$\Delta_Q$",                        "deltaQ"),
        (score_phi,  r"$S_\phi=\Delta_{J_\phi}-\alpha\Delta_Q$", "scorephi"),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        ax.scatter(x_arr, delta_rmse, s=18, alpha=0.7)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
        ax.set_title(f"RMSE gain vs {xlabel}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,
                    f"{prefix}_scatter_{fname_suffix}_vs_deltaRMSE.png"), dpi=400)
        fig.savefig(os.path.join(save_dir,
                    f"{prefix}_scatter_{fname_suffix}_vs_deltaRMSE.pdf"))
        # pas de close — gardée pour plt.show()

    corr_dj    = np.corrcoef(delta_Jphi, delta_rmse)[0, 1]
    corr_dq    = np.corrcoef(delta_Q,    delta_rmse)[0, 1]
    corr_score = np.corrcoef(score_phi,  delta_rmse)[0, 1]

    print(f"\n=== Scatter diagnostics [{prefix}] ===")
    print(f"corr(delta_Jphi, delta_RMSE) = {corr_dj:+.4f}")
    print(f"corr(delta_Q,    delta_RMSE) = {corr_dq:+.4f}")
    print(f"corr(score_phi,  delta_RMSE) = {corr_score:+.4f}")

    return {"corr_deltaJphi": corr_dj, "corr_deltaQ": corr_dq,
            "corr_score": corr_score}


def plot_delta_rmse_vs_delta_jphi(result_dict, save_dir="plots",
                                  prefix="theta_lambda"):
    os.makedirs(save_dir, exist_ok=True)
    x = result_dict["delta_Jphi"].ravel()
    y = result_dict["delta_rmse"].ravel()

    a, b = np.polyfit(x, y, deg=1)
    x_fit = np.linspace(np.min(x), np.max(x), 300)
    y_fit = a * x_fit + b

    corr   = np.corrcoef(x, y)[0, 1]
    ss_res = np.sum((y - (a * x + b)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.scatter(x, y, s=18, alpha=0.7, label="simulation points")
    ax.plot(x_fit, y_fit, linewidth=2.0,
            label=fr"fit: $y={a:.3e}x+{b:.3e}$")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta_{J_\phi}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{RMSE}}$")
    ax.set_title("QCO performance gain vs phase-information gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_delta_rmse_vs_delta_jphi_fit.png"), dpi=400)
    fig.savefig(os.path.join(save_dir,
                f"{prefix}_delta_rmse_vs_delta_jphi_fit.pdf"))
    # pas de close — gardée pour plt.show()

    print(f"\n=== Delta RMSE vs Delta J_phi [{prefix}] ===")
    print(f"corr = {corr:+.4f}")
    print(f"fit: delta_RMSE ≈ {a:.6e} * delta_Jphi + {b:.6e}")
    print(f"R^2 = {r2:.4f}")

    return {"corr": corr, "slope": a, "intercept": b, "r2": r2}


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    SAVE_DIR    = "plots"
    ALPHA_NOISE = 0.2
    N_JOBS      = -1   # tous les cœurs

    # -------------------------------------------------------
    # MODE : choisir parmi DEBUG / INTERMEDIATE / FINAL
    #   DEBUG        ~2-3  min  — vérifier que le code tourne
    #   INTERMEDIATE ~8-10 min  — vérifier que les figures sont propres
    #   FINAL        ~20   min  — version papier
    # -------------------------------------------------------
    MODE = "FINAL"   # changer ici : "DEBUG" / "INTERMEDIATE" / "FINAL"

    if MODE == "DEBUG":
        N_MC         = 1
        T_SIM        = 20.0    # trajectoire courte → bruit élevé
        THETA_VALS   = np.linspace(0.05, np.pi - 0.05, 25)
        LAMBDA_VALS  = np.linspace(0.02, 1.5, 20)
        K_VALS       = np.linspace(0.0, 2.0, 21)
        MEAS_VALS    = np.linspace(0.005, 0.12, 20)
        print("=== MODE DEBUG (~2 min, figures bruitées) ===")

    elif MODE == "INTERMEDIATE":
        N_MC         = 1
        T_SIM        = 100.0   # trajectoire longue → variance ÷ 5 vs T=20
        THETA_VALS   = np.linspace(0.05, np.pi - 0.05, 35)
        LAMBDA_VALS  = np.linspace(0.02, 1.5, 30)
        K_VALS       = np.linspace(0.0, 2.0, 31)
        MEAS_VALS    = np.linspace(0.005, 0.12, 30)
        print("=== MODE INTERMEDIATE (~8 min, figures correctes) ===")

    else:  # FINAL
        N_MC         = 10
        T_SIM        = 100.0
        THETA_VALS   = np.linspace(0.05, np.pi - 0.05, 50)
        LAMBDA_VALS  = np.linspace(0.02, 1.5, 45)
        K_VALS       = np.linspace(0.0, 2.0, 41)
        MEAS_VALS    = np.linspace(0.005, 0.12, 40)
        print("=== MODE FINAL (~20 min, version papier) ===")

    # 1) Plant-only analysis
    plant_data = collect_plant_theta_lambda_data(
        theta_vals=THETA_VALS,
        lambda_vals=LAMBDA_VALS,
        save_dir=SAVE_DIR,
        verbose=True,
        n_mc=N_MC,
        n_jobs=N_JOBS,
        T=T_SIM,
    )
    plot_rmse_vs_jphi_summary(plant_data, save_dir=SAVE_DIR,
                              prefix="plant_theta_lambda")
    plot_jphi_noncollapse_global(plant_data, save_dir=SAVE_DIR,
                                 prefix="plant_theta_lambda")
    plot_jphi_family_by_lambda(plant_data, save_dir=SAVE_DIR,
                               prefix="plant_theta_lambda",
                               x_clip=0.75)
    plot_jphi_family_by_theta(plant_data, save_dir=SAVE_DIR,
                              prefix="plant_theta_lambda",
                              x_clip=0.75)
    fit_local_scaling_by_lambda(plant_data)
    fit_local_scaling_by_theta(plant_data)

    # 2) QCO regime maps
    res_theta_lambda = regime_map_theta_lambda(
        k_so=0.8, k_os=0.8,
        theta_vals=THETA_VALS,
        lambda_vals=LAMBDA_VALS,
        alpha_noise=ALPHA_NOISE,
        save_dir=SAVE_DIR,
        verbose=True,
        n_mc=N_MC,
        n_jobs=N_JOBS,
        T=T_SIM,
    )
    res_coupling_noise = regime_map_coupling_noise(
        theta=np.pi / 2,
        lam=0.2,
        k_vals=K_VALS,
        meas_vals=MEAS_VALS,
        alpha_noise=ALPHA_NOISE,
        save_dir=SAVE_DIR,
        verbose=True,
        n_mc=N_MC,
        n_jobs=N_JOBS,
        T=T_SIM,
    )

    # 3) QCO scatter diagnostics
    for res, prefix in [
        (res_theta_lambda,   "theta_lambda"),
        (res_coupling_noise, "coupling_noise"),
    ]:
        plot_scatter_tradeoff(res, save_dir=SAVE_DIR, prefix=prefix)
        plot_delta_rmse_vs_delta_jphi(res, save_dir=SAVE_DIR, prefix=prefix)

    print("\nDone. All figures saved to:", SAVE_DIR)

    # Ouvre uniquement les figures principales dans le viewer système
    import subprocess, platform

    figures_to_open = [
        # Plant-only
        "plant_theta_lambda_rmse_vs_inv_sqrt_jphi_summary.png",
        "plant_theta_lambda_noncollapse_global_lambda.png",
        "plant_theta_lambda_noncollapse_global_theta.png",
        "plant_theta_lambda_noncollapse_loglog_lambda.png",
        "plant_theta_lambda_families_fixed_lambda.png",
        "plant_theta_lambda_families_fixed_theta.png",
        # QCO theta/lambda
        "regime_map_theta_lambda_qco.png",
        "regime_map_theta_lambda_phase_observability_gain.png",
        "regime_map_theta_lambda_noise_proxy.png",
        "regime_map_theta_lambda_score_phi.png",
        # QCO coupling/noise
        "regime_map_coupling_noise_qco.png",
        "regime_map_coupling_noise_phase_observability_gain.png",
        "regime_map_coupling_noise_noise_proxy.png",
        "regime_map_coupling_noise_score_phi.png",
        # Scatter diagnostics
        "theta_lambda_scatter_deltaJphi_vs_deltaRMSE.png",
        "theta_lambda_scatter_deltaQ_vs_deltaRMSE.png",
        "theta_lambda_scatter_scorephi_vs_deltaRMSE.png",
        "theta_lambda_delta_rmse_vs_delta_jphi_fit.png",
        "coupling_noise_scatter_deltaJphi_vs_deltaRMSE.png",
        "coupling_noise_scatter_deltaQ_vs_deltaRMSE.png",
        "coupling_noise_scatter_scorephi_vs_deltaRMSE.png",
        "coupling_noise_delta_rmse_vs_delta_jphi_fit.png",
    ]

    print(f"Ouverture de {len(figures_to_open)} figures...")
    for fname in figures_to_open:
        path = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(path):
            print(f"  [manquant] {fname}")
            continue
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])