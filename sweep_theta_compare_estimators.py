import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments.compare_estimators import simulate_kalman_and_luenberger


gamma = 1.0
omega = 0.0
g_phi = 2.0
lam = 0.2
kappa = 1.0

thetas = np.linspace(0.02, np.pi - 0.02, 50)

rmse_kf = []
rmse_luen = []

for theta in thetas:
    res = simulate_kalman_and_luenberger(
        gamma=gamma,
        omega=omega,
        g_phi=g_phi,
        lam=lam,
        kappa=kappa,
        theta=theta,
        dt=0.01,
        T=20.0,
        sigma_q=0.02,
        sigma_p=0.02,
        q_phi=0.01,
        meas_std=0.03,
        seed=1,
    )

    rmse_kf.append(res["rmse_phi_kf"])
    rmse_luen.append(res["rmse_phi_luen"])

rmse_kf = np.array(rmse_kf)
rmse_luen = np.array(rmse_luen)

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8.2, 4.8))
plt.plot(thetas, rmse_kf, linewidth=2.2, label="Kalman")
plt.plot(thetas, rmse_luen, linewidth=2.2, label="Luenberger")
plt.xlabel(r"Homodyne angle $\theta$")
plt.ylabel(r"RMSE($\phi$)")
plt.title("Kalman vs Luenberger across homodyne angle")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()

plt.savefig("plots/sweep_theta_compare_estimators.png", dpi=400)
plt.savefig("plots/sweep_theta_compare_estimators.pdf")
plt.show()