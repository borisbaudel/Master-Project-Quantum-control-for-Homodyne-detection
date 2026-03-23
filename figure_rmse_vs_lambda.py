import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments.simulate_ou_kf import simulate_system


def info_proxy(theta: float, g_phi: float) -> float:
    return (g_phi * np.sin(theta)) ** 2


# -----------------------
# PARAMETERS
# -----------------------
gamma = 1.0
omega = 0.0
g_phi = 2.0
lam = 0.2
kappa = 1.0

dt = 0.005
T = 30.0
seed = 0

thetas = np.linspace(0.02, np.pi - 0.02, 80)

rmse_phi = []
infos = []

# -----------------------
# SWEEP
# -----------------------
for theta in thetas:
    res = simulate_system(
        gamma=gamma,
        omega=omega,
        g_phi=g_phi,
        lam=lam,
        kappa=kappa,
        theta=theta,
        T=T,
        dt=dt,
        seed=seed,
    )

    rmse_phi.append(res["phi_rmse"])
    infos.append(info_proxy(theta, g_phi))

rmse_phi = np.array(rmse_phi)
infos = np.array(infos)

rmse_norm = rmse_phi / np.max(rmse_phi)
inv_rmse_norm = (1.0 / rmse_phi) / np.max(1.0 / rmse_phi)
info_norm = infos / np.max(infos)

corr = np.corrcoef(inv_rmse_norm, info_norm)[0, 1]
print(f"Correlation between normalized 1/RMSE and info proxy = {corr:.4f}")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8.2, 4.8))
plt.plot(thetas, rmse_norm, linewidth=2.4, label=r"normalized RMSE($\phi$)")
plt.plot(thetas, info_norm, linewidth=2.4, label=r"normalized $(g_\phi \sin\theta)^2$")
plt.plot(thetas, inv_rmse_norm, "--", linewidth=2.4, label=r"normalized $1/\mathrm{RMSE}(\phi)$")

plt.xlabel(r"Homodyne angle $\theta$")
plt.ylabel("Normalized quantities")
plt.title("Estimation error and physical information proxy")
plt.grid(True, alpha=0.35)
plt.legend(frameon=True)
plt.tight_layout()

plt.savefig("plots/figure_theta_article.png", dpi=400, bbox_inches="tight")
plt.savefig("plots/figure_theta_article.pdf", bbox_inches="tight")
plt.show()