import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from experiments.simulate_ou_kf import simulate_system


def info_proxy(theta: float, g_phi: float) -> float:
    """
    Physical information proxy for the resonant simplified model:
        info ~ (g_phi * sin(theta))^2

    Interpretation:
    - phi drives the informative quadrature with gain g_phi
    - homodyne measures a projection weighted by sin(theta)
    """
    return (g_phi * np.sin(theta)) ** 2


# -----------------------
# PARAMETERS
# -----------------------
gamma = 1.0
omega = 0.0
g_phi = 2.0
lam = 0.2
kappa = 1.0

dt = 0.01
T = 15.0
seed = 0

thetas = np.linspace(0.01, np.pi - 0.01, 60)

rmse_phi = []
infos = []

# -----------------------
# MAIN LOOP
# -----------------------
for theta in thetas:
    res = simulate_system(
        gamma=gamma,
        omega=omega,
        g_phi=g_phi,
        lam=lam,
        kappa=kappa,
        theta=theta,
        dt=dt,
        T=T,
        seed=seed,
    )

    rmse_phi.append(res["phi_rmse"])
    infos.append(info_proxy(theta, g_phi))

rmse_phi = np.array(rmse_phi, dtype=float)
infos = np.array(infos, dtype=float)

# Normalized quantities for visual comparison
rmse_norm = rmse_phi / np.max(rmse_phi)
info_norm = infos / np.max(infos)

# Inverse RMSE is often visually closer to "information"
inv_rmse = 1.0 / rmse_phi
inv_rmse_norm = inv_rmse / np.max(inv_rmse)

os.makedirs("plots", exist_ok=True)

# -----------------------
# PLOT 1: RMSE only
# -----------------------
plt.figure(figsize=(9, 4.8))
plt.plot(thetas, rmse_phi, linewidth=2)
plt.xlabel(r"$\theta$")
plt.ylabel(r"RMSE($\phi$)")
plt.title("Phase estimation error vs homodyne angle")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/rmse_vs_theta.png", dpi=300)
plt.show()

# -----------------------
# PLOT 2: info proxy only
# -----------------------
plt.figure(figsize=(9, 4.8))
plt.plot(thetas, infos, linewidth=2)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$(g_\phi \sin\theta)^2$")
plt.title("Physical information proxy vs homodyne angle")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/info_proxy_vs_theta.png", dpi=300)
plt.show()

# -----------------------
# PLOT 3: overlay comparison
# -----------------------
plt.figure(figsize=(9, 5.2))
plt.plot(thetas, rmse_norm, label=r"normalized RMSE($\phi$)", linewidth=2)
plt.plot(thetas, info_norm, label=r"normalized $(g_\phi \sin\theta)^2$", linewidth=2)
plt.plot(thetas, inv_rmse_norm, label=r"normalized $1/\mathrm{RMSE}(\phi)$", linewidth=2, linestyle="--")

plt.xlabel(r"$\theta$")
plt.ylabel("Normalized quantities")
plt.title("Comparison between estimation error and physical information proxy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/rmse_info_overlay.png", dpi=300)
plt.show()