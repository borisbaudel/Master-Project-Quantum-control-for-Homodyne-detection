import numpy as np
import matplotlib.pyplot as plt

from core.model import build_Aa, build_Ca
from core.observability import (
    observability_condition_number,
    lambda_min_gramian
)

# parameters
gamma = 1.0
omega = 0.0
g_phi = 1.0
lam = 0.5
kappa = 1.0

thetas = np.linspace(0, np.pi, 100)

conds = []
lambdas = []

for theta in thetas:
    A = build_Aa(gamma, omega, g_phi, lam)
    C = build_Ca(kappa, theta)

    conds.append(observability_condition_number(A, C))
    lambdas.append(lambda_min_gramian(A, C))

# plots
plt.figure()
plt.plot(thetas, conds)
plt.title("Observability condition number")
plt.xlabel("theta")
plt.ylabel("cond(O)")
plt.yscale("log")

plt.figure()
plt.plot(thetas, lambdas)
plt.title("Lambda min Gramian")
plt.xlabel("theta")

plt.show()