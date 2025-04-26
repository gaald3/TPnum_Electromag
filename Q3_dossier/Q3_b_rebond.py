import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from Q1_Calcul_Potentiel import relaxation, init_conditions, scale, Nx, Ny
from Q2_Champ_Electrique import calcul_champ_electrique

def main():
    # Constantes physiques
    e = -1.602e-19  # charge de l'électron (C)
    m = 9.109e-31   # masse de l'électron (kg)

    # Paramètres de simulation
    duree_totale = 1.3e-7
    dt = 3e-11
    nb_steps = int(duree_totale / dt)

    # Conditions initiales (en mm)
    x0_mm, y0_mm = 0.0, 3.0
    vx0, vy0 = 0.0, 0.0

    x0 = x0_mm * scale
    y0 = y0_mm * scale

    # Potentiel + champ
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V, tol=1e-3)
    Ex, Ey = calcul_champ_electrique(V)

    # Interpolation du champ
    x = np.arange(Nx)
    y = np.arange(Ny)
    interp_Ex = RegularGridInterpolator((y, x), Ex)
    interp_Ey = RegularGridInterpolator((y, x), Ey)

    # Initialisation
    positions = [(x0, y0)]
    vitesses = [(vx0, vy0)]
    dynode_potentiels = [100, 200, 300, 400]
    rebound_margin = 1

    # Simulation avec rebond
    x, y = x0, y0
    vx, vy = vx0, vy0
    for step in range(nb_steps):
        if not (0 <= x < Nx and 0 <= y < Ny):
            break
        E = np.array([interp_Ex((y, x)), interp_Ey((y, x))])
        ax = (e / m) * E[0] * scale * 1e3
        ay = (e / m) * E[1] * scale * 1e3
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        local_V = V[int(round(y)), int(round(x))]
        if any(abs(local_V - vp) <= rebound_margin for vp in dynode_potentiels):
            vy = -vy
        positions.append((x, y))
        vitesses.append((vx, vy))

    positions = np.array(positions) / scale

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx / scale, 0, Ny / scale])
    plt.colorbar(label="Potentiel (V)")
    plt.plot(positions[:, 0], positions[:, 1], color="cyan", lw=1.5, label="Trajectoire de l'électron")
    plt.scatter([x0_mm], [y0_mm], color="cyan", label="Départ", zorder=5)
    plt.title("Trajectoire de l'électron dans le tube photomultiplicateur avec  rebond (inversion de vitesse)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig("figures_dos/trajectoire_electron_b_rebond.png")
    plt.show()
if __name__ == "__main__":
    main()