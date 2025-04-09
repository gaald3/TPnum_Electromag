import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from Q2_Champ_Electrique import calcul_champ_electrique
from Q1_Calcul_Potentiel import relaxation, init_conditions

# Constantes physiques
e = -1.602e-19  # Charge de l'electron (C)
m = 9.109e-31   # Masse de l'electron (kg)

def trajectoire_electron(V, dt=1e-11, steps=1000, x0=30, y0=10, vx0=0, vy0=0):
    """
    Simule la trajectoire d'un électron dans le tube PM avec la méthode d'Euler.
    - V : matrice du potentiel
    - dt : pas de temps (s)
    - steps : nombre de pas
    - x0, y0 : position initiale (en pixels)
    - vx0, vy0 : vitesse initiale (m/s)
    """
    Ny, Nx = V.shape

    # Calcul du champ E = -grad(V)
    Ex, Ey = calcul_champ_electrique(V)
    
    # Interpolation pour obtenir E(x, y) n'importe où
    interp_Ex = RectBivariateSpline(np.arange(Ny), np.arange(Nx), Ex)
    interp_Ey = RectBivariateSpline(np.arange(Ny), np.arange(Nx), Ey)

    # Position et vitesse initiales
    pos = np.array([x0, y0], dtype=float)
    vel = np.array([vx0, vy0], dtype=float)

    traj = [pos.copy()]

    for _ in range(steps):
        x, y = pos

        # S'arrête si l'électron sort de l'enceinte
        if x < 1 or x >= Nx-1 or y < 1 or y >= Ny-1:
            break

        # Champ E au point actuel (attention : y, x pour les matrices)
        E = np.array([
            interp_Ex(y, x)[0][0],
            interp_Ey(y, x)[0][0]
        ])

        # a = qE/m
        a = (e * E) / m

        # Méthode d'Euler
        vel += a * dt
        pos += vel * dt

        traj.append(pos.copy())

    return np.array(traj)

def main():
    # Générer le potentiel V
    Nx = 60
    Ny = 100
    V_init = np.zeros((Ny, Nx))
    V = init_conditions(V_init)
    V = relaxation(V, tol=1e-3)

    # Lancer la simulation
    traj = trajectoire_electron(V, dt=1e-11, steps=1500, x0=30, y0=5)

    # Affichage
    plt.figure(figsize=(6, 8))
    plt.imshow(V, cmap="inferno", origin="lower")
    plt.plot(traj[:, 0], traj[:, 1], color="cyan")
    plt.title("Trajectoire de l'électron dans le tube PM")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.colorbar(label="Potentiel (V)")
    plt.savefig("figures_dos/trajectoire_electron.png")
    plt.show()

if __name__ == "__main__":
    main()
