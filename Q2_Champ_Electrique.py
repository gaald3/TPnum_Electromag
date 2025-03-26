import numpy as np
import matplotlib.pyplot as plt
from Q1_Calcul_Potentiel import relaxation, init_conditions, scale, Nx, Ny


def calcul_champ_electrique(V):
    # Calcul du champ E = -grad(V)
    Ey, Ex = np.gradient(-V)  # Attention : gradient retourne d'abord l'axe y, ensuite x
    return Ex, Ey


def plot_champ_electrique(V, Ex, Ey):
    plt.figure(figsize=(8, 6))

    # Affichage du potentiel en fond
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx, 0, Ny])
    plt.colorbar(label="Potentiel (V)")

    # Création d'une grille de points plus espacés pour les flèches
    step = 5
    x = np.arange(0, Nx, step)
    y = np.arange(0, Ny, step)
    X, Y = np.meshgrid(x, y)
    
    # Champ électrique échantillonné sur la grille
    Ex_sample = Ex[::step, ::step]
    Ey_sample = Ey[::step, ::step]

    # Superposition du champ vectoriel
    plt.quiver(X, Y, Ex_sample, Ey_sample, color="cyan", scale=100, width=0.003)

    plt.title("Champ électrique dans le tube PM")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.tight_layout()
    plt.savefig("figures_dos/champ_PM.png")
    plt.show()


def main():
    print("Calcul du potentiel pour la question 2...")
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V, tol=1e-3)

    print("Calcul du champ électrique...")
    Ex, Ey = calcul_champ_electrique(V)

    print("Affichage du champ électrique...")
    plot_champ_electrique(V, Ex, Ey)
