import numpy as np
import matplotlib.pyplot as plt
from Q1_Calcul_Potentiel import relaxation, init_conditions, scale, Nx, Ny

# ===================================
# Calcul du champ électrique
# ===================================
def calcul_champ_electrique(V):
    Ey, Ex = np.gradient(-V)  # Gradient retourne d'abord axe y puis x
    return Ex, Ey

# ===================================
# Affichage du champ + potentiel
# ===================================
def plot_champ_electrique(V, Ex, Ey):
    plt.figure(figsize=(10, 4))

    # Affichage du potentiel
    plt.imshow(V, cmap="inferno", origin="lower",
               extent=[0, Nx / scale, 0, Ny / scale])
    plt.colorbar(label="Potentiel (V)")

    # Grille échantillonnée pour le champ (quiver)
    step = 5
    x = np.arange(0, Nx, step)
    y = np.arange(0, Ny, step)
    X, Y = np.meshgrid(x, y)

    Ex_sample = Ex[::step, ::step]
    Ey_sample = Ey[::step, ::step]

    # Affichage du champ
    plt.quiver(X / scale, Y / scale, Ex_sample, Ey_sample,
               color="cyan", scale=100, width=0.003)

    plt.title("Champ électrique dans le tube PM")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.tight_layout()
    plt.savefig("figures_dos/champ_PM.png")
    plt.show()

# ===================================
# Lancement
# ===================================
def main():
    print("Calcul du potentiel pour la question 2...")
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V, tol=1e-3)

    print("Calcul du champ électrique...")
    Ex, Ey = calcul_champ_electrique(V)

    print("Affichage du champ électrique...")
    plot_champ_electrique(V, Ex, Ey)

if __name__ == "__main__":
    main()