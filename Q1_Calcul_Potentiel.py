import numpy as np
import matplotlib.pyplot as plt

# ======================================
# Paramètres géométriques (en mm, convertis en pixels)
# ======================================
scale = 10  # 1 mm = 10 pixels

a = 3 * scale     # Espace entre dynodes et extrémités
b = 2 * scale     # Espace entre dynodes et parois latérales
c = 4 * scale     # Longueur des dynodes
d = 2 * scale     # Distance entre dynodes
e = int(0.2 * scale)  # Épaisseur d'une dynode
f = 6 * scale     # Largeur du tube
N = 4             # Nombre de dynodes

# Taille de la grille
Nx = f
Ny = a * 2 + d * (N - 1) + e * N  # Hauteur : dynodes + espacements
V = np.zeros((Ny, Nx))  # Matrice du potentiel

# ======================================
# Définir les conditions initiales
# ======================================
def init_conditions(V):
    # Bords de l'enceinte à 0 V
    V[:, 0] = V[:, -1] = V[0, :] = V[-1, :] = 0

    # Dynodes du bas
    for i in range(N):
        y = a + i * (d + e)
        V[y:y+e, b:b+c] = 100 * (i + 1)

    # Dynodes du haut (décalées horizontalement)
    for i in range(N):
        y = Ny - (a + i * (d + e) + e)
        V[y:y+e, b + c//2 : b + c + c//2] = 100 * (i + 1)

    return V

# ======================================
# Méthode de relaxation
# ======================================
def relaxation(V, tol=1e-3, max_iter=10000):
    diff = tol + 1
    iterations = 0

    while diff > tol and iterations < max_iter:
        V_old = V.copy()

        # Appliquer la méthode de relaxation
        V[1:-1, 1:-1] = 0.25 * (
            V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2]
        )

        # Réappliquer les conditions aux dynodes
        V = init_conditions(V)

        # Critère d'arrêt basé sur le changement maximal
        diff = np.max(np.abs(V - V_old))
        iterations += 1

    print(f"Convergence atteinte en {iterations} itérations (diff = {diff:.2e})")
    return V

# ======================================
# Affichage et sauvegarde de la figure
# ======================================
def plot_potential(V):
    plt.figure(figsize=(8, 6))
    plt.imshow(V, cmap="inferno", origin="lower",
               extent=[0, Nx/scale, 0, Ny/scale])
    plt.colorbar(label="Potentiel (V)")
    plt.title("Potentiel dans le tube PM")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.savefig("figures_dos/potentiel_PM.png")
    plt.show()

# ======================================
# Exécution principale
# ======================================
def main():
    print("Initialisation de la grille...")

    V = np.zeros((Ny, Nx))  # Potentiel initial
    V = init_conditions(V)
    V = relaxation(V)       # Appliquer la méthode de relaxation
    plot_potential(V)       # Afficher et sauvegarder la figure

if __name__ == "__main__":
    main()
