import numpy as np
import matplotlib.pyplot as plt

# ======================================
# Paramètres géométriques (en mm, convertis en cases)
# ======================================
scale = 10  # 1 mm = 10 cases

a = 3 * scale     # Espace entre dynodes et extrémités du tube (x)
b = 2 * scale     # Espace entre dynodes et bords haut/bas (y)
c = 4 * scale     # Longueur des dynodes (en x)
d = 2 * scale     # Distance entre dynodes (en x)
e = int(0.2 * scale)  # Épaisseur des dynodes (en y)
f = 6 * scale     # Hauteur du tube
N = 4             # Nombre de dynodes

# Taille de la grille (x = largeur, y = hauteur)
Nx = a * 2 + d * (N - 1) + c * N  # dynodes et espacements sur x
Ny = f  # hauteur du tube
V = np.zeros((Ny, Nx))  # grille du potentiel

# ======================================
# Définir les conditions initiales
# ======================================
def init_conditions(V):
    # Parois de l'enceinte à 0 V
    V[:, 0] = V[:, -1] = V[0, :] = V[-1, :] = 0

    # Dynodes du bas
    for i in range(N):
        x = a + i * (c + d)
        V[b:b+e, x:x+c] = 100 * (i + 1)

    # Dynodes du haut, décalées de c/2
    for i in range(N):
        x = a + i * (c + d) + c // 2
        V[-(b+e):-b, x:x+c] = 100 * (i + 1)

    return V

# ======================================
# Méthode de relaxation
# ======================================
def relaxation(V, tol=1e-3, max_iter=10000):
    diff = tol + 1
    iterations = 0

    while diff > tol and iterations < max_iter:
        V_old = V.copy()

        # Mise à jour (sans toucher aux bords)
        V[1:-1, 1:-1] = 0.25 * (
            V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2]
        )

        # Réimposer les conditions (dynodes et parois)
        V = init_conditions(V)

        diff = np.max(np.abs(V - V_old))
        iterations += 1

    print(f"Convergence atteinte en {iterations} itérations (diff = {diff:.2e})")
    return V

# ======================================
# Affichage et sauvegarde
# ======================================
def plot_potential(V):
    plt.figure(figsize=(6, 8))
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx/scale, 0, Ny/scale])
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
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V)
    plot_potential(V)

if __name__ == "__main__":
    main()

