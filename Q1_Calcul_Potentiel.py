import numpy as np
import matplotlib.pyplot as plt

# Paramètres de la grille
Nx, Ny = 100, 100  # Taille de la grille (100x100)
V = np.zeros((Nx, Ny))  # Matrice initiale du potentiel (0V partout)

# Définition des dynodes (positions et potentiels)
def init_conditions(V):
    V[:, 0] = 0  # Bord gauche (enceinte)
    V[:, -1] = 0  # Bord droit (enceinte)
    V[0, :] = 0  # Bord haut (enceinte)
    V[-1, :] = 0  # Bord bas (enceinte)

    # Position des dynodes et potentiels
    dynodes_y = [20, 40, 60, 80]  # Positions sur l'axe y
    for i, y in enumerate(dynodes_y):
        V[y, 10:90] = 100 * (i + 1)  # Dynodes avec potentiels 100V, 200V, ...

    return V

# Méthode de relaxation
def relaxation(V, tol=1e-5, max_iter=10000):
    diff = tol + 1
    iterations = 0

    while diff > tol and iterations < max_iter:
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (V[:-2, 1:-1] + V[2:, 1:-1] + V[1:-1, :-2] + V[1:-1, 2:])
        
        # Vérification du critère d'arrêt
        diff = np.max(np.abs(V - V_old))
        iterations += 1

    print(f"Convergence atteinte en {iterations} itérations.")
    return V

# Affichage du potentiel
def plot_potential(V):
    plt.figure(figsize=(8, 6))
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx, 0, Ny])
    plt.colorbar(label="Potentiel (V)")
    plt.title("Potentiel dans le tube PM")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.show()

# Exécution du programme
def main():
    print("Initialisation de la grille...")
    
    # Définition initiale de V avant de l’envoyer à init_conditions
    Nx, Ny = 100, 100  # Dimensions de la grille
    V = np.zeros((Nx, Ny))  # Initialisation du potentiel à 0V partout

    V = init_conditions(V)  # Maintenant V est bien défini avant d'être utilisé
    V = relaxation(V)  # Application de la méthode de relaxation

    plot_potential(V)  # Affichage du potentiel

