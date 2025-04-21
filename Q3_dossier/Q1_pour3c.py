import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Paramètres géométriques (mm)
# ===============================
scale = 20  # 1 mm = 20 pixels
a = 5.5 * scale   # Distance dynode-extrémité
b = 2 * scale
c = 5 * scale   # Longueur dynode (en x)
d = 11 * scale   # Espacement horizontal entre dynodes
e = int(0.5 * scale)  # Épaisseur dynode (en y)
f = 6 * scale   # Hauteur du tube (en pixels) → 6 mm
N = 4  # Nombre total de dynodes

# ===============================
# Grille
# ===============================
Nx = int(a + 1.5 * (c + d) + c + a)
Ny = f # hauteur du tube
V = np.zeros((Ny, Nx))

# ===============================
# Conditions initiales
# ===============================
def init_conditions(V):
    V[:, 0] = V[:, -1] = V[0, :] = V[-1, :] = 0  # Bords à 0 V

    # Calcul des hauteurs des dynodes pour rebonds de 2 mm
    y_center = Ny // 2  # centre vertical
    y_bas = int(y_center - 1.5 * scale - e // 2)   # dynodes du bas
    y_haut = int(y_center + 1.5 * scale - e // 2)  # dynodes du haut

    # Dynodes du bas
    for i in range(N // 2):
        x_pos = int(a + i * (c + d))
        V[y_bas:y_bas + e, x_pos:x_pos + c] = 100 * (2 * i + 1)

    # Dynodes du haut
    for i in range(N // 2):
        x_pos = int(a + (i + 0.5) * (c + d))
        V[y_haut:y_haut + e, x_pos:x_pos + c] = 100 * (2 * i + 2)

    return V

# ===============================
# Méthode de relaxation
# ===============================
def relaxation(V, tol=1e-3, max_iter=10000):
    diff = tol + 1
    iterations = 0

    while diff > tol and iterations < max_iter:
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (
            V[2:, 1:-1] + V[:-2, 1:-1] +
            V[1:-1, 2:] + V[1:-1, :-2]
        )
        V = init_conditions(V)
        diff = np.max(np.abs(V - V_old))
        iterations += 1

    print(f"Convergence atteinte en {iterations} itérations (diff = {diff:.2e})")
    return V

# ===============================
# Affichage
# ===============================
def plot_potentiel(V):
    Ny, Nx = V.shape
    plt.figure(figsize=(10, 4))
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx/scale, 0, Ny/scale])
    plt.colorbar(label="Potentiel (V)")
    plt.title("Potentiel dans le tube PM (géométrie ajustée)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.savefig("figures_dos/potentiel_PM.png")
    plt.show()

# ===============================
# Lancement
# ===============================
def main():
    print("Initialisation de la géométrie du tube PM...")
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V)
    plot_potentiel(V)

if __name__ == "__main__":
    main()
