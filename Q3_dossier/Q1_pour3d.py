import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Paramètres géométriques (mm)
# ===============================
scale = 10  # 1 mm = 10 cases


# Paramètres ajustés pour garantir rebond vertical de 2 mm sur 4 dynodes
f_mm = 10.86  # Hauteur du tube (mm)
f = int(f_mm * scale)


a = 2.25 * scale   # Distance dynode-extrémité
b = 2 * scale   # Espace dynode-paroi
c = 8 * scale   # Longueur dynode
d = 6.8 * scale   # Distance entre dynodes du même côté
e = int(0.4 * scale)  # Épaisseur dynode (en y)
N = 12  # Nombre total de dynodes (2 en haut, 2 en bas)


# ===============================
# Grille : x = longueur, y = hauteur
# ===============================
Nx = int(a * 2 + N/2 * c + (N/2) * d)  # Largeur suffisante
Ny = f  # Hauteur du tube
V = np.zeros((Ny, Nx))  # Grille du potentiel


# ===============================
# Conditions initiales (géométrie)
# ===============================
def init_conditions(V):
   # Bords du tube à 0 V
   V[:, 0] = V[:, -1] = V[0, :] = V[-1, :] = 0


   # Dynodes du bas (rebond vers le haut)
   for i in range(N // 2):
       x_min = int(a + i * (c + d))
       x_max = x_min + c
       y_min = int(f / 2 - 2 * scale - e / 2)
       y_max = y_min + e
       V[int(y_min):int(y_max), int(x_min):int(x_max)] = 100 * (2 * i + 1)


   # Dynodes du haut (rebond vers le bas)
   for i in range(N // 2):
       x_min = int(a + (i + 0.5) * (c + d))
       x_max = x_min + c
       y_min = int(f / 2 + 2 * scale - e / 2)
       y_max = y_min + e
       V[int(y_min):int(y_max), int(x_min):int(x_max)] = 100 * (2 * i + 2)


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
# Affichage du potentiel
# ===============================
def plot_potential(V):
   plt.figure(figsize=(10, 4))
   plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx/scale, 0, Ny/scale])
   plt.colorbar(label="Potentiel (V)")
   plt.title("Potentiel dans le tube PM (ajusté_d)")
   plt.xlabel("x (mm)")
   plt.ylabel("y (mm)")
   plt.savefig("figures_dos/potentiel_PM.png")
   plt.show()


# ===============================
# Exécution principale
# ===============================
def main():
   print("Initialisation de la géométrie du tube PM (ajustée)...")
   V = np.zeros((Ny, Nx))
   V = init_conditions(V)
   V = relaxation(V)
   plot_potential(V)
