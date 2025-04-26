import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Q1_Calcul_Potentiel import relaxation, init_conditions, scale, Nx, Ny

def calcul_champ_electrique(V):
    # Le pas spatial (chaque pixel vaut 1/scale mm)
    dx = 1 / scale  # en mm
    dy = 1 / scale  # en mm
    # Attention à l’ordre : gradient retourne (axe y, axe x)
    Ey, Ex = np.gradient(-V, dy, dx)
    return Ex, Ey

def plot_champ_electrique(V, Ex, Ey):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Paramètres géométriques
    a = 3 * scale
    b = 2 * scale
    c = 4 * scale
    d = 2 * scale
    e = int(0.2 * scale)
    N = 4

    # Coordonnées des dynodes
    dynode_coords = []
    dynode_potentiels = []
    for i in range(N // 2):
        x = int(a + i * (c + d))
        dynode_coords.append((x, b, c, e))  # dynodes du bas
        dynode_potentiels.append(100 * (2 * i + 1))
    for i in range(N // 2):
        x = int(a + (i + 0.5) * (c + d))
        dynode_coords.append((x, Ny - (b + e), c, e))  # dynodes du haut
        dynode_potentiels.append(100 * (2 * i + 2))

    # Masque pour exclure les flèches sur/trop proches des dynodes
    mask = np.ones_like(V, dtype=bool)
    margin = 3
    for x0, y0, w, h in dynode_coords:
        mask[max(0, y0 - margin):min(Ny, y0 + h + margin),
             max(0, x0 - margin):min(Nx, x0 + w + margin)] = False

    # Grille d’échantillonnage du champ
    step = 4
    x = np.arange(0, Nx, step)
    y = np.arange(0, Ny, step)
    X, Y = np.meshgrid(x, y)

    Ex_sample = Ex[::step, ::step]
    Ey_sample = Ey[::step, ::step]
    magnitude = np.sqrt(Ex_sample**2 + Ey_sample**2)

    mask_sample = mask[::step, ::step]
    Ex_sample[~mask_sample] = np.nan
    Ey_sample[~mask_sample] = np.nan
    magnitude[~mask_sample] = np.nan

    Ex_unit = np.divide(Ex_sample, magnitude, out=np.full_like(Ex_sample, np.nan), where=magnitude != 0)
    Ey_unit = np.divide(Ey_sample, magnitude, out=np.full_like(Ey_sample, np.nan), where=magnitude != 0)

    # Affichage du potentiel en fond
    im = ax.imshow(V, cmap="gray", origin="lower",
                   extent=[0, Nx / scale, 0, Ny / scale], alpha=0.5)

    # Flèches du champ électrique
    mask_plot = ~np.isnan(Ex_unit)
    Q = ax.quiver((X / scale)[mask_plot], (Y / scale)[mask_plot],
                  Ex_unit[mask_plot], Ey_unit[mask_plot],
                  magnitude[mask_plot], cmap='viridis', scale=30, width=0.005)

    # Dessin des dynodes
    for idx, (x0, y0, w, h) in enumerate(dynode_coords):
        val = dynode_potentiels[idx]
        norm_val = val / max(dynode_potentiels)
        rect = patches.Rectangle(
            (x0 / scale, y0 / scale), w / scale, h / scale,
            linewidth=0.5, edgecolor='black',
            facecolor=plt.cm.gray(norm_val), alpha=1.0, zorder=10
        )
        ax.add_patch(rect)

    # Barres de couleur
    cbar_left = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.15)
    cbar_left.set_label('Potentiel (V)')
    cbar_left.ax.yaxis.set_label_position('left')
    cbar_left.ax.yaxis.tick_left()
    cbar_left.ax.yaxis.set_ticks_position('left')

    cbar_right = fig.colorbar(Q, ax=ax, orientation='vertical', fraction=0.046, pad=0.08)
    cbar_right.set_label('|E| (V/mm)')

    ax.set_title("Champ électrique dans le tube photomultiplicateur")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.tight_layout()
    plt.savefig("figures_dos/champ_PM_dynodes_gris_sans_points.png", dpi=300)
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

if __name__ == "__main__":
    main()
