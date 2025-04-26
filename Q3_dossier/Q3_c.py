import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from Q3_dossier.Q1_pour3c import relaxation, init_conditions, scale, Nx, Ny, a, b, c, d, e, f, relaxation, init_conditions
from Q2_Champ_Electrique import calcul_champ_electrique

def main():
    global a,b,c,d,e,f,scale
    # === Affichage des paramètres géométriques ===
    print("\n=== Paramètres géométriques utilisés pour la question 3c ===")
    print(f"a (espace dynode-extrémité) : {a/scale:.2f} mm")
    print(f"b (espace dynode-paroi)     : {b/scale:.2f} mm")
    print(f"c (longueur dynode)         : {c/scale:.2f} mm")
    print(f"d (distance dynodes même côté) : {d/scale:.2f} mm")
    print(f"e (épaisseur dynode)        : {e/scale:.2f} mm")
    print(f"f (hauteur du tube)         : {f/scale:.2f} mm")
    print("===============================================\n")

    # Constantes physiques
    e_charge = -1.602e-19
    m = 9.109e-31

    # Simulation
    dt = 3e-11
    duree_totale = 1.7e-7
    nb_steps = int(duree_totale / dt)

    # Position initiale
    x0_mm = 0.0
    y0_mm = f / (2 * scale)  # départ centré
    vx0, vy0 = 0.0, 0.0

    x0 = x0_mm * scale
    y0 = y0_mm * scale

    # Potentiel et champ
    V = np.zeros((Ny, Nx))
    V = init_conditions(V)
    V = relaxation(V)
    Ex, Ey = calcul_champ_electrique(V)

    # Interpolateurs
    x_vals = np.arange(Nx)
    y_vals = np.arange(Ny)
    interp_Ex = RegularGridInterpolator((y_vals, x_vals), Ex, bounds_error=False, fill_value=0)
    interp_Ey = RegularGridInterpolator((y_vals, x_vals), Ey, bounds_error=False, fill_value=0)

    # Dynodes ordonnées
    def get_dynodes_ordonnes(scale, a, b, c, d, e, f):
        y_center = f // 2
        y_bas = int(y_center - 2 * scale - e // 2)
        y_haut = int(y_center + 2 * scale - e // 2)

        dynodes_ordonnees = []

        x0 = int(a + 0 * (c + d))
        dynodes_ordonnees.append({'x': x0, 'y': y_bas, 'c': c, 'e': e})

        x1 = int(a + (0.5) * (c + d))
        dynodes_ordonnees.append({'x': x1, 'y': y_haut, 'c': c, 'e': e})

        x2 = int(a + 1 * (c + d))
        dynodes_ordonnees.append({'x': x2, 'y': y_bas, 'c': c, 'e': e})

        x3 = int(a + (1.5) * (c + d))
        dynodes_ordonnees.append({'x': x3, 'y': y_haut, 'c': c, 'e': e})

        return dynodes_ordonnees

    dynodes_sequence = get_dynodes_ordonnes(scale, a, b, c, d, e, f)
    dynode_idx = 0

    rebond_pixels = 2 * scale

    # Détection impact dynode
    def electron_touche_dynode(x, y, dynode):
        return (
            dynode['x'] <= x <= dynode['x'] + dynode['c'] and
            dynode['y'] <= y <= dynode['y'] + dynode['e']
        )

    # Simulation
    positions = [(x0, y0)]
    x, y = x0, y0
    vx, vy = vx0, vy0

    for step in range(nb_steps):
        if not (0 <= x < Nx and 0 <= y < Ny):
            break

        E = np.array([interp_Ex((y, x)), interp_Ey((y, x))])
        ax = (e_charge / m) * E[0] * scale * 1e3
        ay = (e_charge / m) * E[1] * scale * 1e3

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        x = min(max(x, 0), Nx - 1)
        y = min(max(y, 0), Ny - 1)

        if dynode_idx < len(dynodes_sequence):
            d = dynodes_sequence[dynode_idx]
            if electron_touche_dynode(x, y, d):
                direction = -1 if vy > 0 else 1
                y += direction * rebond_pixels
                vy = 0
                dynode_idx += 1

        positions.append((x, y))

    # Conversion
    positions = np.array(positions) / scale

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.imshow(V, cmap="inferno", origin="lower", extent=[0, Nx/scale, 0, Ny/scale])
    plt.colorbar(label="Potentiel (V)")
    plt.plot(positions[:, 0], positions[:, 1], color="cyan", lw=1.5, label="Trajectoire")
    plt.scatter([x0_mm], [y0_mm], color="cyan", label="Départ", zorder=5)
    plt.title("Trajectoire de l’électron dans un photomultiplicateur ")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_dos/trajectoire_ordonnée_3c.png")
    plt.show()
if __name__ == "__main__":
    main()













