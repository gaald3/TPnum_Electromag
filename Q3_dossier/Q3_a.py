import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from Q1_Calcul_Potentiel import relaxation, init_conditions, scale, Nx, Ny
from Q2_Champ_Electrique import calcul_champ_electrique

# Constantes physiques
e = -1.602e-19  # charge de l'électron (C)
m = 9.109e-31   # masse de l'électron (kg)

# Paramètres de simulation
duree_totale = 1.3e-7
dt = 3e-11  # Pas de temps (s)
nb_steps = int(duree_totale / dt)  # Nombre de pas de temps

# Conditions initiales (en mm)
x0_mm, y0_mm = 0.0, 3.0  # position initiale
vx0, vy0 = 0.0, 0.0      # vitesse initiale (mm/s)

# Conversion en pixels
x0 = x0_mm * scale
y0 = y0_mm * scale

# Préparer le potentiel et le champ
V = np.zeros((Ny, Nx))
V = init_conditions(V)
V = relaxation(V, tol=1e-3)
Ex, Ey = calcul_champ_electrique(V)

# Création d'interpolateurs pour le champ
x = np.arange(Nx)
y = np.arange(Ny)
interp_Ex = RegularGridInterpolator((y, x), Ex)
interp_Ey = RegularGridInterpolator((y, x), Ey)

# Initialisation des trajectoires
positions = [(x0, y0)]
vitesses = [(vx0, vy0)]

# Potentiels des dynodes
dynode_potentiels = [100, 200, 300, 400]
rebound_margin = 1  # tolérance autour du potentiel (V)

# Simulation avec rebond sur dynodes
x, y = x0, y0
vx, vy = vx0, vy0
for step in range(nb_steps):
    if not (0 <= x < Nx and 0 <= y < Ny):
        break  # L'électron sort du domaine

    # Champ local
    E = np.array([interp_Ex((y, x)), interp_Ey((y, x))])
    ax = (e / m) * E[0] * scale * 1e3  # en mm/s²
    ay = (e / m) * E[1] * scale * 1e3

    # Intégration d'Euler
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt

    # Vérification d'impact sur dynode (rebond)
    local_V = V[int(round(y)), int(round(x))]
    if any(abs(local_V - vp) <= rebound_margin for vp in dynode_potentiels):
        vy = -vy  # rebond vertical

    positions.append((x, y))
    vitesses.append((vx, vy))

# Conversion en mm pour affichage
positions = np.array(positions) / scale
