import numpy as np
from Q1_pour3c import scale

def calcul_champ_electrique(V):
    """
    Calcule le champ électrique Ex et Ey à partir du potentiel V.
    Le champ est retourné en V/mm.
    """
    Ny, Nx = V.shape
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)

    # Calcul de Ex (dérivée selon x)
    Ex[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 / scale)  # V/mm
    Ex[:, 0] = -(V[:, 1] - V[:, 0]) / (1 / scale)
    Ex[:, -1] = -(V[:, -1] - V[:, -2]) / (1 / scale)

    # Calcul de Ey (dérivée selon y)
    Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 / scale)  # V/mm
    Ey[0, :] = -(V[1, :] - V[0, :]) / (1 / scale)
    Ey[-1, :] = -(V[-1, :] - V[-2, :]) / (1 / scale)

    return Ex, Ey
