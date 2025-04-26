# equipe-21-main.py
# Fichier principal du projet - TP Électromagnétisme
# Équipe 21

import Q1_Calcul_Potentiel as q1
import Q2_Champ_Electrique as q2
import Q3_dossier.Q3_a as q3a
import Q3_dossier.Q3_b as q3b
import Q3_dossier.Q3_b_rebond as q3br
import Q3_dossier.Q3_c as q3c
import Q3_dossier.Q3_d as q3d
import Q3_dossier.Q1_pour3c as q3pc
import Q3_dossier.Q1_pour3d as q3pd


def main():
    print("\n=== EXÉCUTION DU TP D'ÉLECTROMAGNÉTISME ===\n")

    print(" Question 1: Calcul du potentiel")
    q1.main()  # Exécute la fonction main() de Q1_Calcul_Potentiel.py

    print("\n Question 2: Calcul du champ électrique")
    q2.main()  # Exécute la fonction main() de Q2_Champ_Electrique.py

    print("\n Question 3b: Résulatas graphiques")
    q3br.main()
    q3b.main()

    print("\n Question 3c: Optimisation de la géométrie")
    q3c.main()

    print("\n Question 3d: PM à 12 dynodes")
    q3d.main()

if __name__ == "__main__":
    main()

