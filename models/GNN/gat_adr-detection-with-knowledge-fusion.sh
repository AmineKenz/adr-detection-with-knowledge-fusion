#!/bin/sh

# Nom de la tâche
#SBATCH --job-name=gat

# Temps maximum d'exécution (ici 100 heures)
#SBATCH --time=100:00:00

# Partition (ou queue) à utiliser
#SBATCH -p kepler

# Nombre de GPU à utiliser
#SBATCH --gres=gpu:3

# Fichiers de sortie et d'erreur
#SBATCH --output=./out_gat.txt
#SBATCH --error=./err_gat.txt

# Notifications par email
#SBATCH --mail-type=ALL # (BEGIN, END, FAIL or ALL)
#SBATCH --mail-user=mohamed-amine.kenzeddine@etu.univ-amu.fr

# Installation des bibliothèques nécessaires
pip install torch-geometric
pip install torch
pip install scikit-learn
pip install pytorch-lightning

# Exécution du script Python
python gat.py