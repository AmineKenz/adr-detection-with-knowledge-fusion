#!/bin/sh

# Nom de la tâche
#SBATCH --job-name=ernie_git_adr-detection-with-knowledge-fusion

# Temps maximum d'exécution (ici 100 heures)
#SBATCH --time=100:00:00

# Partition (ou queue) à utiliser
#SBATCH -p kepler

# Nombre de GPU à utiliser
#SBATCH --gres=gpu:3

# Fichiers de sortie et d'erreur
#SBATCH --output=./out_ernie_git_adr-detection-with-knowledge-fusion.txt
#SBATCH --error=./err_ernie_git_adr-detection-with-knowledge-fusion.txt

# Notifications par email
#SBATCH --mail-type=ALL # (BEGIN, END, FAIL or ALL)
#SBATCH --mail-user=mohamed-amine.kenzeddine@etu.univ-amu.fr



# Installation des bibliothèques nécessaires
pip install collections-extended
pip install pytorch-lightning
pip install scikit-learn
pip install spacy
pip install torch
pip install pandas
pip install seaborn
pip install matplotlib

# Exécution du script Python
python ernie.py