#!/bin/sh

# Nom de la tâche
#SBATCH --job-name=my_dataloader

# Temps maximum d'exécution (ici 100 heures)
#SBATCH --time=100:00:00

# Partition (ou queue) à utiliser
#SBATCH -p kepler

# Nombre de GPU à utiliser
#SBATCH --gres=gpu:3

# Fichiers de sortie et d'erreur
#SBATCH --output=./out_my_dataloader.txt
#SBATCH --error=./err_my_dataloader.txt

# Notifications par email
#SBATCH --mail-type=ALL # (BEGIN, END, FAIL or ALL)
#SBATCH --mail-user=mohamed-amine.kenzeddine@etu.univ-amu.fr


# Installation des bibliothèques nécessaires
pip install torch
pip install spacy
pip install transformers
pip install matplotlib
pip install gensim
pip install pandas
pip install seaborn

# Exécution du script Python
python my_dataloader.py