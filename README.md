# Activité Pratique N°2 — Modèles génératifs avec PyTorch (MNIST)

**Étudiante :** Tawhida AMYAR — EMSI Roudani, 5IDAM
**Cours :** IA Générative Multi-Modale — Pr. Mohamed YOUSSFI

## Objectif
Avec **PyTorch**, créer, entraîner, tester et explorer plusieurs familles de modèles sur le jeu de données **MNIST** (chiffres manuscrits 28×28), afin de comparer l'approche **discriminative** et les approches **génératives**.

## Contenu du dépôt
| Fichier | Description |
|---|---|
| `modeles_generatifs_pytorch.ipynb` | Notebook complet : code, explications et visualisations |
| `pyproject.toml` | Configuration du projet et dépendances (UV) |
| `.gitignore` | Exclut `.venv/` et `data/` |
| `README.md` | Ce document |

## Contenu du notebook
| Partie | Modèle | Ce qui est réalisé |
|---|---|---|
| 0–1 | Préparation | Environnement, chargement de MNIST, `DataLoader` (batch 128), affichage d'exemples |
| I | **Classifieur** (modèle discriminatif) | Réseau 784→128→10, perte `CrossEntropy`, Adam, évaluation : **accuracy ≈ 96 %** sur le test (mesurée à la main et avec `torcheval`) |
| II | **Autoencodeur** | Encodeur/décodeur, espace latent 2D, perte MSE, reconstructions, visualisation de l'espace latent, génération d'un chiffre moyen |
| III | **VAE** | Reparameterization trick, perte = reconstruction (BCE) + divergence KL, génération par échantillonnage de z ~ N(0, I), exploration d'une grille 15×15 de l'espace latent |
| IV | **GAN** | Générateur (100→256→512→784) et discriminateur (784→512→256→1), entraînement adversarial, génération de nouvelles images |
| V | **Modèle de diffusion** | Planning de bruit (T = 200), réseau prédisant le bruit avec embedding du temps, diffusion inverse, courbe de perte |

## Installation avec UV
```bash
uv init --python=3.11
uv venv
# Windows : .venv\Scripts\activate      Linux/Mac : source .venv/bin/activate
uv add ipykernel torch torchvision torcheval matplotlib numpy
```
Ouvrir ensuite le notebook dans VS Code, choisir le noyau `.venv`, puis exécuter les cellules dans l'ordre. MNIST est téléchargé automatiquement dans `./data`.

## Technologies
Python 3.11, PyTorch, torchvision, torcheval, Matplotlib, NumPy, UV, VS Code / Jupyter

## Résumé des enseignements
- Le **classifieur** apprend P(y|x) : il distingue les classes mais ne sait pas générer.
- L'**autoencodeur** apprend une représentation latente, mais l'espace latent n'est pas forcément régulier.
- Le **VAE** régularise l'espace latent (terme KL) et permet de générer de nouveaux chiffres.
- Le **GAN** apprend par compétition entre générateur et discriminateur.
- La **diffusion** apprend à retirer progressivement du bruit pour produire des images.
