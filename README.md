# Adversarial-diffusion-distillation-pokemon

Pokemon Generator with Adversarial Diffusion Distillation

Description
Ce projet vise à générer de nouveaux Pokémon de manière rapide et stylisée en utilisant une approche avancée de compression de modèles de diffusion : Adversarial Diffusion Distillation (ADD). L’objectif est de réduire drastiquement le temps de génération tout en conservant une qualité d’image élevée, le tout à partir de simples prompts textuels décrivant les caractéristiques du Pokémon.

Objectifs
- Finetuner un modèle de diffusion (ex. : Stable Diffusion) sur un dataset de Pokémon.
- Entraîner un modèle student plus rapide via la méthode ADD.
- Générer des Pokémon cohérents avec des prompts tels que "dragon de feu aux ailes transparentes".

Qu'est-ce que l'Adversarial Diffusion Distillation (ADD) ?
L'Adversarial Diffusion Distillation (ADD) est une méthode introduite pour compresser et accélérer les modèles de diffusion, tout en conservant leur qualité de génération. 
Elle repose sur trois piliers principaux :
1. Un modèle 'teacher' (enseignant), qui est un modèle de diffusion pré-entraîné, souvent lent mais très performant.
2. Un modèle 'student' (étudiant), que l'on veut entraîner à générer des images similaires à celles du teacher, mais en beaucoup moins d'étapes.
3. Un discriminateur adversarial, similaire à celui utilisé dans les GANs, qui apprend à distinguer les images générées par le student de celles générées par le teacher.

Le modèle student est ainsi entraîné à :
- Reproduire fidèlement les sorties du teacher à différentes étapes de diffusion (distillation).
- Tromper le discriminateur en produisant des images aussi réalistes que celles du teacher (perte adversariale).
- Réduire le bruit de façon cohérente avec le processus de diffusion inverse.
Cette combinaison permet d’obtenir un modèle student rapide, générant des images de haute qualité en seulement quelques étapes de sampling.

Dataset Pokémon
Pour entraîner notre modèle, nous utilisons le dataset "Pokémon BLIP Captions Dataset". Il s'agit d'un ensemble d'images de Pokémon issues des jeux ou de fan art, accompagnées de descriptions textuelles générées automatiquement à l'aide du modèle BLIP (Bootstrapped Language Image Pretraining). Ce dataset permet de conditionner le modèle sur des prompts textuels variés comme "petit Pokémon rouge avec des ailes de feu" ou "dragon aquatique géant bleu".
Le dataset est nettoyé et filtré pour garantir la cohérence entre les images et les descriptions. Il peut être enrichi avec des données personnalisées (DreamBooth ou LoRA) pour capturer des styles artistiques spécifiques.

Technologies utilisées
- Diffusers (Hugging Face)
- PyTorch & PyTorch Lightning
- Stable Diffusion (modèle teacher)
- Discriminateur adversarial
- CLIP (évaluation de cohérence texte-image)
- Python

Exemple de prompt
"Un Pokémon électrique en forme de loup, avec des éclairs bleus autour de lui."

Article de référence
https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1733935148453/adversarial_diffusion_distillation.pdf

Avantages de l'approche ADD
- Réduction significative du temps de génération (jusqu'à 20x plus rapide).
- Moins de ressources nécessaires pour l'inférence.
- Idéal pour les déploiements en temps réel ou sur des appareils à faible puissance.
- Possibilité de génération massive de contenus avec peu de perte de qualité.

Pipeline d'entraînement
1. Prétraitement du dataset Pokémon avec nettoyage des images et association des captions.
2. Finetuning de Stable Diffusion sur le dataset (modèle teacher).
3. Initialisation du modèle student avec le même backbone que le teacher.
4. Entraînement adversarial avec le discriminateur et la distillation de bruit.
5. Évaluation de la qualité générée avec des scores FID et la similarité CLIP.
6. Export et déploiement du modèle student pour génération rapide.

Évaluation de la qualité
La qualité des Pokémon générés est évaluée à l'aide de plusieurs métriques :
- FID (Fréchet Inception Distance) pour mesurer la similarité distributionnelle.
- CLIP Similarity Score entre le prompt et l'image générée.
- Évaluation humaine via un petit panel de testeurs.

Structure du projet
pokemon-generator/
├── data/                  # Dataset Pokémon (images + captions)
├── models/
│   ├── teacher/           # Modèle Stable Diffusion finetuné
│   └── student/           # Modèle distillé avec ADD
├── training/
│   ├── distillation.py    # Script d'entraînement ADD
│   ├── discriminator.py   # Réseau adversarial
│   └── losses.py          # Fonctions de pertes
├── generate.py            # Génération à partir de prompts
└── README.docx            # Ce document
