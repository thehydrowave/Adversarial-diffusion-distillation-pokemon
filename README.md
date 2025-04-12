# Naruto Generator avec Adversarial Diffusion Distillation

## Description
Ce projet vise à générer de nouvelles images stylisées de l’univers **Naruto** de manière rapide en utilisant une approche avancée de compression de modèles de diffusion : **Adversarial Diffusion Distillation (ADD)**.  
L’objectif est de conserver la qualité artistique des personnages ou scènes tout en accélérant le temps de génération, à partir de simples prompts textuels.

## Objectifs
- Finetuner un modèle de diffusion (ex. : **Stable Diffusion**) sur un dataset basé sur Naruto.
- Entraîner un modèle *student* plus léger via la méthode **ADD**.
- Générer des images cohérentes avec des prompts comme *"ninja aux cheveux blancs avec un bandeau rouge"*.

---

## Qu'est-ce que l'Adversarial Diffusion Distillation (ADD) ?
L'**ADD** est une méthode pour compresser et accélérer les modèles de diffusion tout en conservant une qualité visuelle élevée.  
Elle s'appuie sur :
1. **Un modèle teacher** : modèle Stable Diffusion finetuné sur l’univers Naruto.
2. **Un modèle student** : version plus rapide, distillée à partir du teacher.
3. **Un discriminateur** : apprend à distinguer les images du student de celles du teacher.

Le student est entraîné à :
- **Imiter** les sorties du teacher à différentes étapes.
- **Tromper** le discriminateur avec des images réalistes.
- **Dénoyer** les latents en suivant le processus de diffusion inverse.

---

## Dataset Naruto
Le dataset utilisé contient des images issues de l’univers Naruto (screenshots, fan arts, extraits de manga), accompagnées de **descriptions textuelles** générées ou annotées, décrivant les personnages, leurs actions ou leurs environnements.

Exemples de captions :
- *"ninja en armure sombre lançant un shuriken"*
- *"personnage blond avec des marques sur le visage et des vêtements orange"*

---

## Technologies utilisées
- **Diffusers (Hugging Face)**
- **PyTorch**
- **Stable Diffusion finetuné** (teacher)
- **Modèle UNet réduit** (student)
- **Discriminateur adversarial**
- **CLIP pour l’évaluation**
- **Python (via Notebooks Jupyter)**

---

## Exemple de prompt
*"Un ninja encapuchonné manipulant du feu bleu dans une forêt sombre."*

---

## Article de référence
[Adversarial Diffusion Distillation](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1733935148453/adversarial_diffusion_distillation.pdf)

---

## Avantages de l’approche ADD
- Génération **jusqu’à 20x plus rapide** qu’un modèle classique.
- Optimisé pour **inférence en temps réel** ou sur appareils avec peu de mémoire.
- **Moins de coût de calcul** pour des performances visuelles proches du teacher.

---

## Pipeline d'entraînement
1. **Chargement** des images Naruto et de leurs captions.
2. **Finetuning** de Stable Diffusion pour capturer le style Naruto (*modèle teacher*).
3. **Construction** du modèle student (réduction du UNet).
4. **Entraînement ADD** avec distillation + adversarial loss.
5. **Visualisation** des sorties pour comparer student vs teacher.
6. **Évaluation** avec **FID** et **CLIP Similarity**.
7. **Export** du modèle student pour génération rapide.

---

## Évaluation de la qualité
- **FID** : mesure la similarité de distribution entre vraies et générées.
- **CLIP Similarity** : mesure la cohérence texte-image.
- **Évaluations qualitatives** : inspection visuelle des résultats.

---

## Structure du projet
```
naruto-generator/
├── notebooks/
│   ├── Teacher.ipynb # Stable Diffusion finetune
│   └── Student.ipynb # ADD model distill
└── README.md         # Proposal
```
