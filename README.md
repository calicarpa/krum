# krum

Krum est un projet autour de l’agrégation de gradients robuste en apprentissage distribué, avec un focus sur la résistance aux attaques byzantines et la comparaison de règles d’agrégation.

## Contenu de cette branche

- [ARCHITECTURE.md](ARCHITECTURE.md) : carte du dépôt et description des briques principales.
- [todo.md](todo.md) : feuille de route et pistes d’amélioration.
- [analysis/](analysis) : notes d’analyse sur différentes approches de robustesse.
- [IMPROVEMENTS_DRAFT.md](IMPROVEMENTS_DRAFT.md) : brouillon d’évolutions possibles.
- [aggregators.md](aggregators.md) : liste des aggrégateurs, attaques, pré-aggrégateurs, distributions utilisées dans les papiers

## Papiers listés

- [ByzantineMomentum](analysis/byzantine_momentum_analysis.md)
- [Byzantine-Robust-Gossip](analysis/byzantine_robust_gossip_analysis.md)
- [ByzFL](analysis/byzfl_analysis.md)
- [DECOR](analysis/decor_analysis.md)
- [Robust Collaborative Learning](analysis/robust_collaborative_learning_analysis.md)
- [RPEL-BF2D](analysis/rpel_bf2d_analysis.md)
- [SignGuard](analysis/signguard_analysis.md)

## Dépôts externes

- [Byzantine-Robust-Gossip](https://github.com/renaudgaucher/Byzantine-Robust-Gossip)
- [ByzFL](https://github.com/LPD-EPFL/byzfl)
- [Robust Collaborative Learning](https://github.com/LPD-EPFL/robust-collaborative-learning)
- [SignGuard](https://github.com/JianXu95/SignGuard)
- [DECOR](https://github.com/elfirdoussilab1/DECOR)
- [RPEL-BF2D](https://anonymous.4open.science/r/RPEL-BF2D/readme)

## Objectif

Cette branche sert à documenter et comparer des stratégies de robustesse pour l’entraînement distribué. L’idée est de garder une base lisible pour explorer des agrégateurs, des attaques et des scénarios d’évaluation.

## État actuel

Cette branche est pour l’instant centrée sur la documentation, l’analyse et la préparation d’évolutions. Les points d’entrée de code et les commandes d’exécution seront à compléter quand l’implémentation correspondante sera présente.

## Pour commencer

1. Lire [ARCHITECTURE.md](ARCHITECTURE.md) pour comprendre la structure du code.
2. Parcourir les analyses des papiers relatifs dans [analysis/](analysis).
3. Consulter [todo.md](todo.md) pour voir les prochains chantiers.
