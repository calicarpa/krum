# Tutorial writing TODO

## État des lieux

7 tutoriels existants — bonne couverture usage + extension.

## À faire

- [x] **Decentralised simulation walkthrough** — tutorial pas-à-pas sur `DecentralisedSimulation`. Renommé `centralised_simulation_walkthrough` pour la version centralisée.

- [ ] **Results analysis** — exploiter un `MetricDataFrame` : filtrer, pivoter, plotter avec pandas/matplotlib, exporter en CSV/JSON. Prend le relais de `working_with_orchestrator`.

- [ ] **Systematic benchmark** — lancer N agg × M attacks sur un dataset, produire un tableau comparatif de précision/résilience. Cas d'usage recherche courant.

- [ ] **Understanding Byzantine attacks** — tutoriel conceptuel : threat model, budget byzantin (f < n/2, f < n/3), panorama des attaques et leurs effets.

- [ ] **Choosing an aggregation rule** — quel GAR pour quel scénario. Tableau des hypothèses (n, f, dimension, type d'attaque), guide de sélection.

- [ ] **Custom dataset** — utiliser ses propres données (hors MNIST/CIFAR/Spambase) avec les simulations Krum.
