# Todo : Krum & Robust Federated Learning

## 1. Reproductibilité et Expérience Développeur
- [ ] Écrire un `README.md` court avec les commandes d'exécution minimales (`train.py`, `reproduce.py`).
- [ ] Clarifier la syntaxe du CLI avec des exemples exacts pour passer les arguments hyperparamétriques.

## 2. Refonte de l'Extensibilité - Le cœur du moteur
- [ ] Créer une classe `BaseAggregator` avec un contrat d'interface strict (ex: forcer l'implémentation d'une méthode `aggregate`).
- [ ] Créer une classe `BaseAttack` pour standardiser l'injection de comportements byzantins ou d'empoisonnement.
- [ ] Ajouter des *hooks* de pré-agrégation pour permettre l'insertion de filtres statistiques (ex: filtrage par signe ou amplitude) avant l'agrégateur principal.

## 3. Implémentations Stratégiques ?
- [ ] Intégrer un gestionnaire de topologie dynamique pour supporter des matrices de connectivité arbitraires (graphes bipartis, Erdős-Rényi) au lieu de supposer un graphe complet.
- [ ] Découpler strictement la couche "Transport/Communication" de la couche "Robustesse".
- [ ] Ajouter des opérateurs de compression et de quantification des tenseurs compatibles avec le pipeline d'agrégation robuste.


Peer to peer - lowest priority
Non iid distribution
Distributed execution - lowest priority
Topology -> Client & Server
Documentation:
    - Pytorch -> Flatten / Relink (comment bien les utiliser)
    - Ajout d'un modèle et datasets
    - Annoter et typage !! (Aggregateur, Attaque, Experiments)


-> Message Mahdi
