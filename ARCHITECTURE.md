# Architecture du dépôt krum

Ce document sert de carte de lecture pour le codebase. Le projet est pensé pour des chercheurs qui veulent comparer des règles d'agrégation robustes, simuler des attaques byzantines, lancer des campagnes d'expériences et analyser les résultats ensuite.

L'idée centrale est simple :

1. un script d'entrée construit un modèle, un jeu de données, une fonction de perte, une métrique et un optimiseur ;
2. chaque pas d'entraînement produit des gradients honnêtes ;
3. une attaque fabrique des gradients byzantins ;
4. une règle d'agrégation combine tous les gradients ;
5. le tout est mesuré, tracé et sauvegardé dans des fichiers tabulaires ou JSON ;
6. les scripts de post-traitement relisent ces artefacts pour faire des graphiques.

Le dépôt est volontairement modulaire. La plupart des composants sont chargés automatiquement à partir de leur dossier, ce qui permet à un chercheur de cloner le dépôt, d'ajouter un fichier Python, et de rendre la nouvelle brique immédiatement disponible par nom.

## 1. Vue d'ensemble

### 1.1 Les quatre couches du projet

Le dépôt s'organise naturellement en quatre couches :

- **Orchestration** : [train.py](train.py) lance un entraînement, [reproduce.py](reproduce.py) lance des campagnes de plusieurs entraînements, [histogram.py](histogram.py) charge et visualise les résultats.
- **Briques de recherche** : [aggregators/](aggregators) contient les règles d'agrégation robustes, [attacks/](attacks) contient les attaques byzantines, [experiments/](experiments) contient les wrappers autour de PyTorch et TorchVision.
- **Infrastructure** : [tools/](tools) fournit le logging contextuel, les conversions d'arguments, les helpers de tenseurs, la gestion des jobs et quelques utilitaires généraux.
- **Accélération native** : [native/](native) compile automatiquement les modules C++/CUDA quand ils sont disponibles.

### 1.2 Le flux de données principal

```text
train.py
  -> Configuration de l'environnement
  -> Construction du modèle, des datasets, de la perte et du criterion
  -> Boucle de calcul des gradients honnêtes
  -> Clipping / bruit de confidentialité / attaque / agrégation
  -> Mise à jour du modèle
  -> Écriture des fichiers de résultats

reproduce.py
  -> Préchargement de certaines données
  -> Construction de lots d'expériences
  -> Exécution parallèle sur plusieurs devices
  -> Log des sorties standard et erreur

histogram.py
  -> Lecture des fichiers de résultats
  -> Calcul de colonnes dérivées
  -> Graphiques ligne / histogramme
```

## 2. Carte du dépôt

| Emplacement | Rôle principal | Ce qu'un chercheur y étend en priorité |
| --- | --- | --- |
| [train.py](train.py) | Simulation d'un entraînement sous attaque | Nouvelles mesures par pas, nouveaux artefacts de sortie, nouveaux paramètres CLI |
| [reproduce.py](reproduce.py) | Orchestration de campagnes reproductibles | Grilles d'expériences, nouveaux jeux de paramètres, parallélisme de lancement |
| [histogram.py](histogram.py) | Analyse et visualisation des résultats | Nouvelles colonnes dérivées, nouvelles courbes, nouveaux graphiques |
| [aggregators/](aggregators) | Règles d'agrégation robustes | Nouvelles règles, variantes natives, nouvelles heuristiques d'influence |
| [attacks/](attacks) | Attaques byzantines | Nouvelles directions d'attaque, nouveaux scénarios adversariaux |
| [experiments/](experiments) | Wrappers autour de modèles, datasets, pertes, critères, optimisateurs, checkpoints | Nouveaux modèles, datasets, métriques, options de reprise |
| [native/](native) | Extensions C++/CUDA auto-compilées | Implémentations rapides des règles ou attaques déjà prototypées en Python |
| [tools/](tools) | Utilitaires transverses | Fonctionnalités génériques de logging, parallélisme, parsing, temps, tensor plumbing |

## 3. Les scripts d'entrée

### 3.1 [train.py](train.py)

`train.py` est le cœur du dépôt. Il simule un entraînement distribué simplifié avec workers honnêtes et workers byzantins.

Il gère notamment :

- le parsing d'une très grande surface CLI ;
- la sélection de device pour le modèle et éventuellement un device différent pour la règle d'agrégation ;
- la construction du modèle, du dataset, de la perte, du criterion et de l'optimiseur ;
- la boucle d'entraînement ;
- l'ajout optionnel de bruit de confidentialité gaussien ;
- la sauvegarde des mesures dans le répertoire de résultats ;
- la mesure du temps passé dans chaque phase ;
- l'interruption propre via `SIGINT` et `SIGTERM`.

Le script crée plusieurs files de sortie :

- `config` : résumé textuel de la configuration ;
- `config.json` : version JSON des arguments ;
- `study` : tableau tabulé contenant les métriques détaillées par pas ;
- `eval` : tableau tabulé de l'évaluation cross-accuracy ;
- `perfs.json` : temps accumulés par phase.

Le script est écrit comme un workflow séquentiel : d'abord la configuration, ensuite la préparation, puis la boucle d'apprentissage, enfin la synthèse des temps.

### 3.2 [reproduce.py](reproduce.py)

`reproduce.py` n'entraîne pas lui-même un modèle. Il sert à relancer des expériences de manière industrialisée :

- il prépare les répertoires de données et de figures ;
- il résout les devices disponibles ;
- il précharge certaines données pour éviter des téléchargements concurrents ;
- il construit des commandes `train.py` ;
- il distribue les jobs sur plusieurs threads et plusieurs devices ;
- il archive stdout/stderr de chaque run.

Point important : les jobs sont lancés avec `python3 -OO train.py`. Cela signifie que les assertions et certains chemins de debug sont désactivés dans les exécutions reproductibles, ce qui correspond bien à l'intention de production. C'est aussi la raison pour laquelle les wrappers d'attaque et d'agrégation basculent sur la version `unchecked` quand `__debug__` est faux.

### 3.3 [histogram.py](histogram.py)

`histogram.py` est le script de post-traitement. Il lit les répertoires de résultats, reconstruit les tableaux de métriques et produit des visualisations.

Il contient trois briques utiles :

- `Session` : charge un répertoire de résultats et expose les colonnes sous forme de `DataFrame` ;
- `LinePlot` : construit des courbes avec un ou deux axes Y ;
- `HistPlot` : construit des histogrammes simples.

Le script a été conçu pour une utilisation interactive si GTK est disponible, mais il conserve un mode dégradé pour les environnements sans interface graphique.

## 4. Les classes et objets centraux

### 4.1 Le socle `tools`

Le dossier [tools/](tools) est l'infrastructure commune du projet. C'est le premier endroit à comprendre si l'on veut étendre proprement le dépôt.

#### `tools.UserException`

- Classe de base pour les erreurs utilisateur.
- Elle sert aux validations de paramètres et aux erreurs attendues.
- Le hook d'exception global l'affiche proprement avant de quitter.

#### `tools.Context`

- Gère un contexte d'exécution par thread.
- Ajoute un préfixe de contexte et une couleur aux messages imprimés.
- Sert à structurer les phases de chargement, de configuration, d'entraînement, de performance, etc.
- C'est la convention à suivre pour tout nouveau sous-système qui veut loguer proprement.

#### `tools.ContextIOWrapper`

- Enveloppe `stdout` et `stderr`.
- Préfixe chaque ligne avec le contexte courant.
- Gère le comportement coloré ou non coloré selon le terminal.

#### `tools.TimedContext` et `tools.AccumulatedTimedContext`

- `TimedContext` mesure et affiche le temps écoulé à la sortie d'un bloc.
- `AccumulatedTimedContext` accumule des temps sans impression immédiate.
- `train.py` les utilise pour mesurer gradient, bruit, agrégation et évaluation.

#### `tools.MethodCallReplicator`

- Réplique un appel de méthode sur plusieurs objets.
- Utile pour faire agir plusieurs instances comme une seule façade.

#### `tools.ClassRegister`

- Registre générique de classes nommées.
- Sert de base à plusieurs patterns de découverte par nom.

#### `tools.UnavailableException` et `tools.fatal_unavailable`

- Erreurs standardisées quand une clé demandée n'existe pas dans un registre.
- Très utile pour les modèles, datasets, pertes, critères et optimiseurs.

#### `tools.parse_keyval`

- Parse les arguments `cle:valeur` passés par `--*-args`.
- Convertit automatiquement bool, int, float, string.
- C'est le format de passage de paramètres utilisé par les scripts d'entrée.

#### `tools.onetime`

- Fournit une variable globale protégée par verrou, utilisée ici pour le signal d'arrêt propre.

#### `tools.line_maximize`

- Heuristique de maximisation sur une ligne.
- Sert à choisir un facteur d'attaque quand la meilleure intensité n'est pas connue à l'avance.

#### `tools.pairwise`

- Génère toutes les paires d'indices utiles aux calculs pairwise des règles robustes.

#### `tools.interactive`

- Petit shell interactif intégré.
- Utilisé par `train.py` quand `--user-input-delta` active une pause interactive.

#### `tools.fullqual`

- Renvoie le nom pleinement qualifié d'une classe ou instance.
- Utile pour les messages d'erreur et les registres.

### 4.1.1 `tools.pytorch`

Fichier : [tools/pytorch.py](tools/pytorch.py)

Ce module contient les helpers PyTorch qui rendent le reste du dépôt plus simple à écrire et plus cohérent.

#### Fonctions de manipulation mémoire

- `relink(tensors, common)` : fait pointer plusieurs tenseurs vers un même bloc mémoire contigu.
- `flatten(tensors)` : concatène les tenseurs et les relie à un stockage commun.
- `grad_of(tensor)` : récupère un gradient existant ou en crée un nul.
- `grads_of(tensors)` : version génératrice de `grad_of` pour une collection de paramètres.

#### Fonctions de statistiques et de mesure

- `compute_avg_dev_max(samples)` : calcule moyenne, norme moyenne, écart-type de norme et max coordonnée.
- `AccumulatedTimedContext` : mesure accumulée des temps de sections critiques avec synchronisation CUDA optionnelle.

#### Fonctions utilitaires de recherche

- `weighted_mse_loss(tno, tne, tnw)` et `WeightedMSELoss` : utile pour des expérimentations pondérées.
- `regression(func, vars, data, loss=..., opt=..., steps=...)` : mini-boucle d'optimisation générique pour ajuster des variables libres.
- `pnm(fd, tn)` : export d'un tenseur en flux PGM/PBM, pratique pour sauvegarder des images ou visualiser des tenseurs.

#### Ce que ce module permet architecturalement

- D'aplatir les paramètres et gradients de manière compatible avec les règles d'agrégation.
- De partager un même stockage entre vues tensorielles, ce qui simplifie les copies et les mises à jour.
- De centraliser des helpers très réutilisés pour éviter de dupliquer les mêmes bouts de logique dans `train.py`, `experiments/` et les scripts de recherche.

### 4.1.2 `tools.jobs`

Fichier : [tools/jobs.py](tools/jobs.py)

Ce module gère l'exécution reproductible des expériences depuis `reproduce.py`.

#### Fonctions et classes principales

- `move_directory(path)` : déplace un répertoire existant vers un nom horodaté numéroté avant recréation.
- `dict_to_cmdlist(dp)` : transforme un dictionnaire d'arguments en liste CLI `--clé valeur`.
- `Command` : encapsule une commande de base et ajoute `seed`, `device` et `result-directory` au lancement.
- `Jobs` : orchestre l'exécution parallèle des expériences sur plusieurs devices.

#### Ce que `Jobs` garantit

- Les expériences sont répétées pour chaque seed déclaré.
- Chaque exécution écrit ses sorties dans un répertoire dédié.
- Un run réussi finit dans un dossier final, un run raté dans un dossier `.failed`.
- Les `stdout.log` et `stderr.log` sont archivés pour post-mortem.

#### Ce que ce module permet architecturalement

- De séparer la définition d'une expérience de son exécution réelle.
- De standardiser les conventions de lancement et de répertoire de sortie.
- De rendre la reproduction des campagnes plus sûre, surtout quand plusieurs threads et devices sont impliqués.

### 4.2 `experiments.Configuration`

Fichier : [experiments/configuration.py](experiments/configuration.py)

`Configuration` est un petit conteneur immuable qui transporte les paramètres tensoriels partagés :

- `device` ;
- `dtype` ;
- `non_blocking` ;
- `relink`.

Pourquoi il est important : le code manipule souvent des paramètres et gradients aplatis. La configuration dit si les copies doivent être faites, si on peut relier les vues mémoire, et vers quel device déplacer les batchs.

### 4.3 `experiments.Model`

Fichier : [experiments/model.py](experiments/model.py)

`Model` est l'un des objets les plus importants du dépôt. Ce n'est pas simplement un `torch.nn.Module` encapsulé, c'est un conteneur d'état qui centralise :

- le modèle lui-même ;
- la configuration tensorielle ;
- une version aplatie des paramètres ;
- le gradient aplati ;
- des valeurs par défaut pour `trainset`, `testset`, `loss`, `criterion`, `optimizer`.

#### Découverte automatique des modèles

`Model` construit un registre de constructeurs à partir de :

- `torchvision.models` sous le préfixe `torchvision-<nom>` ;
- [experiments/models/](experiments/models) pour les modèles custom ;
- `__all__` quand il est présent dans les modules custom.

Dans le dépôt courant, [experiments/models/simples.py](experiments/models/simples.py) expose notamment `full`, `conv`, `logit` et `linear`.

#### Méthodes importantes

- `run(data, training=False)` : exécute le forward pass.
- `loss(dataset=None, loss=None, training=None)` : calcule la perte sur un batch.
- `backprop(dataset=None, loss=None, outloss=False)` : calcule le gradient aplati.
- `update(gradient, optimizer=None, relink=None)` : applique un gradient avec l'optimiseur courant.
- `eval(dataset=None, criterion=None)` : calcule la métrique d'évaluation.
- `get()` / `set(params)` : accèdent au vecteur aplati des paramètres.
- `get_gradient()` / `set_gradient(gradient)` : accèdent au gradient aplati.
- `default(name, new=None, erase=False)` : stocke ou lit les dépendances implicites du modèle.

#### Points architecturaux à retenir

- Les paramètres sont aplatis via `tools.flatten`.
- Si le device est CUDA sans index explicite, le modèle peut être enveloppé dans `torch.nn.DataParallel`.
- Le modèle agit donc comme une façade de haut niveau pour des expérimentations distribuées ou semi-distribuées.

### 4.4 `experiments.Dataset`

Fichier : [experiments/dataset.py](experiments/dataset.py)

`Dataset` encapsule trois cas d'usage :

- un dataset nommé récupéré dans les registres ;
- un générateur Python fourni directement par l'utilisateur ;
- un batch unique répété en boucle.

#### Découverte automatique

Le registre est construit à partir de :

- `torchvision.datasets` sous la forme `nom_en_minuscules` ;
- [experiments/datasets/](experiments/datasets) pour les datasets custom ;
- `__all__` dans les modules custom, si présent.

Dans le dépôt courant, [experiments/datasets/svm.py](experiments/datasets/svm.py) expose `phishing`, avec téléchargement et cache local.

#### Fonctions associées

- `get_default_transform(dataset, train)` : donne les transformations par défaut d'un dataset connu.
- `make_datasets(...)` : construit un trainset et un testset cohérents.
- `batch_dataset(inputs, labels, ...)` : transforme des tenseurs bruts en générateurs infinis de minibatchs.
- `make_sampler(loader)` : boucle indéfiniment sur un `DataLoader`.

#### Ce qu'il faut retenir pour de nouvelles recherches

- Si vous ajoutez un dataset, renvoyez idéalement un générateur infini ou un wrapper compatible avec `Dataset.sample`.
- Si vous voulez un dataset téléchargé automatiquement, le pattern déjà présent dans `svm.py` est le bon point de départ.

### 4.5 `experiments.Loss` et `experiments.Criterion`

Fichier : [experiments/loss.py](experiments/loss.py)

Ces deux classes sont souvent confondues, mais elles remplissent des rôles différents.

#### `Loss`

- Objet différentiable.
- Accepte un constructeur PyTorch ou un nom enregistré.
- Peut être combiné avec `+` et `*` pour composer la perte finale.
- Prend aussi en charge deux régularisations internes : `l1` et `l2`.

#### `Criterion`

- Objet de mesure, pas de rétropropagation.
- Retourne un tenseur à deux composantes : nombre de bonnes prédictions et taille du batch.
- Les implémentations intégrées sont `top-k` et `sigmoid`.

#### Pourquoi cette séparation est utile

- La perte sert à calculer les gradients.
- Le criterion sert à évaluer et à journaliser l'évolution du modèle.
- Cette séparation rend la boucle de recherche plus lisible et permet d'ajouter de nouvelles métriques sans casser le calcul de gradient.

### 4.6 `experiments.Optimizer`

Fichier : [experiments/optimizer.py](experiments/optimizer.py)

`Optimizer` encapsule un optimiseur PyTorch choisi par nom ou par constructeur callable.

Points essentiels :

- le constructeur reçoit un `Model` et optimise ses paramètres sous-jacents ;
- `__getattr__` relaie vers l'optimiseur réel ;
- `set_lr(lr)` modifie le learning rate de tous les groupes de paramètres.

Dans `train.py`, le dépôt utilise actuellement `sgd`, mais l'abstraction permet d'en changer sans casser le reste du pipeline.

### 4.7 `experiments.Checkpoint` et `experiments.Storage`

Fichier : [experiments/checkpoint.py](experiments/checkpoint.py)

Le sous-système de checkpoint fournit un mécanisme générique de snapshot/restauration basé sur le protocole `state_dict`.

#### `Checkpoint`

- Stocke des dictionnaires d'état par type d'objet.
- Connaît des transferts spéciaux pour `Model` et `Optimizer` afin de checkpoint le vrai module/optimiseur sous-jacent.
- Propose `snapshot`, `restore`, `load`, `save`.

#### `Storage`

- Hérite de `dict`.
- Implémente `state_dict()` et `load_state_dict()`.
- Permet de rendre un simple dictionnaire compatible avec le pipeline de checkpoint.

Ce sous-système est particulièrement utile si vos recherches demandent des reprises longues, des ablations ou des restaurations fréquentes entre variantes d'expérience.

## 5. Les registres d'agrégation et d'attaque

### 5.1 Contrat des agrégateurs

Fichier : [aggregators/__init__.py](aggregators/__init__.py)

Les agrégateurs sont des fonctions d'agrégation robustes, appelées avec des arguments nommés uniquement.

#### Contrat attendu

- `gradients` : liste non vide de gradients ;
- `f` : nombre de gradients byzantins déclarés ;
- `model` : modèle avec ses valeurs par défaut configurées ;
- arguments additionnels spécifiques à la règle.

#### Ce que `register(...)` fabrique

Pour chaque règle, le registre expose :

- la fonction sélectionnée par défaut ;
- `.checked` : version validante ;
- `.unchecked` : version brute ;
- `.check` : la fonction de validation ;
- `.upper_bound` : borne théorique si disponible ;
- `.influence` : ratio de gradients byzantins acceptés si disponible.

#### Règles fournies

- `average` : moyenne arithmétique simple.
- `median` : médiane coordonnée par coordonnée.
- `brute` : recherche exhaustive de la sous-collection de plus petit diamètre.
- `krum` / `multi-krum` : sélection par score de distances.
- `bulyan` : sélection multi-krum suivie d'un agrégat coordonné plus robuste.

Certaines règles ont aussi une variante native `native-<nom>` si [native/](native) fournit le module correspondant.

### 5.2 Contrat des attaques

Fichier : [attacks/__init__.py](attacks/__init__.py)

Les attaques sont elles aussi des fonctions nommées, appelées en keyword-only.

#### Contrat attendu

- `grad_honests` : liste non vide de gradients honnêtes ;
- `f_decl` : nombre de byzantins déclarés ;
- `f_real` : nombre de byzantins réellement générés ;
- `model` : modèle utilisé pour l'attaque ;
- `defense` : règle d'agrégation à tromper ;
- autres paramètres propres à l'attaque.

#### Ce que `register(...)` garantit

- la version par défaut bascule sur la version validante en mode debug ;
- la version validante vérifie les paramètres et la taille de la sortie ;
- la sortie est toujours une liste de `f_real` gradients.

#### Attaques fournies

- `nan` : fabrique des gradients non finis ;
- `identical` : fabrique plusieurs copies d'un même gradient byzantin ;
- `bulyan`, `empire`, `little` : directions d'attaque issues de travaux connus.

#### Point important

Les gradients retournés ne doivent jamais aliaser les gradients d'entrée. C'est une contrainte forte du projet, à respecter pour éviter des effets de bord dans les variantes natives ou les scripts de recherche.

### 5.3 Règles existantes en détail

#### `average` ([aggregators/average.py](aggregators/average.py))

- Implémentation la plus simple.
- Sert de baseline.
- Son `influence(...)` est simplement la fraction de gradients byzantins dans le lot total.

#### `median` ([aggregators/median.py](aggregators/median.py))

- Agrégation coordonnée par coordonnée.
- Une variante native est proposée si [native/](native) a compilé le module correspondant.
- Adaptée aux données qui contiennent des valeurs non finies.

#### `brute` ([aggregators/brute.py](aggregators/brute.py))

- Parcourt toutes les combinaisons possibles de taille `n - f`.
- Garde la sous-collection de plus petit diamètre.
- C'est un bon point de référence théorique, mais coûteux.

#### `krum` ([aggregators/krum.py](aggregators/krum.py))

- Calcule des scores de distances pairwise.
- Sélectionne les gradients les mieux scorés.
- La variante multi-krum agrège plusieurs gradients sélectionnés.

#### `bulyan` ([aggregators/bulyan.py](aggregators/bulyan.py))

- Combine une phase de sélection type multi-krum et une phase de filtrage coordonnée.
- C'est l'une des règles robustes les plus sophistiquées du dépôt.

## 6. Le sous-système natif

### 6.1 Chargement automatique

Fichier : [native/__init__.py](native/__init__.py)

Le dossier [native/](native) n'est pas un simple dépôt de sources C++/CUDA. Il se compile au moment de l'import Python.

#### Ce que fait le chargeur

- inspecte les sous-dossiers de [native/](native) ;
- reconnaît les préfixes `so_` et `py_` ;
- collecte les fichiers sources correspondant aux extensions autorisées ;
- compile avec `torch.utils.cpp_extension.load` ;
- charge les dépendances déclarées par les fichiers `.deps` ;
- expose les modules Python compilés dans l'espace de noms `native`.

#### Variables d'environnement importantes

- `NATIVE_OPT` : contrôle le mode debug/release du build natif ;
- `NATIVE_STD` : fixe la version du standard C++ ;
- `NATIVE_QUIET` : masque les messages de build en mode release.

#### Dépendances externes

- `ninja` ;
- CUB dans [native/include/cub](native/include/cub) ;
- une installation PyTorch compatible compilation d'extensions.

### 6.2 Pourquoi c'est important pour la recherche

Le dépôt a clairement été pensé pour permettre un prototypage Python rapide, puis un passage en natif quand une idée devient assez stable pour être accélérée.

La logique à retenir est donc :

1. prototype en Python dans `aggregators/` ou `attacks/` ;
2. valider les comportements et les métriques ;
3. déplacer la partie coûteuse dans `native/` si nécessaire ;
4. conserver exactement le même contrat fonctionnel.

## 7. Les classes de post-traitement et de reporting

### 7.1 `histogram.Session`

Fichier : [histogram.py](histogram.py)

`Session` représente un répertoire de résultats d'expérience.

#### Ce qu'elle charge

- le fichier `config` ;
- le fichier `config.json` ;
- le fichier `study` ;
- le fichier `eval`.

#### Méthodes utiles

- `get(...)` : sélectionne une sous-partie des colonnes ;
- `display(...)` : ouvre une vue GTK d'un sous-ensemble ;
- `has_known_ratio()` : vérifie si la règle d'agrégation fournit une borne théorique ;
- `compute_all()` : exécute automatiquement tous les `compute_*` présents ;
- `compute_epoch()` : calcule le nombre d'époques à partir du compteur de points ;
- `compute_lr()` : reconstruit la courbe de learning rate ;

#### Pourquoi `compute_all()` est très utile

Ce mécanisme constitue déjà un point d'extension naturel pour de nouvelles métriques dérivées. Si vous ajoutez une méthode `compute_accuracy_gap`, `compute_margin`, `compute_privacy_budget` ou autre, elle sera appelée automatiquement.

### 7.2 `histogram.LinePlot`

`LinePlot` encapsule les graphiques de courbes.

#### Fonctions principales

- `include(...)` : ajoute plusieurs colonnes du tableau ;
- `include_single(...)` : ajoute une seule courbe identifiée ;
- `include_vline(...)` : ajoute une ligne verticale ;
- `finalize(...)` : fixe les labels, les bornes et la légende ;
- `display()` : affiche la figure ;
- `save(...)` : enregistre la figure sur disque ;
- `close()` : libère la figure.

#### Limites de conception

- au plus deux axes Y ;
- le graphe doit être finalisé avant affichage ou sauvegarde ;
- l'objet est pensé pour du reporting d'expérience, pas pour un framework de visualisation généraliste.

### 7.3 `histogram.HistPlot`

`HistPlot` est le pendant plus simple pour les histogrammes.

Il gère :

- le nombre de bins ;
- l'inclusion des données ;
- la finalisation ;
- l'affichage ;
- l'enregistrement.

### 7.4 Les helpers de tableaux

`histogram.py` contient aussi deux helpers pratiques :

- `select(...)` : sélection sémantique de colonnes ;
- `discard(...)` : suppression sémantique de colonnes.

Ils rendent les post-traitements plus lisibles lorsque les tableaux accumulent beaucoup de colonnes de mesures.

## 8. La gestion des jobs et des campagnes

### 8.1 `tools.jobs.Command`

Fichier : [tools/jobs.py](tools/jobs.py)

`Command` stocke une base de commande et ajoute dynamiquement les arguments de lancement :

- seed ;
- device ;
- result-directory.

Elle sert principalement à faire de la reproduction batch sans réécrire à la main toutes les chaînes CLI.

### 8.2 `tools.jobs.Jobs`

`Jobs` lance un pool de workers qui prennent les expériences en attente et les exécutent.

#### Comportements importants

- parallélisme par device ;
- multiexécution par device via `devmult` ;
- répertoire `.pending` pendant l'exécution ;
- renommage vers un répertoire final à la fin ;
- archivage de `stdout.log` et `stderr.log`.

Cette classe est particulièrement utile pour les campagnes longues, où l'on veut garder une traçabilité simple des runs successifs.

## 9. Artefacts produits par l'exécution

### 9.1 Par `train.py`

Fichiers attendus dans le répertoire de résultats :

- `config` : version lisible par humain ;
- `config.json` : configuration structurée ;
- `study` : métriques détaillées ;
- `eval` : évaluation périodique ;
- `perfs.json` : timings accumulés.

La feuille `study` contient notamment :

- le nombre de pas ;
- le nombre de points de données traités ;
- la perte moyenne ;
- la distance l2 à l'origine ;
- les déviations et normes des gradients honnêtes et adverses ;
- les valeurs absolues maximales par gradient ;
- les cosinus entre gradients honnêtes, adverses et défendus.

### 9.2 Par `reproduce.py`

Pour chaque expérience soumise, `Jobs` conserve :

- le répertoire final de résultats ;
- `stdout.log` ;
- `stderr.log`.

### 9.3 Par les scripts de post-traitement

Les scripts ne génèrent pas uniquement des figures. Ils reconstituent aussi de nouvelles colonnes dérivées pour éviter de dupliquer la logique de calcul dans le script de simulation.

## 10. Conventions à préserver quand on ajoute une recherche

### 10.1 Ajouter une nouvelle règle d'agrégation

- Créer un fichier dans [aggregators/](aggregators).
- Exposer une fonction d'agrégation et une fonction `check`.
- Respecter le contrat keyword-only.
- Ne jamais renvoyer de tenseur qui aliaserait directement un tenseur d'entrée.
- Si une borne théorique ou un ratio d'influence existe, l'exposer aussi.

### 10.2 Ajouter une nouvelle attaque

- Créer un fichier dans [attacks/](attacks).
- Respecter les arguments réservés `grad_honests`, `f_decl`, `f_real`, `model`, `defense`.
- Retourner exactement `f_real` gradients.
- Maintenir la séparation entre vérification et exécution effective.

### 10.3 Ajouter un nouveau modèle

- Créer un module dans [experiments/models/](experiments/models).
- Exporter le constructeur dans `__all__` si possible.
- Renvoyer une vraie instance de `torch.nn.Module`.
- Vérifier que le modèle peut être aplati et remis à jour par `Model.set(...)`.

### 10.4 Ajouter un nouveau dataset

- Créer un module dans [experiments/datasets/](experiments/datasets).
- Exposer un générateur ou une fonction de construction de dataset.
- Prévoir le cache si le dataset est volumineux ou téléchargeable.
- Vérifier que `make_datasets(...)` ou `Dataset(...)` peuvent l'utiliser sans adaptation invasive.

### 10.5 Ajouter une nouvelle métrique

Il y a deux bons emplacements selon le type de métrique :

- **Métrique calculable pendant l'entraînement** : ajouter la mesure dans `train.py` et écrire une colonne supplémentaire dans `study` ou `eval`.
- **Métrique dérivée après coup** : ajouter une méthode `compute_*` dans `Session` pour la reconstruire à partir des colonnes sauvegardées.

Le second cas est souvent le plus propre pour éviter de compliquer la boucle d'entraînement.

### 10.6 Ajouter de nouveaux graphiques

Le code de visualisation actuel est déjà structuré autour de deux classes réutilisables : `LinePlot` et `HistPlot`.

Si les besoins augmentent, il est raisonnable de :

- extraire un petit sous-package dédié à la visualisation ;
- conserver `histogram.py` comme script d'entrée de post-traitement ;
- faire consommer les nouveaux graphiques par `Session` et ses colonnes dérivées ;
- éviter de dupliquer les calculs de métriques dans les scripts de tracé.

## 11. Pièges et points d'attention

### 11.1 Le mode debug change le comportement

Les wrappers des attaques et des agrégateurs choisissent entre version vérifiée et version brute selon `__debug__`. Le mode `-O` ou `-OO` change donc réellement la surface de validation.

### 11.2 La compilation native peut faire échouer l'import

Le dossier [native/](native) compile à l'import. Si un module natif est cassé, l'import peut échouer ou se dégrader en avertissement selon le point de rupture. Il faut donc garder des fallbacks Python corrects.

### 11.3 Les tensors sont souvent aplatis et relinkés

Le projet repose beaucoup sur `tools.flatten(...)` et `tools.relink(...)`. Quand on ajoute des objets qui manipulent les paramètres ou les gradients, il faut être prudent avec les vues, les copies et les opérations inplace.

### 11.4 Les arguments CLI supplémentaires passent sous forme `cle:valeur`

Les arguments supplémentaires `--gar-args`, `--attack-args`, `--model-args`, `--dataset-args`, `--loss-args`, `--criterion-args` sont parsés par `tools.parse_keyval(...)`. Il faut donc documenter les nouvelles options de recherche avec ce format.

### 11.5 Le dépôt n'a pas de suite de tests dédiée

La validation pratique consiste surtout à lancer des commandes ciblées :

- `python train.py --help` ;
- `python reproduce.py --help` ;
- `python histogram.py --help` ;
- un run court sur un dataset simple ;
- un import des modules natifs si vous les modifiez.

## 12. Où commencer quand on arrive dans le dépôt

Si votre but est d'ajouter une nouvelle idée de recherche, l'ordre de lecture le plus rentable est :

1. [train.py](train.py) pour comprendre le protocole expérimental ;
2. [aggregators/__init__.py](aggregators/__init__.py) et [attacks/__init__.py](attacks/__init__.py) pour comprendre les contrats ;
3. [experiments/model.py](experiments/model.py), [experiments/dataset.py](experiments/dataset.py), [experiments/loss.py](experiments/loss.py) et [experiments/optimizer.py](experiments/optimizer.py) pour comprendre les wrappers ;
4. [histogram.py](histogram.py) pour comprendre les colonnes de métriques et les graphiques ;
5. [native/README.md](native/README.md) si vous voulez accélérer une brique lourde.

## 13. Résumé opérationnel

Le dépôt est déjà structuré comme une plateforme de recherche :

- le cœur expérimental est dans `train.py` ;
- les objets modélisés sont dans `experiments/` ;
- les règles adversariales sont branchées dans `aggregators/` et `attacks/` ;
- les artefacts de mesure sont persistés dans des fichiers simples à relire ;
- la visualisation est déjà séparée du calcul ;
- l'accélération native peut être ajoutée sans changer le contrat haut niveau.

Pour améliorer la base de code en vue de nouvelles recherches, les meilleurs investissements sont probablement :

- des métriques mieux structurées ;
- des scripts de visualisation plus modulaires ;
- des helpers de comparaison entre expériences ;
- des templates explicites pour les nouveaux agrégateurs, attaques, datasets et modèles.

Ce document est volontairement centré sur les points qui servent à étendre le dépôt sans casser l'existant. C'est le meilleur point de départ avant d'ajouter des packages plus spécifiques pour les graphiques ou les métriques.

## 14. Lecture concrète des packages

### 14.1 Ce qui se passe réellement quand on lance une expérience

Quand on exécute `train.py`, le dépôt ne charge pas un gros graphe de classes figées. Il reconstruit une expérience à partir de noms passés en CLI.

Le déroulé concret est le suivant :

1. Les arguments `--model`, `--dataset`, `--loss`, `--criterion`, `--gar` et `--attack` sont lus comme des chaînes.
2. Les arguments complémentaires `--model-args`, `--dataset-args`, `--loss-args`, `--criterion-args`, `--gar-args` et `--attack-args` sont convertis en dictionnaires via `tools.parse_keyval(...)`.
3. Le wrapper de modèle résout le constructeur réel correspondant au nom demandé.
4. Le wrapper de dataset résout le générateur ou le dataset correspondant au nom demandé.
5. La perte et le criterion sont instanciés de la même manière.
6. L'attaque et la règle d'agrégation sont récupérées dans leurs registres respectifs.
7. La boucle d'entraînement alterne gradients honnêtes, éventuel bruit de confidentialité, attaque byzantine, agrégation et mise à jour.
8. Les colonnes de résultats sont écrites dans `study` et `eval`, puis relues par `histogram.py`.

Le point important est que la logique scientifique est découpée en briques interchangeables, pas en classes fortement couplées.

### 14.2 Le package `models` et le wrapper `Model`

Le dossier [experiments/models/](experiments/models) ne contient pas l'entraînement lui-même. Il contient des constructeurs de réseaux.

Concrètement, [experiments/model.py](experiments/model.py) fait trois choses :

- il charge les modèles TorchVision disponibles automatiquement sous des noms comme `torchvision-resnet18` ;
- il charge les modèles locaux exposés par [experiments/models/simples.py](experiments/models/simples.py) ou tout autre module exportant des fonctions dans `__all__` ;
- il enveloppe le modèle instancié dans un objet qui sait gérer ses paramètres aplatis, ses gradients, ses valeurs par défaut et son déplacement sur device.

Cela veut dire qu'un modèle n'est pas manipulé comme un simple `torch.nn.Module`, mais comme une brique de recherche prête à l'emploi.

Exemple concret : quand vous demandez `simples-conv`, le registre trouve le constructeur `conv` dans `simples.py`, crée le module correspondant, initialise ses poids, le déplace sur le device demandé, l'enveloppe éventuellement en `DataParallel`, puis aplatie ses paramètres pour que le reste du pipeline puisse le manipuler de façon uniforme.

### 14.3 Le package `datasets` et le wrapper `Dataset`

Le dossier [experiments/datasets/](experiments/datasets) joue le même rôle pour les données.

Concrètement, [experiments/dataset.py](experiments/dataset.py) sait gérer trois sources de données :

- un nom de dataset connu, comme `mnist` ou `cifar10` ;
- un générateur Python déjà prêt à produire des batches ;
- un objet quelconque, qui sera alors répété comme un batch unique.

Quand un dataset est demandé par son nom, le registre construit un générateur de batches à partir de TorchVision ou d'un module local. Le générateur n'est pas consommé une seule fois : il est réutilisé en boucle, ce qui permet à `train.py` de demander indéfiniment de nouveaux minibatchs.

Le cas `phishing` dans [experiments/datasets/svm.py](experiments/datasets/svm.py) est représentatif : le code télécharge si nécessaire, prétraite les données en tenseurs, met en cache le résultat, puis appelle `experiments.batch_dataset(...)` pour obtenir un trainset et un testset qui tournent en continu.

Autrement dit, `Dataset` n'est pas juste un conteneur de données. C'est une abstraction de flux de minibatchs.

### 14.4 Pourquoi ces registres existent

Le projet utilise des registres pour les familles extensibles parce que cela résout plusieurs problèmes en même temps.

- **Découverte par nom** : on peut écrire `--model simples-conv`, `--dataset mnist`, `--gar krum` ou `--attack nan` sans modifier le cœur du code.
- **Extension sans centralisation** : il suffit souvent d'ajouter un fichier dans le bon dossier et d'exporter une fonction ou une classe dans `__all__`.
- **Contrat par famille** : chaque famille a ses propres règles. Une attaque ne retourne pas la même chose qu'un dataset, un criterion ne fait pas la même chose qu'une perte, et un agrégateur ne reçoit pas les mêmes paramètres qu'un modèle.
- **Validation locale** : chaque famille porte son `check()`, donc les erreurs de paramétrage sont détectées au plus près du composant concerné.
- **Bascule Python / native** : une règle peut garder le même nom public tout en passant d'une implémentation Python à une implémentation native quand elle existe.

En pratique, ce n'est pas un registre pour chaque objet au sens OOP, mais un registre par famille de plug-ins. C'est beaucoup plus robuste pour de la recherche, parce que les contrats restent lisibles et que les nouvelles briques s'insèrent sans toucher à un switch central gigantesque.

### 14.5 À quoi sert le package `native`

Le dossier [native/](native) sert d'accélérateur optionnel.

Son rôle est très concret : prendre les parties les plus coûteuses des agrégateurs et les réécrire en C++/CUDA quand cela devient utile.

Le chargeur natif fait les choses suivantes :

- il parcourt les sous-dossiers `so_*` et `py_*` ;
- il compile automatiquement les sources à l'import ;
- il lit les fichiers `.deps` pour connaître l'ordre des dépendances ;
- il ajoute les sources CUDA si le runtime les supporte ;
- il expose les modules Python compilés comme `native.krum`, `native.median`, `native.bulyan`, etc.

Les variantes natives servent surtout pour les chemins chauds : calcul des distances pairwise, sélection de sous-ensembles, médiane coordonnée, combinatoires coûteuses. L'API haut niveau reste la même, ce qui permet de comparer exactement la même règle en version Python et en version native.

L'intérêt pour la recherche est double :

1. prototyper vite en Python ;
2. accélérer ensuite seulement la partie qui devient trop lente.

Si le natif n'est pas disponible, le code Python continue généralement de fonctionner avec des avertissements ou des fallbacks.

### 14.6 Le système de criterion

Le criterion ne sert pas à entraîner le modèle. Il sert à mesurer le modèle.

La différence avec la loss est fondamentale :

- la loss doit être dérivable et participer au calcul de gradient ;
- le criterion n'a pas besoin d'être dérivable et peut se concentrer sur l'évaluation.

Concrètement, le criterion retourne toujours une information de type `[#correct, batch_size]`. Cela permet de sommer plusieurs évaluations puis de calculer une accuracy globale proprement.

Les deux critères intégrés sont :

- `top-k` : une prédiction est correcte si la bonne classe apparaît dans les `k` meilleures sorties du modèle ;
- `sigmoid` : une prédiction est correcte si la sortie est suffisamment proche de la cible binaire.

Dans `train.py`, l'évaluation fonctionne ainsi : le script appelle `model.eval()` sur un ou plusieurs batches de test, additionne les résultats du criterion, puis calcule le ratio `correct / total` pour obtenir l'accuracy. C'est exactement ce qu'on veut pour de la recherche expérimentale : une mesure simple, stable et facile à stocker.

### 14.7 Ce que cela implique pour ajouter de nouvelles briques

Si vous ajoutez une nouvelle idée de recherche, gardez cette règle mentale :

- un nouveau modèle se branche dans `experiments/models/` ;
- un nouveau dataset se branche dans `experiments/datasets/` ;
- une accélération lourde se branche dans `native/` ;
- une nouvelle métrique d'évaluation se branche dans `Criterion` ou dans `histogram.Session` selon qu'elle est calculée pendant l'exécution ou après coup ;
- une nouvelle règle robuste ou une nouvelle attaque se branche dans `aggregators/` ou `attacks/` avec le même contrat de validation.

Autrement dit, les registres ne sont pas un détail d'implémentation : ils sont le mécanisme qui rend le dépôt extensible sans devoir le réécrire à chaque projet de recherche.
