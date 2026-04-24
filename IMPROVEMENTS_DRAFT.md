# Krum - Améliorations

## 1. Problèmes identifiés

### train.py (618 lignes)

| Problème | Impact |
|----------|--------|
| CLI parsing monolithique (~250 lignes) | Difficile à maintenir, duplication si nouveau script |
| Validation dispersée dans `process_commandline()` | Validation mélangée au parsing |
| Gestion des résultats embarquée | Non réutilisable pour d'autres scripts |
| Boucle d'entraînement avec logique intermixée | Difficulté à extraire des sous-parties |

### reproduce.py (244 lignes)

| Problème | Impact |
|----------|--------|
| Boucle for explicite pour les expériences (lignes 140-163) | Lisible mais peu déclaratif |
| Pas de séparation entre définition et exécution | Difficile de versionner les configs |
| Génération de plots codee en dur | Pas de flexibilité |

### histogram.py (715 lignes)

| Problème | Impact |
|----------|--------|
| API `include()` + `finalize()` + `save()` peu intuitive | Courbe d'apprentissage élevée |
| Helpers `select()`/`discard()` dans le fichier principal | Non réutilisables ailleurs |
| Classes `LinePlot` et `HistPlot` trop couplées | Difficile à étendre |

---

## 2. Solutions proposées

### 2.1 Package `tools/cli.py` - Parser CLI réutilisable

**Objectif** : Extraire le parsing CLI dans un module réutilisable.

```python
# -------------------------------------------------------------------------
# API proposée:
# -------------------------------------------------------------------------

class ArgumentParser:
    """Parser CLI avec validation intégrée et contexte de logging."""

    def __init__(self, name, description, context="cmdline"):
        ...

    def add_argument(self, name, type_=None, default=None, help="", validators=[]):
        """Ajouter un argument."""
        ...

    def add_mutually_exclusive_group(self, name):
        """Ajouter un groupe d'arguments mutuellement exclusifs."""
        ...

    def parse(self, argv=None):
        """Parser les arguments."""
        ...

    def validate(self, args):
        """Valider les arguments avec les validators."""
        ...

# -------------------------------------------------------------------------
# Validators prédéfinis:
# -------------------------------------------------------------------------

class Validators:
    @staticmethod
    def positive_int(value):
        """Doit être un entier positif."""
        ...

    @staticmethod
    def positive_float(value):
        """Doit être un flottant positif."""
        ...

    @staticmethod
    def device(value):
        """Valide le device (auto, cpu, cuda:0, etc.)."""
        ...

    @staticmethod
    def range(min_=None, max_=None):
        """Valeur dans un intervalle."""
        ...

    @staticmethod
    def choices(*choices):
        """Valeur doit être dans les choix autorisés."""
        ...

# -------------------------------------------------------------------------
# Utilisation dans train.py:
# -------------------------------------------------------------------------

parser = cli.ArgumentParser(
    "krum-train",
    "Simulate a training session under attack"
)

parser.add_argument("--seed", int, default=-1,
    help="Fixed seed, negative for random")
parser.add_argument("--device", str, default="auto",
    validators=[cli.Validators.device])
parser.add_argument("--nb-workers", int, default=11,
    validators=[cli.Validators.positive_int])

args = parser.parse().validate()
```

**Impact** :
- Réutilisable dans `reproduce.py` et nouveaux scripts
- Validation centralisée et cohérente
- Meilleure documentation des arguments

---

### 2.2 Package `tools/plotting.py` - API matplotlib simplifiée

**Objectif** : Simplifier l'API de visualisation.

```python
# -------------------------------------------------------------------------
# API proposée:
# -------------------------------------------------------------------------

class Figure:
    """Wrapper matplotlib simplifié avec fluent API."""

    def __init__(self, figsize=(10, 6), dpi=100):
        ...

    # Ajout de données
    def line(self, x, y, label=None, **kwargs):
        """Tracer une ligne."""
        ...

    def scatter(self, x, y, label=None, **kwargs):
        """Tracer un scatter plot."""
        ...

    def errorbar(self, x, y, yerr, label=None, **kwargs):
        """Tracer avec barres d'erreur."""
        ...

    def fill_between(self, x, y1, y2, alpha=0.2, **kwargs):
        """Remplir entre deux courbes."""
        ...

    # Lignes de référence
    def hline(self, y, color="black", linestyle="--", **kwargs):
        ...

    def vline(self, x, color="black", linestyle="--", **kwargs):
        ...

    # Labels et titre
    def xlabel(self, label, **kwargs):
        ...

    def ylabel(self, label, **kwargs):
        ...

    def title(self, label, **kwargs):
        ...

    def legend(self, loc="best", **kwargs):
        ...

    # Limites
    def xlim(self, left=None, right=None):
        ...

    def ylim(self, bottom=None, top=None):
        ...

    # Export
    def save(self, path, dpi=200):
        """Sauvegarder la figure."""
        ...

    def show(self):
        """Afficher la figure."""
        ...

class Subplots:
    """Gestionnaire de subplots."""

    def __init__(self, nrows=1, ncols=1, figsize=None, sharex=False, sharey=False):
        ...

    def __getitem__(self, key) -> Figure:
        ...

# -------------------------------------------------------------------------
# Helpers pour histogram.py:
# -------------------------------------------------------------------------

def plot_accuracy(sessions, output_dir):
    """Plot l'accuracy de plusieurs sessions."""
    fig = Figure()
    for session in sessions:
        data = session.get("Step number", "Cross-accuracy")
        fig.line(data["Step number"], data["Cross-accuracy"], label=session.name)
    fig.xlabel("Step")
    fig.ylabel("Accuracy")
    fig.legend()
    fig.save(output_dir / "accuracy.png")

# -------------------------------------------------------------------------
# Utilisation dans histogram.py:
# -------------------------------------------------------------------------

from tools.plotting import Figure

plot = Figure()
plot.line(steps, loss, label="Training Loss")
plot.xlabel("Step")
plot.ylabel("Loss")
plot.title("Training Progress")
plot.grid(True)
plot.save("loss.png")
```

**Impact** :
- API plus intuitive et pythonique
- Réutilisable dans les notebooks et autres scripts
- Moins de code pour des cas courants

---

### 2.3 Config YAML/JSON pour `reproduce.py`

**Objectif** : Déclarer les expériences dans un fichier de config plutôt que du code Python.

```yaml
# -------------------------------------------------------------------------
# Exemple: experiments.yaml
# -------------------------------------------------------------------------

# Configuration de base
base:
  loss: mse
  learning-rate: 2
  criterion: sigmoid
  momentum: 0.99
  evaluation-delta: 50
  nb-steps: 1000
  nb-workers: 11
  nb-decl-byz: 5
  gradient-clip: 0.01
  privacy-delta: 1e-6

# Définition des expériences
experiments:
  - name: "baseline-average"
    description: "Baseline sans attaque"
    gar: average

  - name: "krum-little"
    description: "Krum avec attaque little"
    gar: krum
    attack: little
    attack-args:
      factor: 1.5
      negative: true

  - name: "bulyan-empire"
    gar: bulyan
    attack: empire
    attack-args:
      factor: 1.1

# Grille de variations
grid:
  batch-size: [10, 25, 50, 100, 250, 500]
  privacy-epsilon: [null, 0.1, 0.2, 0.5]
  dataset: [svm-phishing]
  model: [simples-logit]

# Configuration d'exécution
execution:
  seeds: [1, 2, 3, 4, 5]
  devices: auto
  parallel-per-device: 1
  only-plot: false

# Configuration des plots
plots:
  - type: line
    x: "Step number"
    y: "Cross-accuracy"
    output: "accuracy.png"

  - type: line
    x: "Step number"
    y: "Average loss"
    output: "loss.png"
```

```python
# -------------------------------------------------------------------------
# tools/experiment_config.py
# -------------------------------------------------------------------------

class ExperimentConfig:
    """Loader de configuration d'expériences."""

    @staticmethod
    def from_yaml(path) -> "ExperimentConfig":
        """Charger depuis un fichier YAML."""
        ...

    @staticmethod
    def from_json(path) -> "ExperimentConfig":
        """Charger depuis un fichier JSON."""
        ...

    def expand_grid(self):
        """Générer toutes les combinaisons de la grille."""
        ...

    def get_experiments(self):
        """Générateur des expériences individuelles."""
        ...

    def get_plots(self):
        """Générateur des configurations de plots."""
        ...

# -------------------------------------------------------------------------
# Utilisation dans reproduce.py:
# -------------------------------------------------------------------------

config = ExperimentConfig.from_yaml(args.config)
experiments = list(config.expand_grid())

jobs = tools.Jobs(args.data_directory, devices=args.devices)
for exp in experiments:
    jobs.submit(exp.name, make_command(exp.params))

jobs.wait()
```

**Impact** :
- Config versionnable et partageable
- Plus besoin de coder des boucles pour ajouter des variations
- Plus facile de reproduire des expériences publiées

---

### 2.4 Package `tools/metrics.py` - Métriques réutilisables

**Objectif** : Métriques calculables sur les résultats.

```python
# -------------------------------------------------------------------------
# API proposée:
# -------------------------------------------------------------------------

class Metrics:
    """Collection de métriques calculables."""

    @staticmethod
    def compute_epoch(training_points, dataset_name):
        """Calculer le nombre d'époques."""
        ...

    @staticmethod
    def compute_lr(base_lr, decay, step, decay_steps):
        """Calculer le learning rate à un step donné."""
        ...

    @staticmethod
    def moving_average(values, window):
        """Calculer la moyenne mobile."""
        ...

    @staticmethod
    def auc(x, y):
        """Calculer l'aire sous la courbe."""
        ...

    @staticmethod
    def bootstrap_confidence_interval(data, statistic, n_bootstrap=1000, ci=0.95):
        """Intervalle de confiance par bootstrap."""
        ...

# Intégration avec Session
class Session:
    def compute_metric(self, name, *args, **kwargs):
        """Calculer une métrique dérivée."""
        if name == "epoch":
            return Metrics.compute_epoch(...)
        elif name == "lr":
            return Metrics.compute_lr(...)
        ...
```

---

### 2.5 Classes de base `BaseAggregator` et `BaseAttack`

**Objectif** : Standardiser l'interface pour les agrégateurs et attaques afin de faciliter l'ajout de nouvelles méthodes par les chercheurs.

**Problème actuel** : Chaque agrégateur/attaque est une fonction indépendante sans structure commune.

```python
# -------------------------------------------------------------------------
# Exemple actuel (pas de standard):
# -------------------------------------------------------------------------

def krum(gradients, f, model, **kwargs): ...
def median(gradients, f, model, **kwargs): ...
def my_attack(grad_honests, f_decl, f_real, model, defense, **kwargs): ...
```

```python
# -------------------------------------------------------------------------
# API proposée:
# -------------------------------------------------------------------------

class BaseAggregator:
    """Classe de base pour les agrégateurs."""

    def aggregate(self, gradients: List[Tensor], f: int, model, **kwargs) -> Tensor:
        """Agréger les gradients (méthode obligatoire à implémenter)."""
        raise NotImplementedError

    def pre_hook(self, gradients: List[Tensor]) -> List[Tensor]:
        """Hook optionnel exécuté avant l'agrégation (filtrage, etc.)."""
        return gradients

    def post_hook(self, result: Tensor) -> Tensor:
        """Hook optionnel exécuté après l'agrégation."""
        return result

    @property
    def upper_bound(self) -> Optional[float]:
        """Borne théorique du ratio de byzantins acceptés (optionnel)."""
        return None

    @property
    def influence(self) -> Optional[Callable]:
        """Fonction de calcul d'influence (optionnel)."""
        return None

    @classmethod
    def register(cls, name: str):
        """Décorateur pour enregistrer automatiquement l'agrégateur."""
        return aggregators.register(name, cls)


class BaseAttack:
    """Classe de base pour les attaques."""

    def inject(self, grad_honests: List[Tensor], f_decl: int, f_real: int,
               model, defense, **kwargs) -> List[Tensor]:
        """Injecter les gradients byzantins (méthode obligatoire)."""
        raise NotImplementedError

    def pre_hook(self, grad_honests: List[Tensor]) -> List[Tensor]:
        """Hook optionnel exécuté avant la génération des gradients adverses."""
        return grad_honests

    @classmethod
    def register(cls, name: str):
        """Décorateur pour enregistrer automatiquement l'attaque."""
        return attacks.register(name, cls)
```

```python
# -------------------------------------------------------------------------
# Exemple d'utilisation:
# -------------------------------------------------------------------------

@BaseAggregator.register("my-krum")
class MyKrum(BaseAggregator):
    """Exemple d'agrégateur personnalisé."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def aggregate(self, gradients, f, model, **kwargs):
        # Implémentation personnalisée
        return krum_core(gradients, f)

    @property
    def upper_bound(self):
        return f / len(gradients)


@BaseAttack.register("my-attack")
class MyAttack(BaseAttack):
    """Exemple d'attaque personnalisée."""

    def inject(self, grad_honests, f_decl, f_real, model, defense, **kwargs):
        factor = kwargs.get("factor", 1.0)
        # Génération des gradients adverses
        return [g * factor for g in grad_honests[:f_real]]
```

**Impact** :
- Un chercheurs peut ajouter son agrégateur en héritant de `BaseAggregator`
- Accès automatique aux hooks, validation, registration
- Documentation claire des méthodes obligatoires vs optionnelles
- Compatible avec le système de registre existant (`aggregators.register()`)

---

## 3. Résumé des nouveaux fichiers

| Fichier | Description |
|---------|-------------|
| `tools/cli.py` | Parser CLI avec validators |
| `tools/plotting.py` | API matplotlib simplifiée |
| `tools/experiment_config.py` | Loader YAML/JSON |
| `tools/metrics.py` | Métriques réutilisables |
| `tools/interactive.py` | Shell interactif amélioré |

---

## 4. Ordre d'implémentation suggéré

1. **Phase 1** : `cli.py` + `results.py`
   - Extraire le parser et la gestion des fichiers de train.py
   - Impact immédiat sur la maintenabilité

2. **Phase 2** : `plotting.py`
   - Refondre histogram.py
   - Utilisable immédiatement dans les notebooks

3. **Phase 3** : `experiment_config.py`
   - Modifier reproduce.py pour accepter YAML/JSON
   - Déclaratif plutôt qu'impératif

4. **Phase 4** : `metrics.py` (optionnel)
   - Métriques additionnelles
   - Intégration avec Session

---

## 5. Contraintes à respecter

- **Compatibilité CLI** : Tous les arguments existants doivent fonctionner
- **Breaking changes minimales** : API existante préservée
- **Auto-loading** : Les nouveaux modules doivent être auto-chargés comme les autres
- **Tests manuels** : `./train.py --help`, `./reproduce.py --help`, etc.

---

## 6. Exemples d'utilisation future

### Nouveau script d'entraînement minimal

```python
import tools.cli as cli
import tools.results as results

parser = cli.ArgumentParser("my-experiment", "My custom experiment")
parser.add_argument("--epochs", cli.Validators.positive_int, default=100)
args = parser.parse().validate()

res = results.ResultManager("my-results", results.STUDY_SCHEMA)
res.write_config(args.__dict__)

for epoch in range(args.epochs):
    # ... training ...
    res.write("study", epoch, loss, acc)
```

### Nouvelle expérience via config

```yaml
# my-experiments.yaml
base:
  nb-steps: 500

grid:
  gar: [average, krum, bulyan]
  attack: [none, little, empire]
```

```bash
python reproduce.py --config my-experiments.yaml
```
