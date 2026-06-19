# ADR 2026-06-12 — Design de l'orchestration

## Principe

L'utilisateur crée un `Orchestrator` et des `Metric` dans son code.
Tout le reste est interne au package.

```python
orchestrator = Orchestrator("my_orchestrator")

# Ces appels sont valides uniquement à l'intérieur d'un orchestrator.run()
loss = Metric("loss")           # trouve l'orchestrateur actif tout seul
loss.push(step, valeur)         # taggé avec les paramètres du run en cours
df = orchestrator.get("loss")   # → pd.DataFrame
```

## Choix de `pandas.DataFrame`

L’ADR initiale envisageait une classe `MetricDataFrame` dédiée. Pourquoi ne pas utiliser un `pd.DataFrame` standard : il est plus puissant, mieux documenté et évite de réimplémenter le slicing, le groupby, le plotting, etc.

Des fonctions utilitaires pourront être ajoutées plus tard pour faciliter les tranches courantes :
- `slice_by_aggregator(df, Krum)`
- `slice_by_attack(df, Alie)`
- `slice_by_n(df, 13)`
- etc.

Ces helpers ne masqueront pas le DataFrame sous-jacent : l’utilisateur garde toujours accès à `df[df['n'] == 13]` ou à toute opération pandas.

## Package `krum.orchestration`

```txt
__init__.py          # exports Orchestrator, Metric
context.py           # RunContext, ContextManager
metric.py            # Metric
metric_manager.py    # MetricManager
orchestrator.py      # Orchestrator
```

---

## Décisions de design complémentaires

### `Orchestrator.name`

Le paramètre `name` est **obligatoire** et sert **d’identifiant unique** pour l’orchestrateur. Il n’est pas utilisé comme préfixe de métrique ni comme espace de noms dans cette V1, mais permet de différencier plusieurs orchestrateurs lors du débogage ou de futures extensions.

### Contraintes sur les noms de métriques

- **Casse sensible** : `loss` et `Loss` sont deux métriques différentes.
- **Espaces interdits** : un nom ne peut pas contenir d’espaces.
- **Caractères spéciaux autorisés** : `-`, `_`, `.`, `/`, etc. sont acceptés.

### Erreurs levées par `Metric`

1. **Création sans orchestrateur actif** : `Metric("loss")` hors d’un `orchestrator.run()` et sans orchestrateur courant lève une erreur.
2. **Push sans run actif** : `loss.push(...)` hors d’un run actif lève une erreur.
3. **Type incompatible** : si `value` n’est pas une instance de `dtype`, `push()` lève `TypeError`.
4. **dtype mismatch** : créer deux fois une métrique du même nom avec des `dtype` différents lève une erreur.

### `skip_if_exists`

`skip_if_exists=True` évite d’insérer une ligne si le couple **(clé du run, step)** existe déjà pour cette métrique. Cela permet, par exemple, de relancer partiellement une expérience sans doublons.

### Format du DataFrame retourné

`orchestrator.get("loss")` retourne un `pd.DataFrame` dont :
- les colonnes sont `step`, `value`, puis les paramètres du run dans **l’ordre d’insertion** (ordre dans lequel ils ont été vus lors des pushes) ;
- `step` est entier, `value` est typé selon le `dtype` de la métrique, les paramètres sont typés par pandas (souvent `object` pour les classes / strings).

Exemple :

| step | value | n | f | aggregator | attack |
|---|---|---|---|---|---|
| 0 | 2.34 | 10 | 3 | Krum | Alie |
| 1 | 2.10 | 10 | 3 | Krum | Alie |
| 0 | 3.00 | 11 | 2 | Average | SignFlip |

### Pourquoi pas `with` ?

Un `with orchestrator.context(n=10, ...)` pourrait sembler plus idiomatique, mais `orchestrator.run(fn, **params)` est préféré pour deux raisons :

1. **Encapsulation de l'exécution** : `run()` contrôle *quand* et *où* l'expérience s'exécute. Le jour où une queue multi-thread est ajoutée, l'API utilisateur ne change pas. Avec `with`, la ContextVar est posée dans le thread appelant — incompatible avec un worker thread qui doit voir son propre contexte.

2. **L'utilisateur n'a pas à gérer le scope** : avec `with`, l'utilisateur doit appeler son expérience dans le bloc, ce qui est plus verbeux et expose le risque d'oublier l'appel ou de pousser des métriques hors-bloc.

Le `with` serait pertinent si l'utilisateur avait besoin d'interagir directement avec le contexte (ex: lire des params, modifier le run en cours). Ici, le contexte est un détail interne — `run()` le masque correctement.

### Questions laissées ouvertes

- **Gestion des runs en échec** : si `my_experiment` lève une exception, que deviennent les métriques déjà poussées ? Ce point sera discuté lors de l’implémentation de la gestion des runs failed.
- **Multi-run / multi-orchestrateurs** : support de plusieurs orchestrateurs actifs simultanément prévu pour plus tard.

---

## `context.py`

```python
from dataclasses import dataclass
from contextvars import ContextVar


@dataclass(frozen=True)
class RunContext:
    """Paramètres d'un run. Interne."""
    params: dict

    def key(self) -> tuple:
        """Identifiant unique et stable du run.

        Retourne un id dérivé entièrement des paramètres du contexte.
        Deux runs avec les mêmes paramètres produisent la même clé.

        Implémentation suggérée : tuple(sorted(self.params.items()))

        Usages :
        - skip_if_exists : repérer si un step a déjà été poussé pour ce run ;
        - futures extensions : persistance, reprise, déduplication de runs.
        """
        ...


_run_context: ContextVar[RunContext | None] = ContextVar("_run_context", default=None)


class ContextManager:
    """Pile des contextes + gestion de la ContextVar. Interne."""

    def __init__(self):
        self._contexts: list[RunContext] = []
        self._tokens: list[contextvars.Token] = []

    def enter(self, **params) -> None:
        """Crée un RunContext, le stocke, le pose dans la ContextVar."""
        ctx = RunContext(params)
        self._contexts.append(ctx)
        self._tokens.append(_run_context.set(ctx))

    def exit(self) -> None:
        """Dépile proprement la ContextVar et restaure le contexte parent."""
        if self._tokens:
            token = self._tokens.pop()
            _run_context.reset(token)
        if self._contexts:
            self._contexts.pop()
```

`ContextVar` est une variable de contexte isolée par tâche asynchrone ; `Token` est le ticket qui permet de la restaurer à sa valeur antérieure ; `ContextManager` encapsule cette mécanique de pile pour que `enter()` pousse un `RunContext` et `exit()` le dépile proprement, même en cas d'imbrication.

---

## `metric.py`

```python
from contextvars import ContextVar


_current_orchestrator: ContextVar["Orchestrator"] = ContextVar("_current_orchestrator")


class Metric:
    """Collecte des valeurs pendant un run."""

    def __init__(self, name: str, dtype: type = float):
        self._name = name
        self._dtype = dtype
        # via _current_orchestrator → metric_manager.create(name, dtype)
        # lève si aucun orchestrateur n'est actif
        # lève si une métrique du même nom existe avec un dtype différent
        ...

    def push(self, step: int, value, skip_if_exists: bool = False):
        # Lève si aucun run n'est actif.
        # Vérifie que la valeur est compatible avec le dtype déclaré.
        if not isinstance(value, self._dtype):
            raise TypeError(
                f"Metric '{self._name}' expects dtype {self._dtype.__name__}, "
                f"got {type(value).__name__}"
            )
        # via _run_context → paramètres courants
        # stocke (step, value) + paramètres
        # si skip_if_exists=True et (run_key, step) déjà présent → ignore
        ...

    @property
    def name(self) -> str:
        ...
```

---

## `metric_manager.py`

```python
import pandas as pd


class MetricManager:
    """Registre des métriques. Interne."""

    def __init__(self, orchestrator: "Orchestrator"):
        self._orchestrator = orchestrator

    def create(self, name: str, dtype: type) -> "Metric":
        """Retourne l'existant ou crée une nouvelle Metric.

        Valide que le nom ne contient pas d'espace.
        Une métrique est identifiée par son nom au sein de l'orchestrateur.
        Si une métrique du même nom existe déjà avec un dtype différent,
        une ValueError est levée.
        """
        ...

    def get(self, name: str) -> pd.DataFrame:
        """Assemble les pushes dans un DataFrame.

        Colonnes : step, value, + les clés des paramètres du RunContext.
        Ex : step, value, n, f, aggregator, attack
        """
        ...

    def list(self) -> list[str]:
        ...
```

---

## `orchestrator.py`

```python
from krum.orchestration.context import ContextManager
from krum.orchestration.metric_manager import MetricManager


class Orchestrator:
    """Point d'entrée."""

    def __init__(self, name: str):
        self.name = name
        self._contexts = ContextManager()
        self._metrics = MetricManager(self)

    def run(self, experiment, **params):
        """Exécute une expérience dans un contexte."""
        self._contexts.enter(**params)
        try:
            experiment(**params) # In a future work, the experiment will be pushed to the run_queue which is read by the devices
        finally:
            self._contexts.exit()

    def get(self, name: str):
        """Retourne un pd.DataFrame."""
        return self._metrics.get(name)
```

---

## Flux complet

```python
orchestrator = Orchestrator("byzantine_study")
# → instancie ContextManager (pile vide) + MetricManager (registre vide)
# → _current_orchestrator n'est PAS encore posé (pas de run actif)


orchestrator.run(my_experiment, n=10, f=3, aggregator=Krum, attack=Alie, n_steps=100)
```

Détail interne de `orchestrator.run()` :

```txt
1a. ContextManager.enter(n=10, f=3, aggregator=Krum, attack=Alie, n_steps=100)
    → crée RunContext(params={"n": 10, "f": 3, "aggregator": Krum, ...})
    → _run_context.set(ctx)          # ContextVar : tout code dans ce thread voit maintenant ce contexte via Metric.push()
    → empile le token pour restauration future

1b. _current_orchestrator.set(self)  # l'orchestrateur se rend visible pour les Metric() créées dans l'expérience

1c. my_experiment(n=10, f=3, aggregator=Krum, attack=Alie, n_steps=100)
    est appelée avec les paramètres déballés.

    ┌─ À l'intérieur de my_experiment : ─────────────────────────────────────┐
    │                                                                         │
    │  loss = Metric("loss", float)                                           │
    │    → lit _current_orchestrator → trouve orchestrator                    │
    │    → orchestrator._metrics.create("loss", float)                        │
    │      • valide le nom (pas d'espace)                                     │
    │      • si "loss" existe déjà avec un autre dtype → ValueError           │
    │      • sinon enregistre ("loss", float) dans le registre                │
    │                                                                         │
    │  for step in range(n_steps):                                            │
    │      loss.push(step, sim.loss, skip_if_exists=True)                     │
    │        → vérifie isinstance(sim.loss, float) sinon TypeError            │
    │        → lit _run_context.get() → RunContext.params                     │
    │        → calcule run_key = tuple(sorted(params.items()))                │
    │        → si skip_if_exists et (run_key, step) déjà vu → no-op           │
    │        → sinon stocke la ligne :                                        │
    │            {step: 0, value: 2.34, n: 10, f: 3,                          │
    │             aggregator: Krum, attack: Alie}                             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

1d. ContextManager.exit()  (dans le bloc finally)
    → _run_context.reset(token)           # restaure le contexte parent (ou None)
    → _current_orchestrator.reset(token)  # l'orchestrateur n'est plus actif
    → dépile les structures internes
```

```python
for n in [10, 20]:
    for f in [2, 3]:
        for agg in [Krum, Bulyan, Average]:
            for atk in [Alie, SignFlip]:
                orchestrator.run(my_experiment, n=n, f=f, aggregator=agg,
                                 attack=atk, n_steps=100)
# Chaque itération répète le cycle 1a→1d.
# Le MetricManager accumule les lignes de tous les runs dans le même registre.
# La métrique "loss" n'est créée qu'une fois (premier run), réutilisée ensuite.


loss_df = orchestrator.get("loss")
```

```
3a. MetricManager.get("loss")
    → lit toutes les lignes stockées pour "loss"
    → assemble un pd.DataFrame :
        colonnes = [step, value] + clés des params dans l'ordre d'insertion
                   = [step, value, n, f, aggregator, attack]
        types    = step:int, value:float (dtype déclaré), params:object
    → retourne le DataFrame (copie, pas de vue mutable sur le store interne)
```

## Exemple utilisateur complet

cf [orchestration_example.py/](orchestration_example.py)

---

## Invariants à tester / garantir

- `Metric("loss")` sans orchestrateur actif lève une erreur.
- `loss.push(...)` hors run actif lève une erreur.
- `Metric("loss", float)` puis `Metric("loss", int)` lève une erreur de `dtype` mismatch.
- Pousser une valeur de type incompatible avec le `dtype` lève une `TypeError`.
- `orchestrator.get("loss")` contient exactement une ligne par `push` réussi.
- Les colonnes du DataFrame suivent l'ordre d'insertion des paramètres.
- `skip_if_exists=True` évite les doublons pour un même `(run_key, step)`.
- `ContextManager.exit()` restaure correctement le contexte parent en cas de runs imbriqués.
