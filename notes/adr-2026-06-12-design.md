# ADR 2026-06-12 — Design de l'orchestration

## Principe

L'utilisateur crée un `Orchestrator` et des `Metric` dans son code.
Tout le reste est interne au package.

```python
orchestrator = Orchestrator("my_orchestrator")
# Ces appels sont valides uniquement à l'intérieur d'un orchestrator.run()
loss = Metric("loss")      # trouve l'orchestrateur actif tout seul
loss.push(step, valeur)    # taggé avec les paramètres du run en cours
df = orchestrator.get("loss")  # → pd.DataFrame
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
            experiment(**params)
        finally:
            self._contexts.exit()

    def get(self, name: str):
        """Retourne un pd.DataFrame."""
        return self._metrics.get(name)
```

---

## Flux complet

```
orchestrator.run(my_experiment, n=10, f=3, aggregator=Krum)
  1. ContextManager.enter(n=10, f=3, aggregator=Krum)
     → crée RunContext(params={n:10, f:3, aggregator:Krum})
     → _run_context.set(ctx)
  2. my_experiment(n=10, f=3, aggregator=Krum, n_steps=1000)
      a. loss = Metric("loss", float)
         → _current_orchestrator → MetricManager.create("loss", float)
         → stocke (name="loss", dtype=float)
      b. loss.push(step=0, value=2.34)
         → vérifie que 2.34 est un float
         → _run_context → RunContext.params
         → stocke (step=0, value=2.34, n=10, f=3, aggregator=Krum)
  3. ContextManager.exit()
     → _run_context.reset(token) et dépile le contexte parent s'il existe

orchestrator.get("loss")
  → MetricManager.get("loss")
  → assemble toutes les lignes taggées "loss"
  → pd.DataFrame avec colonnes [step, value, n, f, aggregator, attack]
```

## Exemple utilisateur complet

```python
from krum.orchestration import Orchestrator, Metric
from krum.simulations import MySimulation

def my_experiment(n, f, aggregator, attack, n_steps):
    sim = MySimulation(n=n, f=f, aggregator=aggregator, attack=attack)
    loss = Metric("loss", float)  # dtype = float
    for step in range(n_steps):
        sim.step()
        # skip_if_exists permet d'éviter d'insérer deux fois le même step
        loss.push(step, sim.loss, skip_if_exists=True)

orchestrator = Orchestrator("my_orchestrator")

for n in range(10, 100):
    for f in range(2, 5):
        for agg in [Average, Krum, Bulyan]:
            for atk in [Alie, SignFlip]:
                orchestrator.run(my_experiment, n=n, f=f,
                                 aggregator=agg, attack=atk, n_steps=1000)

loss_data = orchestrator.get("loss")
slice = loss_data[loss_data['n'] == 13]
```

---

## Invariants à tester / garantir

- `Metric("loss")` sans orchestrateur actif lève une erreur.
- `loss.push(...)` hors run actif lève une erreur.
- `Metric("loss", float)` puis `Metric("loss", int)` lève une erreur de `dtype` mismatch.
- Pousser une valeur de type incompatible avec le `dtype` lève une `TypeError`.
- `orchestrator.get("loss")` contient exactement une ligne par `push` réussi.
- Les colonnes du DataFrame suivent l’ordre d’insertion des paramètres.
- `skip_if_exists=True` évite les doublons pour un même `(run_key, step)`.
- `ContextManager.exit()` restaure correctement le contexte parent en cas de runs imbriqués.
