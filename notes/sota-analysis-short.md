# SOTA — Executive Summary

## 1. Six libraries, three categories

| Library | Category | Pub | Scope |
|---|---|---|---|
| **Krum** | Byzantine-robust lib | targeting JMLR | 10 agg, 5 att, P2P+PS, faithful sims, orchestration |
| **ByzFL** (EPFL) | Byzantine-robust lib | arXiv 2025 | 12+4 agg, 9 att, PS only, generic FL loop, JSON bench |
| **FedLab** | General FL framework | **JMLR 2023** | 10 FL algos, 9 data splits, compression, **zero Byz** |
| **Blades** | Benchmark suite | IoTDI 2024 | 9 agg, 10 att, YAML+Ray, but fedlib dep mismatch |
| **FL-Byz-Lib** (CRYPTO-KU) | Agg/att collection | TIFS 2024 | **36 agg, 23 att** (biggest), pruning module, no license |
| **ByzPy** | Actor-runtime | none | 12+4 agg, 8 att, **P2P+PS**, 5 actor backends, pre-1.0 |

## 2. Key takeaways

- **Aucune librairie n'est complète.** La plus large (FL-Byz-Lib, 36 agg) n'a pas de license, papier, doc, ou simulation. ByzFL a le plus de features avec un papier mais arXiv seulement, PS only, boucles génériques. FedLab a JMLR mais **zéro** résilience Byzantine.
- **Krum est le seul avec des simulations fidèles aux papiers** — LR schedule, init, weight decay, attack stop exacts. ByzFL/FL-Byz-Lib ont des boucles FedAvg/DSGD génériques.
- **Krum et ByzPy sont les seuls avec du P2P.** ByzPy est pre-1.0, 2 stars, sans papier.
- **Qualité d'ingénierie**: Krum est le seul avec full type checking (`ty`), tests edge-to-edge, ADRs, pre-commit CI. Aucune autre librairie n'approche ce standard.

## 3. Krum's competitive advantages

| Axe | Krum | Concurrent le plus proche |
|---|---|---|
| **Fidélité simulations** | ✅ papiers spécifiques (NIPS17, ICML18, ICML23) | ❌ ByzFL: boucles FedAvg/DSGD génériques |
| **Topologie P2P** | ✅ centralisé + décentralisé | ❌ ByzFL: PS only (ByzPy: oui mais pre-1.0) |
| **Zero-copy model** | ✅ flat tensors shared storage | ❌ tous les autres: flatten/unflatten standard |
| **API stateless** | ✅ @classmethod, pas d'état caché | ❌ ByzFL/ByzPy: instances stateful |
| **Type checking** | ✅ `ty` sur tout le code | ❌ tous: partiel ou inexistant |
| **Tests** | ✅ edge cases, régression, CI | ❌ ByzFL: basique, sans CI |
| **Aggrégateurs uniques** | Bulyan, Brute, Aksel | — |
| **Attaques uniques** | FullGradNeg, SmallPerturb | — |

## 4. Gaps à reconnaître

| Gap | Leader | Détail |
|---|---|---|
| **Couverture agg** | FL-Byz-Lib (36) > ByzFL (12+4) > Krum (10) | Krum manque CAF, SMEA, CenteredClipping, MDA, MeaMed |
| **Couverture att** | FL-Byz-Lib (23) > Blades (10) > ByzFL (9) > Krum (5) | Krum manque IPM, Mimic, LabelFlip |
| **Pre-aggregators** | ByzFL, ByzPy (4) vs Krum (0) | Clipping, NNM, Bucketing, ARC |
| **Data heterogeneity** | FedLab (9 schemes) > ByzFL (3) > Krum (IID only) | |
| **Parallel execution** | FedLab (3 modes) > ByzFL (Pool) > Krum (sync) | |
| **Visualisation** | ByzFL (curves+heatmaps) vs Krum (manual) | |
| **Persistence** | tous sauf Krum (filesystem) vs Krum (in-memory) | |

## 5. Narrative pour le papier

1. **Le paysage est fragmenté** — 6 librairies, aucune complète. Chacune a un angle mort majeur (pas de license, pas de papier, pas de Byz, pas de P2P, pas de CI).
2. **Krum est le premier à combiner** (a) primitives complètes, (b) simulations fidèles aux papiers, (c) orchestration, (d) P2P.
3. **Krum sera le premier JMLR MLOSS** dans le créneau Byzantine resilience — FedLab (JMLR) a zéro Byz, ByzFL est arXiv seulement.
4. **Qualité d'ingénierie** — Krum établit un standard (ty, tests, CI, ADRs) qu'aucune librairie existante ne rencontre.
