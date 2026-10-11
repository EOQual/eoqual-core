# Contribuer à eoqual-core

Ce paquet est une dépendance des autres modules EOQual : chaque ajout se
paie en stabilité. Avant de proposer une brique, vérifier les règles du
README :

- numpy et scipy seulement ;
- fonctions pures, sans entrées/sorties, journalisation ni état ;
- utilisée par **deux modules au moins** (citer lesquels dans la
  proposition) ;
- pas d'indicateur de qualité (ils vont dans `eoqual-metrics`).

## Ajouter une brique

1. Un module par domaine (`fourier`, `edge`, `bayer`...), `__all__`
   explicite, docstrings NumPy.
2. Des tests à vérité **analytique** (gaussienne, Airy, plan...) plutôt
   que des valeurs de référence figées.
3. Une section dans `REFERENCE_TECHNIQUE.md` (principe, conventions,
   pièges) et une ligne dans le tableau du README.
4. Une entrée dans `CHANGELOG.md`. Toute modification de signature ou de
   résultat d'une fonction existante est une rupture d'API : version
   mineure suivante, et mention explicite.

## Licence

Toute implémentation reprise d'un code publié ailleurs doit citer sa
source et sa licence vérifiée (`THIRD_PARTY_LICENSES.md`) ; seules les
licences permissives sont acceptées.

## Tests

```bash
PYTHONPATH=src pytest tests
```
