How to add a new dataset
========================

1. Create a module under ``experiments/datasets/``.
2. Expose a generator or dataset construction function.
3. Plan for caching if the dataset is large or downloadable.
4. Ensure ``make_datasets(...)`` (or ``Dataset(...)``) can consume it without
   invasive changes.
