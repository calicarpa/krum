# ADR 2026-05-05

We assume the user is fluent in python programming, and pytorch.

## To discuss

- parameter server vs worker
    - what is a parameter server? its interface? its knowledge?
    - what is a worker? its interface? its knowledge?
- attacks
    - how to represent attacker's knowledge?
- what is the simulation model?
    - general network topology?
    - scheduler? (asynchrony)
    - checkpoints
    - metrics
- what is the execution model?
    - parallelizable simulations

## Decisions

???
- We focus on 1st order methods only. I.e., gradient aggregations. 0th-order (models aggregation) is excluded.
- We avoid global variables and registers
