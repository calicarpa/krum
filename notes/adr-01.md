# ADR 2026-05-05

We assume the user is fluent in python programming, and pytorch.

## Example

- model distributed learning
- over a dataset
- prepare workers
    - how many?
    - assign dataset slices to each worker
- prepare parameter server
    - which gar to use
- prepare attack
    - attacker's knowledge
        - all gradients from honest workers at time t
- prepare metric
    - which metrics
    - scheduling
- simulation (possibly many simulations)
    - synchrone: for round t in 1, 2, ...
        - parameter server shares the model state at time t to each worker
        - each (honest) worker computes a gradient
        - attacker computes the attack gradients
        - parameter server receives all gradients, aggregates them, and performs a SGD step
        - simulation metric tracking
            - current loss value
            - accuracy (not every time step)
                - caution: centralized vs decentralized
        - checkpoint 
            - metric traces (including metadata)
- collect traces for all simulations
- dataviz bare minimum


## To discuss

- model
    - model function
    - parameter vector 
    - model parameters (possibly up to 1st order)
- parameter server vs worker
    - what is a parameter server? its interface? its knowledge?
    - what is a worker? its interface? its knowledge?
        - local dataset?
- attacks
    - how to represent attacker's knowledge?
- dataset provider
    - how to push/transform data natch for each worker
- what is the simulation model?
    - general network topology?
    - scheduler? (asynchrony)
    - checkpoints
    - metrics
- what is the execution model?
    - parallelizable simulations

## Decisions

- Python library
- We focus on 1st order methods only. I.e., gradient aggregations. 0th-order (models aggregation) is excluded.
- Simulation loop is under the responsibility of the user.

???
- We avoid global variables and registers
