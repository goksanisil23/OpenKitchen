# NEAT Racer Implementation

This implementation evolves a population of feed-forward neural network genomes that
control agents via the raycast hit observations.


## Genome Evaluation

Each genome contains nodes and weighted directed connections. Inputs are normalized
raycast distances, one bias node is fixed to `1`, and outputs are mapped to throttle
and steering.

```text
evaluateGenome(genome, sensor_hits):
    build topological order from enabled connections
    set input node values from normalized sensor distances
    set bias node to 1

    for each non-input node in topological order:
        sum weighted incoming values
        apply sigmoid activation

    return output node values
```

The genome is kept feed-forward by rejecting new connections that would introduce a
cycle.

## Fitness

Fitness is based on progress along the race track centerline.

```text
for each agent:
    find nearest track index to the agent position
    unwrap progress across the start/finish boundary
    keep the best cumulative progress reached during the generation
```

At the end of the generation, each genome's raw fitness becomes its best recorded
progress.

## Speciation

Genomes are grouped by NEAT compatibility distance. The distance compares connection
innovation ids and weights:

```text
compatibility = excess gene cost
              + disjoint gene cost
              + matching weight difference cost
```

Species share fitness by dividing each member's raw fitness by the species size.
Species that stop improving for too long are excluded from reproduction, except for
the species containing the global champion.

## Evolution

```text
evolve():
    keep the best genome unchanged as the elite
    allocate offspring slots to species by total adjusted fitness

    for each species:
        keep only the strongest fraction as possible parents
        create children by copying one parent or crossing over two parents
        mutate each child

    replace the old population with the new one
```

Crossover aligns connection genes by innovation id. Matching genes are inherited from
either parent, while non-matching genes are inherited from the fitter parent.

Mutation can:

- perturb or reset connection weights
- add a valid feed-forward connection
- split an existing connection by inserting a hidden node
- re-enable a disabled connection

## Innovation Tracking

The `InnovationTracker` assigns stable innovation ids to structural changes. This lets
separate genomes that discover the same connection or split align their genes during
compatibility checks and crossover.
