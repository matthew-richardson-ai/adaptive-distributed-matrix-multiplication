# Testing and Benchmark Plan

## Testing Goal

Separate correctness testing from performance benchmarking. A fast result is not useful if the distributed result is numerically wrong.

## Correctness Tests

### Serial Reference

- deterministic small matrices
- random matrices with fixed seeds
- incompatible dimensions
- zero matrices
- identity matrices

### Decomposition

- block coordinates are correct
- all required `(i,j,k)` tasks are generated
- no required task is missing
- local task execution and aggregation equal the serial result

### Protocol

- framing survives partial TCP reads
- task IDs survive round trip
- matrix dimensions survive round trip
- malformed frames are rejected

### Aggregation

- partial products accumulate into the correct block
- completion is detected only after all expected contributions arrive
- duplicate results are detected
- unknown task IDs are rejected

### Multi-Worker

- at least two workers receive tasks
- task completion is not duplicated
- final matrix matches serial result
- worker disconnect does not silently lose work once retry support is enabled

## Development Sizes

Use small matrices for debugging:

- 4 x 4
- 8 x 8
- 100 x 100

Do not use large benchmark matrices until correctness tests pass.

## Benchmark Variables

Primary variables:

| Variable | Candidate Values |
|---|---|
| Matrix size | 100, 500, 1000, 2000, 5000, up to 10000 if practical |
| Worker count | 1, 2, 4, 8 |
| Block size | 128, 256, 512 or practical equivalents |
| Matrix type | dense, block sparse, mixed |
| Pruning | disabled, exact-zero, selected approximate tolerances |

The final experiment matrix may be reduced to fit available hardware and time.

## Systems to Compare

1. Serial reference
2. Distributed dense baseline
3. Adaptive distributed system

Do not compare adaptive performance only against serial. The distributed dense baseline is necessary to isolate the effect of adaptive task selection.

## Metrics

Record per run:

- total wall-clock time
- preprocessing / decomposition time
- communication time
- queue wait time
- worker compute time
- aggregation time
- total tasks created
- total tasks dispatched
- total tasks pruned
- bytes sent
- bytes received
- per-worker completed tasks
- retries
- relative numerical error

Derived metrics:

```text
Speedup = T_serial / T_parallel
Efficiency = Speedup / worker_count
```

## Reproducibility

Every benchmark record should contain:

- timestamp
- git commit SHA
- matrix generation seed
- matrix size
- matrix type
- block size
- worker count
- pruning settings
- machine / CPU information
- Docker version
- Python / NumPy versions

## Benchmark Order

Recommended order:

1. serial baseline
2. one-worker Docker baseline
3. two-worker Docker baseline
4. four-worker Docker baseline
5. eight-worker baseline if practical
6. repeat the same worker sweep with sparse inputs
7. enable exact-zero pruning
8. evaluate approximate pruning only after exact pruning is validated

## Result Acceptance

A benchmark run is valid only when:

- no task is missing
- final matrix is produced
- numerical validation passes for exact modes
- configured error threshold is satisfied for approximate modes
- metrics are complete

Failed runs should be recorded separately rather than silently discarded.
