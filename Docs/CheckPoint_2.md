# Checkpoint 2 Working Draft

## CSC 4770 / 5770 - Adaptive Distributed Matrix Multiplication

### Checkpoint 2 Requirements

By Checkpoint 2, the project should have:

- a drafted abstract
- a drafted background section
- a drafted experimental design
- implementation underway
- a clear explanation of what remains
- a clear explanation of issues or friction encountered

This document is a working meeting guide and paper seed.

## Abstract - Draft

This project investigates the performance tradeoffs of distributed blocked matrix multiplication implemented as TCP-connected Docker microservices. Two input matrices are decomposed into independent block-multiplication tasks, distributed across worker containers, and reconstructed by a dedicated aggregation service. The project first establishes a correct distributed baseline and measures the cost of decomposition, scheduling, communication, worker execution, and aggregation. It then introduces structure-aware task selection, beginning with sparsity-based pruning, to determine whether preprocessing can eliminate enough computation and network traffic to justify its own overhead. Experiments will compare a serial reference, a dense distributed baseline, and an adaptive distributed system across multiple matrix sizes, block sizes, worker counts, and matrix structures. The primary goal is to identify the conditions under which distributed execution and adaptive task selection improve total execution time without unacceptable numerical error.

## Background - Draft

Matrix multiplication is a common operation in scientific computing, machine learning, simulation, and other compute-intensive applications. It is also naturally decomposable because each output block can be expressed as the sum of independent block products.

The course project requires that matrix input, multiplication, and aggregation be implemented as separate microservice roles using Docker containers or virtual machines and TCP socket communication. This creates a useful distributed-systems experiment because performance depends not only on arithmetic cost, but also on decomposition, scheduling, serialization, communication, queueing, and aggregation.

Our project extends the required distributed baseline by asking whether matrix structure can be used to avoid unnecessary work. The first adaptive mechanism is sparsity-aware task pruning. Exact-zero block products can be skipped without changing the final result. Approximate pruning may also be evaluated using a numerical bound, but only after exact behavior is validated.

The project is not intended to replace optimized HPC matrix libraries. It is designed to measure the interaction between computation, communication, scheduling, and matrix structure in a containerized microservice environment.

## Experimental Design - Draft

Three systems will be compared:

1. Serial reference implementation
2. Distributed Docker baseline with dense block transmission and no pruning
3. Adaptive distributed implementation with sparsity-aware task selection

Primary independent variables:

- matrix size
- worker count
- block size
- matrix structure
- pruning tolerance when approximate pruning is enabled

Primary measurements:

- total runtime
- decomposition / preprocessing time
- TCP communication time
- queue wait time
- worker compute time
- aggregation time
- total bytes transferred
- tasks created, dispatched, and pruned
- worker utilization
- numerical error

Dense matrices will serve as the control case. Block-sparse matrices will test whether pruning can remove enough work to offset the cost of adaptive analysis.

## Current Implementation Status

The architecture and repository structure are established. The project is divided into Ingress, Protocol, Worker, Aggregator, Benchmarks, Tests, Results, and Docs areas.

The current implementation priority is the baseline path:

```text
Serial Reference
      |
Decomposition
      |
TCP Framing
      |
One Worker
      |
Aggregation
      |
Multiple Workers
      |
Instrumentation
```

Adaptive pruning is intentionally scheduled after the baseline so that its effect can be measured against a correct distributed comparison.

## What Remains

Core implementation work still includes:

- serial reference and validation
- matrix decomposition
- TCP message framing
- worker execution path
- aggregation service
- multi-worker scheduling
- benchmark instrumentation
- Docker Compose configuration
- correctness and protocol tests

After the baseline is stable:

- exact-zero pruning
- approximate sparsity pruning
- limited worker failure recovery
- low-rank extension if time permits

## Issues and Friction

The primary challenge has been scope control. The graduate extension introduces several possible directions, including adaptive pruning, fault recovery, low-rank compression, and detailed instrumentation. Implementing all of them before the required distributed baseline would put the core project at risk.

The project therefore uses a strict priority:

```text
Correctness
-> Distributed Baseline
-> Multi-Worker Execution
-> Instrumentation
-> Adaptive Sparsity
-> Low-Rank Extension if feasible
```

Additional expected friction includes TCP framing, block-size tradeoffs, worker load imbalance, memory use for large matrices, and the possibility that adaptive preprocessing costs more than the work it removes.

## Immediate Goal Before Final Experimentation

Produce the same correct matrix result through:

```text
Ingress -> TCP -> Docker Worker(s) -> Aggregator -> Final Matrix
```

Once that path is reliable, benchmark it before enabling adaptive behavior.

## Graduate Deliverable Direction

The Checkpoint 2 abstract, background, and experimental design will be expanded into the final 8-page IEEE or ACM graduate paper. The final paper will add measured results, figures, analysis, conclusion, and at least 10 verified references. The project will also require a graduate presentation and live or recorded demonstration during Dead Week.
