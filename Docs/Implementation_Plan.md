# Implementation Plan

## Purpose

This document defines the implementation order for the CSC 4770 / 5770 distributed matrix multiplication project. The goal is to complete the required distributed baseline first, then add adaptive behavior in controlled stages.

## 1. Serial Reference

Implement a trusted single-process multiplication path.

Requirements:

- accept two compatible matrices
- compute the reference result
- record elapsed time
- expose a validation helper using `numpy.allclose`
- fail clearly on invalid dimensions

Exit criterion: the reference implementation passes deterministic unit tests.

## 2. Matrix Decomposition

Implement blocked decomposition in `Ingress/decomposition.py`.

Each task must identify:

- task ID
- output block coordinate `(i, j)`
- reduction index `k`
- left block `A[i,k]`
- right block `B[k,j]`
- block dimensions

The first version should use a fixed configurable block size. Quadtree-specific recursion can be added after the baseline is stable.

Exit criterion: generated tasks reconstruct the same result as the serial reference when evaluated locally.

## 3. Protocol Layer

Implement `Protocol/constants.py`, `Protocol/messages.py`, and `Protocol/framing.py`.

Required behavior:

- fixed protocol version
- explicit message type
- task ID
- payload length
- bounded receive loop
- rejection of malformed or oversized payloads
- binary float64 matrix payloads

Exit criterion: protocol round-trip tests pass for task and result messages.

## 4. One-Worker Distributed Path

Implement the smallest end-to-end networked system:

1. Ingress creates one or more block tasks.
2. Ingress sends tasks to one worker over TCP.
3. Worker decodes the task and computes the block product.
4. Worker sends a result message.
5. Aggregator places the partial result into the correct output block.
6. Final matrix is validated against the serial reference.

Exit criterion: a development-size matrix produces the same result as the serial baseline.

## 5. Multi-Worker Scheduling

Implement a central queue and pull-based scheduling.

Worker state should include:

- worker ID
- connected / disconnected state
- current task
- last activity time
- completed task count

Task state should include:

- QUEUED
- IN_FLIGHT
- COMPLETED
- FAILED

Exit criterion: two or more workers process different tasks from the same multiplication and produce a correct final result.

## 6. Aggregation

Implement `Aggregator/agg_server.py` so that partial products are accumulated by output coordinate.

For each result `T_ikj`:

```text
C[i,j] += T_ikj
```

The aggregator must:

- reject unknown task IDs
- ignore or detect duplicate completions
- track expected contributions per output block
- know when the full result is complete
- remove padding if padding is enabled

## 7. Instrumentation

Add measurements before adaptive optimization.

Record:

- total wall time
- decomposition time
- queue wait time
- TCP send / receive time
- worker compute time
- aggregation time
- bytes sent / received
- task count
- per-worker task count
- retries / failures

The baseline measurements are required so later adaptive results can be compared fairly.

## 8. Adaptive Sparsity

After the distributed baseline is correct, add adaptive checks in `Ingress/adaptive.py`.

Stage 1: exact-zero pruning.

Stage 2: optional approximate pruning using the Frobenius-norm bound:

```text
||A_ik B_kj||_F <= ||A_ik||_F ||B_kj||_F
```

A task may be skipped when the configured bound is below the chosen tolerance.

Record:

- tasks considered
- tasks pruned
- preprocessing time
- bytes avoided
- compute avoided
- final relative error

## 9. Fault Recovery

After multi-worker scheduling is stable:

- detect worker disconnect / timeout
- move unfinished task back to QUEUED
- retry up to a configured limit
- reject duplicate late results

This is task-level recovery only. The project does not require distributed consensus.

## 10. Low-Rank Extension

Low-rank processing is a stretch goal.

Do not begin this phase until:

- baseline is correct
- multi-worker execution is stable
- instrumentation works
- sparsity experiments can run reproducibly

## Recommended Implementation Sequence

```text
Serial Reference
      |
Decomposition
      |
Protocol / Framing
      |
One Worker
      |
Aggregator
      |
Multiple Workers
      |
Instrumentation
      |
Adaptive Sparsity
      |
Fault Recovery
      |
Low-Rank Extension
```

## Core Completion Definition

The required system is considered functionally complete when:

- all required service roles run independently
- services communicate over TCP
- matrices are decomposed into independent tasks
- multiple workers execute tasks
- aggregation reconstructs the final matrix
- the result matches the serial reference
- worker count can be changed for scaling experiments
- baseline performance metrics are recorded
