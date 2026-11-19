# Checkpoint 2 Meeting Summary

**CSC 4770 / 5770 – Adaptive Distributed Matrix Multiplication**  
**James Finnegan and Matthew Richardson**

## Project Goal

Build the required distributed matrix multiplication system using separate Docker microservices over TCP, then measure when parallel execution is actually worth the communication, scheduling, Docker, and aggregation overhead.

Our graduate extension adds adaptive sparsity-aware pruning: if a block multiplication is guaranteed to produce zero, the scheduler skips it instead of sending it to a worker.

## What We Have Implemented

- Ingress / scheduler, Docker worker pool, and aggregation service
- blocked matrix decomposition and padding
- framed TCP protocol with binary float64 matrix transfer
- pull-based multi-worker scheduling
- task/result batching
- task lease / retry support
- exact-zero adaptive pruning
- serial and blocked-serial baselines
- benchmark generation, repeated trials, and result summaries
- controlled BLAS threading for fair worker-count testing
- 20 passing Python tests and successful local Docker end-to-end runs

## Results So Far

**Dense 1000 x 1000, block 256:**  
1 worker: 1.596 s | 2 workers: 1.011 s | 4 workers: 0.896 s | 8 workers: 1.074 s

**Dense 2000 x 2000, block 256:**  
1 worker: 8.686 s | 2 workers: 5.873 s | 4 workers: 6.373 s

**Block-sparse 2000 x 2000 with exact-zero pruning:**  
512 candidate tasks were reduced to about 118–128 dispatched tasks.

- 1 worker: 8.437 s baseline -> 3.460 s adaptive, about 59% reduction
- 2 workers: 5.777 s baseline -> 3.492 s adaptive, about 40% reduction
- 4 workers: 6.018 s baseline -> 3.386 s adaptive, about 44% reduction

Numerical error remained around 5 x 10^-16.

## Main Pain Points / Lessons

- Too many small TCP messages originally dominated runtime. Task/result batching reduced a 500 x 500 test from about 49 s to about 2.07 s.
- NumPy/BLAS internal threading initially made worker-count experiments misleading, so workers are now pinned to one BLAS thread.
- Docker/WSL stability caused some development interruptions.
- A benchmark-only dependency accidentally entered the Ingress service and caused container restarts until it was removed.
- More workers are not automatically better. CPU, memory, communication, and aggregation overhead eventually outweigh additional parallelism.

## What Is Left

- final controlled experiments if needed
- block-size sensitivity and/or approximate pruning if useful
- final figures and tables
- 10 verified references
- 8-page IEEE/ACM graduate paper
- presentation and live or recorded demo

**Main takeaway:** the project is showing that intelligent task elimination can provide a larger and more consistent performance benefit than simply adding more worker containers.
