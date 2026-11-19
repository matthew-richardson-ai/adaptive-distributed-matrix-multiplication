# Checkpoint 2 Summary

**CSC 4770 / 5770 – Adaptive Distributed Matrix Multiplication**  
**Matthew Richardson and James**

## Project Goal

We are building the required distributed matrix multiplication system using separate Docker microservices that communicate over TCP. The matrices are divided into smaller blocks, those block-multiplication jobs are distributed to worker containers, and an aggregation service reconstructs the final matrix.

Our main question is not simply whether more workers are faster. We are testing when distributed execution is worth the added communication, scheduling, Docker, and aggregation overhead. We also added an adaptive scheduler that can identify block products that are guaranteed to be zero and skip them before they are sent to a worker.

## What We Have Implemented

The full distributed path is working end-to-end:

- Ingress / scheduler service
- TCP message framing and binary matrix transfer
- blocked matrix decomposition and padding
- scalable Docker worker containers
- pull-based task scheduling
- aggregation service
- task batching and batched results
- task lease / retry support
- exact-zero adaptive pruning
- serial and blocked-serial baselines
- correctness validation
- repeatable benchmark suites and result summaries
- controlled BLAS threading for fairer worker-count tests

All 20 current Python tests pass, and the Docker system has been tested locally end-to-end.

## Results So Far

For dense 1000 x 1000 matrices with 256 x 256 blocks, the mean distributed runtime improved from about 1.60 seconds with one worker to 0.90 seconds with four workers. Eight workers did not improve performance, showing that adding workers eventually introduces enough contention and overhead to cancel the benefit of additional parallelism.

For dense 2000 x 2000 matrices, two workers performed best on average at about 5.87 seconds. Four workers were slower and more variable.

The strongest result so far is the block-sparse adaptive experiment. With 512 possible block multiplications, exact-zero pruning reduced the number of dispatched tasks to roughly 118–128. Compared with the normal sparse distributed baseline, adaptive pruning reduced mean runtime by about 59% with one worker, 40% with two workers, and 44% with four workers. Numerical error remained around 5 x 10^-16, so exact-zero pruning preserved the correct result.

## Main Pain Points / Lessons

- The original TCP implementation sent too many small messages. A 500 x 500 test initially took about 49 seconds; task and result batching reduced the same workload to about 2.07 seconds.
- NumPy/BLAS internal multithreading initially made worker-count comparisons misleading, so each Docker worker is now restricted to one BLAS thread.
- Docker and WSL caused some development instability.
- A benchmark-only dependency accidentally entered the Ingress service and caused its container to restart until the dependency was removed.
- More workers are not automatically better. Host CPU, memory, communication, and aggregation overhead become limiting factors.

## What Is Left

The core implementation is substantially complete. Remaining work is mainly:

- run any final controlled experiments needed for the paper
- decide whether block-size sensitivity or approximate pruning adds useful evidence
- generate final figures and tables
- complete the 10-reference research set
- write the 8-page IEEE/ACM graduate paper
- prepare the final presentation and live or recorded demo

Low-rank factored transmission remains an optional extension and will not be allowed to delay completion of the core project.

## Main Takeaway

The project is showing that distributed matrix multiplication is a systems tradeoff, not simply a matter of adding more workers. Parallel workers help only while useful computation exceeds communication and coordination overhead. For matrices with exploitable block sparsity, intelligently eliminating unnecessary work can provide a larger benefit than adding more workers.
