# Checkpoint 2 Report

## Adaptive Distributed Matrix Multiplication

**James Finnegan and Matthew Richardson**  
**CSC 4770 / 5770**

---

## 1. Project Overview

This project implements a distributed blocked matrix multiplication system using Docker microservices that communicate through TCP sockets. Two input matrices are divided into smaller block-multiplication tasks, distributed across worker containers, and reconstructed by a dedicated aggregation service.

The project also evaluates an adaptive scheduling strategy for block-sparse matrices. Before a task is dispatched, the scheduler can identify block products that are guaranteed to contribute zero to the final result and skip them entirely.

The central systems question is:

> When does distributed blocked matrix multiplication become worthwhile despite the communication, scheduling, container, and aggregation overhead introduced by distribution?

The graduate extension adds a second question:

> Can matrix structure be used to eliminate enough unnecessary computation and network traffic to improve overall distributed performance?

---

## 2. Background and Motivation

Matrix multiplication is widely used in scientific computing, machine learning, simulation, and numerical analysis. It is naturally parallelizable because an output matrix can be divided into blocks, and each output block is formed from a collection of independent block products.

A distributed implementation introduces additional costs that do not exist in a single-process multiplication. Before a worker performs useful computation, the system must:

- decompose the matrices
- create and schedule tasks
- serialize matrix blocks
- transmit data over TCP
- deserialize data at the worker
- return partial products
- aggregate those results into the final matrix

Because of this overhead, increasing the number of workers does not automatically improve performance.

This project therefore treats matrix multiplication as a distributed-systems experiment rather than an attempt to outperform optimized BLAS or NumPy on a single machine. The objective is to measure the tradeoff between useful parallel computation and the systems overhead required to distribute that computation.

---

## 3. System Architecture

The implemented architecture is shown below.

```text
                Input Matrices A and B
                         |
                         v
                Ingress / Scheduler
                         |
              Block Decomposition
                         |
                Adaptive Decision
                  /           \
             Dispatch        Prune
                 |
                 v
              TCP Tasks
                 |
       +---------+---------+
       |         |         |
       v         v         v
    Worker 1  Worker 2  Worker N
       |         |         |
       +---------+---------+
                 |
                 v
              Aggregator
                 |
                 v
            Final Matrix C
                 |
                 v
        Numerical Validation
```

### Ingress / Scheduler

The Ingress service validates the input matrices, pads them when necessary, divides them into blocks, creates multiplication tasks, applies adaptive pruning when enabled, and schedules work across the worker pool.

### Worker Pool

Each worker runs in its own Docker container. Workers pull tasks from the scheduler, compute block products independently, and return results over TCP.

### Aggregator

The Aggregator receives partial block products and accumulates them into the correct output blocks until the final matrix is complete.

### Benchmark Driver

The benchmark system generates test matrices, runs serial and distributed comparisons, validates numerical correctness, records timing data and task statistics, and stores repeatable JSON results.

---

## 4. Implementation Progress

The core distributed system is now working end-to-end.

### Implemented Components

- NumPy serial reference multiplication
- single-threaded blocked serial comparison
- numerical correctness validation
- blocked matrix decomposition
- padding for non-even block dimensions
- framed TCP protocol
- binary float64 matrix transmission
- Ingress microservice
- central task queue
- pull-based worker scheduling
- Docker worker containers
- dedicated Aggregation microservice
- multi-worker execution
- task batching
- batched result transmission
- task leasing and unfinished-task requeue support
- duplicate-result protection
- exact-zero sparsity pruning
- Frobenius-norm bound support for optional approximate pruning
- dense, block-sparse, and low-rank matrix generators
- repeated benchmark suites
- benchmark summary generation
- configurable Docker worker scaling
- controlled BLAS threading for worker-count experiments
- automated unit tests
- Docker end-to-end smoke testing

The current Python test suite passes all 20 tests.

Low-rank helper functions and kernels have also been implemented, but low-rank factored transmission has not yet been integrated into the TCP network path. It remains an optional extension.

---

## 5. Key Design Decisions

### TCP Framing

TCP is a byte stream and does not preserve message boundaries. The protocol therefore uses explicit framing with message metadata and payload lengths.

Matrix values are transmitted as binary float64 data rather than text.

### Pull-Based Scheduling

Workers request tasks instead of receiving permanently assigned work. This allows faster workers to continue taking new tasks and makes horizontal scaling easier to test.

### Task Batching

The first distributed implementation created too many small network exchanges. Communication overhead became much larger than the useful computation.

The system was changed so multiple tasks and multiple results could be transmitted in batches. This became one of the most important performance improvements in the project.

### Controlled BLAS Threading

NumPy and BLAS libraries can internally create multiple computation threads. Without controlling this, the number of Docker workers would not accurately represent the degree of parallelism being tested.

Each worker container is therefore restricted to one BLAS thread.

### Adaptive Exact-Zero Pruning

For block-sparse matrices, the scheduler checks candidate block products before dispatch.

If a product is guaranteed to be zero, the task is skipped without changing the mathematical result.

This reduces:

- TCP traffic
- task count
- worker computation
- returned result traffic
- aggregation work

---

## 6. Development Challenges

Several development issues directly influenced the design and experimental methodology.

### Excessive Communication Overhead

An early 500 x 500 dense test using block size 64 and two workers took approximately 49 seconds.

After batched task and result communication was implemented, the same general workload completed in approximately 2.07 seconds.

This demonstrated that network granularity can dominate runtime in a microservice system.

### Worker Oversubscription

Initial worker-count results were difficult to interpret because NumPy could internally create additional threads.

Restricting each worker to one BLAS thread made worker-count comparisons much more meaningful.

### Docker and WSL Stability

Docker Desktop and WSL2 occasionally became unstable during development. A heavier benchmark also coincided with a host shutdown, reinforcing the need to treat host CPU and memory limits as part of the experimental environment.

### Benchmark Dependency Failure

A benchmark-only dependency was accidentally imported into the production Ingress service. This caused the Ingress container to repeatedly restart because the package was not installed in that image.

The benchmark dependency was removed from the production service path.

### Scaling Saturation

Repeated experiments showed that adding workers eventually stops improving performance and can increase runtime variability. This is now treated as one of the main findings of the project rather than an implementation failure.

---

## 7. Experimental Design

The current experiments compare three main execution modes:

1. optimized NumPy serial multiplication
2. distributed blocked multiplication without adaptive pruning
3. distributed blocked multiplication with sparsity-aware pruning

### Primary Experimental Variables

- matrix size
- block size
- worker count
- matrix structure
- adaptive pruning mode

### Primary Measurements

- total runtime
- candidate task count
- dispatched task count
- pruned task count
- worker scaling behavior
- numerical error

Dense matrices are used as a control case. Block-sparse matrices are used to evaluate whether pruning removes enough unnecessary work to improve total runtime.

---

## 8. Current Results

### 8.1 Dense Scaling: 1000 x 1000

Configuration:

```text
Matrix size: 1000 x 1000
Block size: 256
Workers: 1, 2, 4, 8
Trials: 5 per worker count
Adaptive pruning: disabled
```

| Workers | Mean Distributed Runtime |
|---:|---:|
| 1 | 1.596 s |
| 2 | 1.011 s |
| 4 | 0.896 s |
| 8 | 1.074 s |

Performance improved from one worker through four workers.

The eight-worker configuration did not improve the mean runtime and produced substantially more variability.

For this workload, four workers produced the best mean runtime.

---

### 8.2 Dense Scaling: 2000 x 2000

Configuration:

```text
Matrix size: 2000 x 2000
Block size: 256
Workers: 1, 2, 4
Trials: 3 per worker count
Adaptive pruning: disabled
```

| Workers | Mean Distributed Runtime |
|---:|---:|
| 1 | 8.686 s |
| 2 | 5.873 s |
| 4 | 6.373 s |

Two workers produced the best mean runtime.

The four-worker configuration was slower and showed greater variability, suggesting contention or coordination overhead at the host level.

---

### 8.3 Block-Sparse Adaptive Pruning: 2000 x 2000

Configuration:

```text
Matrix size: 2000 x 2000
Block size: 256
Matrix type: block sparse
Zero-block fraction: 0.5
Workers: 1, 2, 4
Trials: 3 per configuration
Adaptive epsilon: 0
```

| Workers | Sparse Baseline | Adaptive Sparse | Runtime Reduction |
|---:|---:|---:|---:|
| 1 | 8.437 s | 3.460 s | about 59% |
| 2 | 5.777 s | 3.492 s | about 40% |
| 4 | 6.018 s | 3.386 s | about 44% |

Each run began with 512 candidate block multiplications.

The adaptive scheduler reduced that workload to approximately 118 to 128 dispatched tasks, pruning approximately 384 to 394 tasks.

This removed roughly three quarters of the candidate distributed work.

The relative numerical error remained approximately:

```text
5 x 10^-16
```

Exact-zero pruning therefore reduced runtime substantially without changing the mathematical result.

---

## 9. Preliminary Findings

The current results support several conclusions.

### Distributed Execution Has Significant Fixed Overhead

At the tested matrix sizes, the Docker/TCP implementation remains slower than optimized local NumPy multiplication.

This is expected because NumPy uses highly optimized native linear algebra libraries while the distributed system includes decomposition, serialization, scheduling, TCP communication, Docker isolation, and aggregation.

### Communication Granularity Is Critical

The large improvement after implementing batching shows that protocol design can dominate total runtime.

Too many small messages can eliminate the benefit of parallel computation.

### More Workers Do Not Always Improve Performance

Adding workers improved performance only up to a workload-dependent point.

Beyond that point, contention and coordination overhead caused runtime to flatten or regress.

### Adaptive Task Selection Provides a Strong Benefit

The block-sparse experiments produced the strongest result so far.

Exact-zero pruning reduced distributed runtime by approximately 40% to 59%.

After pruning, runtime was nearly flat across one, two, and four workers:

```text
1 worker: 3.460 s
2 workers: 3.492 s
4 workers: 3.386 s
```

This suggests that removing unnecessary distributed work can be more valuable than simply adding more worker containers.

### Numerical Correctness Is Preserved

Exact-zero pruning removed only block products that mathematically contribute zero.

The final matrices continued to match the serial reference within normal floating-point tolerance.

---

## 10. Remaining Work

The core implementation is substantially complete. Remaining work is primarily experimental refinement and final documentation.

Planned work includes:

- repeat selected final experiments under controlled conditions
- determine whether additional matrix sizes are necessary
- evaluate block-size sensitivity
- improve instrumentation if a more detailed communication/compute breakdown is needed
- evaluate approximate Frobenius-bound pruning only if it adds meaningful evidence
- decide whether low-rank TCP transmission is worth the remaining development time
- generate final figures and tables
- collect and verify at least 10 research references
- complete the 8-page IEEE or ACM graduate paper
- prepare the graduate presentation
- prepare a live or recorded demonstration

Low-rank transmission remains optional and will not delay completion of the validated sparsity-aware system.

---

## 11. Conclusion

The project has progressed from architectural planning into a working and experimentally validated distributed system.

The current results show that distributed matrix multiplication is not improved simply by increasing worker count. Performance depends heavily on communication granularity, available host resources, scheduling overhead, and the amount of useful computation per task.

The strongest finding so far is that adaptive sparsity-aware task elimination can substantially reduce runtime without sacrificing numerical correctness. For the tested block-sparse workload, the scheduler removed roughly three quarters of the candidate tasks and reduced distributed runtime by approximately 40% to 59%.

The project is now entering its final experimentation, analysis, and reporting phase.
