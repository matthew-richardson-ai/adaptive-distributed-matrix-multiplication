# CSC 4770 / 5770 Checkpoint 2 Human Summary

## Adaptive Distributed Matrix Multiplication

### Purpose of This Document

This is a practical Checkpoint 2 briefing document. It is written to help explain the project clearly in conversation before the final IEEE/ACM paper is completed.

The main goals for Checkpoint 2 are to explain:

- what problem the project is trying to solve
- what system has been built
- what has been implemented and verified
- what problems were encountered during development
- what experimental results have been collected so far
- what those results mean
- what remains before the final paper, presentation, and demonstration

---

## 1. What Problem Are We Solving?

The course project requires a distributed matrix multiplication system built from separate microservices. The system must divide large matrices into smaller jobs, distribute those jobs to independent workers, communicate using TCP sockets, and reconstruct the final matrix through an aggregation service.

The basic idea sounds simple: if matrix multiplication can be divided into independent pieces, then adding workers should allow those pieces to be computed in parallel.

The actual systems problem is more complicated.

Distributed execution introduces overhead that does not exist in a single-process matrix multiplication. Before a worker can multiply two blocks, the system has to:

1. divide the matrices into blocks
2. create tasks
3. serialize the task data
4. send it through TCP
5. schedule it to a worker
6. deserialize it
7. compute the block product
8. send the result back
9. aggregate the partial result into the final matrix

Because of those costs, adding more workers does not automatically make the system faster.

The project therefore asks:

> When does distributed blocked matrix multiplication become worthwhile despite the communication, scheduling, container, and aggregation overhead?

The graduate extension adds a second question:

> Can matrix structure be used to identify work that does not need to be performed, so that the system avoids enough computation and communication to justify the cost of making that decision?

The first adaptive mechanism used to answer that question is sparsity-aware task pruning.

---

## 2. What Are We Building?

The system is an adaptive distributed blocked matrix multiplication platform implemented with Docker microservices and TCP socket communication.

The current architecture is:

```text
                Input matrices A and B
                         |
                         v
                Ingress / Scheduler
                         |
              Block decomposition
                         |
                Adaptive decision
                  /           \
             dispatch        prune
                 |
                 v
             TCP tasks
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
            Final matrix C
                 |
                 v
        Numerical validation
```

The major service roles are:

### Ingress / Scheduler

The Ingress service accepts the input matrices, validates them, divides them into blocks, creates multiplication tasks, performs adaptive pruning checks when enabled, and distributes work to available workers.

### Worker Pool

Each worker runs in its own Docker container. Workers request tasks, compute block products independently, and return their partial results over TCP.

The number of workers can be changed so horizontal scaling can be tested.

### Aggregator

The Aggregator receives partial block products and adds them to the appropriate output block until the complete result matrix has been reconstructed.

### Benchmark Driver

The benchmark system generates test matrices, measures serial and distributed execution time, validates numerical correctness, records task statistics, and saves repeatable JSON results.

---

## 3. What Has Been Implemented?

The core distributed system is working end-to-end.

Implemented and verified components include:

- NumPy serial reference multiplication
- single-threaded blocked serial comparison
- numerical correctness validation
- blocked matrix decomposition
- matrix padding for non-even block dimensions
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
- duplicate result protection
- exact-zero sparsity pruning
- Frobenius-norm bound support for optional approximate pruning
- dense matrix generation
- block-sparse matrix generation
- low-rank matrix generation
- benchmark result recording
- repeated benchmark suites
- benchmark summary generation
- configurable Docker worker scaling
- BLAS thread control for fairer worker-count experiments
- automated unit tests
- Docker end-to-end smoke testing

The current Python test suite passes all 20 tests.

Low-rank helper functions and kernels also exist, but low-rank factored transmission has not been integrated into the TCP network path and remains an optional extension.

---

## 4. Important Implementation Decisions

### TCP Framing

TCP is a byte stream and does not preserve application message boundaries. The system therefore uses explicit framing so each message contains metadata describing its type and payload size.

Matrix values are transmitted as binary float64 data instead of text representations.

### Pull-Based Worker Scheduling

Workers request work from the scheduler instead of the scheduler permanently assigning tasks to specific workers.

This makes it easier to add or remove workers and naturally sends more work to workers that become available sooner.

### Task Batching

The first network implementation sent tasks and results too frequently.

That caused TCP communication overhead to dominate the actual matrix multiplication.

The system was changed so multiple tasks and multiple results could be transmitted in batches.

This was one of the most important performance improvements made during implementation.

### Controlled BLAS Threading

NumPy and BLAS libraries can internally use multiple CPU threads.

Without controlling that behavior, four Docker workers might actually create many more than four computation threads, making worker-count experiments misleading.

Worker containers are therefore restricted to one BLAS thread so the Docker worker count represents the parallelism being tested more accurately.

### Adaptive Exact-Zero Pruning

For block-sparse matrices, the scheduler can inspect a candidate block multiplication before sending it to a worker.

If either input block makes the product exactly zero, the multiplication can be skipped without changing the final mathematical result.

This reduces:

- tasks sent over TCP
- matrix blocks transmitted
- worker computation
- results returned
- aggregation work

Because exact-zero pruning is mathematically lossless, it is the first adaptive optimization being evaluated.

---

## 5. Problems and Issues Encountered

Several implementation problems became useful parts of the project because they revealed where distributed overhead actually occurs.

### Excessive TCP Communication

The original implementation created too many small network exchanges.

For a 500 x 500 dense matrix with block size 64 and two workers, an early implementation took approximately:

```text
49.0 seconds
```

After batched task and result communication was implemented, the same general workload completed in approximately:

```text
2.07 seconds
```

This showed that communication granularity can matter more than the arithmetic itself in a microservice implementation.

### Worker Oversubscription

Initial worker-count experiments were difficult to interpret because NumPy could internally create additional computation threads.

BLAS threading was later restricted to one thread per worker container.

### Docker / WSL Stability

Docker Desktop and WSL2 occasionally became unresponsive during development.

One heavier experiment also coincided with a host-computer shutdown, which reinforced the need to treat available CPU and memory as experimental constraints rather than assuming arbitrary worker scaling.

### Benchmark Dependency Issue

A benchmark-only dependency was accidentally imported into the production Ingress service.

The Ingress container repeatedly restarted because that package was not installed in the service image.

The dependency was removed from the service path so production microservices no longer depend on benchmark-specific code.

### Worker Scaling Is Not Automatically Better

Repeated experiments showed that increasing worker count eventually stops helping and can increase variability.

This is not treated as a failure of the project. It is one of the main distributed-systems effects the project is intended to measure.

---

## 6. Current Experimental Results

### Experiment A: Dense Scaling at N = 1000

Configuration:

```text
Matrix size: 1000 x 1000
Block size: 256
Workers: 1, 2, 4, 8
Trials: 5 per worker count
Adaptive pruning: disabled
```

Mean distributed runtime:

| Workers | Mean Runtime |
|---:|---:|
| 1 | 1.596 s |
| 2 | 1.011 s |
| 4 | 0.896 s |
| 8 | 1.074 s |

The runtime improved from one worker through four workers.

Eight workers did not improve the mean runtime and produced substantially more variability.

For this workload, four workers gave the best mean performance.

### Experiment B: Dense Scaling at N = 2000

Configuration:

```text
Matrix size: 2000 x 2000
Block size: 256
Workers: 1, 2, 4
Trials: 3 per worker count
Adaptive pruning: disabled
```

Mean distributed runtime:

| Workers | Mean Runtime |
|---:|---:|
| 1 | 8.686 s |
| 2 | 5.873 s |
| 4 | 6.373 s |

Two workers produced the best mean runtime.

The four-worker configuration showed much greater runtime variability.

These results suggest that the useful worker count depends on workload size and available host resources. Increasing worker count eventually introduces enough contention or coordination overhead to offset the added parallelism.

### Experiment C: Block-Sparse Adaptive Pruning at N = 2000

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

The adaptive test used exact-zero pruning only.

Mean distributed runtime:

| Workers | Sparse Baseline | Adaptive Sparse | Runtime Reduction |
|---:|---:|---:|---:|
| 1 | 8.437 s | 3.460 s | about 59% |
| 2 | 5.777 s | 3.492 s | about 40% |
| 4 | 6.018 s | 3.386 s | about 44% |

Each run began with:

```text
512 candidate block multiplications
```

The adaptive scheduler reduced that to approximately:

```text
118 to 128 dispatched tasks
```

Approximately 384 to 394 candidate tasks were pruned.

That means roughly three quarters of the candidate distributed work was avoided.

The relative numerical error remained approximately:

```text
5 x 10^-16
```

which is consistent with ordinary floating-point accumulation differences.

Because exact-zero pruning does not approximate the matrix product, the runtime improvement was achieved without meaningful loss of numerical correctness.

---

## 7. What Have We Learned So Far?

The experiments currently support several conclusions.

### 1. Distribution Has a Significant Fixed Cost

The distributed implementation is still slower than optimized local NumPy multiplication at the tested matrix sizes.

That is expected because NumPy uses highly optimized native linear algebra libraries while the distributed implementation includes Docker, TCP, serialization, scheduling, and aggregation.

The goal of this project is not to claim that the microservice system replaces optimized BLAS.

The goal is to understand when distributed parallelism and adaptive task elimination overcome their own systems overhead.

### 2. Communication Granularity Matters

The large improvement from task batching demonstrated that protocol design can dominate total runtime.

Sending many small messages can destroy the benefit of parallel computation.

### 3. More Workers Do Not Guarantee Better Performance

The system improved when workers were added up to a point.

After that point, contention and coordination costs caused performance to flatten or regress.

### 4. Adaptive Task Selection Can Matter More Than Adding Workers

This is currently the strongest experimental result.

For block-sparse matrices, exact-zero pruning reduced distributed runtime by approximately 40% to 59%.

After pruning, runtime was almost flat across one, two, and four workers:

```text
1 worker: 3.460 s
2 workers: 3.492 s
4 workers: 3.386 s
```

This suggests that eliminating unnecessary distributed work can provide a larger benefit than simply increasing worker count.

### 5. The Adaptive System Preserved Correctness

Exact-zero pruning removed tasks that mathematically contribute zero to the final result.

The final results continued to match the serial reference within floating-point tolerance.

---

## 8. Current Answer to the Research Question

The current evidence suggests:

> Distributed blocked matrix multiplication benefits from parallel workers only while the useful computation added by those workers exceeds communication, scheduling, aggregation, and host-resource overhead.

The experiments also show:

> When the matrices contain exploitable block sparsity, identifying and removing unnecessary block products before dispatch can substantially reduce distributed runtime.

At the current test scale, intelligent task elimination has produced a more consistent performance benefit than simply adding more worker containers.

This is the central result around which the final paper can be organized.

---

## 9. What Still Remains?

The core implementation is substantially complete.

The remaining work is primarily experimental validation, analysis, documentation, and presentation.

Planned remaining work includes:

- repeat selected final experiments under controlled conditions
- decide whether additional matrix sizes are necessary
- evaluate block-size sensitivity
- improve instrumentation if needed for communication and compute breakdowns
- evaluate approximate Frobenius-bound pruning only if it adds useful evidence
- decide whether low-rank TCP transmission is worth the remaining development time
- generate final figures and tables
- collect and verify at least 10 research references
- write the final 8-page IEEE or ACM graduate paper
- prepare the graduate presentation
- prepare a live or recorded demonstration

The low-rank extension remains optional and should not delay completion of the validated sparsity-aware system.

---

## 10. Checkpoint 2 Short Verbal Explanation

A concise way to explain the project during the checkpoint is:

> We built the required distributed matrix multiplication system using separate Docker microservices for scheduling, computation, and aggregation, with TCP communication between them. The matrices are divided into blocks and the block products are distributed to worker containers. Our main research question is whether the benefit of parallel processing is enough to overcome the communication and coordination overhead of distributing the work. We also added an adaptive scheduler that can detect block multiplications that are guaranteed to produce zero and skip them before they are sent over the network. During implementation, we found that network granularity mattered a lot, so we added task batching, and we also controlled NumPy's internal threading so worker-count experiments were more meaningful. Our current results show that adding workers helps only up to a point, but on block-sparse matrices the adaptive scheduler eliminated roughly three quarters of the candidate tasks and reduced distributed runtime by about 40 to 59 percent without changing the numerical result. The remaining work is final experimentation, figures, references, the IEEE/ACM paper, and the graduate presentation and demo.

---

## 11. Current Project Status

```text
Required distributed architecture        COMPLETE
TCP microservice communication           COMPLETE
Blocked matrix decomposition             COMPLETE
Multi-worker execution                   COMPLETE
Aggregation                              COMPLETE
Correctness validation                   COMPLETE
Task batching                            COMPLETE
Controlled worker threading              COMPLETE
Exact-zero adaptive pruning              COMPLETE
Repeated dense scaling experiments       COMPLETE
Repeated sparse adaptive experiments     COMPLETE
Approximate pruning experiments          OPTIONAL / PENDING
Low-rank network transmission            OPTIONAL / NOT INTEGRATED
Final experimental campaign              PENDING
Final figures and analysis               PENDING
10-reference research set                PENDING
8-page IEEE / ACM paper                  PENDING
Presentation and demo                    PENDING
```

The project has moved from implementation into the final experimental and reporting phase.
