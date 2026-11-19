# Adaptive Distributed Matrix Multiplication

Graduate CSC 5770 project implementing blocked matrix multiplication as a TCP-connected Docker microservice system, then evaluating whether structure-aware task selection can reduce distributed overhead.

## Current Project Status

The core distributed baseline is implemented and has passed automated unit tests and Docker end-to-end smoke testing.

Implemented:

- serial reference and numerical validation
- blocked matrix decomposition and padding
- framed TCP protocol
- Ingress / scheduler service
- scalable Docker workers
- aggregation service
- multi-worker pull scheduling
- task lease / retry support
- exact-zero and Frobenius-bound adaptive pruning
- deterministic benchmark generators
- benchmark recording and suite configuration
- Docker Compose environment
- automated unit and end-to-end tests

Current priority:

1. reproduce the smoke tests locally
2. run the development benchmark suite
3. collect worker-count and block-size comparisons
4. evaluate adaptive pruning on sparse inputs
5. perform repeated final trials and analyze results
6. integrate low-rank transmission only if the measured results justify the extra scope

## Target Architecture

```text
Client
  |
  v
Ingress / Decomposition / Scheduler
  |
  v
Dynamic Task Queue
  |
  +--> Worker 1
  +--> Worker 2
  +--> Worker N
          |
          v
Aggregation Service
  |
  v
Final Matrix
```

The adaptive version inserts a decision step before task dispatch. The baseline version sends all required block products.

## Required Course Features

- separate Dockerized microservice roles
- TCP socket communication
- blocked / divide-and-conquer matrix multiplication
- independently computed submatrix products
- dedicated aggregation service
- horizontal worker scaling
- intelligent workload distribution
- clear failure reporting

## Documentation

- [Checkpoint 1](Docs/CheckPoint_1.md)
- [Checkpoint 2 Working Draft](Docs/CheckPoint_2.md)
- [Implementation Plan](Docs/Implementation_Plan.md)
- [TCP Protocol Specification](Docs/Protocol_Spec.md)
- [Testing and Benchmark Plan](Docs/Testing_and_Benchmarking.md)
- [Local Run Guide](Docs/Local_Run_Guide.md)
- [Research Roadmap](Docs/Roadmap_V3.md)

## Definition of the Core Baseline

The baseline is complete when two input matrices can travel through the full path

```text
Ingress -> TCP -> Docker Worker(s) -> Aggregator -> Final Matrix
```

and the reconstructed result matches the serial reference within the selected numerical tolerance.

Adaptive pruning and low-rank processing are not prerequisites for baseline correctness.


## Quick Start

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
docker compose up -d --build --scale worker=2
python -m Benchmarks.run_benchmarks --size 32 --block-size 16 --workers 2
```

For the full VS Code / Docker workflow, see [Docs/Local_Run_Guide.md](Docs/Local_Run_Guide.md).
