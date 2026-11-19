# Local Run Guide

## Prerequisites

Install:

- Python 3.12 or newer
- Docker Desktop
- Git
- VS Code

Confirm from the VS Code terminal:

```powershell
python --version
docker --version
docker compose version
git --version
```

## 1. Pull the latest repository

```powershell
git pull origin main
```

If cloning for the first time:

```powershell
git clone https://github.com/matthew-richardson-ai/adaptive-distributed-matrix-multiplication.git
cd adaptive-distributed-matrix-multiplication
```

## 2. Create a local Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 3. Run unit tests first

```powershell
python -m pytest -q
```

Do not move to large benchmarks until the test suite passes.

## 4. Build and start the distributed system

Start the Aggregator, Ingress, and two workers:

```powershell
docker compose up -d --build --scale worker=2
```

Check service state:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs -f
```

Press Ctrl+C to stop following logs. The containers continue running.

## 5. Run the first dense end-to-end test

Open another VS Code terminal with the Python environment activated:

```powershell
python -m Benchmarks.run_benchmarks --size 100 --block-size 32 --workers 2
```

A successful exact-mode run should:

- create a serial reference
- submit the matrices to Ingress
- distribute block tasks to workers
- return partial products to the Aggregator
- reconstruct the final matrix
- validate the result against the serial product
- write a benchmark record under `Results/`

## 6. Run the adaptive exact-zero test

```powershell
python -m Benchmarks.run_benchmarks \
  --size 100 \
  --block-size 32 \
  --workers 2 \
  --matrix-type block_sparse \
  --zero-block-fraction 0.5 \
  --adaptive \
  --epsilon 0
```

In PowerShell, if line continuation is inconvenient, run it on one line:

```powershell
python -m Benchmarks.run_benchmarks --size 100 --block-size 32 --workers 2 --matrix-type block_sparse --zero-block-fraction 0.5 --adaptive --epsilon 0
```

The output should show a nonzero number of pruned tasks for a sufficiently sparse generated input.

## 7. Change worker count

```powershell
docker compose up -d --scale worker=1
docker compose up -d --scale worker=2
docker compose up -d --scale worker=4
docker compose up -d --scale worker=8
```

Use the matching `--workers` value when running benchmarks so the stored experiment metadata is correct.

## 8. Run the development benchmark suite

The development configuration uses small cases:

```powershell
python -m Benchmarks.run_suite --config Benchmarks/Configs/dev.json --manage-docker
```

This changes worker scale automatically and records all runs in:

```text
Results/benchmark_results.json
```

## 9. Run the final benchmark suite

Only do this after the development suite is stable:

```powershell
python -m Benchmarks.run_suite --config Benchmarks/Configs/final.json --manage-docker
```

The final configuration is intentionally much larger and may need to be reduced based on local CPU, RAM, Docker limits, and available time.

## 10. Stop the environment

```powershell
docker compose down
```

To also remove anonymous volumes:

```powershell
docker compose down -v
```

## Troubleshooting

### Port already in use

The default ports are:

- Ingress: 9001
- Aggregator: 9002

Check for an existing process or old Docker container using those ports.

### Worker repeatedly reconnects

Run:

```powershell
docker compose logs ingress aggregator worker
```

The worker starts by connecting to Ingress and Aggregator. During startup it may briefly retry while those services become available.

### Benchmark times out

Start with a smaller matrix and confirm:

```powershell
docker compose ps
docker compose logs
```

Then rerun:

```powershell
python -m Benchmarks.run_benchmarks --size 32 --block-size 16 --workers 2 --timeout 60
```

### Exact result mismatch

Stop the benchmark campaign and preserve the logs. Exact modes should match the serial result within the configured NumPy tolerance. Do not use performance results from a run that fails correctness validation.

## Recommended first validation sequence

Run these in order:

1. `python -m pytest -q`
2. `docker compose up -d --build --scale worker=2`
3. `python -m Benchmarks.run_benchmarks --size 32 --block-size 16 --workers 2`
4. `python -m Benchmarks.run_benchmarks --size 100 --block-size 32 --workers 2`
5. adaptive block-sparse run
6. development benchmark suite

Once those pass, increase matrix sizes and worker counts.


## Batched protocol optimization

The worker protocol now requests multiple block tasks at once and returns a batch of partial results in one message. The default batch size is 8 and can be changed before starting Compose:

```powershell
$env:TASK_BATCH_SIZE="16"
docker compose up -d --build --scale worker=2
```

After pulling protocol changes, rebuild the images before benchmarking:

```powershell
git pull origin main
docker compose down
docker compose up -d --build --scale worker=2
```

Use the same benchmark commands after the containers report healthy startup logs.
