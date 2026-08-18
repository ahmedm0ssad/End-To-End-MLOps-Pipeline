# End-To-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-945DD6?style=flat&logo=dvc&logoColor=white)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=flat&logo=githubactions&logoColor=white)](https://github.com/features/actions)

A compact, reproducible MLOps reference implementation built around the **Iris classification** problem. It demonstrates how to move a machine-learning model from a local notebook to a versioned, tracked, containerized, and CI/CD-driven pipeline.

> **Maturity:** production-oriented foundations (data versioning, experiment tracking, CI/CD, containerization) with a clear roadmap for serving and monitoring.

## Highlights

- **Data versioning** with [DVC](https://dvc.org) + DagsHub S3-compatible remote — datasets are fetched deterministically in CI.
- **Experiment tracking** with [MLflow](https://mlflow.org) — every run logs parameters, metrics, tags, the trained model, and a confusion-matrix artifact.
- **CI/CD** with [GitHub Actions](https://github.com/features/actions) — lint → train → deploy → auto-merge, gated to save compute.
- **Containerization** with Docker — a conda-based training image plus a `docker-compose` stack for local MLflow + training.
- **Failsafe design** — error logs are captured and uploaded as artifacts whenever training fails.

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Local Environment](#1-local-environment)
  - [Local Training](#2-local-training)
  - [Docker / MLflow](#3-docker--mlflow-compose)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration & Secrets](#configuration--secrets)
- [Model Lifecycle](#model-lifecycle)
- [Roadmap](#roadmap)
- [License](#license)

---

## How It Works

```
 Data              DVC            Training          Tracking          Delivery
┌─────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
│ sklearn │ → │ dvc add /  │ → │ RandomForest │ → │ MLflow /   │ → │ Docker image │
│ load_   │   │ dvc pull   │   │ Classifier   │   │ DagsHub UI │   │ Docker Hub   │
│ iris    │   │ (remote)   │   │ src/train.py │   │ + artifacts│   │ CI deploy    │
└─────────┘   └────────────┘   └──────────────┘   └────────────┘   └──────────────┘
```

1. **Data ingestion** — `src/save_iris.py` generates `data/iris.csv` from `sklearn.datasets.load_iris`.
2. **Data versioning** — DVC tracks `data/iris.csv` (pointer file `data/iris.csv.dvc`) against a DagsHub remote; CI retrieves it with `dvc pull`.
3. **Training** — `src/train.py` trains a `RandomForestClassifier` on a train/test split.
4. **Evaluation** — computes `accuracy` and saves a confusion matrix to `outputs/confusion_matrix.png`.
5. **Tracking** — MLflow records params, metrics, tags, the model artifact, and the plot.
6. **Delivery** — the CI pipeline builds and pushes the `iris-trainer` Docker image to Docker Hub.

## Tech Stack

| Category | Tooling | Status |
| --- | --- | --- |
| Orchestration | GitHub Actions (`.github/workflows/pipeline.yml`) | Implemented |
| Experiment Tracking | MLflow (local + DagsHub tracking URI) | Implemented |
| Model Registry | MLflow artifacts + `src/get_best_model.py` | Partial |
| Data Versioning | DVC + DagsHub remote | Implemented |
| Containerization | Dockerfile + docker-compose | Implemented |
| CI/CD | GitHub Actions (lint → train → deploy → merge) | Implemented |
| Serving | None (no FastAPI/BentoML app yet) | Planned |
| Monitoring | MLflow history only | Partial |

**Key dependencies:** Python 3.10, `scikit-learn==1.4.0`, `pandas==2.1.0`, `matplotlib==3.8.2`, `mlflow==2.10.0`, `dvc==3.50.0`, `dvc-s3==3.2.0`, `pathspec==0.11.2` (pinned to avoid a DVC import issue).

## Project Structure

```
├── .github/workflows/pipeline.yml   # CI/CD: lint → train → deploy → merge
├── .dvc/                            # DVC config, cache, remote linkage
├── data/
│   └── iris.csv.dvc                 # DVC pointer to the versioned dataset
├── src/
│   ├── save_iris.py                 # one-time dataset generation
│   ├── train.py                     # training + MLflow logging
│   └── get_best_model.py            # retrieve best run's model URI by accuracy
├── outputs/                         # evaluation artifacts (e.g. confusion matrix)
├── mlruns/                          # MLflow run metadata (local/compose)
├── mlartifacts/                     # stored model artifacts
├── dockerfile                       # conda-based training image
├── docker-compose.yml               # local MLflow server + trainer stack
├── requirements.txt                 # pip dependencies
├── environment.yml                  # conda env (mlops-dev)
├── CONTEXT.md                       # deep technical context & ADRs
└── .dockerignore / .dvcignore / .gitignore
```

## Getting Started

### 1. Local Environment

Conda (recommended — mirrors the Docker image):

```bash
conda env create -f environment.yml
conda activate mlops-dev
```

Or pip:

```bash
pip install -r requirements.txt
pip install "dvc==3.50.0" "dvc-s3==3.2.0" "pathspec==0.11.2"
```

### 2. Local Training

```bash
# 1. Generate the dataset (one-time)
python src/save_iris.py

# 2. Version it with DVC (creates data/iris.csv.dvc)
dvc add data/iris.csv

# 3. Pull data from the remote (or after adding a fresh clone)
dvc pull

# 4. Train + track with MLflow
python src/train.py
```

By default the run is logged locally; set `MLFLOW_TRACKING_URI` to a DagsHub remote to centralize tracking:

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
```

### 3. Docker / MLflow (compose)

```bash
docker compose up --build
```

- `train` — runs `src/train.py` and sends logs to the MLflow container.
- `mlflow` — official MLflow server at `http://localhost:5000`.
- Both share the `./mlruns` volume on the `mlops-network` bridge.

## CI/CD Pipeline

`.github/workflows/pipeline.yml` — four gated jobs:

| Job | Trigger | Purpose |
| --- | --- | --- |
| `linter` | every push / PR | flake8 gate on `src/` (syntax + undefined-name checks) |
| `train` | push to `main` **with** `[run-train]` in the commit message | pulls data via DVC, trains, logs to MLflow, uploads artifacts |
| `deploy` | after `train` | builds & pushes `iris-trainer:latest` to Docker Hub |
| `merge` | after `deploy` | force-syncs `main` from `develop` (linear promotion) |

The `train` job is **triple-gated** (linter passes + `main` branch + `[run-train]` keyword) so expensive compute is never wasted. On failure, a structured `error_logs.txt` is uploaded as a GitHub Actions artifact.

### Required Secrets

| Secret | Purpose |
| --- | --- |
| `DAGSHUB_USERNAME` / `DAGSHUB_TOKEN` | DVC remote auth + MLflow tracking |
| `MLFLOW_TRACKING_URI` | remote experiment tracking endpoint |
| `DOCKER_USERNAME` / `DOCKER_PASSWORD` | Docker Hub image push |

> **Security:** credentials are always passed via GitHub secrets or environment variables — never committed to the repository.

## Model Lifecycle

1. **Train** — `src/train.py` fits `RandomForestClassifier`.
2. **Evaluate** — `accuracy` metric + confusion matrix artifact.
3. **Register** — model logged with `mlflow.sklearn.log_model(model, "model")`; run URI written to `model_info.txt`.
4. **Select** — `src/get_best_model.py` identifies the best run by `accuracy`.
5. **Deploy** — CI pushes the container image (no online serving endpoint yet).

## Roadmap

- [ ] Add unit/integration tests with metric-threshold gates.
- [ ] Introduce immutable Docker image tags (per commit/release).
- [ ] Add an online serving layer (FastAPI or equivalent) with health checks.
- [ ] Add formal MLflow registry stages (Staging / Production).
- [ ] Integrate monitoring for model/data drift and service telemetry.
- [ ] Add a feature store and reusable feature pipeline.

## License

This project is provided for educational and reference purposes. See `CONTEXT.md` for detailed architecture notes and architecture decision records (ADRs).