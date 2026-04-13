# CONTEXT.md

## 1) Project Overview

This repository is a compact, reproducible MLOps project built around the Iris classification problem.

Core goals:

- Train and evaluate a baseline model (`RandomForestClassifier`) from a versioned dataset.
- Track experiments, metrics, artifacts, and model outputs in MLflow.
- Version training data with DVC and fetch it in CI.
- Automate validation and container image publishing through GitHub Actions.

Current maturity level:

- Implemented: data versioning, experiment tracking, CI/CD, containerization.
- Partial: model lifecycle automation (best-model script exists, no formal registry stage gate).
- Not yet implemented: online serving API, production monitoring stack, workflow orchestrator.

## 2) Architecture

High-level architecture (current state):

1. Data ingestion

- Source data is generated from `sklearn.datasets.load_iris` via `src/save_iris.py` and persisted as `data/iris.csv`.
- DVC tracks `data/iris.csv` via `data/iris.csv.dvc` and remote storage.

2. Feature engineering

- Minimal transformation only: split features/target from CSV columns.
- No separate feature store or reusable feature pipeline yet.

3. Model training

- `src/train.py` trains a Random Forest on train/test split.
- Training logs params, metrics, tags, model artifact, and confusion matrix into MLflow.

4. Evaluation

- Primary metric: `accuracy`.
- Artifact: confusion matrix image in `outputs/confusion_matrix.png`.

5. Deployment

- CI builds and pushes Docker image (`iris-trainer:latest`) to Docker Hub.
- No model serving endpoint is deployed yet.

6. Monitoring

- Offline monitoring only through MLflow experiment history.
- No live model/data drift or service telemetry stack yet.

## 3) Tech Stack

Status legend:

- Implemented: actively used in code or CI.
- Partial: foundational pieces exist but not full production workflow.
- Planned/Not present: not found in this codebase.

| Category                         | Tooling in This Repo                                       | Status              | Notes                                                               |
| -------------------------------- | ---------------------------------------------------------- | ------------------- | ------------------------------------------------------------------- |
| Orchestration                    | GitHub Actions workflow (`.github/workflows/pipeline.yml`) | Implemented         | Pipeline orchestrates CI validate -> deploy -> merge jobs.          |
| Experiment Tracking              | MLflow (local + DagsHub tracking URI)                      | Implemented         | `src/train.py` logs metrics, params, tags, artifacts, model.        |
| Model Registry                   | MLflow model artifacts + `src/get_best_model.py`           | Partial             | Best-run retrieval exists, but no formal staged registry promotion. |
| Data Versioning                  | DVC (`data/iris.csv.dvc`, `.dvc/config`)                   | Implemented         | Data pulled in CI from configured DVC remote.                       |
| Serving                          | None detected (no FastAPI/BentoML/Seldon app)              | Planned/Not present | Deployment currently builds trainer image only.                     |
| Infrastructure                   | Dockerfile, docker-compose                                 | Implemented         | Local two-service setup (`train` + `mlflow`).                       |
| Cloud Provider / Hosted Services | DagsHub, Docker Hub, GitHub-hosted runners                 | Implemented         | No direct AWS/GCP/Azure IaC in repo.                                |
| CI/CD                            | GitHub Actions                                             | Implemented         | Trigger on push to `develop`, PR to `main`.                         |
| Monitoring                       | MLflow UI/history only                                     | Partial             | No Prometheus/Grafana/Evidently integration yet.                    |

## 4) Project Structure

Top-level layout and intent:

- `src/`
  - Core Python scripts.
  - `train.py`: training + MLflow logging.
  - `save_iris.py`: one-time data creation script.
  - `get_best_model.py`: retrieves top run model URI.

- `data/`
  - Dataset pointer and local data file.
  - `iris.csv.dvc` is the tracked pointer; `iris.csv` is fetched/generated.

- `.dvc/`
  - DVC configuration, cache, remote linkage.

- `.github/workflows/`
  - CI/CD workflow (`pipeline.yml`).

- `mlruns/`
  - MLflow run metadata and artifacts for local/containerized tracking.

- `mlartifacts/`
  - Stored model artifact directories from MLflow runs.

- `outputs/`
  - Generated evaluation outputs (for example confusion matrix image).

- `Dockerfile`
  - Conda-based training image build.

- `docker-compose.yml`
  - Local service composition for trainer and MLflow server.

- `requirements.txt`, `environment.yml`
  - Python dependency and environment definitions.

- `docs/`
  - Supporting project documents (currently lecture material).

## 5) Data Pipeline

Current data flow:

1. Generate initial dataset locally:

- `python src/save_iris.py` writes `data/iris.csv`.

2. Track dataset with DVC:

- `dvc add data/iris.csv` creates/updates `data/iris.csv.dvc`.
- DVC remote points to DagsHub storage.

3. CI data retrieval:

- GitHub Actions sets DVC remote auth using secrets.
- `dvc pull` downloads `data/iris.csv` before training.

4. Training consumption:

- `src/train.py` reads CSV path from `DATA_PATH` (default `data/iris.csv`).

Feature store note:

- No dedicated feature store is implemented.
- Features are computed inline in training code from raw CSV columns.

## 6) Model Lifecycle

Observed lifecycle in this repo:

1. Training

- `src/train.py` fits `RandomForestClassifier`.

2. Evaluation

- Computes `accuracy` and confusion matrix.

3. Registration (current approximation)

- Model artifact logged with `mlflow.sklearn.log_model(model, "model")`.
- URI written to `model_info.txt`.
- `src/get_best_model.py` can identify best run by `accuracy`.

4. Deployment

- CI deploy job builds and pushes Docker image.
- No automated promotion to a model serving endpoint.

Recommended future evolution:

- Introduce explicit model stage transitions (Staging/Production).
- Add deployment target (API or batch inference service) and health checks.

## 7) Environment Setup

Environment sources:

- Conda environment in `environment.yml` (name: `mlops-dev`).
- Pip requirements in `requirements.txt`.

Python version:

- Python 3.10 (both conda and GitHub Actions workflow).

Key dependencies (observed):

- `scikit-learn==1.4.0`
- `pandas==2.1.0`
- `matplotlib==3.8.2`
- `mlflow==2.10.0`
- `dvc==3.50.0`
- `dvc-s3==3.2.0`
- `pathspec==0.11.2` (pin used to avoid DVC/pathspec compatibility issue)

Container runtime:

- Base image: `continuumio/miniconda3`.
- Default container command runs training script.

## 8) Configuration Management

Current approach:

- Environment variables for runtime configuration and secrets.
  - Examples: `MLFLOW_TRACKING_URI`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `DATA_PATH`.
- GitHub Actions secrets for CI credentials.
- DVC remote configuration in `.dvc/config` and local overrides in `.dvc/config.local`.

Important security convention:

- Keep secrets only in environment variables or local non-versioned files.
- Never commit credentials in repository-tracked files.

What is not present:

- No Hydra/OmegaConf-based hierarchical config system.
- No centralized secret manager integration yet.

## 9) Key Conventions

Code and file conventions (observed):

- Source code under `src/`.
- Data artifacts under `data/`, model/eval outputs under `outputs/` and MLflow directories.
- Training run metadata tracked through MLflow tags/params/metrics.

Git branching and automation:

- Workflow triggers on:
  - Push to `develop`.
  - Pull requests targeting `main`.
- Workflow includes automated merge step that force-updates `main` from `develop` after successful deploy.

Versioning policy (effective):

- Data versioning: DVC pointer + remote storage.
- Model versioning: MLflow run IDs and artifacts.
- Container versioning: currently `latest` tag (no immutable semver tags yet).

## 10) Team & Roles

Current visible ownership signals:

- Run tags include `developer=Mossad`.
- Repository appears maintained by a small team (possibly single primary maintainer).

Suggested role ownership map for scaling:

- Data Engineer
  - Own data ingestion quality checks, DVC pipelines, data contracts.
- ML Engineer
  - Own feature logic, training/evaluation scripts, experiment quality.
- MLOps/DevOps Engineer
  - Own CI/CD, image lifecycle, deployment automation, runtime observability.
- Product/Domain Owner
  - Own acceptance criteria, KPI thresholds, release decisions.

## 11) Known Issues & Decisions (ADRs)

ADR-001: Use DVC + DagsHub for dataset reproducibility

- Decision: Track dataset through DVC pointer files and remote storage.
- Benefit: Reproducible data retrieval in CI.
- Trade-off: Requires credential management and DVC setup in pipelines.

ADR-002: Use MLflow as experiment and artifact system

- Decision: Log metrics, artifacts, and model via MLflow.
- Benefit: Centralized experiment lineage.
- Trade-off: No strict registry stage workflow yet.

ADR-003: Deploy container image, not model service

- Decision: CI builds/pushes training image to Docker Hub.
- Benefit: Reproducible runtime packaging.
- Trade-off: No online inference endpoint from this repo currently.

ADR-004: Use force-sync merge automation (`develop` -> `main`)

- Decision: Workflow merge job hard-resets `main` to `origin/develop` and force-pushes.
- Benefit: Simple linear promotion path.
- Trade-off: History rewrite risk and potential overwrite of direct `main` commits.

ADR-005: Pin `pathspec==0.11.2` with DVC

- Decision: Pin dependency to avoid known import compatibility issue.
- Benefit: Stable CI execution.
- Trade-off: Requires periodic review to remove pin when upstream is fixed.

Open issues to prioritize:

- Add tests (unit/integration) for training pipeline and metric thresholds.
- Introduce immutable Docker image tags per commit/release.
- Add serving layer (FastAPI or equivalent) and deployment health checks.
- Integrate monitoring for drift/service metrics.

## 12) Glossary

- CI/CD: Continuous Integration / Continuous Delivery.
- DVC: Data Version Control, tracks large data and model files via pointers.
- DagsHub: Hosted platform integrating Git, DVC remotes, and MLflow tracking.
- MLflow Tracking URI: Endpoint where runs, metrics, and artifacts are logged.
- Artifact: File output from a run (for example model binaries, plots).
- Run: One tracked execution of a training/evaluation job in MLflow.
- Experiment: Group of MLflow runs under a logical project context.
- Model URI: MLflow path identifying where a logged model artifact lives.
- Drift: Statistical shift between training data and production data distributions.

## LLM/Copilot Quick Start Notes

When an AI assistant works on this repo, use this checklist first:

1. Read `src/train.py` to understand the ground-truth training flow and logging contract.
2. Treat DVC as the data source of truth; do not assume `data/iris.csv` is committed.
3. Preserve environment-variable based secrets handling; never hardcode credentials.
4. Keep MLflow parameter/metric/tag names stable unless migration is intentional.
5. If modifying CI, verify compatibility with branch flow (`develop` promotion model).
6. If adding deployment, define clear interface between model artifact selection and serving runtime.
