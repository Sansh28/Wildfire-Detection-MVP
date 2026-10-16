# Wildfire Detection MVP

> Individual coursework project · University of Colorado Boulder

A reproducible machine-learning pipeline for detecting wildfire signals from short Sentinel-2 multispectral sequences. The project compares a static ResNet-18 baseline with a temporal-attention model and records calibration, uncertainty, and threshold-policy behavior.

## What is included

- FIRMS event ingestion, clustering, event-level splitting, and hard-negative metadata generation
- Sentinel-2 extraction hooks through Google Earth Engine
- Static and temporal PyTorch models for `fire` / `not_fire` classification
- PR-AUC, F1, recall, specificity, calibration, MC-dropout uncertainty, and reproducibility manifests
- Unit tests, configs, a project specification, a final report, and one representative reliability diagram

## Key result

On the tracked fast-subset experiments, temporal context was often beneficial but not uniformly dominant. The most actionable finding was that unconstrained validation thresholds could preserve recall while collapsing specificity; bounded and recall-floor threshold policies produced more credible operating points. See [the final report](docs/final_project_report.md) for results and limitations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

The full imagery pipeline requires a NASA FIRMS API key and Google Earth Engine access. Set credentials only as environment variables or local configuration; this repository intentionally excludes credentials, raw data, trained weights, and generated run logs.

```bash
PYTHONPATH=src python -m wildfire_mvp.cli init-layout --config configs/default.toml
PYTHONPATH=src python -m wildfire_mvp.train --config configs/default.toml --model static
PYTHONPATH=src python -m wildfire_mvp.train --config configs/default.toml --model temporal
PYTHONPATH=src python -m wildfire_mvp.evaluate --config configs/default.toml --model temporal
```

## Architecture

`FIRMS anchors → event-level split → Sentinel-2 patches → static/temporal model → threshold-aware evaluation → reproducibility report`

Event-level splitting prevents samples from the same fire event leaking across train and test data. The temporal model uses a shared frame encoder followed by multi-head attention across three date-aligned frames. Evaluation reports uncertainty and calibration alongside classification metrics.

## Scope and responsible use

This is a research MVP, not an operational wildfire-alerting system. Its fast-subset results need larger geographic and temporal validation before any real-world use. Read [PROJECT_SPEC.md](PROJECT_SPEC.md) and the final report before reusing the approach.

## License

Coursework reference implementation. Contact the author before commercial or operational use.
