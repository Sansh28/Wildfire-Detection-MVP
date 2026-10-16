# Wildfire Detection MVP Spec

## Objective

Build a fast, credible wildfire detection MVP that predicts whether a short multispectral sequence contains an active wildfire signal.

## Problem Definition

- Task: binary sequence classification
- Input: `T x C x H x W`
- MVP defaults:
  - `T=3`
  - `C=4` (`RGB + NIR`)
  - `H=W=128`
- Output: probability of `fire`

## Data Strategy

- Positive anchors: FIRMS VIIRS 375 m active fire detections with `nominal` or `high` confidence
- Negative anchors:
  - nearby non-fire Sentinel-2 patches
  - hard negatives from bright or thermally confusing surfaces
- Regions:
  - train/val: western US
  - in-distribution test: held-out western US events
  - OOD test: Canada or Alaska

## MVP Pipeline

1. Ingest FIRMS events
2. Cluster nearby detections into event groups
3. Generate event-level train/val/test splits
4. Retrieve Sentinel-2 imagery for `t-10d`, `t-5d`, `t0`
5. Filter cloudy scenes
6. Save patch tensors and sequence metadata
7. Train static baseline
8. Train temporal model
9. Evaluate uncertainty on OOD split

## Models

- Static baseline:
  - ResNet-18 encoder
  - uses `t0` only
- Temporal model:
  - shared frame encoder
  - temporal attention over frame embeddings
  - dropout enabled for MC dropout inference

## Primary Metrics

- PR-AUC
- F1
- Recall
- Confusion matrix
- ECE
- reliability bins
- variance-error correlation

## Acceptance Criteria

- event-level leakage is prevented
- full pipeline works on one GPU
- temporal model matches or beats static baseline on PR-AUC and recall
- uncertainty is meaningfully larger on mistakes than correct predictions
