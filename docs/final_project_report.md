# Wildfire Detection from Short Multispectral Sequences: A Reproducible MVP with Temporal Modeling and Uncertainty Analysis

## Abstract
Wildfire detection from satellite imagery is operationally important but methodologically challenging because fire signatures are sparse, cloud-affected, and often confounded by bright non-fire surfaces. This project develops and evaluates a reproducible end-to-end machine learning pipeline for binary wildfire detection using short Sentinel-2 multispectral sequences centered on FIRMS active-fire anchors. The pipeline includes FIRMS ingestion, event clustering, event-level split construction, hard-negative generation, Earth Engine extraction of aligned three-frame patches, model training, threshold-aware evaluation, uncertainty diagnostics, and automated report aggregation. Two model families are compared: a static ResNet-18 baseline using the most recent frame and a temporal attention model over three frames. Experiments across in-distribution, out-of-distribution, and incremental fast-subset settings show that temporal modeling is competitive and often favorable, but not uniformly dominant on all metrics across all increments. A key empirical finding is that unconstrained validation-selected thresholds can collapse specificity under corrected split conditions; bounded and recall-floor threshold policies produce more actionable operating points. The result is a complete MVP with reproducibility manifests, calibrated reporting artifacts, and a clear path to stronger large-scale validation.

## 1. Introduction
Wildfires are increasing in frequency, intensity, and socioeconomic impact, creating a practical need for robust, scalable early detection systems. Satellite-based remote sensing is especially attractive because it offers broad geographic coverage and repeated revisits, but wildfire detection from satellite imagery remains difficult in real-world conditions. Several factors contribute to this difficulty: clouds and haze can obscure signal, transient illumination artifacts can mimic fire-like responses, and high-variance background land cover can reduce classifier specificity if datasets are not constructed carefully.

This project addresses that problem by designing and implementing a compact but complete wildfire detection machine learning pipeline that is executable on commodity research hardware. The pipeline focuses on binary classification (fire vs. not-fire) from short multispectral sequences and explicitly tests whether temporal context improves performance relative to a static single-frame baseline. It also evaluates prediction uncertainty, because operational settings require not only high discrimination but also trustworthy confidence signals.

The core hypothesis is that temporal context should improve detection quality for at least part of the data regime by distinguishing persistent fire-relevant evolution from single-frame confounders. A second hypothesis is that uncertainty measures from Monte Carlo dropout should correlate with model errors, especially on more difficult settings. A third practical hypothesis is that threshold policy design is not a cosmetic detail: threshold selection can materially alter recall-specificity tradeoffs and therefore deployment utility.

The work contributes a reproducible, code-complete MVP with:
- event-level split control to reduce leakage risk,
- deterministic hard-negative generation,
- configurable Earth Engine extraction,
- static and temporal neural architectures,
- uncertainty and calibration diagnostics,
- operating-point sweep support,
- manifest-based reproducibility and experiment aggregation.

The project is intentionally scoped as an MVP rather than a final production model. Its purpose is to establish a credible baseline system, quantify trends, and identify where additional research investment is likely to produce meaningful gains.

## 2. Related Work
Wildfire detection and monitoring from satellites is historically tied to thermal anomaly products such as NASA FIRMS, which provide high-value active-fire anchors at broad scale. FIRMS detections are often used as labels or weak supervision targets for downstream modeling because they provide practical fire-event localization at global cadence. In this project, FIRMS is used as the positive-anchor source for event creation, consistent with this common remote-sensing strategy.

From a computer vision standpoint, deep convolutional neural networks remain a strong baseline for image-level remote sensing tasks, and residual architectures such as ResNet are frequently used due to stable optimization and transfer-friendly inductive bias. The static baseline in this project adopts this tradition directly via a ResNet-18 backbone adapted for multispectral input.

Temporal modeling in Earth observation has progressed from recurrent formulations to attention-based designs that can capture cross-time dependencies with fewer recurrence constraints. Transformer-style and attention pooling approaches have shown advantages for sequence understanding in many domains, including remote sensing time series. The temporal model in this work is a lightweight instantiation of that idea: a shared frame encoder followed by multi-head temporal attention and residual normalization.

Calibration and uncertainty estimation are increasingly recognized as first-class requirements in safety-critical ML. Reliability diagrams and expected calibration error (ECE) are now standard diagnostics for confidence quality, while Monte Carlo dropout provides a practical approximate Bayesian uncertainty signal without requiring fully Bayesian neural training. This project follows that practical line by using MC dropout at inference for the temporal model and evaluating variance-error rank correlation as an uncertainty sanity check.

Finally, there is extensive evidence across application domains that threshold selection strongly shapes observed classification performance under imbalance and asymmetric cost constraints. The present project reinforces that point in a wildfire-specific context: unconstrained threshold selection can appear to optimize one metric while degrading operating specificity. Constrained threshold policies are therefore treated here as part of the modeling methodology, not post-hoc reporting.

## 3. Methods
### 3.1 Problem formulation
The task is binary classification over short multispectral patch sequences. Each sample consists of three time steps and four channels (RGB + NIR) at fixed spatial size 128 x 128. The model outputs a scalar probability of fire.

### 3.2 Data pipeline
The pipeline begins with FIRMS CSV ingestion and deduplication over date windows. Detections are clustered into events via a hashed bucket key derived from latitude/longitude/day quantization. For each event, metadata includes spatial coordinates, date, region, label, source, confidence, and split.

Split assignment is event-level and region-aware. OOD regions can be forced into `ood_test` while in-distribution regions are divided into train/val/test with deterministic seeding. This is a key anti-leakage design choice because all derived sequences from an event inherit the same split.

To improve decision-boundary quality, hard negatives are generated around positives by randomized distance/bearing offsets and temporal jitter. Candidate negatives must maintain minimum spatial separation from known positives in the same region. Hard negatives inherit the parent split, preserving split safety.

Sequence metadata is then created using exactly three frame offsets (default: -10, -5, 0 days). Each sequence points to a patch tensor identifier.

### 3.3 Sentinel-2 extraction
Sentinel-2 extraction uses Google Earth Engine. For each event and frame offset, imagery is selected within a configurable temporal search window. Scene selection applies cloud filtering and QA60-based clear-fraction checks with candidate fallback behavior. Patches are downloaded as GeoTIFF, resampled to fixed dimensions, normalized to reflectance-scale [0,1], stacked as sequence tensors, and saved to `.npy`.

### 3.4 Models
Two model classes are implemented.

Static baseline:
- ResNet-18 with adapted first convolution for four-channel input.
- Uses the final temporal frame only.
- Binary logit head with dropout.

Temporal model:
- Shared ResNet-18 frame encoder with identity terminal head.
- Linear projection to hidden dimension.
- Multi-head self-attention across the three temporal embeddings.
- Residual + LayerNorm + mean pooling + binary classifier head.

### 3.5 Training
Training uses BCEWithLogitsLoss with configurable positive class weighting and AdamW optimization. Early stopping is driven by validation PR-AUC, and best checkpoints are saved per model. Device selection supports CUDA, Apple MPS, or CPU.

### 3.6 Evaluation and thresholding
Evaluation computes PR-AUC, F1, precision, recall, specificity, confusion counts, ECE, reliability bins, and variance-error Spearman correlation. Temporal inference supports MC dropout with multiple stochastic passes to estimate predictive variance.

A crucial design element is threshold policy:
- `f1`: choose threshold on validation data maximizing F1.
- `recall_floor_specificity`: among thresholds meeting a target recall, choose highest specificity; otherwise fallback by highest recall then specificity/precision/F1.

Thresholds can also be bounded by min/max values to prevent pathological operating points.

### 3.7 Reproducibility and reporting
Each train/eval run writes a manifest including command log, config snapshot, git commit, environment versions, hardware accelerator state, timestamps, and output artifact paths. A dedicated aggregation script compiles all eval manifests into CSV/Markdown summaries for comparative analysis.

## 4. Experiments
### 4.1 Experimental setup and data regimes
The repository contains two major data regimes relevant to this report:

Full processed split summary:
- 878 total events/sequences.
- Train 307 positive + 307 negative.
- Val 65 positive + 65 negative.
- Test 67 positive + 67 negative.

Fast processed split summary:
- 480 total events/sequences.
- Train 168 positive + 168 negative.
- Val 36 positive + 36 negative.
- Test 36 positive + 36 negative.

Fast balanced run file used by subset run configs (`sequences_balanced.csv`):
- 420 total sequences.
- Train: 149 positive, 177 negative.
- Val: 32 positive, 15 negative.
- Test: 32 positive, 15 negative.

This shows that the balanced fast run is not perfectly class-balanced per split even though earlier pipeline summaries were balanced at upstream stages. That distinction matters when interpreting threshold behavior and metric sensitivity.

### 4.2 Aggregated evaluation outcomes
Across the tracked evaluation artifacts, the temporal model generally remains competitive and frequently superior on PR-AUC/F1 under fast-subset settings, but does not dominate universally.

From incremental comparisons:
- Increment 120:
  - Static: PR-AUC 0.9138, F1 0.7273, Recall 1.0000.
  - Temporal: PR-AUC 1.0000, F1 0.9333, Recall 0.8750.
- Increment 180:
  - Static: PR-AUC 1.0000, F1 0.7059, Recall 1.0000.
  - Temporal: PR-AUC 0.9933, F1 0.9600, Recall 1.0000.
- Increment 240:
  - Static: PR-AUC 0.9962, F1 0.8649, Recall 1.0000.
  - Temporal: PR-AUC 1.0000, F1 1.0000, Recall 1.0000.

Dominance summary from artifacts:
- PR-AUC winner by increment: temporal (120), static (180), temporal (240).
- Recall winner by increment: static (120), tie (180), tie (240).

Interpretation: temporal context appears helpful and often yields better F1-quality operating behavior, but the evidence does not support a strict claim that temporal is always better than static in PR-AUC.

### 4.3 OOD and uncertainty observations
OOD results indicate a larger gap between model families than in-distribution test runs:
- Static OOD: PR-AUC 0.7083, F1 0.0000, Recall 0.0000.
- Temporal OOD: PR-AUC 1.0000, F1 0.6667, Recall 1.0000.

Given likely sample-size limitations and OOD definition constraints, this should be interpreted cautiously but still suggests temporal features may transfer better under distribution shift in this project setup.

Uncertainty summary across runs:
- Static: mean ECE 0.1925, mean variance-error Spearman 0.2800, positive correlation in 4/6 runs.
- Temporal: mean ECE 0.1756, mean variance-error Spearman 0.3533, positive correlation in 3/5 runs.

These results support the claim that uncertainty signals are often directionally informative, but not consistently monotonic across all runs.

### 4.4 Corrected-split threshold pathology and operating-point control
One of the most important experimental findings is the threshold-selection pathology observed on corrected split evaluations.

Unbounded/collapsed-threshold behavior:
- Static corrected split example: threshold about 0.00066, recall 0.9688, specificity 0.0000.
- Temporal corrected split example: threshold about 0.00422, recall 0.9688, specificity 0.0000.

In both cases, near-zero thresholds produce near-all-positive predictions, artificially preserving recall while collapsing specificity.

After threshold bounding/recall-floor policy:
- Static bounded corrected split: threshold about 0.2143, recall 0.5625, specificity 0.7333.
- Temporal bounded corrected split: threshold about 0.2448, recall 0.7813, specificity 0.4000.

Submission-locked operating points further illustrate tradeoff management:
- Static submission-locked: threshold about 0.0520, precision 0.7500, recall 0.7500, specificity 0.4667, F1 0.7500.
- Temporal submission-locked: threshold about 0.0857, precision 0.7568, recall 0.8750, specificity 0.4000, F1 0.8116.

Operating-point sweep artifacts reinforce that static could not satisfy higher recall floors in bounded ranges while retaining strong specificity, whereas temporal offered better recall-oriented tradeoffs at low threshold minima.

### 4.5 Trends and likely explanations
General trends observed:
1. Both architectures can achieve very high in-distribution discrimination under fast subset conditions.
2. Temporal modeling tends to improve F1 and preserve high recall under more constrained threshold policies.
3. OOD behavior is far more sensitive and appears to favor temporal context in current runs.
4. Calibration remains imperfect for both models (ECE not negligible), especially in operating regions where thresholding changes decision balance.

Likely explanations:
- Temporal attention can exploit multi-date consistency unavailable to single-frame static encoding, improving resilience to one-off confounders.
- Small/curated fast subsets can produce optimistic in-distribution scores and unstable calibration.
- Validation-driven threshold optimization without constraints can overfit local score geometry, especially when class balance and score calibration differ across splits.
- The hard-negative construction strategy improves challenge level but may induce distribution structure different from truly natural negatives, affecting transfer and calibration.

### 4.6 Unanswered questions and natural next steps
Several key questions remain unresolved.

First, scale validity: current strongest claims come from fast-subset experiments. It is still unclear how performance, calibration, and threshold stability evolve on larger, more geographically and seasonally diverse corpora.

Second, uncertainty robustness: variance-error correlations are sometimes positive but not uniformly so. A stronger uncertainty study should test alternative estimators (ensembles, deep evidential methods, temperature scaling + conformal wrappers) and evaluate uncertainty under explicit perturbation/OOD stress tests.

Third, OOD definition breadth: current OOD appears region-based and limited in count. A more convincing OOD assessment should include multiple held-out geographies, years, and land-cover contexts.

Fourth, extraction completeness and quality control: the project status notes that full default extraction is not complete for every event, even with fallback tuning. Better scene-selection and cloud/quality mechanisms, plus explicit missingness accounting in training/evaluation, are needed.

Natural next steps are therefore:
- scale data and run confidence intervals via repeated seeds,
- expand OOD partitions and temporal windows,
- compare threshold policies under utility-weighted costs,
- add post-hoc calibration layers and measure decision impact,
- benchmark against stronger temporal baselines (e.g., temporal CNN/ViT variants).

## 5. Conclusions
The main takeaway is that the project successfully delivers a fully reproducible wildfire-detection MVP with end-to-end data, modeling, evaluation, and reporting infrastructure, and demonstrates that temporal modeling is often beneficial but not universally dominant across all metrics and increments. The most practically important insight is that threshold policy strongly determines deployment-relevant behavior: unconstrained validation thresholds can induce specificity collapse, while bounded and recall-floor strategies yield more credible operating points.

The project also raises important ethical and societal considerations. False negatives can delay response in high-risk fire scenarios, while false positives can waste limited emergency resources; both errors have real human and ecological cost. Model confidence outputs therefore need transparent communication and calibration auditing. Regional and temporal data coverage biases can create uneven performance across communities, which implicates fairness and accountability. Operational use would require documentation of failure modes, periodic monitoring for distribution shift, and clear human-in-the-loop governance so model outputs support rather than replace expert judgment.

## 6. Bibliography
[1] He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.

[2] Vaswani, A., Shazeer, N., Parmar, N., et al. Attention Is All You Need. NeurIPS, 2017.

[3] Gal, Y., and Ghahramani, Z. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML, 2016.

[4] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML, 2017.

[5] NASA FIRMS (Fire Information for Resource Management System), VIIRS active fire products. https://firms.modaps.eosdis.nasa.gov/

[6] Google Earth Engine Team. Google Earth Engine: Planetary-scale geospatial analysis platform. https://earthengine.google.com/

[7] Sentinel-2 User Handbook. European Space Agency (ESA).
