# Incremental Comparison Summary

Inc-only comparison across the fast subset expansion runs.

| increment | model | PR-AUC | F1 | Recall | ECE | Var-Err Spearman | delta PR-AUC | delta F1 | delta Recall | delta ECE | delta Var-Err Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | static | 0.9138 | 0.7273 | 1.0000 | 0.4285 | 0.5253 | n/a | n/a | n/a | n/a | n/a |
| 120 | temporal | 1.0000 | 0.9333 | 0.8750 | 0.0754 | 0.1473 | n/a | n/a | n/a | n/a | n/a |
| 180 | static | 1.0000 | 0.7059 | 1.0000 | 0.3994 | -0.2129 | +0.0862 | -0.0214 | +0.0000 | -0.0291 | -0.7381 |
| 180 | temporal | 0.9933 | 0.9600 | 1.0000 | 0.1302 | -0.1338 | -0.0067 | +0.0267 | +0.1250 | +0.0549 | -0.2811 |
| 240 | static | 0.9962 | 0.8649 | 1.0000 | 0.1762 | 0.5675 | -0.0038 | +0.1590 | +0.0000 | -0.2231 | +0.7804 |
| 240 | temporal | 1.0000 | 1.0000 | 1.0000 | 0.0531 | -0.0469 | +0.0067 | +0.0400 | +0.0000 | -0.0772 | +0.0869 |