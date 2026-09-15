| Model | Fine SB-F1 | Coarse SB-F1 | Cross-lineage error | Fine-outside-parent given coarse-correct | Oracle F1 | Hierarchy gap |
|---|---:|---:|---:|---:|---:|---:|
| Plain MLP | 0.136 | 0.335 | 0.491 | 0.176 | 0.378 | 0.242 |
| + parent-mass | 0.136 | 0.334 | 0.496 | 0.179 | 0.378 | 0.242 |
| + parent-mass + consistency | 0.139 | 0.337 | 0.495 | 0.178 | 0.380 | 0.241 |
| + KD | 0.140 | 0.335 | 0.492 | 0.172 | 0.384 | 0.244 |
| + parent-mass + KD | 0.139 | 0.332 | 0.492 | 0.178 | 0.375 | 0.236 |
| Full HBM | 0.138 | 0.336 | 0.495 | 0.180 | 0.372 | 0.235 |
