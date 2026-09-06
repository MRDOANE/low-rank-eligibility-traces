# External-validation cascade report

Final scientific status: **YELLOW**

The protocol was valid, but at least one scientific criterion failed.

Selected global-SVD rank: **4**

## ICLR continuation gate

| Criterion | Pass |
|---|---:|
| within exact performance tolerance | yes |
| gradient cosine at least 0 95 in every audit cell | yes |
| credit memory reduction at least 3x | yes |
| paired advantage for one comparator at two or more conditions | no |

## Protocol-validity checks

| Check | Pass |
|---|---:|
| audit has predictions after delayed feedback in every cell | yes |
| yearbook oracle is above floor | yes |
| criteo oracle beats constant predictor | yes |

## Matched exact-credit audit

| Benchmark | Delay | Accuracy gap (pp) | Relative excess loss | Median cosine | Predictions after feedback | Cell pass |
|---|---:|---:|---:|---:|---:|---:|
| criteo | natural | -0.1373 | -0.0092 | 0.9978 | 523776 | yes |
| yearbook | 10 | 0.2087 | -0.0289 | 0.9999 | 14976 | yes |
| yearbook | 100 | -0.0024 | -0.0007 | 0.9994 | 3456 | yes |
| yearbook | 50 | 0.0781 | -0.0152 | 0.9995 | 9856 | yes |

## Learner-competence diagnostics

Yearbook immediate-oracle mean balanced accuracy by delay: 10=0.6421, 100=0.6421, 50=0.6421

Criteo immediate-oracle mean relative test-log-loss gain over the constant predictor: 0.2234

## Confirmatory paired intervals

| Condition | Comparator | Metric | Mean advantage | 95% CI | Positive |
|---|---|---|---:|---:|---:|
| yearbook:delay=10 | equal_byte_replay | online_accuracy_advantage | -0.136978 | [-0.148068, -0.125888] | no |
| yearbook:delay=10 | random_projection | online_accuracy_advantage | 0.000707 | [-0.019494, 0.020909] | no |
| yearbook:delay=100 | equal_byte_replay | online_accuracy_advantage | -0.029028 | [-0.034438, -0.023618] | no |
| yearbook:delay=100 | random_projection | online_accuracy_advantage | -0.005695 | [-0.009187, -0.002203] | no |
| yearbook:delay=50 | equal_byte_replay | online_accuracy_advantage | -0.076863 | [-0.082470, -0.071257] | no |
| yearbook:delay=50 | random_projection | online_accuracy_advantage | 0.000988 | [-0.002838, 0.004813] | no |
| criteo:test_natural_delay | equal_byte_replay | negative_log_loss_advantage | -0.115426 | [-0.316177, 0.085325] | no |
| criteo:test_natural_delay | random_projection | negative_log_loss_advantage | -0.048618 | [-0.310404, 0.213169] | no |

Positive-CI confirmatory conditions by comparator: equal_byte_replay=none; random_projection=none

The trial-level CSV contains predictive performance, credit error, gradient cosine, actual state bytes, peak RAM/VRAM, update latency, and throughput. Delay-stratified metrics remain in each JSON trial file.
