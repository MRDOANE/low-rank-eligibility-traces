# Review of the v1.0 decision

The v1.0 run completed all 240 scheduled trials and passed its file-integrity checks. Its printed `GREEN` decision does not support the intended continuation claim, because three reporting and audit defects made the gate too permissive. Version 1.1 supersedes that decision while preserving v1.0 as an auditable exploratory run.

## What remained encouraging

- On Yearbook, global SVD retained strong local gradient geometry: the median per-trial cosine was about 0.992 at rank 2, 0.996 at rank 4, and 0.999 at rank 8.
- On Criteo, cosine rose from about 0.675 at rank 2 to 0.831 at rank 4 and 0.948 at rank 8. Rank 8 still reduced per-event credit storage by about 3.18 times, placing it close to the preregistered joint geometry/memory boundary.
- Predictive results for compressed credit remained close to exact credit in the short audit used by v1.0.

## Why the green decision was invalid

1. The cosine gate pooled 15 Yearbook trials with five Criteo trials. Strong Yearbook values raised the pooled rank-2 median to 0.991 even though the Criteo rank-2 median was about 0.675. The gate now evaluates every benchmark/delay cell separately.
2. The 4,096-example Yearbook audit ended before any labels arrived at delays 50 and 100, whose first feedback was due after 6,400 and 12,800 examples. Exact and compressed credit therefore appeared identical without exercising their updates. The v1.1 audit uses 16,384 examples and verifies post-feedback predictions directly.
3. All six positive confidence-interval votes came from Criteo delay bins. Those bins were defined using conversion timing: short-delay bins contained converters, while the 30-day bin contained non-converters. They were correlated outcome strata from one test stream rather than six independent delay conditions. Version 1.1 treats the entire Criteo test stream as one confirmatory condition and requires two wins against the same comparator.

The v1.0 immediate-feedback models were also weak: Yearbook balanced accuracy stayed near 0.50 and Criteo predicted the majority class. Version 1.1 adds disjoint-seed, method-neutral learning-rate calibration, an intercept inside the traced parameterization, and explicit oracle-competence checks. These changes are protocol repairs declared before rerunning the benchmarks; they are not a reinterpretation of the v1.0 thresholds.
