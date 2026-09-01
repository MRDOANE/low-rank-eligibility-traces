# Frozen protocol: N08+O10 stacked eligibility sketch v1.0

## Evidence entering the bridge

N08 v2 was positive only after its prospectively allowed rank-four rescue. At 34.03% of exact
persistent trace state, rank four passed every frozen gate. It retained essentially all full-trace
learning, had mean gradient cosine 0.99996, was statistically noninferior to both matched controls,
and achieved a 5.50% mean long-horizon gain over replay with a 95% bootstrap interval of
2.99-8.07%. Its latent-rank-eight boundary was noncatastrophic but no longer inferentially positive.

O10 v2 passed its premise and method-specific calibration. Its equal-layer rank-eight trace used
26.56% of exact persistent state, retained approximately 99% of full-trace learning, and had mean
gradient cosine near 0.987. Its mean replay advantage grew from 1.9% at 1x to 27.2% at 16x. At 32x,
one teacher/latent-rank cell degraded sharply. Across five teachers, the replay and global-control
confidence intervals crossed zero, so layerwise specificity was not established.

## Combined hypothesis

For each episode, vertically stack the layer eligibility matrices into one block-row matrix. The
candidate estimates one shared right basis from a left-energy-weighted covariance of the event
right factors. It stores that basis together with layer-preserving coefficients. Under the same
allocated state as four independent rank-eight sketches, the representation has rank 13 and uses
4,160 of 4,352 allocated floats. The hypothesis is that this shared adaptive subspace reduces the
brittle equal-layer allocation seen by O10 without giving up N08's low-rank delayed-credit
advantage.

The current reference uses the materialized event factors twice: first to select the right basis,
then to form the layer coefficients. It tests the information representation cleanly before an
online subspace tracker is engineered. It is not itself a one-pass streaming algorithm.

## Controls

- exact full eligibility trace;
- equal-layer rank-eight two-sided sketch;
- exact batch rank-13 global SVD as an information-state oracle, not a systems baseline;
- prospectively selected matched-state replay: gradient-ranked replay at 8x and 16x, reservoir
  replay at 32x.

All learned methods receive identical teachers, event batches, optimizer steps, parameter counts,
evaluation examples, and allocated persistent-state budgets. Each method uses a frozen learning
rate; only the new candidate is calibrated, using teachers excluded from all later stages.

## Noise and false-negative safeguards

- The second half of each training trajectory is averaged prospectively for the primary result.
- Final endpoint weights remain a secondary diagnostic.
- Confirmation uses eight independent teachers and two nested training replicates per teacher.
- Bootstrap intervals resample teacher-level aggregates, not individual nested fits.
- Differences are normalized by the error reduction available to the exact trace. This avoids the
  unstable relative ratios that amplified O10's single 32x outlier.
- The cascade may stop after a clear screen failure, but an ambiguous screen proceeds to the full
  confirmation.

## Confirmation interpretation

A strong positive requires the candidate to retain full-trace learning, remain noninferior to
replay and the global information oracle, and have a positive teacher-bootstrap lower bound over
the equal-layer sketch overall and at 32x. A supportive result meets all retention and
noninferiority gates without proving superiority over equal-layer allocation. Supportive results
still justify carrying the candidate into external benchmarks; they do not justify a stacked-rank
advantage claim.

## Claim boundary

Persistent-state accounting is exact for the retained basis and layer coefficients. The reference
implementation materializes event factors and makes two passes over them to isolate the
representation question. Therefore, this experiment cannot by itself support claims about
one-pass online operation, end-to-end peak CUDA memory, wall-clock acceleration, or
deployment-ready online learning. Those claims require a later online subspace tracker and systems
benchmark.
