# Frozen protocol: N08+O10 optimizer-fair stacked eligibility repair v1.1

## Why one repair is scientifically allowed

The v1.0 screen was strong, but its confirmation was invalid because the exact-reference optimizer
failed its own cell-level learning gate and only the candidate had been calibrated. This is an
apparatus failure, not a valid mechanism failure. v1.1 changes optimizer selection and numerical
analysis only. It does not change the candidate, controls, persistent-state budgets, task family,
primary horizons, confirmation size, or substantive pass thresholds.

The source result and diagnosis are frozen in `docs/N08_O10_STACKED_V10_RESULT.md`. The input
archive SHA-256 is
`ff5391547a6411ef98ae7c3b344a05a2d07f16aa4e87b9a8ee0577ddaad380e4`.

## Methods and state

The five methods remain:

- exact full eligibility trace;
- stacked shared-right-subspace candidate;
- equal-layer rank-eight sketch;
- batch global-SVD information oracle;
- prospectively selected matched-state replay: gradient replay at 8x and 16x, reservoir replay at
  32x.

The candidate still uses 4,160 floats of 4,352 allocated, versus 16,384 floats for the exact trace.
The representation remains a two-pass materialized-factor reference and does not establish online
operation or a systems speedup.

## Stage 1: premise and provenance

The v1.0 structural, finite-difference, trace-quality, and state-accounting checks are rerun. The
new gate additionally verifies that every fresh teacher set is disjoint, that the known failure
teachers are audit-only, and that the frozen optimizer grid contains the v1.0 candidate and
control rates.

## Stage 2: optimizer-fair calibration

Every method is calibrated separately at each of 8x, 16x, and 32x on teachers 1709 and 1801,
latent ranks 2 and 4, and the common frozen grid
`[0.00025, 0.0005, 0.001, 0.002, 0.004]`. Every fit uses 350 steps and the same task, batch,
averaging, weight-decay, and evaluation rules.

For each method-horizon pair, select the rate with the lowest worst normalized averaged-checkpoint
MSE across calibration cells, then lowest mean, then lowest median. Exact ties choose the lower
rate. Normalization is by each cell's initial MSE.

## Stage 3: clean calibration holdout

Teacher 1901 is excluded from tuning. All five methods run at their selected rates for 350 steps
at both latent ranks and all three horizons. The stage passes only if:

- every result is finite;
- no selected method has MSE above 1.25 times initial MSE in any cell;
- the exact trace learns by at least 60% overall, 45% at every horizon, and 20% in every cell.

This stage validates optimizer selection only and never contributes to candidate inference.

## Stage 4: known-failure reference audit

Only the exact trace is rerun on v1.0 teachers 907 and 1201, latent rank 2, horizons 8x/16x/32x,
and both original nested training replicates. The v1.0 task, evaluation, and training seed namespaces
are reproduced exactly; only the independently selected exact-trace rate changes. All twelve cells
must be finite and the exact trace must again clear the 60% overall, 45% per-horizon, and 20%
per-cell learning gates.

These diagnosed teachers are permanently excluded from performance analysis. Audit failure stops
v1.1 as invalid with no mechanism inference.

## Stage 5: fresh screen

Teachers 2003, 2111, and 2203 are new. The screen uses latent ranks 2 and 4, horizons 8x and 32x,
one training replicate, 350 steps, and all five methods at their selected rates (60 fits).

An invalid exact reference stops without inference. A valid screen advances unless it is decisively
bad under the unchanged retention, cosine, control, state, and collapse thresholds.

## Stage 6: fresh confirmation

Teachers 2309, 2411, 2503, 2609, 2707, 2801, 2903, and 3001 are new. Confirmation uses latent
ranks 2 and 4, horizons 8x/16x/32x, two nested training replicates, 550 steps, and all five methods
(480 fits). Independent teachers are the bootstrap units; nested replicates stay within teacher.

The exact reference must learn by at least 60% overall, 45% at every horizon, and 20% in every
cell. If any reference condition fails, no control bootstrap or candidate decision is computed.

## Stable metrics and frozen decision rules

For each cell:

- method learning fraction = `(initial MSE - method MSE) / initial MSE`;
- candidate gain over a control = `(control MSE - candidate MSE) / initial MSE`;
- candidate/full gain retention = `candidate learning fraction / full learning fraction`, computed
  for inference only after the exact-reference gate passes in every cell.

This removes the v1.0 singular denominator while retaining the original scientific thresholds.

A supportive confirmation requires:

- mean gain retention at least 0.95, teacher-bootstrap lower bound at least 0.90, and every-horizon
  retention at least 0.85;
- mean candidate gradient cosine at least 0.90 and every-horizon cosine at least 0.85;
- teacher-bootstrap noninferiority to replay, global SVD, and equal-layer controls, including the
  longest horizon, with margins 0.03, 0.03, and 0.02 of initial MSE;
- the unchanged 27% state ceiling, matched compressed-state allocation, parameter count, optimizer
  steps, selected-rate application, and prospective weight averaging.

A strong positive additionally requires mean gain over equal-layer of at least 0.01 of initial
MSE and positive teacher-bootstrap lower bounds both overall and at 32x. A supportive result does
not establish a stacked-allocation advantage.

## Stage 7: conditional boundary

Only a supportive confirmation runs teachers 3109, 3203, and 3301 at latent ranks 8 and 12,
horizons 16x and 32x, one replicate, and 450 steps (60 fits). The boundary is characterization;
it cannot rescue the primary confirmation.

## Terminal rules

- Invalid holdout, audit, screen, or confirmation: preserve N08 and O10; make no stacked inference.
- Valid negative screen or confirmation: stop the stacked bridge permanently and build the
  N08-centered bounded paper.
- Supportive but not strong: preserve the combined method only under the no-specific-advantage
  wording and inspect the conditional boundary.
- Strong positive: build the online subspace tracker and then external benchmarks. Systems claims
  remain prohibited until those later stages are measured.

No further learning-rate expansion, rank search, task modification, control removal, or mechanism
rescue is authorized after a valid v1.1 result.
