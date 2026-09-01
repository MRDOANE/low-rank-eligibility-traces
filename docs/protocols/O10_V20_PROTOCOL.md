# O10 v2.0: Actual-Horizon Layerwise Eligibility Traces

## Main research question

When terminal feedback is delayed, do separate fixed-state traces at several network layers become
more useful than an equally budgeted replay memory as the horizon on which the model actually
learns becomes longer?

## Relationship to the frozen v1.0 result

O10 v1.0 remains a failure of its original broad gate. A rank-eight layerwise trace retained 99.31%
of exact-trace gain with 26.56% of the persistent state, but it was 49.68% worse than reservoir
replay on the pooled horizon metric and won only two of six cells. It did, however, reduce MSE by
24.10% against replay on the prespecified 8x evaluation slice.

That slice did not prove delayed credit assignment at 8x. Every method had trained on 4x streams;
8x was only a length extrapolation after learning. All methods also shared one learning rate and the
five-seed confirmation never ran. v2.0 asks a new prospectively frozen question and does not
retroactively pass v1.0.

## False-negative protections added in v2.0

- Models train independently at actual 1x, 4x, and 8x delayed-feedback horizons.
- Every method and horizon selects among five learning rates on separate development teachers.
- Five confirmation replicates use five independently drawn teachers, not five optimizer seeds on
  one teacher.
- The same teacher is paired across horizons inside a replicate, making the horizon interaction a
  within-replicate comparison.
- Replay is represented by reservoir, stratified, and online gradient-prioritized policies. The best
  replay family at each horizon is chosen on development data before confirmation is scored.
- The old pooled ablation is replaced by a stronger global truncated-SVD oracle. It retains layer
  position and allocates rank freely across the vertically stacked layer-gradient matrix.
- Whole teacher replicates, rather than correlated task cells, are resampled for confidence
  intervals.
- An informative capacity failure receives a frozen rank-twelve rescue. A promising horizon trend
  instead receives actual 16x and 32x training.
- A full-trace failure makes the result invalid rather than negative evidence against O10.

## Frozen task family

- Four frozen nonlinear layers of width 64.
- One trainable 64 by 64 residual adapter at each layer.
- Latent-rank-2 and latent-rank-4 teacher families.
- Base sequence length 128; actual training lengths 128, 512, and 1,024 in the primary stage.
- One scalar aggregate target is disclosed only after the complete sequence.
- Square-root-length normalization keeps target scale comparable across horizons.
- Evaluation uses IID, increased input noise, increased latent scale, and their joint shift at the
  same horizon on which each model trained.
- Development teacher draws are disjoint from the five confirmation teacher draws.

## Frozen rank-eight methods and state accounting

| Method | Allocated floats | Used floats | Role |
|---|---:|---:|---|
| Exact full trace | 16,384 | 16,384 | Learnability and upper-bound reference |
| Separate layerwise rank-8 traces | 4,352 | 4,352 | O10 candidate |
| Global rank-13 truncated SVD | 4,352 | 4,173 | Strong same-budget global low-rank oracle |
| Reservoir replay, 66 events | 4,352 | 4,290 | Unbiased replay control |
| Stratified replay, 66 events | 4,352 | 4,290 | Horizon-coverage replay control |
| Gradient-prioritized replay, 66 events | 4,352 | 4,290 | Online importance replay control |
| Eight multi-timescale factor traces | 4,352 | 4,096 | Generic trace control |

The candidate uses 26.56% of full-trace persistent state. Allocated rather than used state is matched,
so unused rounding capacity cannot favor a control. The exact global SVD is intentionally stronger
than a practical streaming sketch. If O10 cannot compete with the best information that a global
same-state factorization could store, layer separation is not established.

## Stage 0: implementation and premise audit

The run stops as invalid unless:

- paired horizons contain identical base matrices and teacher adapters;
- manual layer gradients match finite differences;
- rank-eight layerwise and global approximations are finite and informative;
- permuting layer identity materially changes the gradient direction;
- all matched methods respect the same allocated state;
- targets are finite and the zero-adapter learner has nontrivial headroom.

No performance comparison against replay is used to screen the task before training.

## Stage 1: optimizer calibration

For every method and each actual horizon, learning rates 0.002, 0.004, 0.008, 0.016, and 0.032
run for 200 steps on two independent development teachers in both latent-rank families. The selected
rate minimizes median held-out MSE normalized by zero-student MSE. The replay family with the best
development objective is frozen separately at each horizon.

This stage contains 420 training runs. A numerical failure of the entire calibration grid is an
implementation failure, not scientific evidence.

## Stage 2: five-replicate actual-horizon primary

All seven methods train for 550 matched updates at 1x, 4x, and 8x on five independent teacher
replicates and two latent-rank families. There are 210 confirmation runs. The main unit of
uncertainty is the paired teacher replicate.

The primary trace-crossover gate requires all of the following:

- the exact trace removes at least 60% of initial error overall, 45% at every horizon, and 20% in
  every individual teacher/rank/horizon cell;
- candidate mean exact-gain retention is at least 95%, with a paired-replicate lower 95% bound of
  at least 90% and at least 85% retention at every horizon;
- at 8x, candidate MSE is at least 5% lower than prospectively selected replay, its lower confidence
  bound is positive, and at least 70% of rank-by-replicate cells are wins;
- candidate replay advantage improves by at least five percentage points from 1x to 8x, with a
  positive lower confidence bound;
- candidate is noninferior to the stronger global low-rank oracle within a three-point margin at 8x;
- candidate gradient cosine averages at least 0.88 and is at least 0.80 at every horizon;
- parameters, optimizer steps, and allocated state remain matched.

A separate layer-specificity flag requires at least a 3% candidate gain over the global oracle, a
positive lower confidence bound, and the same 70% cell-win requirement. Thus a result may establish
a compressed-trace crossover without establishing that separate per-layer storage caused it.

## Conditional branches

### Rank-twelve capacity rescue

Rank twelve is allowed only if rank eight retains meaningful exact-gradient information but fails
because gain retention or gradient cosine suggests inadequate rank. Rank twelve uses 6,720 floats,
41.02% of the full trace, and repeats calibration plus all five paired primary replicates. Merely
losing to a strong control while already retaining the exact trace does not authorize more rank.

### Actual 16x and 32x extension

The extension runs if the primary gate passes or if rank eight shows a noncatastrophic positive
horizon trend. The selected rank trains on sequences of 2,048 and 4,096 events using the same five
teacher families. Learning rates and replay identity are recalibrated on development teachers at
both horizons. Multi-timescale memory is omitted only from this conditional extension after having
served as a primary control; exact trace, candidate, global SVD, and all three replay policies remain.

The combined 1x-to-32x analysis must satisfy the same frozen gate and have positive point-estimate
replay gains at both 16x and 32x. Failure after a primary pass is reported as a longer-horizon
boundary, not used to erase the narrower 8x result.

## Interpretation

| Outcome | Meaning | Next action |
|---|---|---|
| Layerwise long-delay crossover | Candidate beats replay increasingly with delay and beats the global oracle | Strong O10 component for the combined N08/O10 paper |
| Trace crossover without layer specificity | Candidate beats replay but not the global oracle | Preserve as combined trace evidence; do not claim layer separation |
| 8x-only positive | Primary passes but 16x/32x does not | Report a bounded effect and redesign the combined test cautiously |
| Decisive failure | Exact trace learns, candidate is informative enough to test, but no justified rescue or horizon trend remains | Freeze O10 and move to N10 |
| Invalid reference | Exact learner, premise, or calibration fails | Repair the apparatus; make no negative mechanism inference |

## Novelty boundary after the August 29, 2026 collision check

O10 does not claim to invent eligibility traces, reduced-rank online credit assignment, deep
gradient eligibility traces, layer-local learning, or long-delay state-space traces. Relevant prior
lines include e-prop, KeRNL, Deep Reinforcement Learning with Gradient Eligibility Traces,
Cascading Eligibility Traces, Traces Propagation, Selective Eligibility Traces for RLVR, and ARCA.

The residual claim is narrower: under an exactly matched persistent-state budget, determine whether
summarizing all event-level adapter gradients crosses over against storing a shrinking event sample,
and whether reserving low-rank capacity separately by layer improves that crossover. The paired
actual-training-horizon interaction and the same-budget global SVD oracle are the distinguishing
tests.

- https://openreview.net/forum?id=ryGfnoC5KQ
- https://arxiv.org/abs/2506.14598
- https://arxiv.org/abs/2507.09087
- https://arxiv.org/abs/2509.13053
- https://arxiv.org/abs/2605.05965
- https://arxiv.org/abs/2606.00257

## Claim boundary

The implementation measures persistent eligibility information, not peak CUDA memory or wall-clock
efficiency. For batched reference comparisons it materializes event factors, and the global control
uses an exact truncated SVD. A positive result must be followed by a true online kernel and a natural
delayed-feedback benchmark before making systems or practical-memory claims.

## Frozen pre-run odds

- P(paper-worthy positive): 58%
- P(Big-3 main track | positive): 40%
- P(TMLR | positive): 70%
- P(JMLR | positive): 38%

The new design is not evidence and therefore does not change these priors.
