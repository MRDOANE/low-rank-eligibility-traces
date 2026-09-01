# N08 v2.0: Accuracy-Memory-Delay Pareto Test

## Main research question in one sentence

Can a neural network keep a small compressed record of earlier gradient information and still learn
almost as well as a network that stores the full record when the feedback arrives much later?

## What this run specifically tests

The old N08 screen asked the compressed trace to beat replay by 10%. It missed that bar, even though
it retained 99.77% of the full trace's learning gain with 17.01% of the persistent state. That old
failure remains frozen. v2.0 asks a new question: does the compressed trace lie on a better
accuracy-memory-delay tradeoff curve than exact storage, replay, or generic multi-timescale state?

In second-year CS terms, imagine that a program processes a long list and receives one error message
only at the end. The exact algorithm saves a large matrix describing how every weight contributed.
N08 saves only the most important one, two, or four matrix directions. This experiment checks how
much accuracy that compression costs as the list becomes 1, 4, or 8 times longer.

## Why this is less likely to produce a false negative

- Every method receives its own learning-rate calibration on separate development tasks.
- The confirmation uses five fresh seeds, none reused from v1.0.
- The primary gate tests rank two at 1x, 4x, and 8x learning horizons, not merely longer evaluation.
- The comparison uses replay and multi-timescale controls with no more persistent state.
- Statistical intervals resample whole seeds, so correlated task cells are not treated as independent.
- Rank two is allowed a preregistered rank-four rescue only when it is clearly informative but appears
  capacity-limited.
- A successful rank-two result automatically maps ranks one and four and tests a harder latent-rank-eight
  boundary.
- A task is not allowed to count against N08 unless the exact full trace first demonstrates adequate
  learning headroom.

## Frozen stages

| Stage | Work | Purpose |
|---|---|---|
| Reference audit | Finite differences, spectral coverage, state accounting | Prove that the task and implementation are valid before training |
| Calibration | Three learning rates per method, rank, and horizon on two development seeds | Avoid rejecting a method because one shared optimizer setting favored another |
| Primary | Five seeds, latent ranks 2 and 4, horizons 1x/4x/8x, trace rank 2 | Test near-lossless compression and noninferiority at 17% state |
| Capacity rescue | Rank 4 on the same five-seed grid, only if rank 2 is informative but fails | Distinguish a dead mechanism from an overly tight compression budget |
| Frontier | Ranks 1 and 4 on two prespecified confirmation seeds, if rank 2 passes | Check that the result is part of a smooth tradeoff rather than one lucky budget |
| Boundary | Three new seeds at latent rank 8 and horizons 4x/8x | Characterize where compression begins to break |

## Primary continuation gate

The rank-two result advances only if all of these frozen checks pass:

- the exact trace removes at least 50% of available error overall and at least 35% at every horizon;
- mean compressed-trace gain retention is at least 97%, with a seed-bootstrap lower bound of 94%;
- every latent-rank/horizon cell retains at least 90% of exact-trace gain;
- the upper 95% bound on excess error is at most 3% versus replay and 5% versus the strongest control;
- mean gradient cosine is at least 0.90 and at least 0.85 at every horizon;
- rank-two state remains at most 20% of full state;
- parameters and optimizer-step counts match.

A stronger Pareto signal additionally requires at least a 3% mean replay advantage at long horizons,
a positive lower confidence bound, and at least two long-horizon task cells clearing 3%.

## Automatic rank-four rescue

Rank four runs only if rank two fails the full gate while retaining at least 80% of exact learning,
maintaining at least 0.75 gradient cosine, and passing the implementation controls. These conditions
mean that the mechanism is carrying useful information and a larger rank is a scientifically motivated
capacity test. Rank four is capped at 36% of full state. A clearly uninformative rank-two result does
not receive further task-specific rescue.

## Interpretation

| Outcome | Meaning | Next action |
|---|---|---|
| Strong Pareto positive | Compression matches exact learning and beats replay as delay grows | Continue to O10, then test the combined layerwise mechanism |
| Compression positive | Compression is near-lossless and statistically noninferior, but not superior | Continue to O10; treat N08 as an efficiency component rather than a standalone win |
| Rank-four rescue positive | The idea works, but 17% state was too restrictive | Continue with the 34% state formulation and report the boundary honestly |
| Decisive failure | The exact learner works but both justified compression budgets fail | Stop N08 v2.0 and move directly to O10 |
| Invalid reference | The exact learner does not establish a valid test | Repair the benchmark; do not count this as evidence against N08 |

## Important claim boundary

The state figures describe the eligibility information that must persist until feedback arrives. The
reference code materializes event factors to run batched comparisons efficiently, so this experiment
does not yet claim lower measured peak GPU memory or faster wall-clock training. A true online kernel
and natural delayed-feedback benchmark are required before a systems claim.

## August 2026 collision check

Recent adjacent work narrows the claim but does not duplicate this test. Hanut and Kadmon compress
the dimensionality of the error signal used for feedback alignment, while ARCA derives token salience
from a LoRA adapter's hidden-state residual. Neither studies a streaming truncated-SVD summary of an
accumulated layerwise eligibility matrix under a fixed persistent-state budget. KeRNL remains the
closest established reduced-rank eligibility-trace precedent, so a positive N08 result must be framed
as an empirical state-delay Pareto frontier rather than the invention of low-rank credit assignment.

- https://arxiv.org/abs/2502.20580
- https://arxiv.org/abs/2606.00257
- https://openreview.net/forum?id=ryGfnoC5KQ

## Frozen pre-run odds

These subjective estimates are recorded before seeing v2.0 results:

- `P(+) = 67%`
- `P(Big 3 main track | +) = 38%`
- `P(TMLR | +) = 72%`
- `P(JMLR | +) = 40%`

The experiment design itself does not count as positive evidence and therefore does not change these
priors.
