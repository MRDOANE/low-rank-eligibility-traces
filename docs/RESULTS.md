# Frozen Results

## Reading the metrics

All performance comparisons use error normalized by the initial error in the same cell. A positive candidate gain means the candidate had lower error than the named control. Gain retention is the candidate learning fraction divided by the exact full-trace learning fraction. Gradient cosine measures agreement between the candidate and exact trace updates.

The protocols, gates, and bootstrap units were fixed before each confirmatory stage. Nested training replicates remained within the independent teacher used as the bootstrap unit. The external validation used paired seed-level Student-t intervals fixed before the run.

## External validation v1.1: fixed and natural label delays

The external protocol tested the strongest N08 global state on two chronological streams. Yearbook supplied 37,921 physical images and fixed delays of 10, 50, and 100 batches. Criteo supplied 5,947,563 retained clicks, chronological conversion timestamps, and a declared 30-day maturity point for non-conversions. Five main seeds were used throughout.

### Rank audit against exact credit

| Rank | Minimum cell median cosine | Minimum exact/global state ratio | Result |
|---:|---:|---:|---|
| 2 | 0.9300 | 12.6842x | Failed the Criteo cosine requirement |
| 4 | 0.9978 | 6.3519x | Smallest rank passing every cell |
| 8 | 0.9999 | 3.1784x | Passed but used more state |

For the selected rank, all four cells passed the exact-performance tolerance, minimum cosine 0.95, minimum feedback exposure, and threefold state-reduction criteria.

| Benchmark | Delay | Accuracy gap from exact (pp) | Relative excess loss | Median cosine | Predictions after feedback |
|---|---:|---:|---:|---:|---:|
| Criteo | natural | -0.1373 | -0.0092 | 0.9978 | 523,776 |
| Yearbook | 10 | 0.2087 | -0.0289 | 0.9999 | 14,976 |
| Yearbook | 50 | 0.0781 | -0.0152 | 0.9995 | 9,856 |
| Yearbook | 100 | -0.0024 | -0.0007 | 0.9994 | 3,456 |

### Full competitive comparison

The frozen gate required a positive paired 95% interval at two independent conditions against the same comparator. No comparator comparison met that rule.

| Condition | Comparator | Global-SVD advantage | 95% interval |
|---|---|---:|---:|
| Yearbook delay 10 | equal-byte replay | -0.136978 accuracy | [-0.148068, -0.125888] |
| Yearbook delay 50 | equal-byte replay | -0.076863 accuracy | [-0.082470, -0.071257] |
| Yearbook delay 100 | equal-byte replay | -0.029028 accuracy | [-0.034438, -0.023618] |
| Yearbook delay 10 | random projection | 0.000707 accuracy | [-0.019494, 0.020909] |
| Yearbook delay 50 | random projection | 0.000988 accuracy | [-0.002838, 0.004813] |
| Yearbook delay 100 | random projection | -0.005695 accuracy | [-0.009187, -0.002203] |
| Criteo test | equal-byte replay | -0.115426 negative log loss | [-0.316177, 0.085325] |
| Criteo test | random projection | -0.048618 negative log loss | [-0.310404, 0.213169] |

Positive values favor global SVD. The Criteo intervals were wide and inconclusive. On Yearbook, replay dominated global SVD at all delays; random projection was statistically tied at delays 10 and 50 and better at delay 100.

### Secondary representations and systems measurements

The O10 layerwise and shared variants did not reverse the competitive result. On Criteo, mean test log loss was 3.2708 for global SVD, 3.1053 for O10 layerwise, 3.2792 for O10 shared, 2.9409 for random projection, and 3.1817 for equal-byte replay. The immediate-feedback oracle reached 0.3045, confirming that the representation and labels contained learnable signal.

The reference SVD path processed roughly 1,200 to 1,300 examples per second. Replay and the randomized or O10 codecs generally processed about 18,000 to 28,000 examples per second. Median feedback-update latency for SVD was short, but the per-example decomposition dominated full-pass throughput. No systems-efficiency claim is supported.

### External decision

The terminal decision was **YELLOW**: all protocol-validity, exact-fidelity, gradient-cosine, and memory criteria passed; the competitive paired-interval criterion failed. The result supports a compression-fidelity claim and rejects the preregistered superiority claim.

## N08 v2: streaming low-rank trace

N08 tested whether a streaming rank-constrained eligibility trace could retain delayed-credit learning across horizons while using less persistent state.

| Quantity | Frozen result |
|---|---:|
| Selected rank | 4 |
| Candidate state | 588 floats |
| Exact state | 1,728 floats |
| Candidate fraction | 34.0278% |
| Mean gain retention | 1.02150 |
| 95% interval for retention | [1.00585, 1.04302] |
| Mean gradient cosine | 0.999957 |
| Long-horizon gain over replay | 0.05504 |
| 95% interval for long-horizon replay gain | [0.02986, 0.08068] |

The frozen decision was `n08_v2_strong_pareto_positive_continue_to_o10`. The rank-four rescue passed every state, learning, retention, gradient-alignment, and matched-process check. The designed rank-eight boundary preserved learning but reduced gradient alignment, providing a capacity boundary rather than another superiority result.

## O10 v2: equal-layer rank allocation

O10 tested whether preserving separate layer blocks at equal rank improved delayed-credit learning relative to replay and a matched global low-rank oracle.

| Quantity | Frozen result |
|---|---:|
| Selected rank per layer | 8 |
| Candidate state | 4,352 floats |
| Exact state | 16,384 floats |
| Candidate fraction | 26.5625% |
| Mean gain retention | 0.99361 |
| 95% interval for retention | [0.98438, 1.00024] |
| Mean gradient cosine | 0.98714 |
| 32x gain over replay | 0.12845 |
| 32x gain over global oracle | -0.71804 |

The 32x confidence intervals crossed zero for both replay and global comparisons. O10 therefore stopped under its frozen rule and did not establish layerwise specificity. It remained useful as evidence that compressed traces retained approximately 99% of exact-trace learning at one-quarter state and as motivation for testing global budget allocation directly.

## Stacked v1.1: optimizer-fair shared right subspace

The combined candidate estimated one right subspace across layer blocks and retained layer-specific coefficient blocks. It used 4,160 floats from a matched allocation of 4,352, or 25.3906% of the 16,384-float exact trace.

The earlier v1.0 confirmation was invalid because the exact reference failed its learning gate while only the candidate had been optimizer-calibrated. v1.1 calibrated every method using the same frozen grid and worst-cell-first rule. Every method-horizon pair selected a learning rate of 0.0005. The clean holdout and the 12-cell exact-reference repair audit passed before fresh candidate inference began.

### Primary confirmation

| Quantity | Mean | 95% teacher-bootstrap interval |
|---|---:|---:|
| Full-trace gain retention | 1.000363 | [1.000048, 1.000929] |
| Gradient cosine | 0.978646 | [0.977143, 0.980139] |
| Gain over global oracle | -0.000042 | [-0.000092, -0.000006] |
| Gain over equal-layer | 0.000436 | [0.000019, 0.001164] |
| Gain over replay | 0.002185 | [0.000189, 0.005765] |
| 32x gain over equal-layer | 0.000564 | [0.000040, 0.001438] |
| 32x gain over replay | 0.005429 | [0.000454, 0.014556] |

The candidate beat the full trace in 61 of 96 cells, the global oracle in 40, equal-layer in 67, and replay in 72. No cell collapsed. Every supportive check passed. The strong gate failed because the mean equal-layer gain was 0.000436 of initial error, below the frozen 0.01 effect-size threshold. Its sign and confidence bounds were positive.

The frozen decision was `stacked_trace_optimizer_fair_supportive_without_specific_advantage_preserve_claim_boundary`.

### Prespecified high-rank boundary

The boundary increased latent rank to 8 and 12 at 16x and 32x horizons.

| Quantity | Mean | 95% teacher-bootstrap interval |
|---|---:|---:|
| Full-trace gain retention | 1.001212 | [0.999985, 1.002922] |
| Gradient cosine | 0.780435 | [0.778609, 0.783430] |
| Gain over global oracle | -0.005631 | [-0.016126, -0.000204] |
| Gain over equal-layer | 0.046169 | [0.002303, 0.126750] |
| Gain over replay | 0.012900 | [0.000924, 0.033302] |

The candidate beat equal-layer and replay in all 12 cells and lost to the global oracle in all 12. The boundary did not pass the supportive cosine threshold. It shows that global allocation matters as effective rank rises, while the approximate stacked update stops following the exact gradient direction closely.

## Overall interpretation

N08 provides a positive controlled Pareto result. O10 establishes an allocation control and negative specificity boundary. Stacked v1.1 shows that a shared-right-subspace representation can preserve exact synthetic-task learning at approximately one-quarter persistent state. External v1.1 confirms that rank-four global SVD can reproduce exact observation-time credit at 6.35x lower per-event state across two real streams.

The external comparison also identifies the main scientific boundary. Exact-credit fidelity is not equivalent to delayed-learning utility. Replay can benefit from recomputing the Jacobian under the later model, and a rough sketch can alter the update in ways that are at least as useful as reconstructing the earlier Jacobian. The current evidence supports the compression result, the fidelity result, and this separation; it does not support competitive or systems superiority.
