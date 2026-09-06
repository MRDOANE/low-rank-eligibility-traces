# Claim Boundary

## External validation v1.1

### Supported

- The protocol completed validly on the Yearbook and Criteo streams with five main seeds.
- A rank-four global SVD state passed the frozen exact-credit performance, gradient-cosine, feedback-exposure, and memory criteria in every audit cell.
- The exact/global per-event credit-state ratio was 6.3519x.
- Audit median gradient cosine was 0.9978 on Criteo and at least 0.9993 at every Yearbook delay.
- The largest mean audit accuracy gap from exact credit was 0.2087 percentage points.
- Global SVD failed the frozen competitive gate: it did not obtain a positive paired 95% interval against either equal-byte replay or random projection at two independent conditions.
- Equal-byte replay was better on Yearbook by 13.70, 7.69, and 2.90 accuracy points at delays 10, 50, and 100. Random projection was tied at delays 10 and 50 and better at delay 100. Criteo comparator intervals were inconclusive.
- The supplied implementation records literal persistent-state tensor bytes, process RSS, PyTorch allocator peaks, synchronized feedback-update latency, and full-pass throughput.

### Unsupported

- Superiority to replay, random projection, Frequent Directions, naïve delayed learning, or an immediate-feedback oracle.
- Higher end-to-end throughput or lower peak RAM/VRAM in the supplied implementation.
- State-of-the-art Yearbook or Criteo prediction.
- Reproduction of the NeurIPS 2024 paper’s full-network optimization results; the present study reuses its stream ordering and delay convention with a frozen feature producer and an auditable adapter.
- Commercial use of the Criteo data, which are distributed under CC BY-NC-SA 4.0 and are not included here.
- Generality beyond the tested adapter, feature boundary, ranks, delays, and data streams.

## Controlled N08/O10 program

### Supported

- A streaming rank-four trace preserved delayed-credit learning at 34.03% of the exact persistent-state count in N08 v2.
- The rank-four N08 trace produced a strong long-horizon advantage over matched replay under its frozen synthetic-task gate.
- Equal-layer rank-eight traces retained approximately 99% of exact-trace learning at 26.56% state in O10 v2.
- The optimizer-fair stacked rank-13 representation retained full-trace learning at 25.39% state in the final synthetic confirmation.
- The stacked method had small positive primary-confirmation gains over equal-layer and replay controls.
- The global-SVD information oracle remained slightly better in the primary confirmation and clearly better at the high-rank boundary.
- The prespecified boundary identifies a regime where equal allocation degrades, stacked allocation preserves task learning, and exact-gradient alignment declines.

### Unsupported

- A large stacked-specific advantage in the primary synthetic regime.
- Universal effectiveness across arbitrary architectures, tasks, or reinforcement-learning benchmarks.
- Treating the invalid stacked v1.0 confirmation as evidence.

## Terminology boundary

The external object is the per-example observation-time logit Jacobian retained until a delayed binary label arrives. The manuscript calls this a **delayed-credit state** and relates it to eligibility traces. It should not imply that the method introduced eligibility traces, low-rank gradient approximation, RTRL compression, or SVD-based matrix approximation.

## State and systems boundary

Persistent-state ratios refer to the retained information at the declared post-feature boundary. Frozen Yearbook feature extraction and deterministic Criteo hashing are preprocessing. Replay stores the exact post-preprocessing float16 token values and recomputes the adapter Jacobian at feedback time; credit methods store float32 credit payloads and the observation-time logit.

The earlier stacked reference implementation materializes event factors and uses two passes to estimate and apply a shared subspace. The external global-SVD implementation forms each exact adapter Jacobian and performs a per-example decomposition before storage. These are reference implementations for scientific comparison. Their state ratios do not establish an efficient production kernel.

## Treatment of superseded runs

- Stacked v1.0 is an apparatus diagnosis. Its confirmation was invalid because the exact trace failed 10 of 96 cell-level learning checks while only the candidate had received learning-rate calibration.
- External validation v1.0 is also diagnostic. Its gate pooled repeated Yearbook cells over Criteo, used Yearbook audit windows shorter than delays 50 and 100, and counted outcome-defined Criteo strata as independent confirmations.
- Only stacked v1.1 and external validation v1.1 support confirmatory inference.
