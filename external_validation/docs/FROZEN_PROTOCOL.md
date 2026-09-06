# Frozen protocol specification

`config/frozen_protocol.json` is the authoritative v1.1 protocol. The launcher copies it into every full result directory and places its canonical SHA-256 digest in every trial record. This version supersedes v1.0; `V1_0_REVIEW.md` records why.

## Method state boundary

The online learner begins at the adapter's float16-quantized token tensor. Frozen image feature extraction and deterministic Criteo hashing are preprocessing; the quantized values are promoted to float32 for arithmetic. Replay retains the exact post-preprocessing float16 values and recomputes the adapter Jacobian when feedback arrives. Credit methods retain float32 credit payloads plus the observation-time logit. The byte tables therefore compare persistent online state at one declared boundary. Model parameters and optimizer buffers are reported separately from credit/replay state.

The exact per-example state is `L × d × d` plus one logit. Global SVD reshapes it to `(L d) × d` and retains `U_r`, `s_r`, `V_r^T`, and the logit. Random projection uses one two-sided sketch of the stacked matrix. O10 layerwise uses independent two-sided sketches per layer. O10 shared stores one adaptive right basis with layer-specific coefficients. Projection matrices are counted once as static state.

## Prequential ordering

At each stream step the model first predicts the current examples. The simulator then admits the corresponding label-free credit or replay state and releases feedback whose due key has arrived. Updates therefore cannot affect the prediction on the same record. The oracle receives its label directly after that prediction. Remaining feedback is flushed after the final prediction so state and latency accounting includes settlement.

Yearbook delays are measured in batches of 128 exactly as the authors' implementation multiplies the configured delay by effective batch size. Criteo feedback is scheduled in one-hour processing buckets while its raw delay is retained in seconds for stratification.

## Chronological splits

- Yearbook uses the upstream loader's default `train` stream for prequential evaluation and its separate `test` split for learning-rate calibration on two disjoint seeds. Ordering is ascending year with a deterministic within-year permutation. The physical archive contains 37,921 images; the manifest reports the number actually referenced by each upstream split.
- Criteo uses days 0–21 for retrospective historical initialization, days 21–25.5 for rank audit/validation, and the remainder for test reporting. These are fixed timestamp boundaries, not quantiles computed from outcomes. Positive feedback uses observed conversion times; a negative becomes known after the declared 30-day maturity window.

## Calibration and learner competence

Learning rates `0.0005`, `0.001`, `0.003`, and `0.01` are compared using seeds 167 and 181. The selected rate is shared by exact credit, every compression method, replay controls, and the oracle within a benchmark. Main results use seeds 83, 97, 113, 131, and 149. The Yearbook calibration arm uses the separate upstream test split; Criteo calibration uses only the historical training interval.

The full-run gate checks that the immediate-feedback Yearbook oracle reaches mean balanced accuracy 0.55 at every fixed delay and that the immediate-feedback Criteo oracle improves held-out log loss over the test-set constant predictor by at least 0.5% relative. A failed competence check makes the scientific comparison invalid and produces red.

## Audit and promotion

Ranks 2, 4, and 8 are evaluated against exact credit on the same leading chronological audit window for every seed and delay. The Yearbook audit contains 16,384 records, so delay 100 has more than 1,024 predictions after feedback has begun. The Criteo audit contains 524,288 validation records. Promotion selects the smallest rank passing the exact-performance tolerance, median cosine 0.95, minimum feedback exposure, and threefold per-event state reduction in every benchmark/delay cell. If no rank passes, the most cosine-aligned rank that still meets memory advances only to characterize the failure.

Full paired intervals use the five main seeds. Yearbook advantage is the global-SVD online-accuracy difference at each of delays 10, 50, and 100. Criteo advantage is comparator log loss minus global-SVD log loss on the complete held-out test stream. Student-t intervals are two-sided 95% intervals over paired seed differences. The continuation rule requires a strictly positive lower endpoint at two or more independent conditions against one fixed comparator: either equal-byte replay or global random projection. Criteo delay bins remain in trial JSON for descriptive diagnosis; they are outcome-linked and cannot provide separate confirmatory votes.
