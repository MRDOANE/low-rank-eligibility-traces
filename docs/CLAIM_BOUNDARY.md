# Claim Boundary

## Supported by the experiments

- A streaming rank-four trace preserved delayed-credit learning at 34.03% of the exact persistent-state count in N08 v2.
- The rank-four trace produced a strong long-horizon advantage over matched replay under the frozen N08 gate.
- Equal-layer rank-eight traces retained approximately 99% of exact-trace learning at 26.56% state in O10 v2.
- The optimizer-fair stacked rank-13 representation retained full-trace learning at 25.39% state in the final confirmation.
- The stacked method had small positive primary-confirmation gains over equal-layer and replay controls.
- The global-SVD oracle remained slightly better in the primary confirmation and clearly better at the high-rank boundary.
- The prespecified boundary identifies a regime where equal allocation degrades, stacked allocation preserves task learning, and exact-gradient alignment declines.

## Not supported by the experiments

- One-pass online operation.
- Lower peak host or CUDA memory in the supplied reference implementation.
- Lower end-to-end wall time or higher throughput.
- Superiority to the batch global-SVD information oracle.
- A large stacked-specific advantage in the primary confirmation regime.
- Universal effectiveness across arbitrary architectures, tasks, or real-world reinforcement-learning benchmarks.

## Why the systems claims are excluded

The stacked reference implementation materializes per-event factors and uses two passes: one to estimate the shared subspace and another to apply it. Persistent-state accounting describes the retained eligibility representation, not the peak temporary memory of this implementation. A production claim requires a genuinely online tracker and direct peak-memory, throughput, and wall-time measurements.

## Treatment of v1.0

Stacked v1.0 is retained solely as an apparatus diagnosis. Its confirmation was invalid because the exact trace failed 10 of 96 cell-level learning checks while only the candidate had received learning-rate calibration. No mechanism inference or v1.0 effect size should be reported as confirmatory evidence. The single authorized v1.1 repair applied the same optimizer-selection procedure to every method and used fresh teachers for inference.
