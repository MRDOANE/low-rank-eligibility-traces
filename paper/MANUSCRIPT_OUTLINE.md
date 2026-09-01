# Manuscript Outline

## Working title

**Low-Rank Eligibility Traces Preserve Delayed Credit Assignment at Reduced Persistent State**

## Abstract structure

1. Delayed feedback requires retaining credit information across many events.
2. Exact eligibility traces scale with parameter count and can be expensive to retain.
3. Introduce streaming low-rank and shared-right-subspace trace representations.
4. Report the N08 Pareto result, O10 allocation boundary, and optimizer-fair stacked confirmation.
5. State the information-representation contribution and the online-systems limitation.

## 1. Introduction

- Delayed credit assignment and eligibility traces.
- Persistent-state cost of exact traces.
- Why low-rank structure may retain useful credit information.
- Contributions and prespecified claim boundary.

## 2. Related work

- Eligibility traces and delayed reinforcement learning.
- Low-rank gradient and optimizer-state compression.
- Replay-based credit assignment.
- Layerwise versus global rank allocation.

## 3. Methods

- Synthetic delayed-outcome task family.
- Exact trace and state accounting.
- N08 streaming low-rank trace.
- O10 equal-layer sketch and matched global oracle.
- Stacked shared-right-subspace representation.
- Replay controls, optimizer calibration, averaging, and evaluation metrics.

## 4. Experimental design

- Prospective gates and stopping rules.
- Independent teacher partitions and nested replicates.
- N08 v2, O10 v2, invalid stacked v1.0 diagnosis, and authorized v1.1 repair.
- Teacher-level bootstrap intervals and claim thresholds.

## 5. Results

- N08 strong Pareto result at 34.03% state.
- O10 performance preservation and failed layerwise-specificity claim.
- Stacked v1.1 reference repair and optimizer-fair confirmation at 25.39% state.
- Small gains over equal-layer and replay; slight loss to global oracle.
- Prespecified high-rank boundary.

## 6. Discussion

- Low-rank delayed-credit information appears highly compressible in the tested family.
- Allocation across layers matters as effective rank rises.
- Task performance can remain stable after exact-gradient alignment declines.
- Implications for a future one-pass online tracker.

## 7. Limitations

- Synthetic task family.
- Materialized factors and two-pass implementation.
- No demonstrated peak-memory, throughput, or wall-time improvement.
- No superiority to the global-SVD information oracle.

## 8. Conclusion

Summarize the state-performance Pareto result and the empirically mapped allocation boundary without extending beyond the frozen evidence.
