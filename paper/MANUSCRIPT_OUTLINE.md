# TMLR Manuscript Outline

## Working title

**Low-Rank Delayed-Credit States: Gradient Fidelity, Memory Compression, and Competitive Limits**

## One-sentence claim

A rank-four SVD representation compresses per-example observation-time credit by 6.35x while closely reproducing exact delayed updates across controlled and external streams, yet equal-byte replay and random sketching show that exact-credit fidelity alone is insufficient to choose the best delayed learner.

## Abstract structure

1. Delayed labels force an online learner to decide what information to retain between prediction and feedback.
2. Define the exact delayed-credit state as the observation-time logit Jacobian and compare compressed credit with input replay/recomputation under measured byte budgets.
3. Introduce global SVD, streaming, shared-right-subspace, layerwise, and two-sided random-sketch representations.
4. Summarize the controlled N08/O10 findings and preregistered external validation on Yearbook and Criteo.
5. Report the main positive result: rank four, 6.35x per-event compression, median cosine 0.9978 to 0.9999, and performance within 0.21 percentage points of exact credit.
6. Report the competitive boundary: zero confirmatory wins, large Yearbook deficits to equal-byte replay, and unfavorable SVD throughput.
7. State the lesson: representational fidelity and delayed-learning utility require separate evaluation.

## 1. Introduction

- Motivate prediction streams in which outcomes arrive after parameters have changed.
- State the operational choice: retain inputs for later recomputation, retain observation-time credit, or retain a compressed sketch.
- Define why exact observation-time credit may differ from label-time recomputation after model drift.
- Present the paper as an empirical study of retained information under delayed supervision.
- List contributions using scoped language and no priority claim.

## 2. Problem formulation

- Observation at time t, model parameters theta_t, logit f(x_t; theta_t), label reveal time rho_t.
- Exact delayed-credit state J_t = partial f(x_t; theta_t) / partial theta.
- At reveal, form the stored-credit update from the residual and J_t.
- Define replay/recomputation using x_t and the later parameters theta_rho_t.
- Define state boundary, byte accounting, static state, pending queues, latency, and throughput.
- Explain that “delayed-credit state” is related to eligibility traces and differs from the continuously evolving RTRL influence matrix.

## 3. Related work

- Delayed online learning and label-delay continual learning.
- Eligibility traces, e-prop, OSTL, and expected traces.
- RTRL, NoBackTrack, UORO, KF-RTRL, optimal Kronecker-sum approximation, and SK-RTRL.
- Low-rank gradient communication compression and streaming matrix sketches.
- Replay and recomputation under delayed labels.
- Delayed conversion modeling.
- Close with the exact gap from `docs/NOVELTY_AUDIT.md`.

## 4. Methods

- Four-matrix adapter and analytic outer-product Jacobian.
- Exact per-event state.
- Streaming N08 trace.
- Batch global SVD of the vertically stacked layer Jacobian.
- O10 equal-layer and shared-right representations.
- Matched-state two-sided random projection.
- Full replay, equal-byte replay, naïve delayed learning, and immediate-feedback oracle.
- Frozen optimization, state-budget, resource, and predictive metrics.

## 5. Experimental design

### 5.1 Controlled development program

- Synthetic delayed-outcome task family.
- N08 v2, O10 v2, invalid stacked v1.0 diagnosis, and optimizer-fair stacked v1.1 repair.
- Prospective gates, independent teachers, and bootstrap units.

### 5.2 External validation

- Yearbook provenance, physical archive count, official loader, chronological order, deterministic feature extraction, and delays 10/50/100.
- Criteo source revision, click filter, safe-feature allow-list, forbidden outcome fields, chronological split, observed positive reveal time, and 30-day negative maturity.
- Separate calibration seeds and five main seeds.
- Rank audit against exact credit and full paired comparison.
- Frozen green/yellow/red decision rule.

## 6. Results

### 6.1 Controlled compression and allocation

- N08 rank-four Pareto result at 34.03% state.
- O10 rank-eight preservation at 26.56% state and failed layerwise-specificity claim.
- Stacked v1.1 preservation at 25.39% state, small control gains, and slight global-oracle deficit.

### 6.2 External exact-credit fidelity

- Rank sweep and rank-four selection.
- 6.3519x memory reduction.
- Four audit cells, exact-performance gaps, cosine, relative credit error, and post-feedback exposure.

### 6.3 External predictive utility

- Yearbook comparison at each fixed delay.
- Whole-stream Criteo log-loss comparison; delay bins remain descriptive.
- Paired intervals and failed competitive gate.

### 6.4 Systems behavior

- Literal state bytes and peak queue state.
- Peak RAM/VRAM.
- Median and p95 feedback-update latency.
- Full-pass throughput and SVD decomposition cost.

### 6.5 Secondary O10 analyses

- Layerwise and shared variants as secondary results.
- No reversal of the main competitive conclusion.

## 7. Discussion

- Explain why low-rank structure preserved observation-time Jacobians.
- Develop the distinction between approximation fidelity and algorithmic utility.
- Explain how recomputation at label time can exploit the updated model and why this can help under drift.
- Discuss why random projection may act as useful regularization even with lower cosine.
- Derive practical representation-selection rules from feature bytes, credit bytes, delay, recomputation cost, and replay restrictions.
- Explain why the synthetic N08 advantage did not transfer to Yearbook/Criteo.
- Frame the YELLOW gate as the result of the preregistered test.

## 8. Limitations

- Small traced adapter rather than full-network credit.
- Frozen feature producers and declared post-feature state boundary.
- Per-example SVD reference implementation has poor throughput.
- Criteo uses retrospective historical initialization and a policy-defined negative maturity time.
- Two external datasets and one adapter family.
- No theoretical guarantee for the downstream effect of credit reconstruction error.
- The study does not implement Frequent Directions in the returned v1.1 run; the two-sided random sketch is the confirmatory sketch comparator.

## 9. Conclusion

Summarize the measured compression and fidelity, the failed competitive gate, and the resulting guidance: a delayed-credit compressor should be evaluated against replay and sketching at an explicit state boundary, because matching exact credit does not by itself establish the best learner.

## Recommended main-paper displays

1. **Figure 1:** observation-time credit storage versus label-time replay/recomputation timeline.
2. **Figure 2:** rank versus memory reduction, median cosine, and exact-credit performance gap across the four audit cells.
3. **Figure 3:** paired global-SVD advantage with 95% intervals for both comparators and all four confirmatory conditions.
4. **Figure 4:** state bytes versus throughput/latency for global SVD, replay, random projection, and O10 variants.
5. **Table 1:** method definitions and actual retained objects.
6. **Table 2:** controlled N08/O10/stacked results.
7. **Table 3:** external audit and full predictive results.

## TMLR submission notes

- Use the official TMLR LaTeX template and anonymize the manuscript and supplementary archive.
- Do not link the double-blind submission to this author-identified repository. Prepare a separate anonymous supplementary ZIP if code is submitted during review.
- Keep the invalid v1.0 runs in the reproducibility history and exclude them from confirmatory tables.
- State that Criteo is CC BY-NC-SA 4.0 and was downloaded rather than redistributed.
- Use approximately 35 to 50 carefully selected references, with substantial coverage of RTRL approximations, delayed labels, replay, matrix sketching, and conversion-delay modeling.
